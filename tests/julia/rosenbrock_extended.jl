using OrdinaryDiffEqRosenbrock: ROS2, ROS3, Rodas3, Rodas4, Rodas5P, Rosenbrock32, RosenbrockW6S4OS

function rust_extended_rosenbrock_endpoints()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example rosenbrock_extended_compliance`
    rows = split(chomp(read(command, String)), '\n')
    Dict(
        first(fields) => parse(Float64, fields[2])
        for fields in split.(strip.(rows), ',')
    )
end

function extended_rosenbrock_reference(algorithm; adaptive)
    function stiff_nonautonomous!(du, u, _, time)
        du[1] = -1000.0 * (u[1] - cos(time)) - sin(time)
    end
    if adaptive
        problem = ODEProblem(stiff_nonautonomous!, [1.0], (0.0, 1.0))
        solution = solve(
            problem,
            algorithm;
            abstol = 1.0e-8,
            reltol = 1.0e-8,
            dt = 0.1,
            save_everystep = false,
        )
    else
        function exponential!(du, u, _, _)
            du[1] = u[1]
        end
        problem = ODEProblem(exponential!, [1.0], (0.0, 1.0))
        solution = solve(
            problem,
            algorithm;
            adaptive = false,
            dt = 0.01,
            save_everystep = false,
        )
    end
    only(solution.u[end])
end

@testset "Extended Rosenbrock compliance" begin
    rust = rust_extended_rosenbrock_endpoints()
    algorithms = Dict(
        "ros2" => ROS2(),
        "rodas3" => Rodas3(),
        "rosenbrock32" => Rosenbrock32(),
        "rodas4" => Rodas4(),
        "rodas5p" => Rodas5P(),
        "rosenbrockw6s4os" => RosenbrockW6S4OS(),
    )
    julia = Dict{String, Float64}()
    for (name, algorithm) in algorithms
        if name == "rosenbrockw6s4os"
            julia["$(name)_fixed"] = extended_rosenbrock_reference(algorithm; adaptive = false)
            continue
        end
        julia["$(name)_adaptive"] = extended_rosenbrock_reference(algorithm; adaptive = true)
        julia["$(name)_fixed"] = extended_rosenbrock_reference(algorithm; adaptive = false)
    end
    julia["ros3_adaptive"] = extended_rosenbrock_reference(ROS3(); adaptive = true)

    @test Set(keys(rust)) == Set(keys(julia))
    for name in keys(julia)
        @testset "$name endpoint" begin
            if endswith(name, "_fixed")
                @test rust[name] ≈ julia[name] rtol = 8.0e-12 atol = 8.0e-13
            else
                # Controller details differ, but both endpoints are independently
                # tolerance controlled using the same embedded estimator.
                @test rust[name] ≈ julia[name] rtol = 4.0e-6 atol = 5.0e-8
            end
            if endswith(name, "_fixed")
                @test rust[name] ≈ ℯ rtol = 3.0e-5 atol = 5.0e-8
            else
                @test rust[name] ≈ cos(1.0) rtol = 4.0e-6 atol = 5.0e-8
            end
        end
    end
end
