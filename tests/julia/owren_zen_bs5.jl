using OrdinaryDiffEqLowOrderRK: BS5, OwrenZen3, OwrenZen4, OwrenZen5

function rust_owren_zen_bs5_endpoints()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example owren_zen_bs5_compliance`
    rows = split(chomp(read(command, String)), '\n')
    Dict(
        first(fields) => parse(Float64, fields[2])
        for fields in split.(strip.(rows), ',')
    )
end

function high_order_exponential_reference(algorithm; adaptive)
    function exponential!(du, u, _, _)
        du[1] = u[1]
    end
    problem = ODEProblem(exponential!, [1.0], (0.0, 1.0))
    if adaptive
        solution = solve(
            problem,
            algorithm;
            abstol = 1.0e-9,
            reltol = 1.0e-9,
            save_everystep = false,
        )
    else
        solution = solve(
            problem,
            algorithm;
            adaptive = false,
            dt = 0.05,
            save_everystep = false,
        )
    end
    only(solution.u[end])
end

@testset "Owren-Zennaro and BS5 compliance" begin
    rust = rust_owren_zen_bs5_endpoints()
    algorithms = Dict(
        "owren_zen3" => OwrenZen3(),
        "owren_zen4" => OwrenZen4(),
        "owren_zen5" => OwrenZen5(),
        "bs5" => BS5(),
    )
    julia = Dict{String, Float64}()
    for (name, algorithm) in algorithms
        julia["$(name)_adaptive"] = high_order_exponential_reference(algorithm; adaptive = true)
        julia["$(name)_fixed"] = high_order_exponential_reference(algorithm; adaptive = false)
    end

    @test Set(keys(rust)) == Set(keys(julia))
    for name in keys(julia)
        @testset "$name endpoint" begin
            if endswith(name, "_fixed")
                @test rust[name] ≈ julia[name] rtol = 2.0e-13 atol = 2.0e-14
            else
                @test rust[name] ≈ julia[name] rtol = 2.0e-7 atol = 2.0e-9
            end
            @test rust[name] ≈ ℯ rtol = 3.0e-5 atol = 2.0e-9
        end
    end
end
