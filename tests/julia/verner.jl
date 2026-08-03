using OrdinaryDiffEqVerner: Vern6, Vern7, Vern8, Vern9

function rust_verner_endpoints()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example verner_compliance`
    rows = split(chomp(read(command, String)), '\n')
    Dict(
        first(fields) => parse.(Float64, fields[2:end])
        for fields in split.(strip.(rows), ',')
    )
end

function verner_reference(algorithm; adaptive)
    function oscillator!(du, u, _, time)
        du[1] = u[2]
        du[2] = -u[1] + 0.1 * time
    end
    problem = ODEProblem(oscillator!, [1.0, 0.0], (0.0, 2.0))
    if adaptive
        solution = solve(
            problem,
            algorithm;
            abstol = 1.0e-10,
            reltol = 1.0e-10,
            dt = 0.5,
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
    collect(solution.u[end])
end

@testset "Verner-family compliance" begin
    rust = rust_verner_endpoints()
    algorithms = Dict(
        "vern6" => Vern6(),
        "vern7" => Vern7(),
        "vern8" => Vern8(),
        "vern9" => Vern9(),
    )
    julia = Dict{String, Vector{Float64}}()
    for (name, algorithm) in algorithms
        julia["$(name)_adaptive"] = verner_reference(algorithm; adaptive = true)
        julia["$(name)_fixed"] = verner_reference(algorithm; adaptive = false)
    end

    @test Set(keys(rust)) == Set(keys(julia))
    for name in keys(julia)
        @testset "$name endpoint" begin
            if endswith(name, "_fixed")
                @test rust[name] ≈ julia[name] rtol = 4.0e-13 atol = 4.0e-14
            else
                # The repositories use different step-size controllers; compare
                # both independently tolerance-controlled endpoints.
                @test rust[name] ≈ julia[name] rtol = 2.0e-9 atol = 2.0e-10
            end
        end
    end
end
