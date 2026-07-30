using OrdinaryDiffEqAdamsBashforthMoulton: AB3, AB4, AB5, ABM32, ABM43, ABM54

function rust_adams_endpoints()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example adams_compliance`
    rows = split(chomp(read(command, String)), '\n')
    Dict(
        first(fields) => parse(Float64, fields[2])
        for fields in split.(strip.(rows), ',')
    )
end

function adams_reference(algorithm)
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
    only(solution.u[end])
end

@testset "Fixed Adams compliance" begin
    rust = rust_adams_endpoints()
    julia = Dict(
        "ab3" => adams_reference(AB3()),
        "ab4" => adams_reference(AB4()),
        "ab5" => adams_reference(AB5()),
        "abm32" => adams_reference(ABM32()),
        "abm43" => adams_reference(ABM43()),
        "abm54" => adams_reference(ABM54()),
    )

    @test Set(keys(rust)) == Set(keys(julia))
    for name in keys(julia)
        @testset "$name endpoint" begin
            @test rust[name] ≈ julia[name] rtol = 2.0e-10 atol = 2.0e-12
            @test rust[name] ≈ ℯ rtol = 2.0e-6
        end
    end
end
