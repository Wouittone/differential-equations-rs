using OrdinaryDiffEqSDIRK: ImplicitEuler, ImplicitMidpoint, Trapezoid

function rust_implicit_endpoints()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example implicit_compliance`
    rows = split(chomp(read(command, String)), '\n')
    Dict(
        first(fields) => parse.(Float64, fields[2:end])
        for fields in split.(strip.(rows), ',')
    )
end

function implicit_reference(algorithm)
    function stiff_linear!(du, u, _, _)
        du[1] = -10.0 * u[1] + u[2]
        du[2] = -u[2]
    end
    problem = ODEProblem(stiff_linear!, [1.0, 1.0], (0.0, 1.0))
    solution = solve(
        problem,
        algorithm;
        adaptive = false,
        dt = 0.01,
        save_everystep = false,
    )
    solution.u[end]
end

@testset "Fixed implicit ODE compliance" begin
    rust = rust_implicit_endpoints()
    julia = Dict(
        "implicit_euler" => implicit_reference(ImplicitEuler()),
        "implicit_midpoint" => implicit_reference(ImplicitMidpoint()),
        "trapezoid" => implicit_reference(Trapezoid()),
    )

    @test Set(keys(rust)) == Set(keys(julia))
    for name in keys(julia)
        @testset "$name endpoint" begin
            @test rust[name] ≈ julia[name] rtol = 2.0e-9 atol = 2.0e-11
        end
    end
end
