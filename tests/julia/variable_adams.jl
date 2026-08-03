using OrdinaryDiffEqAdamsBashforthMoulton: VCAB3, VCAB4, VCAB5, VCABM3, VCABM4, VCABM5

function rust_variable_adams_rows()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example variable_adams_compliance`
    rows = split(chomp(read(command, String)), '\n')
    Dict(
        (fields[1], fields[2]) => (
            values = parse.(Float64, split(fields[3], ';')),
            accepted = parse(Int, fields[4]),
            rejected = parse(Int, fields[5]),
        )
        for fields in split.(strip.(rows), ',')
    )
end

function variable_adams_problems()
    function exponential!(du, u, _, _)
        du[1] = u[1]
    end
    function vector_nonautonomous!(du, u, _, time)
        du[1] = -0.4 * u[1] + sin(time)
        du[2] = u[1] - 0.2 * u[2] + cos(time)
    end
    Dict(
        "forward" => ODEProblem(exponential!, [1.0], (0.0, 1.0)),
        "vector" => ODEProblem(vector_nonautonomous!, [0.3, -0.7], (0.0, 2.0)),
    )
end

function solve_variable_adams(problem, algorithm)
    solve(
        problem,
        algorithm;
        abstol = 1.0e-9,
        reltol = 1.0e-9,
        dt = 0.013,
        dtmax = 0.2,
        save_everystep = false,
    )
end

@testset "Variable-coefficient Adams compliance" begin
    # Algorithms and formulae are pinned to OrdinaryDiffEq.jl commit
    # 211142263781255a9aa2f910f6760b9f18ec29c8. The current Julia VCAB3
    # controller has a noticeably larger global error on exp(u) than its
    # requested local tolerance; test each port against an independent Tsit5
    # reference as well as against the corresponding pinned-family endpoint.
    rust = rust_variable_adams_rows()
    problems = variable_adams_problems()
    algorithms = Dict(
        "vcab3" => VCAB3(),
        "vcab4" => VCAB4(),
        "vcab5" => VCAB5(),
        "vcabm3" => VCABM3(),
        "vcabm4" => VCABM4(),
        "vcabm5" => VCABM5(),
    )

    @test Set(keys(rust)) == Set((name, case) for name in keys(algorithms) for case in keys(problems))
    for (name, algorithm) in algorithms, (case, problem) in problems
        @testset "$name $case" begin
            row = rust[(name, case)]
            julia_solution = solve_variable_adams(problem, algorithm)
            julia_endpoint = collect(julia_solution.u[end])
            reference = reference_endpoint(problem)
            rust_rtol = startswith(name, "vcabm") ? 3.0e-6 : 5.0e-8
            julia_rtol = name == "vcab3" ? 2.0e-5 : (startswith(name, "vcabm") ? 3.0e-6 : 5.0e-8)

            @test length(row.values) == length(reference)
            @test row.values ≈ reference rtol = rust_rtol atol = 2.0e-10
            @test julia_endpoint ≈ reference rtol = julia_rtol atol = 2.0e-10
            @test row.values ≈ julia_endpoint rtol = max(rust_rtol, julia_rtol) atol = 3.0e-10
            @test row.accepted > 0
            @test row.rejected >= 0
        end
    end
end
