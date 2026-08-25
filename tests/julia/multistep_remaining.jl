import SciMLBase

using OrdinaryDiffEqAdamsBashforthMoulton: VCABM
using OrdinaryDiffEqBDF: IMEXEuler, IMEXEulerARK, SBDF, SBDF2, SBDF3, SBDF4
using OrdinaryDiffEqIMEXMultistep: CNAB2, CNLF2
using SciMLBase: SplitODEProblem

function rust_remaining_multistep_rows()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example multistep_remaining_compliance`
    Dict(fields[1] => parse(Float64, fields[2]) for fields in split.(strip.(split(chomp(read(command, String)), '\n')), ','))
end

function remaining_variable_problem()
    function rhs!(du, u, _, time)
        du[1] = u[1] + time
    end
    ODEProblem(rhs!, [1.0], (0.0, 1.0))
end

function remaining_split_problem()
    function implicit!(du, u, _, _)
        du[1] = -2.0 * u[1]
    end
    function explicit!(du, u, _, time)
        du[1] = 0.5 * u[1] + sin(time)
    end
    SciMLBase.SplitODEProblem(implicit!, explicit!, [1.0], (0.0, 1.0))
end

@testset "Remaining Adams/BDF/IMEX multistep compliance" begin
    # Formulae are pinned to OrdinaryDiffEq.jl commit
    # 211142263781255a9aa2f910f6760b9f18ec29c8.
    rust = rust_remaining_multistep_rows()
    algorithms = Dict(
        "imex_euler" => IMEXEuler(),
        "imex_euler_ark" => IMEXEulerARK(),
        "sbdf" => SBDF(2),
        "sbdf2" => SBDF2(),
        "sbdf3" => SBDF3(),
        "sbdf4" => SBDF4(),
        "cnab2" => CNAB2(),
        "cnlf2" => CNLF2(),
    )
    @test Set(keys(rust)) == union(Set(keys(algorithms)), Set(["vcabm"]))

    variable = remaining_variable_problem()
    julia_vcabm = solve(
        variable,
        VCABM();
        abstol = 1.0e-9,
        reltol = 1.0e-9,
        dt = 0.001,
        dtmax = 0.05,
        save_everystep = false,
    )
    exact_variable = 2exp(1.0) - 2
    @test rust["vcabm"] ≈ exact_variable rtol = 3.0e-6 atol = 3.0e-9
    @test only(julia_vcabm.u[end]) ≈ exact_variable rtol = 4.0e-5 atol = 3.0e-9
    @test rust["vcabm"] ≈ only(julia_vcabm.u[end]) rtol = 4.0e-5 atol = 6.0e-9

    problem = remaining_split_problem()
    exact_split = exp(-1.5) + (1.5sin(1.0) - cos(1.0)) / 3.25 + exp(-1.5) / 3.25
    for (name, algorithm) in algorithms
        @testset "$name endpoint" begin
            solution = solve(
                problem,
                algorithm;
                adaptive = false,
                dt = 0.0025,
                save_everystep = false,
            )
            julia_endpoint = only(solution.u[end])
            tolerance = name in ("sbdf3", "sbdf4") ? 2.0e-1 :
                (startswith(name, "imex_euler") ? 5.0e-3 : 5.0e-5)
            @test rust[name] ≈ exact_split rtol = tolerance atol = tolerance
            @test julia_endpoint ≈ exact_split rtol = tolerance atol = tolerance
            @test rust[name] ≈ julia_endpoint rtol = 2.0e-10 atol = 2.0e-11
        end
    end
end
