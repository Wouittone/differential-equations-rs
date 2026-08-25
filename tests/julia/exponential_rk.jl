import SciMLBase

using OrdinaryDiffEqExponentialRK: EPIRK4s3A, EPIRK4s3B, EPIRK5P1, EPIRK5P2,
    EPIRK5s3, ETD1, ETD2, ETDRK2, ETDRK3, ETDRK4, EXPRB53s3, Exp4, Exprb32,
    Exprb43, HochOst4, LawsonEuler, NorsettEuler
using SciMLBase: ODEFunction, SplitFunction, SplitODEProblem
using SciMLOperators: ScalarOperator

function rust_exponential_rk_results()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example exponential_compliance`
    rows = split(chomp(read(command, String)), '\n')
    Dict(first(fields) => parse(Float64, fields[2]) for fields in split.(strip.(rows), ','))
end

function exponential_linear_problem()
    function rhs!(du, u, _, _)
        du[1] = -2.0 * u[1]
    end
    function jac!(jacobian, _, _, _)
        jacobian[1, 1] = -2.0
    end
    ODEProblem(ODEFunction(rhs!; jac = jac!), [1.0], (0.0, 1.0))
end

function exponential_endpoint(algorithm)
    solution = solve(
        exponential_linear_problem(),
        algorithm;
        adaptive = false,
        dt = 0.2,
        save_everystep = false,
    )
    only(solution.u[end])
end

function etd2_endpoint()
    function nonlinear!(du, _, _, _)
        fill!(du, 0.0)
    end
    split = SplitFunction(ScalarOperator(-2.0), nonlinear!)
    problem = SciMLBase.SplitODEProblem(split, [1.0], (0.0, 1.0))
    solution = solve(
        problem,
        ETD2();
        adaptive = false,
        dt = 0.2,
        save_everystep = false,
    )
    only(solution.u[end])
end

@testset "Exponential Runge--Kutta compliance" begin
    rust = rust_exponential_rk_results()
    algorithms = Dict(
        "lawson_euler" => LawsonEuler(krylov = true),
        "norsett_euler" => NorsettEuler(krylov = true),
        "etd1" => ETD1(krylov = true),
        "etdrk2" => ETDRK2(krylov = true),
        "etdrk3" => ETDRK3(krylov = true),
        "etdrk4" => ETDRK4(krylov = true),
        "hoch_ost4" => HochOst4(krylov = true),
        "exp4" => Exp4(),
        "epirk4s3a" => EPIRK4s3A(),
        "epirk4s3b" => EPIRK4s3B(),
        "epirk5s3" => EPIRK5s3(),
        "exprb53s3" => EXPRB53s3(),
        "epirk5p1" => EPIRK5P1(),
        "epirk5p2" => EPIRK5P2(),
        "exprb32" => Exprb32(),
        "exprb43" => Exprb43(),
    )
    @test Set(keys(rust)) == union(Set(keys(algorithms)), Set(["etd2"]))
    exact = exp(-2.0)
    julia_etd2 = etd2_endpoint()
    @test rust["etd2"] ≈ julia_etd2 rtol = 2.0e-11 atol = 2.0e-13
    @test julia_etd2 ≈ exact rtol = 2.0e-11 atol = 2.0e-13
    for (name, algorithm) in algorithms
        julia = exponential_endpoint(algorithm)
        @test rust[name] ≈ julia rtol = 2.0e-11 atol = 2.0e-13
        @test julia ≈ exact rtol = 2.0e-11 atol = 2.0e-13
    end
end
