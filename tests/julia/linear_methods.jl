import SciMLBase

using OrdinaryDiffEqLinear: CayleyEuler, CG2, CG3, CG4a, LieEuler, LieRK4,
    LinearExponential, MagnusAdapt4, MagnusGauss4, MagnusGL4, MagnusGL6,
    MagnusGL8, MagnusLeapfrog, MagnusMidpoint, MagnusNC6, MagnusNC8, RKMK2, RKMK4
using SciMLBase: ODEProblem, SplitFunction, SplitODEProblem
using SciMLOperators: MatrixOperator
using LinearAlgebra: mul!

function rust_linear_method_results()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example linear_methods_compliance`
    rows = split(chomp(read(command, String)), '\n')
    Dict(first(fields) => parse.(Float64, fields[2:end]) for fields in split.(strip.(rows), ','))
end

function constant_generator()
    MatrixOperator([0.0 -1.0; 1.0 0.0])
end

function vector_endpoint(algorithm)
    problem = ODEProblem(constant_generator(), [1.0, 0.0], (0.0, 1.0))
    solution = solve(problem, algorithm; adaptive = false, dt = 0.2, save_everystep = false)
    collect(solution.u[end])
end

function cayley_endpoint()
    generator = constant_generator()
    right! = (du, u, _, _) -> mul!(du, u, [0.0 -1.0; 1.0 0.0], -1.0, 0.0)
    split = SplitFunction(generator, right!, _func_cache = zeros(2, 2))
    problem = SciMLBase.SplitODEProblem(split, [2.0 0.5; 0.5 -1.0], (0.0, 1.0))
    solution = solve(problem, CayleyEuler(); adaptive = false, dt = 0.2, save_everystep = false)
    vec(copy(solution.u[end]))
end

@testset "Exact linear and Lie-group method compliance" begin
    rust = rust_linear_method_results()
    algorithms = Dict(
        "lie_euler" => LieEuler(),
        "linear_exponential" => LinearExponential(krylov = :off),
        "magnus_midpoint" => MagnusMidpoint(),
        "magnus_leapfrog" => MagnusLeapfrog(),
        "rkmk2" => RKMK2(),
        "rkmk4" => RKMK4(),
        "lie_rk4" => LieRK4(),
        "cg2" => CG2(),
        "cg3" => CG3(),
        "cg4a" => CG4a(),
        "magnus_adapt4" => MagnusAdapt4(),
        "magnus_gauss4" => MagnusGauss4(),
        "magnus_gl4" => MagnusGL4(),
        "magnus_gl6" => MagnusGL6(),
        "magnus_nc6" => MagnusNC6(),
        "magnus_gl8" => MagnusGL8(),
        "magnus_nc8" => MagnusNC8(),
    )
    @test Set(keys(rust)) == union(Set(keys(algorithms)), Set(["cayley_euler"]))
    for (name, algorithm) in algorithms
        julia = vector_endpoint(algorithm)
        @test rust[name] ≈ julia rtol = 3.0e-11 atol = 3.0e-13
    end
    @test rust["cayley_euler"] ≈ cayley_endpoint() rtol = 3.0e-11 atol = 3.0e-13
end
