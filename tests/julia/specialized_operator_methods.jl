import SciMLBase

using OrdinaryDiffEqAMF: AMF, build_amf_function
using OrdinaryDiffEqRKIP: RKIP
using OrdinaryDiffEqStabilizedIRK: IRKC
using OrdinaryDiffEqRosenbrock: Rosenbrock23
using SciMLBase: ODEProblem, SplitFunction, SplitODEProblem
using SciMLOperators: MatrixOperator

import LinearAlgebra

# Pinned OrdinaryDiffEqRKIP's own MatrixOperator tests provide this dispatch;
# SciMLOperators otherwise returns the backing matrix instead of an operator.
LinearAlgebra.exp(operator::MatrixOperator, t) = MatrixOperator(exp(t * operator.A))

function rust_specialized_operator_results()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example specialized_operator_compliance`
    rows = split(chomp(read(command, String)), '\n')
    Dict(first(fields) => parse(Float64, fields[2]) for fields in split.(strip.(rows), ','))
end

function amf_endpoint()
    rhs! = (du, u, _, _) -> (du[1] = -3.0u[1])
    jac = MatrixOperator(reshape([-3.0], 1, 1))
    j1 = MatrixOperator(reshape([-1.0], 1, 1))
    j2 = MatrixOperator(reshape([-2.0], 1, 1))
    amf_function = build_amf_function(rhs!; jac = jac, split = (j1, j2))
    problem = ODEProblem(amf_function, [1.0], (0.0, 0.5))
    only(solve(problem, AMF(Rosenbrock23); adaptive = false, dt = 0.01, save_everystep = false).u[end])
end

function rkip_endpoint()
    linear = MatrixOperator(reshape([-2.0], 1, 1))
    nonlinear! = (du, _, _, _) -> (du[1] = 1.0)
    problem = SciMLBase.SplitODEProblem(SplitFunction(linear, nonlinear!), [1.0], (0.0, 1.0))
    only(solve(problem, RKIP(0.1, 0.2; nb_of_cache_step = 2); adaptive = false, dt = 0.1, save_everystep = false).u[end])
end

function irkc_endpoint()
    implicit! = (du, u, _, _) -> (du[1] = -u[1])
    explicit! = (du, u, _, _) -> (du[1] = -100.0u[1])
    problem = SciMLBase.SplitODEProblem(SplitFunction(implicit!, explicit!), [1.0], (0.0, 0.1))
    estimate! = integrator -> (integrator.eigen_est = 100.0)
    only(solve(problem, IRKC(eigen_est = estimate!); adaptive = false, dt = 0.001, save_everystep = false).u[end])
end

@testset "AMF, RKIP, and IRKC pinned compliance" begin
    rust = rust_specialized_operator_results()
    @test Set(keys(rust)) == Set(["amf", "rkip", "irkc"])
    @test isapprox(rust["amf"], amf_endpoint(); rtol = 2.0e-8, atol = 2.0e-11)
    @test isapprox(rust["rkip"], rkip_endpoint(); rtol = 3.0e-11, atol = 3.0e-13)
    @test isapprox(rust["irkc"], irkc_endpoint(); rtol = 4.0e-3, atol = 3.0e-9)
end
