using OrdinaryDiffEqTaylorSeries: ExplicitTaylor, ExplicitTaylor2, ExplicitTaylorAdaptiveOrder
using SciMLBase: ODEProblem, solve

function rust_taylor_results()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example taylor_compliance`
    rows = split(chomp(read(command, String)), '\n')
    Dict(first(fields) => parse(Float64, fields[2]) for fields in split.(strip.(rows), ','))
end

function taylor_endpoint(algorithm, dt)
    rhs! = (du, u, _, _) -> (du[1] = u[1])
    problem = ODEProblem(rhs!, [1.0], (0.0, 1.0))
    only(solve(problem, algorithm; adaptive = false, dt, save_everystep = false).u[end])
end

@testset "Taylor-series pinned compliance" begin
    rust = rust_taylor_results()
    @test Set(keys(rust)) == Set([
        "ExplicitTaylor2", "ExplicitTaylor", "ExplicitTaylorAdaptiveOrder",
    ])
    @test isapprox(rust["ExplicitTaylor2"], taylor_endpoint(ExplicitTaylor2(), 0.01); rtol = 2.0e-11, atol = 2.0e-13)
    @test isapprox(rust["ExplicitTaylor"], taylor_endpoint(ExplicitTaylor(order = Val(8)), 0.1); rtol = 2.0e-11, atol = 2.0e-13)
    @test isapprox(
        rust["ExplicitTaylorAdaptiveOrder"],
        taylor_endpoint(ExplicitTaylorAdaptiveOrder(min_order = Val(6), max_order = Val(7)), 0.1);
        rtol = 2.0e-9, atol = 2.0e-11,
    )
end
