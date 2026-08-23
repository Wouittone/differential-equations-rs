using OrdinaryDiffEqBDF: FBDF, QBDF, QNDF

function rust_variable_bdf_results()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example bdf_variable_compliance`
    Dict(first(fields) => fields[2:end] for fields in split.(strip.(split(chomp(read(command, String)), '\n')), ','))
end

@testset "variable-order regular ODE BDF compliance" begin
    rust = rust_variable_bdf_results()
    function stiff!(du, u, _, t)
        du[1] = -15.0 * (u[1] - cos(t)) - sin(t)
    end
    problem = ODEProblem(stiff!, [1.0], (0.0, 1.0))

    for (name, algorithm) in (("qndf", QNDF()), ("qbdf", QBDF()), ("fbdf", FBDF()))
        fixed = solve(problem, algorithm; adaptive = false, dt = 0.01, save_everystep = false)
        adaptive = solve(
            problem,
            algorithm;
            abstol = 1.0e-8,
            reltol = 1.0e-8,
            save_everystep = false,
        )
        @test parse(Float64, rust[name][1]) ≈ only(fixed.u[end]) rtol = 5.0e-4 atol = 5.0e-7
        @test parse(Float64, rust[name][2]) ≈ only(adaptive.u[end]) rtol = 5.0e-6 atol = 5.0e-8
        @test parse(Float64, rust[name][2]) ≈ cos(1.0) rtol = 5.0e-6 atol = 5.0e-8
        @test parse(Int, rust[name][3]) > 0
    end
end
