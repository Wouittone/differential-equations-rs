using OrdinaryDiffEqBDF: QBDF2, QNDF2

function rust_qndf2_results()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example qndf2_compliance`
    Dict(first(fields) => fields[2:end] for fields in split.(strip.(split(chomp(read(command, String)), '\n')), ','))
end

@testset "QNDF2 regular ODE compliance" begin
    rust = rust_qndf2_results()
    function stiff!(du, u, _, t)
        du[1] = -15.0 * (u[1] - cos(t)) - sin(t)
    end
    problem = ODEProblem(stiff!, [1.0], (0.0, 1.0))
    fixed = solve(problem, QNDF2(); adaptive = false, dt = 0.01, save_everystep = false)
    @test parse(Float64, only(rust["qndf2_fixed"])) ≈ only(fixed.u[end]) rtol = 3.0e-4 atol = 3.0e-7
    adaptive = solve(problem, QNDF2(); abstol = 1.0e-8, reltol = 1.0e-8, save_everystep = false)
    @test parse(Float64, rust["qndf2"][1]) ≈ only(adaptive.u[end]) rtol = 8.0e-4 atol = 8.0e-7
    @test parse(Float64, rust["qndf2"][1]) ≈ cos(1.0) rtol = 8.0e-4 atol = 8.0e-7
    @test parse(Int, rust["qndf2"][2]) > 0

    qbdf_fixed = solve(problem, QBDF2(); adaptive = false, dt = 0.01, save_everystep = false)
    @test parse(Float64, only(rust["qbdf2_fixed"])) ≈ only(qbdf_fixed.u[end]) rtol = 3.0e-4 atol = 3.0e-7
    qbdf_adaptive = solve(problem, QBDF2(); abstol = 1.0e-8, reltol = 1.0e-8, save_everystep = false)
    @test parse(Float64, rust["qbdf2"][1]) ≈ only(qbdf_adaptive.u[end]) rtol = 8.0e-4 atol = 8.0e-7
    @test parse(Int, rust["qbdf2"][2]) > 0
end
