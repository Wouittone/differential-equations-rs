using OrdinaryDiffEqBDF: QBDF1, QNDF1

function rust_qndf1_results()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example qndf1_compliance`
    Dict(
        first(fields) => fields[2:end]
        for fields in split.(strip.(split(chomp(read(command, String)), '\n')), ',')
    )
end

@testset "QNDF1 regular ODE compliance" begin
    rust = rust_qndf1_results()
    function stiff!(du, u, _, t)
        du[1] = -15.0 * (u[1] - cos(t)) - sin(t)
    end
    problem = ODEProblem(stiff!, [1.0], (0.0, 1.0))
    fixed = solve(problem, QNDF1(); adaptive = false, dt = 0.01, save_everystep = false)
    # Both implementations are order-one QNDF1; the pinned nonlinear solve
    # and the Rust frozen Newton tolerance differ by a small O(dt) term.
    @test parse(Float64, only(rust["qndf1_fixed"])) ≈ only(fixed.u[end]) rtol = 2.0e-4 atol = 2.0e-7

    adaptive = solve(
        problem,
        QNDF1();
        abstol = 1.0e-8,
        reltol = 1.0e-8,
        save_everystep = false,
    )
    @test parse(Float64, rust["qndf1"][1]) ≈ only(adaptive.u[end]) rtol = 5.0e-4 atol = 5.0e-7
    @test parse(Float64, rust["qndf1"][1]) ≈ cos(1.0) rtol = 5.0e-4 atol = 5.0e-7
    @test parse(Int, rust["qndf1"][2]) > 0

    qbdf_fixed = solve(problem, QBDF1(); adaptive = false, dt = 0.01, save_everystep = false)
    @test parse(Float64, only(rust["qbdf1_fixed"])) ≈ only(qbdf_fixed.u[end]) rtol = 2.0e-4 atol = 2.0e-7
    qbdf_adaptive = solve(
        problem,
        QBDF1();
        abstol = 1.0e-8,
        reltol = 1.0e-8,
        save_everystep = false,
    )
    @test parse(Float64, rust["qbdf1"][1]) ≈ only(qbdf_adaptive.u[end]) rtol = 5.0e-4 atol = 5.0e-7
    @test parse(Int, rust["qbdf1"][2]) > 0
end
