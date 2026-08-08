using OrdinaryDiffEqBDF: ABDF2

function rust_abdf2_results()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example abdf2_compliance`
    Dict(
        first(fields) => fields[2:end]
        for fields in split.(strip.(split(chomp(read(command, String)), '\n')), ',')
    )
end

@testset "ABDF2 regular ODE compliance" begin
    rust = rust_abdf2_results()
    function stiff!(du, u, _, t)
        du[1] = -15.0 * (u[1] - cos(t)) - sin(t)
    end
    problem = ODEProblem(stiff!, [1.0], (0.0, 1.0))
    fixed = solve(problem, ABDF2(); adaptive = false, dt = 0.01, save_everystep = false)
    @test parse(Float64, only(rust["abdf2_fixed"])) ≈ only(fixed.u[end]) rtol = 2.0e-10 atol = 2.0e-12

    adaptive = solve(
        problem,
        ABDF2();
        abstol = 1.0e-8,
        reltol = 1.0e-8,
        save_everystep = false,
    )
    @test parse(Float64, rust["abdf2"][1]) ≈ only(adaptive.u[end]) rtol = 1.0e-5 atol = 5.0e-9
    @test parse(Float64, rust["abdf2"][1]) ≈ cos(1.0) rtol = 1.0e-5 atol = 5.0e-9
    @test parse(Int, rust["abdf2"][2]) > 0
end
