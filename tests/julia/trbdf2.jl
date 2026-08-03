using OrdinaryDiffEqSDIRK: TRBDF2

function rust_trbdf2_results()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example trbdf2_compliance`
    Dict(
        first(fields) => fields[2:end]
        for fields in split.(strip.(split(chomp(read(command, String)), '\n')), ',')
    )
end

function trbdf2_vector_problem()
    function rhs!(du, u, _, time)
        du[1] = -30.0 * (u[1] - cos(time)) - sin(time)
        du[2] = -2.0 * u[2] + time
    end
    ODEProblem(rhs!, [1.0, 0.0], (0.0, 1.0))
end

@testset "TRBDF2 compliance" begin
    rust = rust_trbdf2_results()

    function stiff!(du, u, _, time)
        du[1] = -1000.0 * (u[1] - cos(time)) - sin(time)
    end
    stiff = ODEProblem(stiff!, [1.0], (0.0, 1.0))
    adaptive = solve(
        stiff,
        TRBDF2();
        abstol = 1.0e-7,
        reltol = 1.0e-7,
        save_everystep = false,
    )
    rust_adaptive = parse(Float64, rust["adaptive"][1])
    @test rust_adaptive ≈ only(adaptive.u[end]) rtol = 4.0e-6 atol = 4.0e-9
    @test rust_adaptive ≈ cos(1.0) rtol = 4.0e-6 atol = 4.0e-9
    @test parse(Int, rust["adaptive"][2]) > 0

    fixed = solve(
        trbdf2_vector_problem(),
        TRBDF2();
        adaptive = false,
        dt = 0.025,
        save_everystep = false,
    )
    rust_fixed = parse.(Float64, rust["fixed"])
    @test rust_fixed ≈ collect(fixed.u[end]) rtol = 2.0e-10 atol = 2.0e-12

    backward_problem = ODEProblem((u, _, _) -> u, [exp(1.0)], (1.0, 0.0))
    backward = solve(
        backward_problem,
        TRBDF2();
        adaptive = false,
        dt = 0.025,
        save_everystep = false,
    )
    @test parse(Float64, only(rust["backward"])) ≈ only(backward.u[end]) rtol = 2.0e-10 atol = 2.0e-12
end
