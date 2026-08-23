using OrdinaryDiffEqPRK: KuttaPRK2p5
using OrdinaryDiffEqQPRK: QPRK98

function rust_parallel_rk_results()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example parallel_rk_compliance`
    rows = split(chomp(read(command, String)), '\n')
    Dict(
        first(fields) => parse.(Float64, fields[2:end])
        for fields in split.(strip.(rows), ',')
    )
end

function parallel_rk_nonautonomous()
    function rhs!(du, u, _, time)
        du[1] = u[1] + time
    end
    ODEProblem(rhs!, [1.0], (0.0, 1.0))
end

function parallel_rk_oscillator()
    function rhs!(du, u, _, time)
        du[1] = u[2]
        du[2] = -u[1] + 0.1 * time
    end
    ODEProblem(rhs!, [1.0, 0.0], (0.0, 2.0))
end

function parallel_rk_fixed_endpoint(algorithm)
    solution = solve(
        parallel_rk_nonautonomous(),
        algorithm;
        adaptive = false,
        dt = 0.01,
        save_everystep = false,
    )
    only(solution.u[end])
end

@testset "Parallel explicit Runge--Kutta compliance" begin
    rust = rust_parallel_rk_results()

    @test only(rust["kutta_prk2p5_fixed"]) ≈ parallel_rk_fixed_endpoint(KuttaPRK2p5()) rtol = 4.0e-13 atol = 5.0e-14
    # QPRK98 is intended for quadruple precision. The exact large rational
    # coefficients cancel heavily in Float64, so Rust/Julia operation-order
    # differences appear around the last 9--10 reliable digits.
    @test only(rust["qprk98_fixed"]) ≈ parallel_rk_fixed_endpoint(QPRK98()) rtol = 1.0e-10 atol = 5.0e-12

    julia_adaptive = solve(
        parallel_rk_oscillator(),
        QPRK98();
        abstol = 1.0e-10,
        reltol = 1.0e-10,
        dt = 0.25,
        save_everystep = false,
    )
    @test rust["qprk98_adaptive"] ≈ collect(julia_adaptive.u[end]) rtol = 1.0e-8 atol = 2.0e-10
end
