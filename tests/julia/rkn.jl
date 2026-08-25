import SciMLBase

using OrdinaryDiffEqRKN:
    DPRKN4,
    DPRKN5,
    DPRKN6,
    DPRKN6FM,
    DPRKN8,
    DPRKN12,
    ERKN4,
    ERKN5,
    ERKN7,
    FineRKN4,
    FineRKN5,
    IRKN3,
    IRKN4,
    Nystrom4,
    Nystrom4VelocityIndependent,
    Nystrom5VelocityIndependent,
    RKN4

function rust_rkn_endpoints()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example rkn_compliance`
    Dict(
        first(fields) => parse.(Float64, fields[2:end])
        for fields in split.(strip.(split(chomp(read(command, String)), '\n')), ',')
    )
end

function rkn_oscillator_endpoint(algorithm)
    function acceleration!(dv, _, q, _, _)
        dv[1] = -q[1]
    end
    problem = SciMLBase.SecondOrderODEProblem(acceleration!, [0.0], [1.0], (0.0, 1.0))
    solution = solve(problem, algorithm; adaptive = false, dt = 0.01, save_everystep = false)
    [only(solution.u[end].x[1]), only(solution.u[end].x[2])]
end

function irkn_oscillator_endpoint(algorithm)
    function acceleration!(dv, _, q, _, _)
        dv[1] = -q[1]
    end
    problem = SciMLBase.SecondOrderODEProblem(acceleration!, [0.0], [1.0], (0.0, 1.0))
    solution = solve(problem, algorithm; adaptive = false, dt = 0.125, save_everystep = false)
    [only(solution.u[end].x[1]), only(solution.u[end].x[2])]
end

function rkn_adaptive_oscillator_endpoint(algorithm)
    function acceleration!(dv, _, q, _, _)
        dv[1] = -q[1]
    end
    problem = SciMLBase.SecondOrderODEProblem(acceleration!, [0.0], [1.0], (0.0, 1.0))
    solution = solve(
        problem, algorithm; adaptive = true, dt = 0.5, dtmax = 0.5,
        abstol = 1.0e-10, reltol = 1.0e-10, save_everystep = false,
    )
    [only(solution.u[end].x[1]), only(solution.u[end].x[2])]
end

function rkn_velocity_dependent_endpoint(algorithm = Nystrom4(); adaptive = false)
    function acceleration!(dv, v, q, _, time)
        dv[1] = -q[1] - 0.2 * v[1] + 0.1 * time
    end
    problem = SciMLBase.SecondOrderODEProblem(acceleration!, [0.25], [1.0], (0.0, 1.0))
    solution = solve(
        problem, algorithm; adaptive, dt = adaptive ? 0.5 : 0.01, dtmax = 0.5,
        abstol = 1.0e-10, reltol = 1.0e-10, save_everystep = false,
    )
    [only(solution.u[end].x[1]), only(solution.u[end].x[2])]
end

function dprkn6_dense_midpoint()
    function acceleration!(dv, _, q, _, _)
        dv[1] = -q[1]
    end
    problem = SciMLBase.SecondOrderODEProblem(acceleration!, [0.0], [1.0], (0.0, 1.0))
    solution = solve(
        problem, DPRKN6(); adaptive = false, dt = 1.0,
        saveat = [0.0, 0.5, 1.0], save_everystep = false,
    )
    [only(solution.u[2].x[1]), only(solution.u[2].x[2])]
end

@testset "fixed Runge-Kutta-Nystrom compliance" begin
    rust = rust_rkn_endpoints()
    julia = Dict(
        "nystrom4" => rkn_oscillator_endpoint(Nystrom4()),
        "nystrom4_velocity_independent" =>
            rkn_oscillator_endpoint(Nystrom4VelocityIndependent()),
        "nystrom5_velocity_independent" =>
            rkn_oscillator_endpoint(Nystrom5VelocityIndependent()),
        "rkn4" => rkn_oscillator_endpoint(RKN4()),
        "nystrom4_velocity_dependent" => rkn_velocity_dependent_endpoint(),
        "dprkn4_fixed" => rkn_oscillator_endpoint(DPRKN4()),
        "dprkn5_fixed" => rkn_oscillator_endpoint(DPRKN5()),
        "dprkn6_fixed" => rkn_oscillator_endpoint(DPRKN6()),
        "dprkn6fm_fixed" => rkn_oscillator_endpoint(DPRKN6FM()),
        "dprkn8_fixed" => rkn_oscillator_endpoint(DPRKN8()),
        "dprkn12_fixed" => rkn_oscillator_endpoint(DPRKN12()),
        "erkn4_fixed" => rkn_oscillator_endpoint(ERKN4()),
        "erkn5_fixed" => rkn_oscillator_endpoint(ERKN5()),
        "erkn7_fixed" => rkn_oscillator_endpoint(ERKN7()),
        "dprkn4_adaptive" => rkn_adaptive_oscillator_endpoint(DPRKN4()),
        "dprkn5_adaptive" => rkn_adaptive_oscillator_endpoint(DPRKN5()),
        "dprkn6_adaptive" => rkn_adaptive_oscillator_endpoint(DPRKN6()),
        "dprkn6fm_adaptive" => rkn_adaptive_oscillator_endpoint(DPRKN6FM()),
        "dprkn8_adaptive" => rkn_adaptive_oscillator_endpoint(DPRKN8()),
        "dprkn12_adaptive" => rkn_adaptive_oscillator_endpoint(DPRKN12()),
        "erkn4_adaptive" => rkn_adaptive_oscillator_endpoint(ERKN4()),
        "erkn5_adaptive" => rkn_adaptive_oscillator_endpoint(ERKN5()),
        "erkn7_adaptive" => rkn_adaptive_oscillator_endpoint(ERKN7()),
        "finerkn4_fixed" => rkn_velocity_dependent_endpoint(FineRKN4()),
        "finerkn5_fixed" => rkn_velocity_dependent_endpoint(FineRKN5()),
        "finerkn4_adaptive" => rkn_velocity_dependent_endpoint(FineRKN4(); adaptive = true),
        "finerkn5_adaptive" => rkn_velocity_dependent_endpoint(FineRKN5(); adaptive = true),
        "dprkn6_dense_midpoint" => dprkn6_dense_midpoint(),
        "irkn3_fixed" => irkn_oscillator_endpoint(IRKN3()),
        "irkn4_fixed" => irkn_oscillator_endpoint(IRKN4()),
    )
    @test Set(keys(rust)) == Set(keys(julia))
    for name in keys(julia)
        if endswith(name, "_adaptive")
            @test isapprox(rust[name], julia[name]; rtol = 2.0e-9, atol = 2.0e-10)
        else
            @test isapprox(rust[name], julia[name]; rtol = 3.0e-12, atol = 3.0e-13)
        end
    end
end
