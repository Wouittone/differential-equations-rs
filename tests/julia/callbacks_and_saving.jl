using SciMLBase: ContinuousCallback, DiscreteCallback, terminate!
using OrdinaryDiffEqLowOrderRK: RK4

function rust_callback_rows()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example callback_compliance`
    Dict(
        first(fields) => parse.(Float64, fields[2:end])
        for fields in split.(strip.(split(chomp(read(command, String)), '\n')), ',')
    )
end

function unit_rate!(du, _, _, _)
    du[1] = 1.0
end

@testset "Callback and save-at compliance" begin
    rust = rust_callback_rows()

    event_problem = ODEProblem(unit_rate!, [0.0], (0.0, 2.0))
    condition(u, _, _) = u[1] - 0.75
    function affect!(integrator)
        integrator.u[1] = 42.0
        terminate!(integrator)
    end
    event = solve(
        event_problem,
        RK4();
        adaptive = false,
        dt = 0.5,
        callback = ContinuousCallback(condition, affect!),
        save_everystep = false,
    )
    @test rust["event"] ≈ [event.t[end], event.u[end][1]] rtol = 1.0e-13 atol = 1.0e-13

    discrete_problem = ODEProblem(unit_rate!, [0.0], (0.0, 1.0))
    discrete_condition(u, t, _) = t >= 0.5 && u[1] < 5.0
    discrete_affect!(integrator) = integrator.u[1] += 10.0
    discrete = solve(
        discrete_problem,
        RK4();
        adaptive = false,
        dt = 0.25,
        callback = DiscreteCallback(discrete_condition, discrete_affect!),
        save_everystep = false,
    )
    @test rust["discrete"] ≈ [discrete.t[end], discrete.u[end][1]] rtol = 1.0e-13 atol = 1.0e-13

    save_problem = ODEProblem(unit_rate!, [0.0], (0.0, 1.0))
    saved = solve(
        save_problem,
        RK4();
        adaptive = false,
        dt = 0.3,
        saveat = [0.2, 0.5, 0.8],
    )
    julia_values = collect(Iterators.flatten(zip(saved.t, only.(saved.u))))
    @test rust["save_at"] ≈ julia_values rtol = 1.0e-13 atol = 1.0e-13
end
