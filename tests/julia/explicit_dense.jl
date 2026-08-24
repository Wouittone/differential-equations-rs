using Test
using SciMLBase: ContinuousCallback, ODEProblem, solve, terminate!
using OrdinaryDiffEqLowOrderRK: DP5, OwrenZen3, OwrenZen4, OwrenZen5, RK4
using OrdinaryDiffEqTsit5: Tsit5

@testset "RK4 Hermite save-at reference" begin
    function cubic!(du, _, _, t)
        du[1] = 3t^2
    end

    problem = ODEProblem(cubic!, [0.0], (0.0, 1.0))
    solution = solve(problem, RK4(); dt = 1.0, saveat = [0.25, 0.75])

    @test isapprox(solution.u[1][1], 0.015625; atol = 1.0e-14)
    @test isapprox(solution.u[2][1], 0.421875; atol = 1.0e-14)
end

@testset "free explicit-RK continuous extensions" begin
    function exponential!(du, u, _, _)
        du[1] = u[1]
    end

    problem = ODEProblem(exponential!, [1.0], (0.0, 1.0))
    algorithms = (DP5(), OwrenZen3(), OwrenZen4(), OwrenZen5())
    dense_samples = (
        [1.221470148631057, 1.7331417529458806, 2.4595887579652946],
        [1.216, 1.706291666666667, 2.413],
        [1.22112999591419, 1.7325755385254473, 2.460288616366704],
        [1.2213113186813191, 1.7330660720366877, 2.4590016672390163],
    )
    root_times = (
        0.5877863812955013,
        0.5881037090300367,
        0.5877840693475775,
        0.5877869725203198,
    )

    for (algorithm, expected_samples, expected_root) in
        zip(algorithms, dense_samples, root_times)
        solution = solve(
            problem,
            algorithm;
            adaptive = false,
            dt = 1.0,
            saveat = [0.2, 0.55, 0.9],
        )
        @test isapprox(
            [state[1] for state in solution.u],
            expected_samples;
            rtol = 0.0,
            atol = 3.0e-14,
        )

        condition(u, _, _) = u[1] - 1.8
        callback = ContinuousCallback(condition, terminate!)
        event_solution = solve(
            problem,
            algorithm;
            adaptive = false,
            dt = 0.25,
            callback = callback,
            save_everystep = false,
            abstol = 1.0e-13,
        )
        @test isapprox(event_solution.t[end], expected_root; rtol = 0.0, atol = 5.0e-12)
        @test isapprox(event_solution.u[end][1], 1.8; rtol = 0.0, atol = 2.0e-12)
    end
end


@testset "Tsit5 method-specific save-at reference" begin
    function exponential!(du, u, _, _)
        du[1] = u[1]
    end

    problem = ODEProblem(exponential!, [1.0], (0.0, 1.0))
    solution = solve(
        problem,
        Tsit5();
        adaptive = false,
        dt = 1.0,
        saveat = [0.25, 0.5, 0.75],
    )

    @test isapprox(solution.u[1][1], 1.2840130541696058; atol = 2.0e-14)
    @test isapprox(solution.u[2][1], 1.6484577270499763; atol = 2.0e-14)
    @test isapprox(solution.u[3][1], 2.1166342629770343; atol = 2.0e-14)

    backward = solve(
        ODEProblem(exponential!, [exp(1.0)], (1.0, 0.0)),
        Tsit5();
        adaptive = false,
        dt = 1.0,
        saveat = [0.75, 0.25],
    )
    @test isapprox(backward.u[1][1], 2.1160005261297776; atol = 2.0e-14)
    @test isapprox(backward.u[2][1], 1.283736699170017; atol = 2.0e-14)
end
