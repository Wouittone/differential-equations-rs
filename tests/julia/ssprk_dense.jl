using SciMLBase: ContinuousCallback, ODEProblem, solve, terminate!
using OrdinaryDiffEqSSPRK:
    SSPRK22, SSPRK33, SSPRK43, SSPRK432, SSPRK53, SSPRKMSVS32, pRRK22

@testset "SSPRK specialized quadratic dense output" begin
    function exponential!(du, u, _, _)
        du[1] = u[1]
    end

    problem = ODEProblem(exponential!, [1.0], (0.0, 1.0))
    algorithms = (SSPRK22(), SSPRK33(), SSPRK43(), SSPRK432())
    expected_samples = (
        [1.2200000000000002, 1.7012500000000002, 2.305],
        [1.2266666666666666, 1.751666666666667, 2.44],
        [1.2275, 1.7579687500000003, 2.456875],
        [1.2275, 1.7579687500000003, 2.456875],
    )
    expected_roots = (
        0.6124515496597099,
        0.5775918047351754,
        0.5737122778180594,
        0.5737122778180594,
    )
    callback = ContinuousCallback((u, _, _) -> u[1] - 1.8, terminate!)

    for (algorithm, samples, root) in zip(algorithms, expected_samples, expected_roots)
        solution = solve(
            problem,
            algorithm;
            adaptive = false,
            dt = 1.0,
            saveat = [0.2, 0.55, 0.9],
        )
        @test isapprox([state[1] for state in solution.u], samples; rtol = 0.0, atol = 3.0e-14)

        event_solution = solve(
            problem,
            algorithm;
            adaptive = false,
            dt = 1.0,
            callback = callback,
            save_everystep = false,
        )
        @test isapprox(event_solution.t[end], root; rtol = 0.0, atol = 2.0e-13)
        @test isapprox(event_solution.u[end][1], 1.8; rtol = 0.0, atol = 2.0e-12)
    end
end

@testset "SSPRK generic Hermite dispatch" begin
    function quadratic!(du, _, _, t)
        du[1] = 2t
    end

    problem = ODEProblem(quadratic!, [0.0], (0.0, 1.0))
    callback = ContinuousCallback((u, _, _) -> u[1] - 0.36, terminate!)

    # One shared explicit-tableau kernel and one multistep SSP kernel lock the
    # generic OrdinaryDiffEqCore Hermite dispatch used by the remaining types.
    for algorithm in (SSPRK53(), SSPRKMSVS32())
        solution = solve(
            problem,
            algorithm;
            adaptive = false,
            dt = 1.0,
            saveat = [0.3, 0.7],
        )
        @test isapprox(
            [state[1] for state in solution.u],
            [0.09, 0.49];
            rtol = 0.0,
            atol = 1.0e-14,
        )

        event_solution = solve(
            problem,
            algorithm;
            adaptive = false,
            dt = 1.0,
            callback = callback,
            save_everystep = false,
        )
        @test isapprox(event_solution.t[end], 0.6; rtol = 0.0, atol = 2.0e-14)
    end

    # The pinned pRRK22 cache supplies the same generic Hermite samples, but
    # its upstream callback path currently raises because the cache has no
    # `tmp` field. Keep the matched sample without pretending the Julia root
    # path is executable at this revision.
    relaxation = solve(
        problem,
        pRRK22();
        adaptive = false,
        dt = 1.0,
        saveat = [0.3, 0.7],
    )
    @test isapprox(
        [state[1] for state in relaxation.u],
        [0.09, 0.49];
        rtol = 0.0,
        atol = 1.0e-14,
    )
end
