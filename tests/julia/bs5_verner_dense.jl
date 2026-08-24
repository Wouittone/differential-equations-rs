using Test
using SciMLBase: ContinuousCallback, ODEProblem, solve, terminate!
using OrdinaryDiffEqLowOrderRK: BS5
using OrdinaryDiffEqVerner: Vern6, Vern7, Vern8, Vern9

function dense_exponential!(du, u, _, _)
    du[1] = u[1]
end

dense_problem = ODEProblem(dense_exponential!, [1.0], (0.0, 1.0))

@testset "BS5 and Verner method-specific dense output" begin
    algorithms = (BS5(), Vern6(), Vern7(), Vern8(), Vern9())
    expected_samples = (
        [1.2214004531632976, 1.7332401956162267, 2.4595777580872276],
        [1.2213906976267277, 1.7332688761483586, 2.459584705061208],
        [1.2214027749928802, 1.733252953490458, 2.4596033527459955],
        [1.2214025169262221, 1.733252914957541, 2.4596031147569155],
        [1.2214027544790693, 1.7332529995165806, 2.4596031102113534],
    )
    expected_roots = (
        0.5877866641397586,
        0.5877866649209265,
        0.587786664891878,
        0.5877866649024726,
        0.5877866649021184,
    )

    for (algorithm, reference_samples, reference_root) in
        zip(algorithms, expected_samples, expected_roots)
        sampled = solve(
            dense_problem,
            algorithm;
            adaptive = false,
            dt = 1.0,
            saveat = [0.2, 0.55, 0.9],
        )
        samples = [state[1] for state in sampled.u]
        @test isapprox(samples, reference_samples; rtol = 0.0, atol = 3.0e-10)

        condition(u, _, _) = u[1] - 1.8
        callback = ContinuousCallback(condition, terminate!)
        event_solution = solve(
            dense_problem,
            algorithm;
            adaptive = false,
            dt = 0.25,
            callback = callback,
            save_everystep = false,
            abstol = 1.0e-13,
        )
        @test isapprox(event_solution.t[end], reference_root; rtol = 0.0, atol = 7.0e-12)
        @test isapprox(event_solution.u[end][1], 1.8; rtol = 0.0, atol = 2.0e-12)
    end
end
