using Test
using SciMLBase: ContinuousCallback, ODEProblem, solve, terminate!
using OrdinaryDiffEqRosenbrock: ROS2, Rodas4, Rodas5P, Rodas6P, Rosenbrock23

function rosenbrock_dense_exponential!(du, u, _, _)
    du[1] = u[1]
end

rosenbrock_dense_problem =
    ODEProblem(rosenbrock_dense_exponential!, [1.0], (0.0, 1.0))

@testset "Rosenbrock/Rodas dense output" begin
    sample_cases = (
        (Rosenbrock23(), 1.0, [1.2056854249492381, 1.7581349186104047, 2.555584412271571]),
        (Rodas4(), 1.0, [1.2223065548277507, 1.7391385379449833, 2.4640010504603405]),
        (Rodas5P(), 1.0, [1.2214034175316406, 1.7331826623028603, 2.4594699855442346]),
        (Rodas6P(), 0.05, [1.221402758160135, 1.7332530178672585, 2.4596031111566306]),
    )
    for (algorithm, step, expected) in sample_cases
        solution = solve(
            rosenbrock_dense_problem,
            algorithm;
            adaptive = false,
            dt = step,
            saveat = [0.2, 0.55, 0.9],
        )
        @test isapprox(
            [state[1] for state in solution.u],
            expected;
            rtol = 0.0,
            atol = 2.0e-9,
        )
    end

    root_cases = (
        (Rosenbrock23(), 0.25, 0.5865779168719102),
        (Rodas4(), 0.25, 0.5877877517967383),
        (Rodas5P(), 0.25, 0.587786630367156),
        (Rodas6P(), 0.05, 0.5877866649019579),
        (ROS2(), 0.25, 0.7767698682101478),
    )
    condition(u, _, _) = u[1] - 1.8
    callback = ContinuousCallback(condition, terminate!; abstol = 1.0e-13)
    for (algorithm, step, expected) in root_cases
        solution = solve(
            rosenbrock_dense_problem,
            algorithm;
            adaptive = false,
            dt = step,
            callback = callback,
            save_everystep = false,
        )
        @test isapprox(solution.t[end], expected; rtol = 0.0, atol = 8.0e-11)
        @test isapprox(solution.u[end][1], 1.8; rtol = 0.0, atol = 2.0e-12)
    end
end
