using Test
using SciMLBase: ODEProblem, solve
using OrdinaryDiffEqLowOrderRK: RK4

@testset "RK4 Hermite save-at reference" begin
    function cubic!(du, _, _, t)
        du[1] = 3t^2
    end

    problem = ODEProblem(cubic!, [0.0], (0.0, 1.0))
    solution = solve(problem, RK4(); dt = 1.0, saveat = [0.25, 0.75])

    @test isapprox(solution.u[1][1], 0.015625; atol = 1.0e-14)
    @test isapprox(solution.u[2][1], 0.421875; atol = 1.0e-14)
end
