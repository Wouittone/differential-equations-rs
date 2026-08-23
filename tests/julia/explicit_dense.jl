using Test
using SciMLBase: ODEProblem, solve
using OrdinaryDiffEqLowOrderRK: RK4
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
