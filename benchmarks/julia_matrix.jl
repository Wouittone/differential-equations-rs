using SciMLBase: ODEProblem, solve
using OrdinaryDiffEqAdamsBashforthMoulton: AB3, AB4, AB5, ABM32, ABM43, ABM54
using OrdinaryDiffEqLowOrderRK:
    Alshina2, Alshina3, Euler, Midpoint, Heun, Ralston, Ralston4, RK4, RKM, BS3, DP5
using OrdinaryDiffEqRosenbrock: Rosenbrock23
using OrdinaryDiffEqSDIRK: ImplicitEuler, ImplicitMidpoint, Trapezoid
using OrdinaryDiffEqSSPRK: SSPRK22, SSPRK33, SSPRK43
using OrdinaryDiffEqTsit5: Tsit5

function problem(dimension, stiffness, endpoint)
    rates = [stiffness * (1 + (index - 1) / dimension) for index in 1:dimension]
    function decay!(derivative, state, rates, _)
        @inbounds @simd for index in eachindex(state)
            derivative[index] = -rates[index] * state[index]
        end
    end
    ODEProblem(decay!, ones(dimension), (0.0, endpoint), rates)
end

function benchmark(name, problem, algorithm, repetitions; adaptive)
    options = if adaptive
        (; abstol = 1.0e-7, reltol = 1.0e-7, save_everystep = false)
    else
        (; adaptive = false, dt = 0.01, save_everystep = false)
    end

    solve(problem, algorithm; options...)
    GC.gc()
    measurement = @timed begin
        checksum = 0.0
        rhs_evaluations = 0
        for _ in 1:repetitions
            solution = solve(problem, algorithm; options...)
            checksum += solution.u[end][1]
            rhs_evaluations += solution.stats.nf
        end
        (checksum / repetitions, rhs_evaluations / repetitions)
    end
    checksum, rhs_evaluations = measurement.value
    println(
        "julia,$name,$(length(problem.u0)),$(1.0e9 * measurement.time / repetitions)," *
            "$(measurement.bytes / repetitions),NaN,$rhs_evaluations,$checksum"
    )
end

function main()
    repetitions = length(ARGS) == 1 ? parse(Int, only(ARGS)) : 20
    nonstiff = problem(128, 0.2, 2.0)
    stiff = problem(8, 20.0, 1.0)

    println(
        "language,algorithm,dimension,nanoseconds_per_solve,bytes_allocated_per_solve," *
            "allocations_per_solve,rhs_evaluations_per_solve,checksum"
    )
    benchmark("Tsit5", nonstiff, Tsit5(), repetitions; adaptive = true)
    benchmark("Midpoint", nonstiff, Midpoint(), repetitions; adaptive = true)
    benchmark("Heun", nonstiff, Heun(), repetitions; adaptive = true)
    benchmark("Ralston", nonstiff, Ralston(), repetitions; adaptive = true)
    benchmark("BS3", nonstiff, BS3(), repetitions; adaptive = true)
    benchmark("DP5", nonstiff, DP5(), repetitions; adaptive = true)
    benchmark("Euler", nonstiff, Euler(), repetitions; adaptive = false)
    benchmark("RK4", nonstiff, RK4(), repetitions; adaptive = false)
    benchmark("RKM", nonstiff, RKM(), repetitions; adaptive = false)
    benchmark("Ralston4", nonstiff, Ralston4(), repetitions; adaptive = false)
    benchmark("Alshina2", nonstiff, Alshina2(), repetitions; adaptive = false)
    benchmark("Alshina3", nonstiff, Alshina3(), repetitions; adaptive = false)
    benchmark("AB3", nonstiff, AB3(), repetitions; adaptive = false)
    benchmark("AB4", nonstiff, AB4(), repetitions; adaptive = false)
    benchmark("AB5", nonstiff, AB5(), repetitions; adaptive = false)
    benchmark("ABM32", nonstiff, ABM32(), repetitions; adaptive = false)
    benchmark("ABM43", nonstiff, ABM43(), repetitions; adaptive = false)
    benchmark("ABM54", nonstiff, ABM54(), repetitions; adaptive = false)
    benchmark("SSPRK22", nonstiff, SSPRK22(), repetitions; adaptive = false)
    benchmark("SSPRK33", nonstiff, SSPRK33(), repetitions; adaptive = false)
    benchmark("SSPRK43", nonstiff, SSPRK43(), repetitions; adaptive = true)
    benchmark("ImplicitEuler", stiff, ImplicitEuler(), repetitions; adaptive = false)
    benchmark("ImplicitMidpoint", stiff, ImplicitMidpoint(), repetitions; adaptive = false)
    benchmark("Trapezoid", stiff, Trapezoid(), repetitions; adaptive = false)
    benchmark("Rosenbrock23", stiff, Rosenbrock23(), repetitions; adaptive = true)
end

main()
