include(joinpath(@__DIR__, "..", "tests", "julia", "pinned_environment.jl"))
check_pins()

using SciMLBase: ODEProblem, solve
using OrdinaryDiffEqAdamsBashforthMoulton: AB3, AB4, AB5, ABM32, ABM43, ABM54
using OrdinaryDiffEqLowOrderRK:
    Alshina2, Alshina3, Euler, Midpoint, Heun, Ralston, Ralston4, RK4, RKM, BS3, DP5
using OrdinaryDiffEqRosenbrock: Rodas4P, Rodas5P, Rodas5Pr, Rosenbrock23
using OrdinaryDiffEqSDIRK:
    ImplicitEuler, ImplicitMidpoint, KenCarp5, Kvaerno5, TRBDF2, Trapezoid
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

function solve_batch(problem, algorithm, options, repetitions)
    checksum = 0.0
    rhs_evaluations = 0
    for _ in 1:repetitions
        solution = solve(problem, algorithm; options...)
        checksum += solution.u[end][1]
        rhs_evaluations += solution.stats.nf
    end
    checksum / repetitions, rhs_evaluations / repetitions
end

function benchmark(name, problem, algorithm, repetitions; adaptive, mode)
    options = if adaptive
        (; abstol = 1.0e-7, reltol = 1.0e-7, save_everystep = false)
    else
        (; adaptive = false, dt = 0.01, save_everystep = false)
    end

    solve(problem, algorithm; options...)
    GC.gc()
    if mode == "timing"
        result = nothing
        elapsed = @elapsed begin
            result = solve_batch(problem, algorithm, options, repetitions)
        end
        checksum, rhs_evaluations = result
        bytes = NaN
    elseif mode == "allocation"
        measurement = @timed begin
            solve_batch(problem, algorithm, options, repetitions)
        end
        elapsed = measurement.time
        bytes = measurement.bytes
        checksum, rhs_evaluations = measurement.value
    else
        error("mode must be timing or allocation")
    end
    println(
        "julia,$name,$(length(problem.u0)),$(1.0e9 * elapsed / repetitions)," *
            "$(bytes / repetitions),NaN,$rhs_evaluations,$checksum"
    )
end

function main()
    repetitions = 20
    selected = nothing
    mode = "timing"
    index = 1
    while index <= length(ARGS)
        argument = ARGS[index]
        if argument == "--repetitions"
            index += 1
            repetitions = parse(Int, ARGS[index])
        elseif argument == "--algorithm"
            index += 1
            selected = ARGS[index]
        elseif argument == "--mode"
            index += 1
            mode = ARGS[index]
        elseif !startswith(argument, "-")
            repetitions = parse(Int, argument)
        else
            error("unknown argument: $argument")
        end
        index += 1
    end
    nonstiff = problem(128, 0.2, 2.0)
    stiff = problem(8, 20.0, 1.0)

    println(
        "language,algorithm,dimension,nanoseconds_per_solve,bytes_allocated_per_solve," *
            "allocations_per_solve,rhs_evaluations_per_solve,checksum"
    )
    maybe(name, problem, algorithm; adaptive) =
        (isnothing(selected) || selected == name) &&
        (benchmark(name, problem, algorithm, repetitions; adaptive, mode); true)
    ran = false
    ran |= maybe("Tsit5", nonstiff, Tsit5(); adaptive = true)
    ran |= maybe("Midpoint", nonstiff, Midpoint(); adaptive = true)
    ran |= maybe("Heun", nonstiff, Heun(); adaptive = true)
    ran |= maybe("Ralston", nonstiff, Ralston(); adaptive = true)
    ran |= maybe("BS3", nonstiff, BS3(); adaptive = true)
    ran |= maybe("DP5", nonstiff, DP5(); adaptive = true)
    ran |= maybe("Euler", nonstiff, Euler(); adaptive = false)
    ran |= maybe("RK4", nonstiff, RK4(); adaptive = false)
    ran |= maybe("RKM", nonstiff, RKM(); adaptive = false)
    ran |= maybe("Ralston4", nonstiff, Ralston4(); adaptive = false)
    ran |= maybe("Alshina2", nonstiff, Alshina2(); adaptive = false)
    ran |= maybe("Alshina3", nonstiff, Alshina3(); adaptive = false)
    ran |= maybe("AB3", nonstiff, AB3(); adaptive = false)
    ran |= maybe("AB4", nonstiff, AB4(); adaptive = false)
    ran |= maybe("AB5", nonstiff, AB5(); adaptive = false)
    ran |= maybe("ABM32", nonstiff, ABM32(); adaptive = false)
    ran |= maybe("ABM43", nonstiff, ABM43(); adaptive = false)
    ran |= maybe("ABM54", nonstiff, ABM54(); adaptive = false)
    ran |= maybe("SSPRK22", nonstiff, SSPRK22(); adaptive = false)
    ran |= maybe("SSPRK33", nonstiff, SSPRK33(); adaptive = false)
    ran |= maybe("SSPRK43", nonstiff, SSPRK43(); adaptive = true)
    ran |= maybe("ImplicitEuler", stiff, ImplicitEuler(); adaptive = false)
    ran |= maybe("ImplicitMidpoint", stiff, ImplicitMidpoint(); adaptive = false)
    ran |= maybe("Trapezoid", stiff, Trapezoid(); adaptive = false)
    ran |= maybe("Rosenbrock23", stiff, Rosenbrock23(); adaptive = true)
    ran |= maybe("TRBDF2", stiff, TRBDF2(); adaptive = true)
    ran |= maybe("Kvaerno5", stiff, Kvaerno5(); adaptive = true)
    ran |= maybe("KenCarp5", stiff, KenCarp5(); adaptive = true)
    ran |= maybe("Rodas4P", stiff, Rodas4P(); adaptive = true)
    ran |= maybe("Rodas5P", stiff, Rodas5P(); adaptive = true)
    ran |= maybe("Rodas5Pr", stiff, Rodas5Pr(); adaptive = true)
    !isnothing(selected) && !ran && error("unknown benchmark algorithm: $selected")
end

main()
