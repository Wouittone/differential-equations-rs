using Test
using SciMLBase: ODEProblem, solve
using OrdinaryDiffEqTsit5: Tsit5

const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const ABSOLUTE_TOLERANCE = 1.0e-10
const RELATIVE_TOLERANCE = 1.0e-10

function rust_endpoints()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example compliance_fixture`
    rows = split(chomp(read(command, String)), '\n')

    Dict(
        first(fields) => parse.(Float64, fields[2:end])
        for fields in split.(strip.(rows), ',')
    )
end

function reference_endpoint(problem)
    solution = solve(
        problem,
        Tsit5();
        abstol = ABSOLUTE_TOLERANCE,
        reltol = RELATIVE_TOLERANCE,
        save_everystep = false,
    )
    collect(solution.u[end])
end

function exponential_problem()
    function exponential!(du, u, rate, _)
        du[1] = rate * u[1]
    end
    ODEProblem(exponential!, [0.5], (0.0, 1.0), 1.01)
end

function oscillator_problem()
    function oscillator!(du, u, _, _)
        du[1] = u[2]
        du[2] = -u[1]
    end
    ODEProblem(oscillator!, [1.0, 0.0], (0.0, 2π))
end

function logistic_problem()
    function logistic!(du, u, parameters, _)
        rate, capacity = parameters
        du[1] = rate * u[1] * (1 - u[1] / capacity)
    end
    ODEProblem(logistic!, [0.25], (0.0, 5.0), (1.3, 10.0))
end

function lorenz_problem()
    function lorenz!(du, u, _, _)
        du[1] = 10.0 * (u[2] - u[1])
        du[2] = u[1] * (28.0 - u[3]) - u[2]
        du[3] = u[1] * u[2] - (8.0 / 3.0) * u[3]
    end
    ODEProblem(lorenz!, [1.0, 0.0, 0.0], (0.0, 1.0))
end

@testset "Rust Tsit5 compliance with OrdinaryDiffEqTsit5.jl" begin
    rust = rust_endpoints()
    problems = Dict(
        "exponential" => exponential_problem(),
        "oscillator" => oscillator_problem(),
        "logistic" => logistic_problem(),
        "lorenz" => lorenz_problem(),
    )

    @test Set(keys(rust)) == Set(keys(problems))
    for (name, problem) in problems
        @testset "$name endpoint" begin
            julia = reference_endpoint(problem)
            @test length(rust[name]) == length(julia)
            @test rust[name] ≈ julia rtol = 5.0e-8 atol = 5.0e-10
        end
    end
end

include("low_order_rk.jl")
include("adams.jl")
include("implicit.jl")
include("ssprk.jl")
include("rosenbrock.jl")
include("callbacks_and_saving.jl")
include("explicit_dense.jl")
include("owren_zen_bs5.jl")
include("trbdf2.jl")
include("sdirk2.jl")
include("abdf2.jl")
include("mebdf2.jl")
include("second_order.jl")
include("verner.jl")
include("variable_adams.jl")
include("rosenbrock_extended.jl")
include("ssprk_extended.jl")
include("low_storage_rk.jl")
