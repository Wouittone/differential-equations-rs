using Test
using SciMLBase: ODEProblem, solve
using OrdinaryDiffEqRosenbrock: ROS34PW3

const ROS34PW3_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

function rust_ros34pw3_endpoints()
    manifest = joinpath(ROS34PW3_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example ros34pw3_compliance`
    Dict(first(fields) => parse(Float64, fields[2]) for fields in split.(readlines(command), ','))
end

function reference_endpoint(; adaptive)
    if adaptive
        function stiff!(du, u, _, time)
            du[1] = -1000.0 * (u[1] - cos(time)) - sin(time)
        end
        problem = ODEProblem(stiff!, [1.0], (0.0, 1.0))
        solution = solve(
            problem, ROS34PW3(); abstol = 1.0e-8, reltol = 1.0e-8,
            dt = 0.1, save_everystep = false,
        )
    else
        function exponential!(du, u, _, _)
            du[1] = u[1]
        end
        problem = ODEProblem(exponential!, [1.0], (0.0, 1.0))
        solution = solve(problem, ROS34PW3(); adaptive = false, dt = 0.01, save_everystep = false)
    end
    only(solution.u[end])
end

@testset "ROS34PW3 regular ODE compliance" begin
    rust = rust_ros34pw3_endpoints()
    @test Set(keys(rust)) == Set(("ros34pw3_adaptive", "ros34pw3_fixed"))
    @test rust["ros34pw3_adaptive"] ≈ reference_endpoint(adaptive = true) rtol = 4.0e-6 atol = 5.0e-8
    @test rust["ros34pw3_fixed"] ≈ reference_endpoint(adaptive = false) rtol = 8.0e-12 atol = 8.0e-13
end
