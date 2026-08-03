using OrdinaryDiffEqSymplecticRK:
    LeapfrogDriftKickDrift, SymplecticEuler, VelocityVerlet, VerletLeapfrog
using SciMLBase: SecondOrderODEProblem

function rust_second_order_endpoints()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example second_order_compliance`
    rows = split(chomp(read(command, String)), '\n')
    Dict(
        first(fields) => parse.(Float64, fields[2:end])
        for fields in split.(strip.(rows), ',')
    )
end

function oscillator_endpoint(algorithm)
    function acceleration!(dv, _, q, _, _)
        dv[1] = -q[1]
    end
    problem = SecondOrderODEProblem(acceleration!, [0.0], [1.0], (0.0, 1.0))
    solution = solve(
        problem,
        algorithm;
        adaptive = false,
        dt = 0.01,
        save_everystep = false,
    )
    # SciML's ArrayPartition stores velocity before position.
    [only(solution.u[end].x[1]), only(solution.u[end].x[2])]
end

function velocity_dependent_endpoint()
    function acceleration!(dv, v, q, _, time)
        dv[1] = -q[1] - 0.2 * v[1] + 0.1 * time
    end
    problem = SecondOrderODEProblem(acceleration!, [0.25], [1.0], (0.0, 1.0))
    solution = solve(
        problem,
        LeapfrogDriftKickDrift();
        adaptive = false,
        dt = 0.01,
        save_everystep = false,
    )
    [only(solution.u[end].x[1]), only(solution.u[end].x[2])]
end

@testset "Second-order symplectic compliance" begin
    rust = rust_second_order_endpoints()
    julia = Dict(
        "symplectic_euler" => oscillator_endpoint(SymplecticEuler()),
        "velocity_verlet" => oscillator_endpoint(VelocityVerlet()),
        "verlet_leapfrog" => oscillator_endpoint(VerletLeapfrog()),
        "leapfrog_dkd" => oscillator_endpoint(LeapfrogDriftKickDrift()),
        "leapfrog_dkd_velocity_dependent" => velocity_dependent_endpoint(),
    )

    @test Set(keys(rust)) == Set(keys(julia))
    for name in keys(julia)
        @testset "$name endpoint" begin
            @test rust[name] ≈ julia[name] rtol = 2.0e-13 atol = 2.0e-14
        end
    end
end
