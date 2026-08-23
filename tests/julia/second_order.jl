using OrdinaryDiffEqSymplecticRK:
    CalvoSanz4,
    CandyRoz4,
    KahanLi6,
    KahanLi8,
    LeapfrogDriftKickDrift,
    McAte2,
    McAte3,
    McAte4,
    McAte42,
    McAte5,
    McAte8,
    PseudoVerletLeapfrog,
    Ruth3,
    SofSpa10,
    SymplecticEuler,
    VelocityVerlet,
    VerletLeapfrog,
    Yoshida6
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
        "pseudo_verlet_leapfrog" => oscillator_endpoint(PseudoVerletLeapfrog()),
        "mcate2" => oscillator_endpoint(McAte2()),
        "ruth3" => oscillator_endpoint(Ruth3()),
        "mcate3" => oscillator_endpoint(McAte3()),
        "candy_roz4" => oscillator_endpoint(CandyRoz4()),
        "mcate4" => oscillator_endpoint(McAte4()),
        "calvo_sanz4" => oscillator_endpoint(CalvoSanz4()),
        "mcate42" => oscillator_endpoint(McAte42()),
        "mcate5" => oscillator_endpoint(McAte5()),
        "yoshida6" => oscillator_endpoint(Yoshida6()),
        "kahan_li6" => oscillator_endpoint(KahanLi6()),
        "mcate8" => oscillator_endpoint(McAte8()),
        "kahan_li8" => oscillator_endpoint(KahanLi8()),
        "sof_spa10" => oscillator_endpoint(SofSpa10()),
    )

    @test Set(keys(rust)) == Set(keys(julia))
    for name in keys(julia)
        @testset "$name endpoint" begin
            @test rust[name] ≈ julia[name] rtol = 2.0e-13 atol = 2.0e-14
        end
    end
end
