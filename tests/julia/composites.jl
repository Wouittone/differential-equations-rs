using SciMLBase: init, solve!
using OrdinaryDiffEqDefault: DefaultImplicitODEAlgorithm, DefaultODEAlgorithm
using OrdinaryDiffEqExplicitRK: ExplicitRK
using OrdinaryDiffEqRosenbrock: Rodas5P
using OrdinaryDiffEqTsit5: AutoTsit5
using OrdinaryDiffEqVerner: AutoVern6, AutoVern7, AutoVern8, AutoVern9

function rust_composite_endpoints()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example composites_compliance`
    rows = split(chomp(read(command, String)), '\n')
    Dict(
        first(fields) => parse.(Float64, fields[2:end])
        for fields in split.(strip.(rows), ',')
    )
end

function composite_reference(algorithm)
    function rhs!(du, u, _, t)
        du[1] = u[1] + t
    end
    solution = solve(
        ODEProblem(rhs!, [1.0], (0.0, 1.0)),
        algorithm;
        abstol = 1.0e-10,
        reltol = 1.0e-10,
        save_everystep = false,
    )
    only(solution.u[end])
end

function switching_reference(; stiff_first)
    function tracking_rate(t)
        if stiff_first
            return t < 0.5 ? 500.0 : 1.0
        end
        return t < 0.5 ? 1.0 : 500.0
    end
    function rhs!(du, u, _, t)
        rate = tracking_rate(t)
        du[1] = -rate * (u[1] - cos(t)) - sin(t)
    end

    problem = ODEProblem(rhs!, [1.0], (0.0, 1.0))
    # The pinned SciML implementation treats maxstiffstep/maxnonstiffstep as
    # maxima and switches only after `count > maxstiffstep` (or its negative
    # counterpart). Rust names these settings "confirmations" and switches
    # when that exact count is reached. Zeros here and ones in the Rust fixture
    # therefore give both implementations one-confirmation behavior. Rust also
    # has an explicit minimum-residence and rejection-confirmation policy that
    # is intentionally not part of this matched fixed-step case.
    algorithm = AutoTsit5(
        Rodas5P();
        maxstiffstep = 0,
        maxnonstiffstep = 0,
        nonstifftol = 1 // 5,
        stifftol = 1 // 10,
        dtfac = 1,
        stiffalgfirst = stiff_first,
    )
    integrator = init(
        problem,
        algorithm;
        adaptive = false,
        dt = 1 / 256,
        tstops = [0.5],
        save_everystep = false,
    )
    solution = solve!(integrator)
    # SciML exposes the final composite-cache index, but not a stable public
    # cumulative switch counter. The direction is consequently checked from
    # the known initial branch and this final indicator; Rust's own switch
    # count is asserted independently below.
    return (
        endpoint = only(solution.u[end]),
        rhs_evaluations = integrator.stats.nf,
        accepted_steps = integrator.stats.naccept,
        rejected_steps = integrator.stats.nreject,
        final_branch = integrator.cache.current - 1,
    )
end

@testset "automatic/default composite compliance" begin
    rust = rust_composite_endpoints()
    julia = Dict(
        "auto_tsit5" => composite_reference(AutoTsit5(Rodas5P())),
        "auto_vern6" => composite_reference(AutoVern6(Rodas5P())),
        "auto_vern7" => composite_reference(AutoVern7(Rodas5P())),
        "auto_vern8" => composite_reference(AutoVern8(Rodas5P())),
        "auto_vern9" => composite_reference(AutoVern9(Rodas5P())),
        "default_ode_algorithm" => composite_reference(DefaultODEAlgorithm()),
        "default_implicit_ode_algorithm" => composite_reference(DefaultImplicitODEAlgorithm()),
        "explicit_rk" => composite_reference(ExplicitRK()),
    )
    switching_names = Set(("switch_nonstiff_to_stiff", "switch_stiff_to_nonstiff"))
    @test Set(keys(rust)) == union(Set(keys(julia)), switching_names)
    for name in keys(julia)
        @testset "$name endpoint" begin
            @test only(rust[name]) ≈ julia[name] rtol = 2.0e-7 atol = 2.0e-9
        end
    end
end


@testset "time-varying automatic switching compliance" begin
    rust = rust_composite_endpoints()
    cases = (
        (
            name = "switch_nonstiff_to_stiff",
            reference = switching_reference(; stiff_first = false),
            final_branch = 1,
        ),
        (
            name = "switch_stiff_to_nonstiff",
            reference = switching_reference(; stiff_first = true),
            final_branch = 0,
        ),
    )

    for case in cases
        @testset "$(case.name)" begin
            # Rust fields are endpoint, RHS evaluations, accepted steps,
            # rejected steps, switch count, and final branch respectively.
            observed = rust[case.name]
            @test observed[1] ≈ case.reference.endpoint rtol = 2.0e-6 atol = 2.0e-8
            @test observed[1] ≈ cos(1.0) rtol = 2.0e-6 atol = 2.0e-8
            @test observed[2] > observed[3]
            @test Int(observed[3]) == case.reference.accepted_steps == 256
            @test Int(observed[4]) == case.reference.rejected_steps == 0
            @test Int(observed[5]) == 1
            @test Int(observed[6]) == case.final_branch
            @test case.reference.rhs_evaluations > case.reference.accepted_steps
            @test case.reference.final_branch == case.final_branch
        end
    end
end
