using OrdinaryDiffEqStabilizedRK: ESERK4, ESERK5, RKC, RKG1, RKG2, RKL1, RKL2, RKMC2, ROCK2, ROCK4, SERK2, TSRKC2, TSRKC3

function rust_stabilized_results()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example stabilized_compliance`
    rows = split(chomp(read(command, String)), '\n')
    Dict(
        first(fields) => parse(Float64, fields[2])
        for fields in split.(strip.(rows), ',')
    )
end

function stabilized_stiff_forced()
    function rhs!(du, u, _, time)
        du[1] = -40.0 * (u[1] - cos(time)) - sin(time)
    end
    ODEProblem(rhs!, [1.0], (0.0, 1.0))
end

function stabilized_nonautonomous()
    function rhs!(du, u, _, time)
        du[1] = u[1] + time
    end
    ODEProblem(rhs!, [1.0], (0.0, 1.0))
end

function stabilized_exponential()
    function rhs!(du, u, _, _)
        du[1] = u[1]
    end
    ODEProblem(rhs!, [1.0], (0.0, 1.0))
end

function stabilized_fixed_endpoint(algorithm, problem; step)
    solution = solve(
        problem,
        algorithm;
        adaptive = false,
        dt = step,
        save_everystep = false,
    )
    only(solution.u[end])
end

function stabilized_adaptive_endpoint(algorithm)
    solution = solve(
        stabilized_nonautonomous(),
        algorithm;
        abstol = 1.0e-7,
        reltol = 1.0e-7,
        dt = 0.1,
        save_everystep = false,
    )
    only(solution.u[end])
end

function stabilized_convergence_ratio(algorithm)
    exact = exp(1.0)
    coarse = stabilized_fixed_endpoint(algorithm, stabilized_exponential(); step = 0.1)
    fine = stabilized_fixed_endpoint(algorithm, stabilized_exponential(); step = 0.05)
    abs(coarse - exact) / abs(fine - exact)
end

@testset "Stabilized explicit Runge--Kutta compliance" begin
    rust = rust_stabilized_results()
    stiff_eigen_estimate = integrator -> integrator.eigen_est = 48.0
    mild_eigen_estimate = integrator -> integrator.eigen_est = 1.2
    algorithms = Dict(
        "eserk4" => (
            ESERK4(eigen_est = stiff_eigen_estimate),
            ESERK4(eigen_est = mild_eigen_estimate),
            4,
        ),
        "eserk5" => (
            ESERK5(eigen_est = stiff_eigen_estimate),
            ESERK5(eigen_est = mild_eigen_estimate),
            5,
        ),
        "rkc" => (
            RKC(eigen_est = stiff_eigen_estimate),
            RKC(eigen_est = mild_eigen_estimate),
            2,
        ),
        "rkl1" => (
            RKL1(eigen_est = stiff_eigen_estimate),
            RKL1(eigen_est = mild_eigen_estimate),
            1,
        ),
        "rkl2" => (
            RKL2(eigen_est = stiff_eigen_estimate),
            RKL2(eigen_est = mild_eigen_estimate),
            2,
        ),
        "rkg1" => (
            RKG1(eigen_est = stiff_eigen_estimate),
            RKG1(eigen_est = mild_eigen_estimate),
            1,
        ),
        "rkg2" => (
            RKG2(eigen_est = stiff_eigen_estimate),
            RKG2(eigen_est = mild_eigen_estimate),
            2,
        ),
        "rkmc2" => (
            RKMC2(eigen_est = stiff_eigen_estimate),
            RKMC2(eigen_est = mild_eigen_estimate),
            2,
        ),
        "rock2" => (
            ROCK2(eigen_est = stiff_eigen_estimate),
            ROCK2(eigen_est = mild_eigen_estimate),
            2,
        ),
        "rock4" => (
            ROCK4(eigen_est = stiff_eigen_estimate),
            ROCK4(eigen_est = mild_eigen_estimate),
            4,
        ),
        "serk2" => (
            SERK2(eigen_est = stiff_eigen_estimate),
            SERK2(eigen_est = mild_eigen_estimate),
            2,
        ),
        "tsrkc2" => (
            TSRKC2(eigen_est = stiff_eigen_estimate),
            TSRKC2(eigen_est = mild_eigen_estimate),
            2,
        ),
        "tsrkc3" => (
            TSRKC3(eigen_est = stiff_eigen_estimate),
            TSRKC3(eigen_est = mild_eigen_estimate),
            3,
        ),
    )
    nonautonomous_exact = 2.0 * exp(1.0) - 2.0

    @test length(rust) == 3 * length(algorithms)
    for (name, (stiff_algorithm, mild_algorithm, order)) in algorithms
        @testset "$name" begin
            fixed = stabilized_fixed_endpoint(
                stiff_algorithm,
                stabilized_stiff_forced();
                step = 0.05,
            )
            adaptive = stabilized_adaptive_endpoint(mild_algorithm)
            rust_fixed = rust["$(name)_fixed_stiff"]
            rust_adaptive = rust["$(name)_adaptive_nonautonomous"]
            rust_ratio = rust["$(name)_convergence_ratio"]
            julia_ratio = stabilized_convergence_ratio(mild_algorithm)

            @test rust_fixed ≈ fixed rtol = 2.0e-10 atol = 2.0e-12
            @test rust_adaptive ≈ adaptive rtol = 2.0e-4 atol = 2.0e-6
            @test rust_fixed ≈ cos(1.0) rtol = order == 1 ? 2.0e-2 : 5.0e-3
            @test fixed ≈ cos(1.0) rtol = order == 1 ? 2.0e-2 : 5.0e-3
            @test rust_adaptive ≈ nonautonomous_exact rtol = 5.0e-4 atol = 5.0e-6
            @test adaptive ≈ nonautonomous_exact rtol = 5.0e-4 atol = 5.0e-6
            @test log2(rust_ratio) > order - 0.5
            @test log2(julia_ratio) > order - 0.5
        end
    end
end
