using OrdinaryDiffEqFeagin: Feagin10, Feagin12, Feagin14
using OrdinaryDiffEqHighOrderRK: DP8, PFRK87, TanYam7, TsitPap8
using OrdinaryDiffEqVerner: RKV76IIa

function rust_high_order_results()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example high_order_compliance`
    rows = split(chomp(read(command, String)), '\n')
    Dict(
        first(fields) => parse.(Float64, fields[2:end])
        for fields in split.(strip.(rows), ',')
    )
end

function high_order_nonautonomous(initial, span)
    function rhs(u, _, time)
        [u[1] + time]
    end
    ODEProblem(rhs, [initial], span)
end

function high_order_oscillator()
    function rhs(u, _, time)
        [u[2], -u[1] + 0.1 * time]
    end
    ODEProblem(rhs, [1.0, 0.0], (0.0, 2.0))
end

function high_order_exponential()
    function rhs(u, _, _)
        [u[1]]
    end
    ODEProblem(rhs, [1.0], (0.0, 2.0))
end

function high_order_fixed_endpoint(algorithm, problem; step)
    solution = solve(
        problem,
        algorithm;
        adaptive = false,
        dt = step,
        save_everystep = false,
    )
    collect(solution.u[end])
end

function high_order_adaptive_endpoint(algorithm, tolerance)
    solution = solve(
        high_order_oscillator(),
        algorithm;
        abstol = tolerance,
        reltol = tolerance,
        dt = 0.5,
        save_everystep = false,
    )
    collect(solution.u[end])
end

function high_order_convergence_ratio(algorithm)
    exact = exp(2.0)
    coarse = only(high_order_fixed_endpoint(algorithm, high_order_exponential(); step = 1.0))
    fine = only(high_order_fixed_endpoint(algorithm, high_order_exponential(); step = 0.5))
    abs(coarse - exact) / abs(fine - exact)
end

@testset "High-order explicit Runge--Kutta compliance" begin
    rust = rust_high_order_results()
    algorithms = Dict(
        "dp8" => (DP8(), 8, 1.0e-10),
        "feagin10" => (Feagin10(), 10, 1.0e-10),
        "feagin12" => (Feagin12(), 12, 1.0e-10),
        "feagin14" => (Feagin14(), 14, 1.0e-10),
        # OrdinaryDiffEq's PFRK87 estimator stalls at tighter tolerances.
        "pfrk87" => (PFRK87(), 8, 1.0e-4),
        "rkv76iia" => (RKV76IIa(), 7, 1.0e-10),
        "tanyam7" => (TanYam7(), 7, 1.0e-10),
        "tsitpap8" => (TsitPap8(), 8, 1.0e-10),
    )
    backward_initial = 2.0 * exp(2.0) - 3.0

    @test length(rust) == 4 * length(algorithms)
    for (name, (algorithm, order, adaptive_tolerance)) in algorithms
        @testset "$name" begin
            forward = high_order_fixed_endpoint(
                algorithm,
                high_order_nonautonomous(1.0, (0.0, 2.0));
                step = 0.25,
            )
            backward = high_order_fixed_endpoint(
                algorithm,
                high_order_nonautonomous(backward_initial, (2.0, 0.0));
                step = 0.25,
            )
            adaptive = high_order_adaptive_endpoint(algorithm, adaptive_tolerance)
            adaptive_comparison_tolerance = max(3.0e-9, 0.05 * adaptive_tolerance)
            julia_ratio = high_order_convergence_ratio(algorithm)
            rust_ratio = only(rust["$(name)_convergence_ratio"])

            @test isapprox(
                rust["$(name)_fixed_forward"], forward; rtol = 2.0e-12, atol = 2.0e-13
            )
            @test isapprox(
                rust["$(name)_fixed_backward"], backward; rtol = 2.0e-12, atol = 2.0e-13
            )
            @test isapprox(
                rust["$(name)_adaptive_vector"],
                adaptive;
                rtol = adaptive_comparison_tolerance,
                atol = 0.1 * adaptive_comparison_tolerance,
            )
            @test log2(rust_ratio) > order - 2
            @test log2(julia_ratio) > order - 2
        end
    end
end
