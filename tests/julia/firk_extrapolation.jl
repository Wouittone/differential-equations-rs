using OrdinaryDiffEqFIRK: AdaptiveRadau, GaussLegendre, RadauIIA3, RadauIIA5, RadauIIA9
using OrdinaryDiffEqExtrapolation: AitkenNeville, ExtrapolationMidpointDeuflhard,
    ExtrapolationMidpointHairerWanner, ImplicitDeuflhardExtrapolation,
    ImplicitEulerBarycentricExtrapolation, ImplicitEulerExtrapolation,
    ImplicitHairerWannerExtrapolation

function rust_firk_extrapolation_results()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example firk_extrapolation_compliance`
    rows = split(chomp(read(command, String)), '\n')
    Dict(first(fields) => parse(Float64, fields[2]) for fields in split.(strip.(rows), ','))
end

function firk_extrapolation_problem()
    function rhs!(du, u, _, _)
        du[1] = u[1]
    end
    ODEProblem(rhs!, [1.0], (0.0, 1.0))
end

function firk_extrapolation_endpoint(algorithm)
    solution = solve(
        firk_extrapolation_problem(),
        algorithm;
        adaptive = false,
        dt = 0.1,
        save_everystep = false,
    )
    only(solution.u[end])
end

@testset "FIRK and extrapolation compliance" begin
    rust = rust_firk_extrapolation_results()
    algorithms = Dict(
        "radau_iia3" => RadauIIA3(),
        "radau_iia5" => RadauIIA5(),
        "radau_iia9" => RadauIIA9(),
        "adaptive_radau" => AdaptiveRadau(min_order = 5, max_order = 5),
        "gauss_legendre" => GaussLegendre(num_stages = 2),
        "aitken_neville" => AitkenNeville(),
        "midpoint_deuflhard" => ExtrapolationMidpointDeuflhard(),
        "midpoint_hairer_wanner" => ExtrapolationMidpointHairerWanner(),
        "implicit_euler" => ImplicitEulerExtrapolation(),
        "implicit_deuflhard" => ImplicitDeuflhardExtrapolation(),
        "implicit_hairer_wanner" => ImplicitHairerWannerExtrapolation(),
        "implicit_euler_barycentric" => ImplicitEulerBarycentricExtrapolation(),
    )
    @test Set(keys(rust)) == Set(keys(algorithms))
    for (name, algorithm) in algorithms
        julia = firk_extrapolation_endpoint(algorithm)
        # Fixed-step endpoints compare the actual pinned family constructors.
        # Extrapolation implementations use algebraically equivalent Neville
        # versus barycentric evaluation, so their floating-point paths need a
        # slightly looser tolerance than fixed Radau collocation.
        tolerance = startswith(name, "radau") || name in ("adaptive_radau", "gauss_legendre") ? 2.0e-10 : 2.0e-7
        @test isapprox(rust[name], julia; rtol = tolerance, atol = tolerance * 1.0e-2)
    end
end
