using OrdinaryDiffEqNewmark: GeneralizedAlpha, NewmarkBeta
import SciMLBase

# The pinned generic algorithm remaker reconstructs structs through keyword
# fields. GeneralizedAlpha's public constructor names those coefficients as
# positional arguments, so the generic path fails once differentiation loads.
function SciMLBase.remake(
        algorithm::GeneralizedAlpha;
        autodiff = algorithm.autodiff,
        kwargs...,
    )
    @assert isempty(kwargs)
    GeneralizedAlpha(
        algorithm.αm,
        algorithm.αf,
        algorithm.β,
        algorithm.γ,
        algorithm.nlsolve,
        autodiff,
        algorithm.thread,
        algorithm.concrete_jac,
    )
end

function rust_newmark_results()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example newmark_compliance`
    Dict(
        first(fields) => parse.(Float64, fields[2:end])
        for fields in split.(strip.(split(chomp(read(command, String)), '\n')), ',')
    )
end

function newmark_harmonic_problem()
    function acceleration!(output, velocity, position, _, _)
        output[1] = -position[1]
    end
    function position_rate!(output, velocity, _, _, _)
        output[1] = velocity[1]
    end
    SciMLBase.DynamicalODEProblem(acceleration!, position_rate!, [1.0], [0.0], (0.0, 1.0))
end

function newmark_endpoint(algorithm)
    solution = solve(
        newmark_harmonic_problem(),
        algorithm;
        adaptive = false,
        dt = 0.05,
        save_everystep = false,
    )
    [solution.u[end].x[1][1], solution.u[end].x[2][1]]
end

@testset "Newmark structural compliance" begin
    rust = rust_newmark_results()
    @test isapprox(
        rust["newmark_beta"],
        newmark_endpoint(NewmarkBeta());
        rtol = 2.0e-11,
        atol = 2.0e-12,
    )
    @test isapprox(
        rust["generalized_alpha"],
        newmark_endpoint(GeneralizedAlpha(rho_inf = 0.5));
        rtol = 2.0e-11,
        atol = 2.0e-12,
    )
end
