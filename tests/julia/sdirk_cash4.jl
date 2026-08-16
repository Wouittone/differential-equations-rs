using OrdinaryDiffEqSDIRK: Cash4

function rust_sdirk_cash4_results()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example sdirk_cash4_compliance`
    Dict(
        first(fields) => fields[2:end]
        for fields in split.(strip.(split(chomp(read(command, String)), '\n')), ',')
    )
end

@testset "Cash4 regular ODE compliance" begin
    rust = rust_sdirk_cash4_results()
    function stiff!(du, u, _, t)
        du[1] = -15.0 * (u[1] - cos(t)) - sin(t)
    end
    problem = ODEProblem(stiff!, [1.0], (0.0, 1.0))
    reference = solve(problem, Cash4(); adaptive = false, dt = 0.01, save_everystep = false)
    @test haskey(rust, "cash4_fixed")
    @test parse(Float64, only(rust["cash4_fixed"])) ≈ only(reference.u[end]) rtol = 4.0e-9 atol = 2.0e-11
    @test parse(Float64, rust["cash4"][2]) ≈ cos(1.0) rtol = 2.0e-6 atol = 2.0e-8
    @test parse(Int, rust["cash4"][3]) > 0
end
