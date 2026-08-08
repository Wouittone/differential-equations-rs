using OrdinaryDiffEqSDIRK: SDIRK2

function rust_sdirk2_results()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example sdirk2_compliance`
    Dict(
        first(fields) => fields[2:end]
        for fields in split.(strip.(split(chomp(read(command, String)), '\n')), ',')
    )
end

@testset "SDIRK2 regular ODE compliance" begin
    rust = rust_sdirk2_results()
    function stiff!(du, u, _, t)
        du[1] = -15.0 * (u[1] - cos(t)) - sin(t)
    end
    problem = ODEProblem(stiff!, [1.0], (0.0, 1.0))
    reference = solve(
        problem,
        SDIRK2();
        adaptive = false,
        dt = 0.01,
        save_everystep = false,
    )
    @test haskey(rust, "sdirk2_fixed")
    @test parse(Float64, only(rust["sdirk2_fixed"])) ≈ only(reference.u[end]) rtol = 2.0e-10 atol = 2.0e-12
    @test parse(Float64, rust["sdirk2"][2]) ≈ cos(1.0) rtol = 2.0e-6 atol = 2.0e-8
    @test parse(Int, rust["sdirk2"][3]) > 0
end
