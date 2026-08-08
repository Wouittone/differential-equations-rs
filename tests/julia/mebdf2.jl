using OrdinaryDiffEqBDF: MEBDF2

function rust_mebdf2_results()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example mebdf2_compliance`
    Dict(first(fields) => fields[2:end] for fields in split.(strip.(split(chomp(read(command, String)), '\n')), ','))
end

@testset "MEBDF2 regular ODE compliance" begin
    rust = rust_mebdf2_results()
    function stiff!(du, u, _, t)
        du[1] = -15.0 * (u[1] - cos(t)) - sin(t)
    end
    problem = ODEProblem(stiff!, [1.0], (0.0, 1.0))
    reference = solve(problem, MEBDF2(); adaptive = false, dt = 0.01, save_everystep = false)
    @test parse(Float64, rust["mebdf2"][1]) ≈ only(reference.u[end]) rtol = 2.0e-10 atol = 2.0e-12
    @test parse(Int, rust["mebdf2"][2]) == 100
end

