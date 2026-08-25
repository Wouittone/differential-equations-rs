using OrdinaryDiffEqSIMDRK: MER5v2, MER6v2, RK6v4
using SciMLBase: ODEProblem, solve

function rust_simd_rk_results()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example simd_rk_compliance`
    rows = split(chomp(read(command, String)), '\n')
    Dict(first(fields) => parse(Float64, fields[2]) for fields in split.(strip.(rows), ','))
end

function simd_endpoint(algorithm)
    # The pinned SIMD package intentionally exposes only its constant-cache
    # out-of-place path; packed stage lanes flow through this scalar identity.
    problem = ODEProblem((u, _, _) -> u, 1.0, (0.0, 1.0))
    solve(problem, algorithm; adaptive = false, dt = 0.05, save_everystep = false).u[end]
end

@testset "SIMD Runge--Kutta pinned compliance" begin
    rust = rust_simd_rk_results()
    @test Set(keys(rust)) == Set(["MER5v2", "MER6v2", "RK6v4"])
    @test isapprox(rust["MER5v2"], simd_endpoint(MER5v2()); rtol = 3.0e-12, atol = 3.0e-14)
    @test isapprox(rust["MER6v2"], simd_endpoint(MER6v2()); rtol = 3.0e-12, atol = 3.0e-14)
    @test isapprox(rust["RK6v4"], simd_endpoint(RK6v4()); rtol = 3.0e-12, atol = 3.0e-14)
end
