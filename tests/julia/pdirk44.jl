using OrdinaryDiffEqPDIRK: PDIRK44

function rust_pdirk44_results()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example pdirk44_compliance`
    fields = split(chomp(read(command, String)), ',')
    @test first(fields) == "pdirk44_fixed"
    parse(Float64, fields[2])
end

function pdirk44_rhs!(du, u, p, t)
    du[1] = u[1]
end

pdirk44_problem = ODEProblem(pdirk44_rhs!, [1.0], (0.0, 1.0))
pdirk44_solution = solve(
    pdirk44_problem,
    PDIRK44(threading = false),
    adaptive = false,
    dt = 0.05,
    save_everystep = false,
)

@test isapprox(rust_pdirk44_results(), pdirk44_solution.u[end][1]; atol = 2.0e-12, rtol = 2.0e-12)
