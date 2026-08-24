using OrdinaryDiffEqNordsieck: AN5, JVODE, JVODE_Adams, JVODE_BDF

function rust_nordsieck_results()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example nordsieck_compliance`
    rows = split(chomp(read(command, String)), '\n')
    Dict(first(fields) => parse(Float64, fields[2]) for fields in split.(strip.(rows), ','))
end

function nordsieck_endpoint(algorithm)
    rhs(u, _, _) = u
    problem = ODEProblem(rhs, 1.0, (0.0, 1.0))
    solution = solve(problem, algorithm; adaptive = false, dt = 0.001, save_everystep = false)
    solution.u[end]
end

@testset "Nordsieck compliance" begin
    rust = rust_nordsieck_results()
    algorithms = Dict(
        "an5" => AN5(),
        "jvode" => JVODE(),
        "jvode_adams" => JVODE_Adams(),
        "jvode_bdf" => JVODE_BDF(),
    )
    @test Set(keys(rust)) == Set(keys(algorithms))
    for (name, algorithm) in algorithms
        @testset "$name endpoint" begin
            julia = nordsieck_endpoint(algorithm)
            @test isapprox(rust[name], julia; rtol = 6.0e-4, atol = 2.0e-7)
        end
    end
end
