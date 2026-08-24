using OrdinaryDiffEqMultirate: MIS, MRAB, MREEF, MRIGARKERK22a, MRIGARKERK22b,
    MRIGARKERK33a, MRIGARKERK45a, MRIGARKESDIRK34a, MRIGARKIRK21a

function rust_multirate_results()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example multirate_compliance`
    rows = split(chomp(read(command, String)), '\n')
    Dict(first(fields) => parse(Float64, fields[2]) for fields in split.(strip.(rows), ','))
end

function multirate_problem()
    SplitODEProblem(
        (u, _, _) -> -0.9 * u,
        (u, _, _) -> -0.1 * u,
        1.0,
        (0.0, 1.0),
    )
end

function multirate_endpoint(algorithm)
    solution = solve(
        multirate_problem(),
        algorithm;
        adaptive = false,
        dt = 0.05,
        save_everystep = false,
    )
    only(solution.u[end:end])
end

@testset "multirate and MRI-GARK compliance" begin
    rust = rust_multirate_results()
    algorithms = Dict(
        "mis" => MIS(m = 8),
        "mrab" => MRAB(k = 3, m = 8),
        "mreef" => MREEF(),
        "erk22a" => MRIGARKERK22a(m = 8),
        "erk22b" => MRIGARKERK22b(m = 8),
        "erk33a" => MRIGARKERK33a(m = 8),
        "erk45a" => MRIGARKERK45a(m = 8),
        "esdirk34a" => MRIGARKESDIRK34a(m = 8),
        "irk21a" => MRIGARKIRK21a(m = 8),
    )
    @test Set(keys(rust)) == Set(keys(algorithms))
    for (name, algorithm) in algorithms
        @testset "$name endpoint" begin
            julia = multirate_endpoint(algorithm)
            @test isapprox(rust[name], julia; rtol = 2.0e-9, atol = 2.0e-11)
        end
    end
end
