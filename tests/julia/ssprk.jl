using OrdinaryDiffEqSSPRK:
    KYK2014DGSSPRK_3S2, SSPRK22, SSPRK33, SSPRK43, SSPRK432, pRRK22, pRRK33

function rust_ssprk_endpoints()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example ssprk_compliance`
    rows = split(chomp(read(command, String)), '\n')
    Dict(
        first(fields) => parse(Float64, fields[2])
        for fields in split.(strip.(rows), ',')
    )
end

function ssprk_reference(algorithm; adaptive)
    function exponential!(du, u, _, _)
        du[1] = u[1]
    end
    problem = ODEProblem(exponential!, [1.0], (0.0, 1.0))
    if adaptive
        solution = solve(
            problem,
            algorithm;
            abstol = 1.0e-9,
            reltol = 1.0e-9,
            save_everystep = false,
        )
    else
        solution = solve(
            problem,
            algorithm;
            adaptive = false,
            dt = 0.01,
            save_everystep = false,
        )
    end
    only(solution.u[end])
end

@testset "SSP Runge–Kutta compliance" begin
    rust = rust_ssprk_endpoints()
    julia = Dict(
        "ssprk22" => ssprk_reference(SSPRK22(); adaptive = false),
        "prrk22" => ssprk_reference(pRRK22(); adaptive = false),
        "prrk33" => ssprk_reference(pRRK33(); adaptive = false),
        "ssprk33" => ssprk_reference(SSPRK33(); adaptive = false),
        "ssprk43" => ssprk_reference(SSPRK43(); adaptive = true),
        "ssprk432" => ssprk_reference(SSPRK432(); adaptive = true),
        "kyk2014dgssprk_3s2" => ssprk_reference(KYK2014DGSSPRK_3S2(); adaptive = false),
    )

    @test Set(keys(rust)) == Set(keys(julia))
    for name in keys(julia)
        @testset "$name endpoint" begin
            @test rust[name] ≈ julia[name] rtol = 2.0e-7 atol = 2.0e-9
        end
    end
end
