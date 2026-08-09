using OrdinaryDiffEqLowOrderRK:
    Alshina2, Alshina3, Alshina6, Anas5, Euler, Midpoint, Heun, Ralston, Ralston4, RK4, RKM, RKO65, MSRK5, MSRK6, FRK65, PSRK3p5q4, PSRK3p6q5, BS3, DP5

function rust_low_order_endpoints()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example low_order_compliance`
    rows = split(chomp(read(command, String)), '\n')
    Dict(
        first(fields) => parse(Float64, fields[2])
        for fields in split.(strip.(rows), ',')
    )
end

function exponential_reference(algorithm; adaptive = true, dt = nothing)
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
            dt,
            save_everystep = false,
        )
    end
    only(solution.u[end])
end

@testset "Low-order explicit RK compliance" begin
    rust = rust_low_order_endpoints()
    julia = Dict(
        "euler" => exponential_reference(Euler(); adaptive = false, dt = 0.001),
        "rk4" => exponential_reference(RK4(); adaptive = false, dt = 0.01),
        "rkm" => exponential_reference(RKM(); adaptive = false, dt = 0.01),
        "rko65" => exponential_reference(RKO65(); adaptive = false, dt = 0.01),
        "ralston4" => exponential_reference(Ralston4(); adaptive = false, dt = 0.01),
        "alshina2" => exponential_reference(Alshina2(); adaptive = false, dt = 0.01),
        "alshina3" => exponential_reference(Alshina3(); adaptive = false, dt = 0.01),
        "alshina6" => exponential_reference(Alshina6(); adaptive = false, dt = 0.01),
        "anas5" => exponential_reference(Anas5(); adaptive = false, dt = 0.01),
        "msrk5" => exponential_reference(MSRK5(); adaptive = false, dt = 0.01),
        "msrk6" => exponential_reference(MSRK6(); adaptive = false, dt = 0.01),
        "frk65" => exponential_reference(FRK65(); adaptive = false, dt = 0.01),
        "psrk3p5q4" => exponential_reference(PSRK3p5q4(); adaptive = false, dt = 0.01),
        "psrk3p6q5" => exponential_reference(PSRK3p6q5(); adaptive = false, dt = 0.01),
        "midpoint" => exponential_reference(Midpoint()),
        "heun" => exponential_reference(Heun()),
        "ralston" => exponential_reference(Ralston()),
        "bs3" => exponential_reference(BS3()),
        "dp5" => exponential_reference(DP5()),
    )

    @test Set(keys(rust)) == Set(keys(julia))
    for name in keys(julia)
        @testset "$name endpoint" begin
            @test rust[name] ≈ julia[name] rtol = 2.0e-7 atol = 2.0e-9
            @test rust[name] ≈ ℯ rtol = 8.0e-4 atol = 2.0e-9
        end
    end
end
