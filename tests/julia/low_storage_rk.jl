using OrdinaryDiffEqLowStorageRK:
    CarpenterKennedy2N54,
    DGLDDRK73_C,
    DGLDDRK84_C,
    DGLDDRK84_F,
    NDBLSRK124,
    NDBLSRK134,
    NDBLSRK144,
    ORK256,
    SHLDDRK64

function rust_low_storage_endpoints()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example low_storage_rk_compliance`
    rows = split(chomp(read(command, String)), '\n')
    Dict(
        first(fields) => parse(Float64, fields[2])
        for fields in split.(strip.(rows), ',')
    )
end

function low_storage_reference(algorithm)
    function nonautonomous!(du, u, _, t)
        du[1] = u[1] + t
    end
    problem = ODEProblem(nonautonomous!, [1.0], (0.0, 1.0))
    solution = solve(
        problem,
        algorithm;
        adaptive = false,
        dt = 0.01,
        save_everystep = false,
    )
    only(solution.u[end])
end

@testset "Low-storage Runge--Kutta compliance" begin
    rust = rust_low_storage_endpoints()
    julia = Dict(
        "ork256" => low_storage_reference(ORK256()),
        "carpenter_kennedy_2n54" => low_storage_reference(CarpenterKennedy2N54()),
        "shlddrk64" => low_storage_reference(SHLDDRK64()),
        "dglddrk73_c" => low_storage_reference(DGLDDRK73_C()),
        "dglddrk84_c" => low_storage_reference(DGLDDRK84_C()),
        "dglddrk84_f" => low_storage_reference(DGLDDRK84_F()),
        "ndblsrk124" => low_storage_reference(NDBLSRK124()),
        "ndblsrk134" => low_storage_reference(NDBLSRK134()),
        "ndblsrk144" => low_storage_reference(NDBLSRK144()),
    )

    @test Set(keys(rust)) == Set(keys(julia))
    for name in keys(julia)
        @testset "$name nonautonomous endpoint" begin
            @test rust[name] ≈ julia[name] rtol = 5.0e-13 atol = 5.0e-14
        end
    end
end
