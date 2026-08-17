using OrdinaryDiffEqSSPRK:
    SSPRK53,
    SSPRK53_2N1,
    SSPRK53_2N2,
    SSPRK53_H,
    SSPRK54,
    SSPRK63,
    SSPRK73,
    SSPRK83,
    SSPRK104,
    SSPRK932,
    SSPRKMSVS32

function rust_extended_ssprk_endpoints()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example ssprk_extended_compliance`
    rows = split(chomp(read(command, String)), '\n')
    Dict(
        first(fields) => parse(Float64, fields[2])
        for fields in split.(strip.(rows), ',')
    )
end

function extended_ssprk_reference(algorithm)
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

function extended_ssprk_reference(algorithm::SSPRK932)
    function nonautonomous(u, _, t)
        [u[1] + t]
    end
    # The pinned SSPRK932 in-place cache applies a different recurrence than
    # its constant-cache path. Compare the documented method via that path.
    problem = ODEProblem(nonautonomous, [1.0], (0.0, 1.0))
    solution = solve(
        problem,
        algorithm;
        adaptive = false,
        dt = 0.01,
        save_everystep = false,
    )
    only(solution.u[end])
end

@testset "Extended SSP Runge--Kutta compliance" begin
    rust = rust_extended_ssprk_endpoints()
    julia = Dict(
        "ssprk53" => extended_ssprk_reference(SSPRK53()),
        "ssprk53_2n1" => extended_ssprk_reference(SSPRK53_2N1()),
        "ssprk53_2n2" => extended_ssprk_reference(SSPRK53_2N2()),
        "ssprk53_h" => extended_ssprk_reference(SSPRK53_H()),
        "ssprk63" => extended_ssprk_reference(SSPRK63()),
        "ssprk73" => extended_ssprk_reference(SSPRK73()),
        "ssprk83" => extended_ssprk_reference(SSPRK83()),
        "ssprk54" => extended_ssprk_reference(SSPRK54()),
        "ssprk104" => extended_ssprk_reference(SSPRK104()),
        "ssprk932" => extended_ssprk_reference(SSPRK932()),
        "ssprkmsvs32" => extended_ssprk_reference(SSPRKMSVS32()),
    )

    @test Set(keys(rust)) == Set(keys(julia))
    for name in keys(julia)
        @testset "$name nonautonomous endpoint" begin
            @test rust[name] ≈ julia[name] rtol = 5.0e-13 atol = 5.0e-14
        end
    end
end
