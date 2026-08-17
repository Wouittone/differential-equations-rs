using OrdinaryDiffEqDefault: DefaultImplicitODEAlgorithm, DefaultODEAlgorithm
using OrdinaryDiffEqExplicitRK: ExplicitRK
using OrdinaryDiffEqRosenbrock: Rodas5P
using OrdinaryDiffEqTsit5: AutoTsit5
using OrdinaryDiffEqVerner: AutoVern6, AutoVern7, AutoVern8, AutoVern9

function rust_composite_endpoints()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example composites_compliance`
    rows = split(chomp(read(command, String)), '\n')
    Dict(first(fields) => parse(Float64, fields[2]) for fields in split.(strip.(rows), ','))
end

function composite_reference(algorithm)
    function rhs!(du, u, _, t)
        du[1] = u[1] + t
    end
    solution = solve(
        ODEProblem(rhs!, [1.0], (0.0, 1.0)),
        algorithm;
        abstol = 1.0e-10,
        reltol = 1.0e-10,
        save_everystep = false,
    )
    only(solution.u[end])
end

@testset "automatic/default composite compliance" begin
    rust = rust_composite_endpoints()
    julia = Dict(
        "auto_tsit5" => composite_reference(AutoTsit5(Rodas5P())),
        "auto_vern6" => composite_reference(AutoVern6(Rodas5P())),
        "auto_vern7" => composite_reference(AutoVern7(Rodas5P())),
        "auto_vern8" => composite_reference(AutoVern8(Rodas5P())),
        "auto_vern9" => composite_reference(AutoVern9(Rodas5P())),
        "default_ode_algorithm" => composite_reference(DefaultODEAlgorithm()),
        "default_implicit_ode_algorithm" => composite_reference(DefaultImplicitODEAlgorithm()),
        "explicit_rk" => composite_reference(ExplicitRK()),
    )
    @test Set(keys(rust)) == Set(keys(julia))
    for name in keys(julia)
        @testset "$name endpoint" begin
            @test rust[name] ≈ julia[name] rtol = 2.0e-7 atol = 2.0e-9
        end
    end
end
