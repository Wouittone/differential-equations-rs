using OrdinaryDiffEqLowOrderRK: SplitEuler
using SciMLBase: SplitODEProblem

function rust_split_euler_endpoint()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example split_euler_compliance`
    fields = split(strip(read(command, String)), ',')
    parse(Float64, fields[2])
end

@testset "SplitEuler typed split-problem compliance" begin
    function explicit!(du, u, _, _)
        du[1] = u[1]
    end
    function implicit!(du, _, _, time)
        du[1] = time
    end
    problem = SplitODEProblem(explicit!, implicit!, [1.0], (0.0, 1.0))
    solution = solve(
        problem,
        SplitEuler();
        adaptive = false,
        dt = 0.01,
        save_everystep = false,
    )
    @test rust_split_euler_endpoint() ≈ only(solution.u[end]) rtol = 2.0e-14 atol = 2.0e-14
end
