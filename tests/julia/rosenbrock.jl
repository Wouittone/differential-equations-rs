using OrdinaryDiffEqRosenbrock: Rosenbrock23

function rust_rosenbrock_result()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example rosenbrock_compliance`
    fields = split(strip(read(command, String)), ',')
    (endpoint = parse(Float64, fields[2]), rhs_evaluations = parse(Int, fields[3]))
end

function rosenbrock_reference()
    function stiff_nonautonomous!(du, u, _, time)
        du[1] = -1000.0 * (u[1] - cos(time)) - sin(time)
    end
    problem = ODEProblem(stiff_nonautonomous!, [1.0], (0.0, 1.0))
    solve(
        problem,
        Rosenbrock23();
        abstol = 1.0e-7,
        reltol = 1.0e-7,
        save_everystep = false,
    )
end

@testset "Rosenbrock23 compliance" begin
    rust = rust_rosenbrock_result()
    julia = rosenbrock_reference()

    @test rust.endpoint ≈ only(julia.u[end]) rtol = 3.0e-7 atol = 3.0e-9
    @test rust.endpoint ≈ cos(1.0) rtol = 3.0e-7 atol = 3.0e-9
    @test rust.rhs_evaluations > 0
end
