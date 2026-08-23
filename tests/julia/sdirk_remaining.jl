using OrdinaryDiffEqSDIRK:
    ARS222,
    ARS232,
    ARS343,
    ARS443,
    BHR553,
    CFNLIRK3,
    ESDIRK325L2SA,
    ESDIRK436L2SA2,
    ESDIRK437L2SA,
    ESDIRK547L2SA2,
    ESDIRK54I8L2SA,
    ESDIRK659L2SA,
    Hairer4,
    Hairer42,
    IMEXSSP222,
    IMEXSSP2322,
    IMEXSSP3332,
    IMEXSSP3433,
    KenCarp3,
    KenCarp4,
    KenCarp47,
    KenCarp5,
    KenCarp58,
    Kvaerno3,
    Kvaerno4,
    Kvaerno5,
    SDIRK22,
    SFSDIRK4,
    SFSDIRK5,
    SFSDIRK6,
    SFSDIRK7,
    SFSDIRK8,
    SSPSDIRK2

function rust_remaining_sdirk_endpoints()
    manifest = joinpath(REPOSITORY_ROOT, "Cargo.toml")
    command = `cargo run --quiet --release --manifest-path $manifest --example sdirk_remaining_compliance`
    rows = split(chomp(read(command, String)), '\n')
    Dict(
        first(fields) => parse(Float64, fields[2])
        for fields in split.(strip.(rows), ',')
    )
end

function remaining_sdirk_reference(algorithm)
    function nonautonomous!(du, u, _, time)
        du[1] = u[1] + time
    end
    solution = solve(
        ODEProblem(nonautonomous!, [1.0], (0.0, 1.0)),
        algorithm;
        adaptive = false,
        dt = 0.01,
        save_everystep = false,
    )
    only(solution.u[end])
end

@testset "Remaining SDIRK/ESDIRK/IMEX compliance" begin
    rust = rust_remaining_sdirk_endpoints()
    julia = Dict(
        "ars222" => remaining_sdirk_reference(ARS222()),
        "ars232" => remaining_sdirk_reference(ARS232()),
        "ars343" => remaining_sdirk_reference(ARS343()),
        "ars443" => remaining_sdirk_reference(ARS443()),
        "bhr553" => remaining_sdirk_reference(BHR553()),
        "cfnlirk3" => remaining_sdirk_reference(CFNLIRK3()),
        "esdirk325l2sa" => remaining_sdirk_reference(ESDIRK325L2SA()),
        "esdirk436l2sa2" => remaining_sdirk_reference(ESDIRK436L2SA2()),
        "esdirk437l2sa" => remaining_sdirk_reference(ESDIRK437L2SA()),
        "esdirk547l2sa2" => remaining_sdirk_reference(ESDIRK547L2SA2()),
        "esdirk54i8l2sa" => remaining_sdirk_reference(ESDIRK54I8L2SA()),
        "esdirk659l2sa" => remaining_sdirk_reference(ESDIRK659L2SA()),
        "hairer4" => remaining_sdirk_reference(Hairer4()),
        "hairer42" => remaining_sdirk_reference(Hairer42()),
        "imexssp222" => remaining_sdirk_reference(IMEXSSP222()),
        "imexssp2322" => remaining_sdirk_reference(IMEXSSP2322()),
        "imexssp3332" => remaining_sdirk_reference(IMEXSSP3332()),
        "imexssp3433" => remaining_sdirk_reference(IMEXSSP3433()),
        "kencarp3" => remaining_sdirk_reference(KenCarp3()),
        "kencarp4" => remaining_sdirk_reference(KenCarp4()),
        "kencarp47" => remaining_sdirk_reference(KenCarp47()),
        "kencarp5" => remaining_sdirk_reference(KenCarp5()),
        "kencarp58" => remaining_sdirk_reference(KenCarp58()),
        "kvaerno3" => remaining_sdirk_reference(Kvaerno3()),
        "kvaerno4" => remaining_sdirk_reference(Kvaerno4()),
        "kvaerno5" => remaining_sdirk_reference(Kvaerno5()),
        "sdirk22" => remaining_sdirk_reference(SDIRK22()),
        "sfsdirk4" => remaining_sdirk_reference(SFSDIRK4()),
        "sfsdirk5" => remaining_sdirk_reference(SFSDIRK5()),
        "sfsdirk6" => remaining_sdirk_reference(SFSDIRK6()),
        "sfsdirk7" => remaining_sdirk_reference(SFSDIRK7()),
        "sfsdirk8" => remaining_sdirk_reference(SFSDIRK8()),
        "sspsdirk2" => remaining_sdirk_reference(SSPSDIRK2()),
    )

    @test Set(keys(rust)) == Set(keys(julia))
    for name in keys(julia)
        @testset "$name nonautonomous endpoint" begin
            @test isapprox(rust[name], julia[name]; rtol = 2.0e-10, atol = 2.0e-12)
        end
    end
end
