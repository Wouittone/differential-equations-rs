using OrdinaryDiffEqLowStorageRK:
    CFRLDDRK64,
    CKLLSRK43_2,
    CKLLSRK54_3C,
    CKLLSRK54_3C_3R,
    CKLLSRK54_3M_3R,
    CKLLSRK54_3M_4R,
    CKLLSRK54_3N_3R,
    CKLLSRK54_3N_4R,
    CKLLSRK65_4M_4R,
    CKLLSRK75_4M_5R,
    CKLLSRK85_4C_3R,
    CKLLSRK85_4FM_4R,
    CKLLSRK85_4M_3R,
    CKLLSRK85_4P_3R,
    CKLLSRK95_4C,
    CKLLSRK95_4M,
    CKLLSRK95_4S,
    CarpenterKennedy2N54,
    DGLDDRK73_C,
    DGLDDRK84_C,
    DGLDDRK84_F,
    NDBLSRK124,
    NDBLSRK134,
    NDBLSRK144,
    ORK256,
    RDPK3Sp35,
    RDPK3Sp49,
    RDPK3Sp510,
    RDPK3SpFSAL35,
    RDPK3SpFSAL49,
    RDPK3SpFSAL510,
    RK46NL,
    SHLDDRK_2N,
    SHLDDRK52,
    SHLDDRK64,
    TSLDDRK74,
    ParsaniKetchesonDeconinck3S32,
    ParsaniKetchesonDeconinck3S53,
    ParsaniKetchesonDeconinck3S173,
    ParsaniKetchesonDeconinck3S184,
    ParsaniKetchesonDeconinck3S105,
    ParsaniKetchesonDeconinck3S82,
    ParsaniKetchesonDeconinck3S94,
    ParsaniKetchesonDeconinck3S205

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

function low_storage_reference(algorithm::SHLDDRK_2N)
    function nonautonomous(u, _, t)
        [u[1] + t]
    end
    # The pinned in-place and constant-cache SHLDDRK_2N recurrences differ.
    # Compare the documented method through the constant-cache path.
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
        "parsani_ketcheson_deconinck_3s32" =>
            low_storage_reference(ParsaniKetchesonDeconinck3S32()),
        "parsani_ketcheson_deconinck_3s53" =>
            low_storage_reference(ParsaniKetchesonDeconinck3S53()),
        "parsani_ketcheson_deconinck_3s173" =>
            low_storage_reference(ParsaniKetchesonDeconinck3S173()),
        "parsani_ketcheson_deconinck_3s184" =>
            low_storage_reference(ParsaniKetchesonDeconinck3S184()),
        "parsani_ketcheson_deconinck_3s105" =>
            low_storage_reference(ParsaniKetchesonDeconinck3S105()),
        "parsani_ketcheson_deconinck_3s82" =>
            low_storage_reference(ParsaniKetchesonDeconinck3S82()),
        "parsani_ketcheson_deconinck_3s94" =>
            low_storage_reference(ParsaniKetchesonDeconinck3S94()),
        "parsani_ketcheson_deconinck_3s205" =>
            low_storage_reference(ParsaniKetchesonDeconinck3S205()),
        "cfrlddrk64" => low_storage_reference(CFRLDDRK64()),
        "ckllsrk43_2" => low_storage_reference(CKLLSRK43_2()),
        "ckllsrk54_3c" => low_storage_reference(CKLLSRK54_3C()),
        "ckllsrk54_3c_3r" => low_storage_reference(CKLLSRK54_3C_3R()),
        "ckllsrk54_3m_3r" => low_storage_reference(CKLLSRK54_3M_3R()),
        "ckllsrk54_3m_4r" => low_storage_reference(CKLLSRK54_3M_4R()),
        "ckllsrk54_3n_3r" => low_storage_reference(CKLLSRK54_3N_3R()),
        "ckllsrk54_3n_4r" => low_storage_reference(CKLLSRK54_3N_4R()),
        "ckllsrk65_4m_4r" => low_storage_reference(CKLLSRK65_4M_4R()),
        "ckllsrk75_4m_5r" => low_storage_reference(CKLLSRK75_4M_5R()),
        "ckllsrk85_4c_3r" => low_storage_reference(CKLLSRK85_4C_3R()),
        "ckllsrk85_4fm_4r" => low_storage_reference(CKLLSRK85_4FM_4R()),
        "ckllsrk85_4m_3r" => low_storage_reference(CKLLSRK85_4M_3R()),
        "ckllsrk85_4p_3r" => low_storage_reference(CKLLSRK85_4P_3R()),
        "ckllsrk95_4c" => low_storage_reference(CKLLSRK95_4C()),
        "ckllsrk95_4m" => low_storage_reference(CKLLSRK95_4M()),
        "ckllsrk95_4s" => low_storage_reference(CKLLSRK95_4S()),
        "rdpk3sp35" => low_storage_reference(RDPK3Sp35()),
        "rdpk3sp49" => low_storage_reference(RDPK3Sp49()),
        "rdpk3sp510" => low_storage_reference(RDPK3Sp510()),
        "rdpk3spfsal35" => low_storage_reference(RDPK3SpFSAL35()),
        "rdpk3spfsal49" => low_storage_reference(RDPK3SpFSAL49()),
        "rdpk3spfsal510" => low_storage_reference(RDPK3SpFSAL510()),
        "rk46nl" => low_storage_reference(RK46NL()),
        "shlddrk_2n" => low_storage_reference(SHLDDRK_2N()),
        "shlddrk52" => low_storage_reference(SHLDDRK52()),
        "tslddrk74" => low_storage_reference(TSLDDRK74()),
    )

    @test Set(keys(rust)) == Set(keys(julia))
    for name in keys(julia)
        @testset "$name nonautonomous endpoint" begin
            @test rust[name] ≈ julia[name] rtol = 5.0e-13 atol = 5.0e-14
        end
    end
end
