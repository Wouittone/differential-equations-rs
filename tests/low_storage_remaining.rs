const LOW_STORAGE_SOURCE: &str = include_str!("../src/low_storage_rk.rs");

#[test]
fn remaining_low_storage_family_names_are_declared() {
    let names = [
        "RK46NL",
        "CFRLDDRK64",
        "TSLDDRK74",
        "SHLDDRK52",
        "SHLDDRK_2N",
        "RDPK3Sp35",
        "RDPK3Sp49",
        "RDPK3Sp510",
        "RDPK3SpFSAL35",
        "RDPK3SpFSAL49",
        "RDPK3SpFSAL510",
        "CKLLSRK43_2",
        "CKLLSRK54_3C",
        "CKLLSRK95_4S",
        "CKLLSRK95_4C",
        "CKLLSRK95_4M",
        "CKLLSRK54_3C_3R",
        "CKLLSRK54_3M_3R",
        "CKLLSRK54_3N_3R",
        "CKLLSRK85_4C_3R",
        "CKLLSRK85_4M_3R",
        "CKLLSRK85_4P_3R",
        "CKLLSRK54_3N_4R",
        "CKLLSRK54_3M_4R",
        "CKLLSRK65_4M_4R",
        "CKLLSRK85_4FM_4R",
        "CKLLSRK75_4M_5R",
    ];

    for name in names {
        assert!(
            LOW_STORAGE_SOURCE.contains(name),
            "missing low-storage family {name}"
        );
    }
}
