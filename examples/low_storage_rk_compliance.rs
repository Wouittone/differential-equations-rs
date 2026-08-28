use differential_equations::solvers::explicit::*;
use differential_equations::*;

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn problem() -> OdeProblem<TestRhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), time: f64) {
        du[0] = u[0] + time;
    }
    OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
}

fn endpoint<A: OdeAlgorithm>(algorithm: A) -> f64 {
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.01),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    solve(&problem(), algorithm, &options).unwrap().last_state()[0]
}

fn main() {
    println!("ork256,{:.17e}", endpoint(Ork256));
    println!(
        "carpenter_kennedy_2n54,{:.17e}",
        endpoint(CarpenterKennedy2N54)
    );
    println!(
        "parsani_ketcheson_deconinck_3s32,{:.17e}",
        endpoint(ParsaniKetchesonDeconinck3S32)
    );
    println!(
        "parsani_ketcheson_deconinck_3s53,{:.17e}",
        endpoint(ParsaniKetchesonDeconinck3S53)
    );
    println!(
        "parsani_ketcheson_deconinck_3s173,{:.17e}",
        endpoint(ParsaniKetchesonDeconinck3S173)
    );
    println!(
        "parsani_ketcheson_deconinck_3s184,{:.17e}",
        endpoint(ParsaniKetchesonDeconinck3S184)
    );
    println!(
        "parsani_ketcheson_deconinck_3s105,{:.17e}",
        endpoint(ParsaniKetchesonDeconinck3S105)
    );
    println!(
        "parsani_ketcheson_deconinck_3s82,{:.17e}",
        endpoint(ParsaniKetchesonDeconinck3S82)
    );
    println!(
        "parsani_ketcheson_deconinck_3s94,{:.17e}",
        endpoint(ParsaniKetchesonDeconinck3S94)
    );
    println!(
        "parsani_ketcheson_deconinck_3s205,{:.17e}",
        endpoint(ParsaniKetchesonDeconinck3S205)
    );
    println!("shlddrk64,{:.17e}", endpoint(Shlddrk64));
    println!("dglddrk73_c,{:.17e}", endpoint(Dglddrk73C));
    println!("dglddrk84_c,{:.17e}", endpoint(Dglddrk84C));
    println!("dglddrk84_f,{:.17e}", endpoint(Dglddrk84F));
    println!("ndblsrk124,{:.17e}", endpoint(Ndblsrk124));
    println!("ndblsrk134,{:.17e}", endpoint(Ndblsrk134));
    println!("ndblsrk144,{:.17e}", endpoint(Ndblsrk144));
    println!("cfrlddrk64,{:.17e}", endpoint(CFRLDDRK64));
    println!("ckllsrk43_2,{:.17e}", endpoint(CKLLSRK43_2));
    println!("ckllsrk54_3c,{:.17e}", endpoint(CKLLSRK54_3C));
    println!("ckllsrk54_3c_3r,{:.17e}", endpoint(CKLLSRK54_3C_3R));
    println!("ckllsrk54_3m_3r,{:.17e}", endpoint(CKLLSRK54_3M_3R));
    println!("ckllsrk54_3m_4r,{:.17e}", endpoint(CKLLSRK54_3M_4R));
    println!("ckllsrk54_3n_3r,{:.17e}", endpoint(CKLLSRK54_3N_3R));
    println!("ckllsrk54_3n_4r,{:.17e}", endpoint(CKLLSRK54_3N_4R));
    println!("ckllsrk65_4m_4r,{:.17e}", endpoint(CKLLSRK65_4M_4R));
    println!("ckllsrk75_4m_5r,{:.17e}", endpoint(CKLLSRK75_4M_5R));
    println!("ckllsrk85_4c_3r,{:.17e}", endpoint(CKLLSRK85_4C_3R));
    println!("ckllsrk85_4fm_4r,{:.17e}", endpoint(CKLLSRK85_4FM_4R));
    println!("ckllsrk85_4m_3r,{:.17e}", endpoint(CKLLSRK85_4M_3R));
    println!("ckllsrk85_4p_3r,{:.17e}", endpoint(CKLLSRK85_4P_3R));
    println!("ckllsrk95_4c,{:.17e}", endpoint(CKLLSRK95_4C));
    println!("ckllsrk95_4m,{:.17e}", endpoint(CKLLSRK95_4M));
    println!("ckllsrk95_4s,{:.17e}", endpoint(CKLLSRK95_4S));
    println!("rdpk3sp35,{:.17e}", endpoint(RDPK3Sp35));
    println!("rdpk3sp49,{:.17e}", endpoint(RDPK3Sp49));
    println!("rdpk3sp510,{:.17e}", endpoint(RDPK3Sp510));
    println!("rdpk3spfsal35,{:.17e}", endpoint(RDPK3SpFSAL35));
    println!("rdpk3spfsal49,{:.17e}", endpoint(RDPK3SpFSAL49));
    println!("rdpk3spfsal510,{:.17e}", endpoint(RDPK3SpFSAL510));
    println!("rk46nl,{:.17e}", endpoint(RK46NL));
    println!("shlddrk_2n,{:.17e}", endpoint(SHLDDRK_2N));
    println!("shlddrk52,{:.17e}", endpoint(SHLDDRK52));
    println!("tslddrk74,{:.17e}", endpoint(TSLDDRK74));
}
