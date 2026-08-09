use differential_equations::{
    CarpenterKennedy2N54, Dglddrk73C, Dglddrk84C, Dglddrk84F, Ndblsrk124, Ndblsrk134, Ndblsrk144,
    OdeAlgorithm, OdeProblem, Ork256, ParsaniKetchesonDeconinck3S32, ParsaniKetchesonDeconinck3S53,
    ParsaniKetchesonDeconinck3S82, ParsaniKetchesonDeconinck3S94, ParsaniKetchesonDeconinck3S105,
    ParsaniKetchesonDeconinck3S173, ParsaniKetchesonDeconinck3S184, ParsaniKetchesonDeconinck3S205,
    SaveMode, Shlddrk64, SolveOptions, solve,
};

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
}
