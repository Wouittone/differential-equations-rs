use differential_equations::{
    OdeAlgorithm, OdeProblem, SaveMode, SolveOptions, SspRk53, SspRk53H, SspRk53TwoN1,
    SspRk53TwoN2, SspRk54, SspRk63, SspRk73, SspRk83, SspRk104, SspRk932, SspRkMsvs32, SspRkMsvs43,
    solve,
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
    println!("ssprk53,{:.17e}", endpoint(SspRk53));
    println!("ssprk53_2n1,{:.17e}", endpoint(SspRk53TwoN1));
    println!("ssprk53_2n2,{:.17e}", endpoint(SspRk53TwoN2));
    println!("ssprk53_h,{:.17e}", endpoint(SspRk53H));
    println!("ssprk63,{:.17e}", endpoint(SspRk63));
    println!("ssprk73,{:.17e}", endpoint(SspRk73));
    println!("ssprk83,{:.17e}", endpoint(SspRk83));
    println!("ssprk54,{:.17e}", endpoint(SspRk54));
    println!("ssprk104,{:.17e}", endpoint(SspRk104));
    println!("ssprk932,{:.17e}", endpoint(SspRk932));
    println!("ssprkmsvs32,{:.17e}", endpoint(SspRkMsvs32));
    println!("ssprkmsvs43,{:.17e}", endpoint(SspRkMsvs43));
}
