use differential_equations::{
    OdeProblem, SaveMode, SolveOptions, SspRk22, SspRk33, SspRk43, solve,
};

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn problem() -> OdeProblem<TestRhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
        du[0] = u[0];
    }
    OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
}

fn fixed_options() -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(0.01),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

fn adaptive_options() -> SolveOptions {
    SolveOptions {
        absolute_tolerance: 1.0e-9,
        relative_tolerance: 1.0e-9,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

fn main() {
    let rk22 = solve(&problem(), SspRk22, &fixed_options()).unwrap();
    println!("ssprk22,{:.17e}", rk22.last_state()[0]);

    let rk33 = solve(&problem(), SspRk33, &fixed_options()).unwrap();
    println!("ssprk33,{:.17e}", rk33.last_state()[0]);

    let rk43 = solve(&problem(), SspRk43, &adaptive_options()).unwrap();
    println!("ssprk43,{:.17e}", rk43.last_state()[0]);
}
