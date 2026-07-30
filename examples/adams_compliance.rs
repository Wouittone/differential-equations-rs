use differential_equations::{Ab3, Ab4, Ab5, OdeProblem, SaveMode, SolveOptions, solve};

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn problem() -> OdeProblem<TestRhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
        du[0] = u[0];
    }
    OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
}

fn options() -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(0.01),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

fn main() {
    let ab3 = solve(&problem(), Ab3, &options()).unwrap();
    println!("ab3,{:.17e}", ab3.last_state()[0]);

    let ab4 = solve(&problem(), Ab4, &options()).unwrap();
    println!("ab4,{:.17e}", ab4.last_state()[0]);

    let ab5 = solve(&problem(), Ab5, &options()).unwrap();
    println!("ab5,{:.17e}", ab5.last_state()[0]);
}
