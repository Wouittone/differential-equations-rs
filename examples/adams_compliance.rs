use differential_equations::{
    Ab3, Ab4, Ab5, Abm32, Abm43, Abm54, OdeProblem, SaveMode, SolveOptions, solve,
};

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

    let abm32 = solve(&problem(), Abm32, &options()).unwrap();
    println!("abm32,{:.17e}", abm32.last_state()[0]);

    let abm43 = solve(&problem(), Abm43, &options()).unwrap();
    println!("abm43,{:.17e}", abm43.last_state()[0]);

    let abm54 = solve(&problem(), Abm54, &options()).unwrap();
    println!("abm54,{:.17e}", abm54.last_state()[0]);
}
