use differential_equations::algorithms::taylor::{
    ExplicitTaylor, ExplicitTaylor2, ExplicitTaylorAdaptiveOrder,
};
use differential_equations::{OdeAlgorithm, OdeProblem, SaveMode, SolveOptions};

fn endpoint<A: OdeAlgorithm>(algorithm: A, step: f64) -> f64 {
    let problem = OdeProblem::new(
        |output: &mut [f64], state: &[f64], _: &(), _: f64| output[0] = state[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    algorithm
        .solve(
            &problem,
            &SolveOptions {
                adaptive: false,
                initial_step: Some(step),
                save: SaveMode::Endpoints,
                ..SolveOptions::default()
            },
        )
        .unwrap()
        .last_state()[0]
}

fn main() {
    println!("ExplicitTaylor2,{:.17e}", endpoint(ExplicitTaylor2, 0.01));
    println!(
        "ExplicitTaylor,{:.17e}",
        endpoint(ExplicitTaylor::new(8), 0.1)
    );
    println!(
        "ExplicitTaylorAdaptiveOrder,{:.17e}",
        endpoint(ExplicitTaylorAdaptiveOrder::new(6, 7), 0.1)
    );
}
