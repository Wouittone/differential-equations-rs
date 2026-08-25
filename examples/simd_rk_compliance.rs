use differential_equations::algorithms::simd::{MER5v2, MER6v2, RK6v4};
use differential_equations::{OdeAlgorithm, OdeProblem, SaveMode, SolveOptions};

fn endpoint<A: OdeAlgorithm>(algorithm: A) -> f64 {
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
                initial_step: Some(0.05),
                save: SaveMode::Endpoints,
                ..SolveOptions::default()
            },
        )
        .unwrap()
        .last_state()[0]
}

fn main() {
    println!("MER5v2,{:.17e}", endpoint(MER5v2));
    println!("MER6v2,{:.17e}", endpoint(MER6v2));
    println!("RK6v4,{:.17e}", endpoint(RK6v4));
}
