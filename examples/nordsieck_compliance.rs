use differential_equations::algorithms::multistep::{AN5, JVODE, JVODE_Adams, JVODE_BDF};
use differential_equations::{OdeAlgorithm, OdeProblem, SaveMode, SolveOptions, solve};

fn endpoint<A: OdeAlgorithm>(algorithm: A) -> f64 {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(0.001)
        .with_save(SaveMode::Endpoints);
    solve(&problem, algorithm, &options).unwrap().last_state()[0]
}

fn main() {
    for (name, value) in [
        ("an5", endpoint(AN5)),
        ("jvode", endpoint(JVODE::default())),
        ("jvode_adams", endpoint(JVODE_Adams::default())),
        ("jvode_bdf", endpoint(JVODE_BDF::default())),
    ] {
        println!("{name},{value:.17e}");
    }
}
