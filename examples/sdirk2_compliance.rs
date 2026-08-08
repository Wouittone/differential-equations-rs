use differential_equations::{OdeProblem, SaveMode, Sdirk2, SolveOptions, solve};

fn main() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), t: f64| {
            du[0] = -15.0 * (u[0] - t.cos()) - t.sin();
        },
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions {
        absolute_tolerance: 1.0e-8,
        relative_tolerance: 1.0e-8,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let solution = solve(&problem, Sdirk2, &options).expect("SDIRK2 solve");
    println!(
        "sdirk2 final t={:.6} u={:.12e} accepted={} rejected={}",
        solution.times().last().copied().unwrap_or_default(),
        solution.last_state()[0],
        solution.stats().accepted_steps,
        solution.stats().rejected_steps
    );
}
