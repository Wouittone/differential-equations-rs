use differential_equations::{OdeProblem, Qbdf2, Qndf2, SaveMode, SolveOptions, solve};

fn main() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), t: f64| {
            du[0] = -15.0 * (u[0] - t.cos()) - t.sin();
        },
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let fixed_options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.01),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let fixed = solve(&problem, Qndf2, &fixed_options).expect("QNDF2 fixed solve");
    let qbdf_fixed = solve(&problem, Qbdf2, &fixed_options).expect("QBDF2 fixed solve");
    let adaptive_options = SolveOptions {
        absolute_tolerance: 1.0e-8,
        relative_tolerance: 1.0e-8,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let adaptive = solve(&problem, Qndf2, &adaptive_options).expect("QNDF2 adaptive solve");
    let qbdf_adaptive = solve(&problem, Qbdf2, &adaptive_options).expect("QBDF2 adaptive solve");
    println!("qndf2_fixed,{:.17e}", fixed.last_state()[0]);
    println!("qbdf2_fixed,{:.17e}", qbdf_fixed.last_state()[0]);
    println!(
        "qndf2,{:.17e},{},{}",
        adaptive.last_state()[0],
        adaptive.stats().accepted_steps,
        adaptive.stats().rejected_steps
    );
    println!(
        "qbdf2,{:.17e},{},{}",
        qbdf_adaptive.last_state()[0],
        qbdf_adaptive.stats().accepted_steps,
        qbdf_adaptive.stats().rejected_steps
    );
}
