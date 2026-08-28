use differential_equations::solvers::multistep::*;
use differential_equations::*;

fn main() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), t: f64| {
            du[0] = -15.0 * (u[0] - t.cos()) - t.sin();
        },
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let fixed = SolveOptions {
        adaptive: false,
        initial_step: Some(0.01),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let adaptive = SolveOptions {
        absolute_tolerance: 1.0e-8,
        relative_tolerance: 1.0e-8,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };

    for (name, fixed_endpoint, adaptive_endpoint, accepted, rejected) in [
        {
            let fixed_solution = solve(&problem, QNDF, &fixed).expect("QNDF fixed solve");
            let adaptive_solution = solve(&problem, QNDF, &adaptive).expect("QNDF adaptive solve");
            (
                "qndf",
                fixed_solution.last_state()[0],
                adaptive_solution.last_state()[0],
                adaptive_solution.stats().accepted_steps,
                adaptive_solution.stats().rejected_steps,
            )
        },
        {
            let fixed_solution = solve(&problem, QBDF, &fixed).expect("QBDF fixed solve");
            let adaptive_solution = solve(&problem, QBDF, &adaptive).expect("QBDF adaptive solve");
            (
                "qbdf",
                fixed_solution.last_state()[0],
                adaptive_solution.last_state()[0],
                adaptive_solution.stats().accepted_steps,
                adaptive_solution.stats().rejected_steps,
            )
        },
        {
            let fixed_solution = solve(&problem, FBDF, &fixed).expect("FBDF fixed solve");
            let adaptive_solution = solve(&problem, FBDF, &adaptive).expect("FBDF adaptive solve");
            (
                "fbdf",
                fixed_solution.last_state()[0],
                adaptive_solution.last_state()[0],
                adaptive_solution.stats().accepted_steps,
                adaptive_solution.stats().rejected_steps,
            )
        },
    ] {
        println!("{name},{fixed_endpoint:.17e},{adaptive_endpoint:.17e},{accepted},{rejected}");
    }
}
