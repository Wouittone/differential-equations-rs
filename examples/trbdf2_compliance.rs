use differential_equations::{OdeProblem, SaveMode, SolveOptions, Trbdf2, solve};

fn main() {
    let stiff = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), time: f64| {
            du[0] = -1000.0 * (u[0] - time.cos()) - time.sin();
        },
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let adaptive = SolveOptions {
        absolute_tolerance: 1.0e-7,
        relative_tolerance: 1.0e-7,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let stiff_solution = solve(&stiff, Trbdf2, &adaptive).unwrap();
    println!(
        "adaptive,{:.17e},{},{}",
        stiff_solution.last_state()[0],
        stiff_solution.stats().accepted_steps,
        stiff_solution.stats().rejected_steps
    );

    let vector = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), time: f64| {
            du[0] = -30.0 * (u[0] - time.cos()) - time.sin();
            du[1] = -2.0 * u[1] + time;
        },
        vec![1.0, 0.0],
        (0.0, 1.0),
        (),
    );
    let fixed = SolveOptions {
        adaptive: false,
        initial_step: Some(0.025),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let vector_solution = solve(&vector, Trbdf2, &fixed).unwrap();
    println!(
        "fixed,{:.17e},{:.17e}",
        vector_solution.last_state()[0],
        vector_solution.last_state()[1]
    );

    let backward = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
        vec![std::f64::consts::E],
        (1.0, 0.0),
        (),
    );
    let backward_solution = solve(&backward, Trbdf2, &fixed).unwrap();
    println!("backward,{:.17e}", backward_solution.last_state()[0]);
}
