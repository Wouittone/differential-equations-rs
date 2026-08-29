use differential_equations::solvers::explicit::Rk4;
use differential_equations::{OdeProblem, SaveMode, SolveOptions, solve};

#[test]
fn public_fixed_step_solver_hits_time_stops_and_resumes_its_step() {
    let problem = OdeProblem::new(
        |derivative: &mut [f64], _: &[f64], _: &(), _: f64| derivative[0] = 1.0,
        [0.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(0.4)
        .with_save(SaveMode::EveryStep)
        .with_time_stops([0.25, 0.5]);

    let solution = solve(&problem, Rk4, &options).unwrap();

    assert_eq!(solution.times(), &[0.0, 0.25, 0.5, 0.9, 1.0]);
    assert!((solution.last_state()[0] - 1.0).abs() < 1.0e-14);
}
