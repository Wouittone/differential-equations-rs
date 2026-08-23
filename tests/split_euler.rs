use differential_equations::algorithms::*;
use differential_equations::*;

#[test]
fn split_euler_advances_both_typed_components() {
    let problem = SplitOdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
        |du: &mut [f64], _: &[f64], _: &(), time: f64| du[0] = time,
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.25),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let solution = solve_split_euler(&problem, SplitEuler, &options).unwrap();
    assert_eq!(solution.last_state(), &[2.882_812_5]);
    assert_eq!(solution.stats().rhs_evaluations, 4);
}

#[test]
fn split_euler_retains_fixed_step_contract() {
    let problem = SplitOdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = 0.0,
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    assert_eq!(
        solve_split_euler(&problem, SplitEuler, &SolveOptions::default()).unwrap_err(),
        SolveError::AdaptiveStepUnsupported
    );
}
