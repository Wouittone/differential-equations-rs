use differential_equations::solvers::explicit::*;
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

    let empty = SplitOdeProblem::new(
        |_: &mut [f64], _: &[f64], _: &(), _: f64| {},
        |_: &mut [f64], _: &[f64], _: &(), _: f64| {},
        Vec::new(),
        (0.0, 1.0),
        (),
    );
    assert_eq!(
        SplitOdeAlgorithm::solve(&SplitEuler, &empty, &fixed(0.1)),
        Err(SolveError::EmptyState)
    );
}

#[test]
fn split_euler_applies_initial_and_step_callbacks() {
    let terminated = SplitOdeProblem::new(
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = 1.0,
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = 0.0,
        vec![0.0],
        (0.0, 1.0),
        (),
    )
    .with_discrete_callback(
        |_, _, time| time == 0.0,
        |state, _, _| {
            state[0] = 3.0;
            CallbackAction::Terminate
        },
    );
    let solution = solve_split_euler(&terminated, SplitEuler, &fixed(0.5)).unwrap();
    assert_eq!(solution.last_state(), &[3.0]);
    assert_eq!(solution.stats().callback_invocations, 1);
    assert_eq!(solution.stats().rhs_evaluations, 0);

    let continuing = SplitOdeProblem::new(
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = 1.0,
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = 0.0,
        vec![0.0],
        (0.0, 1.0),
        (),
    )
    .with_discrete_callback(
        |state, _, time| time >= 0.5 && state[0] < 5.0,
        |state, _, _| {
            state[0] += 10.0;
            CallbackAction::Continue
        },
    );
    let solution = solve_split_euler(&continuing, SplitEuler, &fixed(0.5)).unwrap();
    assert!((solution.last_state()[0] - 11.0).abs() < 1.0e-14);
    assert_eq!(solution.stats().callback_invocations, 1);
}

#[test]
fn split_euler_localizes_continuous_callbacks_and_retains_dense_output() {
    let problem = SplitOdeProblem::new(
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = 0.75,
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = 0.25,
        vec![0.0],
        (0.0, 1.0),
        (),
    )
    .with_continuous_callback(
        |state, _, _| state[0] - 0.5,
        |state, _, _| {
            state[0] = 2.0;
            CallbackAction::Terminate
        },
    );
    let solution = solve_split_euler(
        &problem,
        SplitEuler,
        &fixed(1.0)
            .with_dense_output(true)
            .with_event_tolerance(1.0e-12),
    )
    .unwrap();
    assert!((solution.times().last().unwrap() - 0.5).abs() < 1.0e-11);
    assert_eq!(solution.last_state(), &[2.0]);
    assert!((solution.interpolate(0.25).unwrap()[0] - 0.25).abs() < 1.0e-12);
    assert_eq!(solution.stats().callback_invocations, 1);
}

fn fixed(step: f64) -> SolveOptions {
    SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(step)
        .with_save(SaveMode::Endpoints)
}
