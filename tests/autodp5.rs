use differential_equations::{
    AutoDp5, CallbackAction, Dp5, OdeProblem, Rodas5P, SaveMode, SolveOptions, solve,
};

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn exponential_rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
    du[0] = u[0];
}

fn exponential(initial: f64, span: (f64, f64)) -> OdeProblem<TestRhs, ()> {
    OdeProblem::new(exponential_rhs, vec![initial], span, ())
}

fn fixed(step: f64) -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

#[test]
fn fixed_and_adaptive_match_dp5_forward_and_backward() {
    let fixed_forward = solve(
        &exponential(1.0, (0.0, 1.0)),
        AutoDp5::new(Rodas5P),
        &fixed(0.01),
    )
    .unwrap();
    let dp5_forward = solve(&exponential(1.0, (0.0, 1.0)), Dp5, &fixed(0.01)).unwrap();
    assert_eq!(fixed_forward.last_state(), dp5_forward.last_state());

    let adaptive = SolveOptions {
        absolute_tolerance: 1.0e-10,
        relative_tolerance: 1.0e-10,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let auto = solve(
        &exponential(1.0, (0.0, 1.0)),
        AutoDp5::new(Rodas5P),
        &adaptive,
    )
    .unwrap();
    let reference = solve(&exponential(1.0, (0.0, 1.0)), Dp5, &adaptive).unwrap();
    assert_eq!(auto.last_state(), reference.last_state());

    let backward = solve(
        &exponential(std::f64::consts::E, (1.0, 0.0)),
        AutoDp5::new(Rodas5P),
        &fixed(0.01),
    )
    .unwrap();
    assert!((backward.last_state()[0] - 1.0).abs() < 1.0e-8);
}

#[test]
fn save_at_and_callback_semantics_are_preserved() {
    let save_at = solve(
        &exponential(1.0, (0.0, 1.0)),
        AutoDp5::new(Rodas5P),
        &SolveOptions {
            adaptive: false,
            initial_step: Some(0.5),
            save: SaveMode::Endpoints,
            save_at: vec![0.25, 0.75],
            ..SolveOptions::default()
        },
    )
    .unwrap();
    assert_eq!(save_at.times(), &[0.25, 0.75]);

    let callback_problem = OdeProblem::new(
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = 1.0,
        vec![0.0],
        (0.0, 2.0),
        (),
    )
    .with_continuous_callback(
        |state, _: &(), _| state[0] - 0.75,
        |state, _: &(), _| {
            state[0] = 42.0;
            CallbackAction::Terminate
        },
    );
    let callback = solve(&callback_problem, AutoDp5::new(Rodas5P), &fixed(0.5)).unwrap();
    assert_eq!(callback.last_state(), &[42.0]);
    assert_eq!(callback.stats().callback_invocations, 1);
}

#[test]
fn fifth_order_convergence_is_retained() {
    let endpoint = |step| {
        solve(
            &exponential(1.0, (0.0, 1.0)),
            AutoDp5::new(Rodas5P),
            &fixed(step),
        )
        .unwrap()
        .last_state()[0]
    };
    let coarse_error = (endpoint(0.1) - std::f64::consts::E).abs();
    let fine_error = (endpoint(0.05) - std::f64::consts::E).abs();
    assert!(coarse_error / fine_error > 20.0);
}
