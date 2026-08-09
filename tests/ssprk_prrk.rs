use differential_equations::{
    CallbackAction, OdeProblem, Prrk22, Prrk33, SaveMode, SolveOptions, solve,
};

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn exponential() -> OdeProblem<TestRhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
        du[0] = u[0];
    }
    OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
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
fn prrk22_default_matches_second_order_fixed_convergence() {
    let coarse = solve(&exponential(), Prrk22::default(), &fixed(0.2)).unwrap();
    let fine = solve(&exponential(), Prrk22::default(), &fixed(0.1)).unwrap();
    let exact = 1.0f64.exp();
    let coarse_error = (coarse.last_state()[0] - exact).abs();
    let fine_error = (fine.last_state()[0] - exact).abs();
    assert!(fine_error < coarse_error / 3.0);
}

#[test]
fn prrk22_supports_relaxation_backward_and_termination() {
    let options = fixed(0.05);
    let relaxed = solve(&exponential(), Prrk22::new(0.5), &options).unwrap();
    assert!(relaxed.last_state()[0].is_finite());

    let backward = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
        vec![1.0f64.exp()],
        (1.0, 0.0),
        (),
    )
    .with_discrete_callback(
        |_, _, t| (t - 0.5).abs() < 1.0e-12,
        |_u, _, _| CallbackAction::Continue,
    );
    let backward_solution = solve(&backward, Prrk22::default(), &options).unwrap();
    assert!((backward_solution.last_state()[0] - 1.0).abs() < 2.0e-3);

    let terminating = exponential().with_discrete_callback(
        |_, _, t| t >= 0.25 - 1.0e-12,
        |_u, _, _| CallbackAction::Terminate,
    );
    let stopped = solve(&terminating, Prrk22::default(), &options).unwrap();
    assert!(stopped.times().last().copied().unwrap() <= 0.25 + 1.0e-12);
}

#[test]
fn prrk33_default_has_third_order_fixed_convergence() {
    let coarse = solve(&exponential(), Prrk33::default(), &fixed(0.2)).unwrap();
    let fine = solve(&exponential(), Prrk33::default(), &fixed(0.1)).unwrap();
    let exact = 1.0f64.exp();
    let coarse_error = (coarse.last_state()[0] - exact).abs();
    let fine_error = (fine.last_state()[0] - exact).abs();
    assert!(fine_error < coarse_error / 6.0);
}

#[test]
fn prrk33_supports_relaxation_backward_callbacks_and_save_at() {
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.05),
        save: SaveMode::EveryStep,
        save_at: vec![0.25, 0.5, 0.75],
        ..SolveOptions::default()
    };
    let relaxed = solve(&exponential(), Prrk33::new(0.5), &options).unwrap();
    assert!(relaxed.last_state()[0].is_finite());
    assert!(
        relaxed
            .times()
            .iter()
            .any(|time| (*time - 0.5).abs() < 1.0e-12)
    );

    let backward = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
        vec![1.0f64.exp()],
        (1.0, 0.0),
        (),
    );
    let backward_solution = solve(&backward, Prrk33::default(), &fixed(0.05)).unwrap();
    assert!((backward_solution.last_state()[0] - 1.0).abs() < 5.0e-5);

    let terminating = exponential().with_discrete_callback(
        |_, _, t| t >= 0.25 - 1.0e-12,
        |_u, _, _| CallbackAction::Terminate,
    );
    let stopped = solve(&terminating, Prrk33::default(), &fixed(0.05)).unwrap();
    assert!(stopped.times().last().copied().unwrap() <= 0.25 + 1.0e-12);
}
