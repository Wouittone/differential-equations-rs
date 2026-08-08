use differential_equations::{
    CallbackAction, OdeProblem, Qndf1, SaveMode, SolveError, SolveOptions, solve,
};

fn fixed(step: f64) -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

#[test]
fn fixed_step_has_first_order_convergence() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let coarse = solve(&problem, Qndf1, &fixed(0.1)).unwrap();
    let fine = solve(&problem, Qndf1, &fixed(0.05)).unwrap();
    let exact = (-1.0f64).exp();
    let coarse_error = (coarse.last_state()[0] - exact).abs();
    let fine_error = (fine.last_state()[0] - exact).abs();
    assert!(
        fine_error < coarse_error * 0.65,
        "errors {coarse_error} and {fine_error}"
    );
}

#[test]
fn adaptive_stiff_decay_and_backward_callback_work() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), t: f64| {
            du[0] = -15.0 * (u[0] - t.cos()) - t.sin();
        },
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions {
        absolute_tolerance: 1.0e-7,
        relative_tolerance: 1.0e-7,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let solution = solve(&problem, Qndf1, &options).unwrap();
    assert!((solution.last_state()[0] - 1.0f64.cos()).abs() < 2.0e-4);
    assert!(solution.stats().rejected_steps > 0);

    let backward = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -u[0],
        vec![(-1.0f64).exp()],
        (1.0, 0.0),
        (),
    )
    .with_discrete_callback(
        |_, _, t| (t - 0.5).abs() < 1.0e-12,
        |_u, _, _| CallbackAction::Continue,
    );
    let result = solve(&backward, Qndf1, &fixed(0.05)).unwrap();
    assert!((result.last_state()[0] - 1.0).abs() < 2.0e-2);
}

#[test]
fn reports_nonfinite_rhs_and_singular_systems() {
    let bad = OdeProblem::new(
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = f64::NAN,
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    assert_eq!(
        solve(&bad, Qndf1, &fixed(0.1)),
        Err(SolveError::NonFiniteDerivative)
    );

    let singular = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    )
    .with_jacobian(|j, _, _, _| j[0] = 1.0);
    assert_eq!(
        solve(&singular, Qndf1, &fixed(1.0)),
        Err(SolveError::SingularLinearSystem)
    );
}
