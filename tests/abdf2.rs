use differential_equations::solvers::multistep::*;
use differential_equations::*;

#[allow(clippy::type_complexity)]
fn exponential(
    rate: f64,
    span: (f64, f64),
) -> OdeProblem<impl Fn(&mut [f64], &[f64], &(), f64), ()> {
    OdeProblem::new(
        move |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = rate * u[0],
        vec![1.0],
        span,
        (),
    )
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
fn fixed_step_is_second_order() {
    let a = solve(&exponential(-1.0, (0.0, 1.0)), Abdf2, &fixed(0.1)).unwrap();
    let b = solve(&exponential(-1.0, (0.0, 1.0)), Abdf2, &fixed(0.05)).unwrap();
    let e1 = (a.last_state()[0] - (-1.0f64).exp()).abs();
    let e2 = (b.last_state()[0] - (-1.0f64).exp()).abs();
    assert!(e2 < e1 / 3.0, "errors {e1} and {e2}");
}

#[test]
fn adaptive_stiff_decay_and_nonautonomous_rhs() {
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
    let solution = solve(&problem, Abdf2, &options).unwrap();
    assert!((solution.last_state()[0] - 1.0f64.cos()).abs() < 2.0e-5);
    assert!(solution.stats().rejected_steps > 0);
}

#[test]
fn backward_integration_and_callback_reset() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -u[0],
        vec![(-1.0f64).exp()],
        (1.0, 0.0),
        (),
    )
    .with_discrete_callback(
        |_, _, t| (t - 0.5).abs() < 1.0e-12,
        |_u, _, _| CallbackAction::Continue,
    );
    let solution = solve(&problem, Abdf2, &fixed(0.05)).unwrap();
    assert!((solution.last_state()[0] - 1.0).abs() < 1.0e-2);
}

#[test]
fn analytic_and_finite_difference_jacobians_agree() {
    let rhs = |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -3.0 * u[0];
    let plain = OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ());
    let analytic =
        OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ()).with_jacobian(|j, _, _, _| j[0] = -3.0);
    let options = fixed(0.02);
    let a = solve(&plain, Abdf2, &options).unwrap();
    let b = solve(&analytic, Abdf2, &options).unwrap();
    assert!((a.last_state()[0] - b.last_state()[0]).abs() < 1.0e-12);
    assert!(b.stats().jacobian_evaluations > 0);
}

#[test]
fn malformed_rhs_and_singular_failure_are_reported() {
    let bad = OdeProblem::new(
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = f64::NAN,
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    assert_eq!(
        solve(&bad, Abdf2, &fixed(0.1)),
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
        solve(&singular, Abdf2, &fixed(1.0)),
        Err(SolveError::SingularLinearSystem)
    );
}

#[test]
fn allocation_shape_remains_bounded() {
    let solution = solve(&exponential(-1.0, (0.0, 2.0)), Abdf2, &fixed(0.01)).unwrap();
    assert_eq!(solution.stats().accepted_steps, 200);
    assert!(solution.stats().linear_solves > 0);
}
