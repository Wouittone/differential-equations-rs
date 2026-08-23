use differential_equations::algorithms::*;
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
    let a = solve(&exponential(-1.0, (0.0, 1.0)), Mebdf2, &fixed(0.1)).unwrap();
    let b = solve(&exponential(-1.0, (0.0, 1.0)), Mebdf2, &fixed(0.05)).unwrap();
    let e1 = (a.last_state()[0] - (-1.0f64).exp()).abs();
    let e2 = (b.last_state()[0] - (-1.0f64).exp()).abs();
    assert!(e2 < e1 / 3.0, "errors {e1} and {e2}");
}

#[test]
fn stiff_nonautonomous_and_backward() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), t: f64| du[0] = -15.0 * (u[0] - t.cos()) - t.sin(),
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let result = solve(&problem, Mebdf2, &fixed(0.01)).unwrap();
    assert!((result.last_state()[0] - 1.0f64.cos()).abs() < 2.0e-5);

    let backward = solve(&exponential(-1.0, (1.0, 0.0)), Mebdf2, &fixed(0.02)).unwrap();
    assert!((backward.last_state()[0] - 1.0f64.exp()).abs() < 1.0e-3);
}

#[test]
fn callbacks_and_jacobians_are_safe() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    )
    .with_discrete_callback(
        |_, _, t| (t - 0.5).abs() < 1.0e-12,
        |_u, _, _| CallbackAction::Continue,
    )
    .with_jacobian(|j, _, _, _| j[0] = -1.0);
    let result = solve(&problem, Mebdf2, &fixed(0.05)).unwrap();
    assert!(result.stats().jacobian_evaluations > 0);
    assert!(result.stats().linear_solves > 0);
}

#[test]
fn malformed_rhs_and_fixed_configuration_fail() {
    let bad = OdeProblem::new(
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = f64::NAN,
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    assert_eq!(
        solve(&bad, Mebdf2, &fixed(0.1)),
        Err(SolveError::NonFiniteDerivative)
    );
    let options = SolveOptions {
        adaptive: true,
        ..SolveOptions::default()
    };
    assert_eq!(
        solve(&exponential(-1.0, (0.0, 1.0)), Mebdf2, &options),
        Err(SolveError::AdaptiveStepUnsupported)
    );
}
