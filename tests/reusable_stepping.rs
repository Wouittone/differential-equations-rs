use differential_equations::stepping::{ExplicitRungeKuttaStepper, StepError, StepFailure};
use differential_equations::tableau::parse_tableau;
use std::convert::Infallible;

#[test]
fn reusable_rk_lifecycle_fsal_and_signed_endpoints() {
    let tableau = parse_tableau(
        include_str!("../src/tableau/resources/explicit/tsit5.json"),
        "Tsit5",
    )
    .unwrap();
    let mut stepper = ExplicitRungeKuttaStepper::new(&tableau, 0., &[1.]).unwrap();
    let mut calls = 0;
    let mut rhs = |_: f64, y: &[f64], out: &mut [f64]| {
        calls += 1;
        out[0] = y[0];
        Ok::<_, Infallible>(())
    };
    stepper.inject_derivative(&[1.]).unwrap();
    let view = stepper.attempt(0.1, &mut rhs).unwrap();
    assert_eq!(view.stage(0).unwrap(), &[1.]);
    assert!(view.component_error.unwrap()[0].is_finite());
    assert!((view.candidate[0] - 0.1_f64.exp()).abs() < 1e-9);
    assert_eq!(stepper.state(), &[1.]);
    stepper.reject().unwrap();
    stepper.attempt(0.05, &mut rhs).unwrap();
    stepper.accept().unwrap();
    stepper.attempt_to(0.1, 0.2, &mut rhs).unwrap();
    stepper.accept().unwrap();
    assert_eq!(stepper.time(), 0.1);
    assert!((stepper.state()[0] - 0.1_f64.exp()).abs() < 1e-10);
    stepper.attempt(-0.1, &mut rhs).unwrap();
    stepper.accept().unwrap();
    assert!((stepper.state()[0] - 1.).abs() < 1e-9);
    stepper.attempt(0., &mut rhs).unwrap();
    stepper.accept().unwrap();
    assert_eq!(calls, 4 * (tableau.stages() - 1));
    assert_eq!(stepper.statistics().rejected_steps, 1);
    assert_eq!(stepper.statistics().accepted_steps, 4);
}

#[test]
fn preserves_typed_failure_and_restarts_without_committing() {
    let tableau = parse_tableau(
        include_str!("../src/tableau/resources/explicit/rk4.json"),
        "Rk4",
    )
    .unwrap();
    let mut s = ExplicitRungeKuttaStepper::new(&tableau, 0., &[2.]).unwrap();
    let error = s
        .attempt(0.1, &mut |_: f64, _: &[f64], _: &mut [f64]| {
            Err::<(), _>(42)
        })
        .unwrap_err();
    assert!(matches!(error, StepError::User(42)));
    assert_eq!(s.state(), &[2.]);
    assert_eq!(s.accept(), Err(StepFailure::NoCandidate));
    s.reset(1., &[3.]).unwrap();
    let mut rhs = |_: f64, y: &[f64], out: &mut [f64]| {
        out[0] = -y[0];
        Ok::<_, Infallible>(())
    };
    s.attempt(0.1, &mut rhs).unwrap();
    assert!(matches!(
        s.attempt(0.1, &mut rhs),
        Err(StepError::Solver(StepFailure::PendingCandidate))
    ));
    s.accept().unwrap();
    assert_eq!(s.time(), 1.1);
    assert!(s.current_derivative().is_none());
    s.reset(1e20, &[3.]).unwrap();
    assert!(matches!(
        s.attempt(1., &mut rhs),
        Err(StepError::Solver(StepFailure::TimeResolution))
    ));
}

#[test]
fn rkn_drag_stages_backward_and_rejection_cache() {
    use differential_equations::stepping::{AccelerationPolicy, RknStepper};
    let tab = differential_equations::tableau::parse_rkn_tableau(
        include_str!("../src/tableau/resources/second_order/fine-rkn4.json"),
        "FineRkn4",
    )
    .unwrap();
    let mut s = RknStepper::new(
        &tab,
        AccelerationPolicy::VelocityDependent,
        0.,
        &[0.],
        &[1.],
    )
    .unwrap();
    let mut rhs = |_: f64, _: &[f64], v: &[f64], a: &mut [f64]| {
        a[0] = -v[0];
        Ok::<_, Infallible>(())
    };
    s.attempt(0.1, &mut rhs).unwrap();
    s.reject().unwrap();
    assert_eq!(s.position(), &[0.]);
    for _ in 0..20 {
        s.attempt(0.05, &mut rhs).unwrap();
        s.accept().unwrap();
    }
    assert!((s.velocity()[0] - (-1_f64).exp()).abs() < 2e-8);
    assert!((s.position()[0] - (1. - (-1_f64).exp())).abs() < 2e-8);
    assert_eq!(s.statistics().rhs_evaluations, 21 * tab.stages() - 1);
    for _ in 0..20 {
        s.attempt(-0.05, &mut rhs).unwrap();
        s.accept().unwrap();
    }
    assert!(s.position()[0].abs() < 5e-8);
    assert!((s.velocity()[0] - 1.).abs() < 5e-8);
}

