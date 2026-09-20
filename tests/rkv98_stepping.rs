use differential_equations::{
    stepping::{AdaptiveController, ControllerConfig, ExplicitRungeKuttaStepper},
    tableau::RungeKuttaCoefficients,
};

#[path = "../tableau-core/tests/fixtures/numeris_rkv98.rs"]
#[allow(dead_code)]
mod numeris_rkv98;

fn tableau() -> differential_equations::tableau::RungeKuttaTableau {
    let rows: Vec<&[f64]> = numeris_rkv98::A[..16]
        .iter()
        .map(|row| &row[..16])
        .collect();
    let mut coefficients = RungeKuttaCoefficients::explicit(
        "Numeris RKV98",
        9,
        &rows,
        &numeris_rkv98::B[..16],
        &numeris_rkv98::C[..16],
    );
    coefficients.embedded_order = Some(8);
    coefficients.b_hat = Some(&numeris_rkv98::BHAT[..16]);
    coefficients.build().unwrap()
}

#[test]
fn rkv98_statistics_reset_preserves_reusable_workspace() {
    let tableau = tableau();
    let mut stepper = ExplicitRungeKuttaStepper::new(&tableau, 0.0, &[1.0]).unwrap();
    let mut rhs = |_: f64, state: &[f64], derivative: &mut [f64]| {
        derivative[0] = -state[0];
        Ok::<_, ()>(())
    };
    stepper.attempt(0.1, &mut rhs).unwrap();
    stepper.accept().unwrap();
    assert_eq!(stepper.statistics().accepted_steps, 1);
    stepper.clear_statistics();
    assert_eq!(stepper.statistics(), Default::default());
    assert_eq!(stepper.time(), 0.1);
    assert_eq!(stepper.state(), &[(-0.1f64).exp()]);
}

#[test]
fn rkv98_rejection_keeps_state_and_counts_work() {
    let tableau = tableau();
    let mut stepper = ExplicitRungeKuttaStepper::new(&tableau, 0.0, &[1.0]).unwrap();
    let mut rhs = |_: f64, state: &[f64], derivative: &mut [f64]| {
        derivative[0] = -state[0];
        Ok::<_, ()>(())
    };
    stepper.attempt(0.1, &mut rhs).unwrap();
    stepper.reject().unwrap();
    let statistics = stepper.statistics();
    assert_eq!(statistics.accepted_steps, 0);
    assert_eq!(statistics.rejected_steps, 1);
    assert_eq!(stepper.time(), 0.0);
    assert_eq!(stepper.state(), &[1.0]);
}

#[test]
fn rkv98_controller_rejection_is_explicit() {
    let mut config = ControllerConfig::proportional(8).unwrap();
    config.rejection_maximum = 0.5;
    let mut controller = AdaptiveController::new(config, 0.1).unwrap();
    let decision = controller.assess(0.1, 2.0).unwrap();
    assert!(!decision.accepted);
    assert_eq!(decision.next_step, 0.05);
    assert!(controller.state().rejected_since_acceptance);
}
