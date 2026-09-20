use differential_equations::{solvers::explicit::Tsit5, stepping::*};
use std::convert::Infallible;
fn rhs(_: f64, y: &[f64], d: &mut [f64]) -> Result<(), Infallible> {
    d[0] = y[0];
    Ok(())
}
#[test]
fn moving_continuation_preserves_bitwise_state_and_exact_fsal_counts() {
    let tab = Tsit5.tableau().unwrap();
    let mut continuous = ExplicitRungeKuttaStepper::new(tab, 0., &[1.]).unwrap();
    for _ in 0..100 {
        continuous.attempt(0.01, &mut rhs).unwrap();
        continuous.accept().unwrap();
    }
    let mut split = ExplicitRungeKuttaStepper::new(tab, 0., &[1.]).unwrap();
    for _ in 0..40 {
        split.attempt(0.01, &mut rhs).unwrap();
        split.accept().unwrap();
    }
    let mut controller =
        AdaptiveController::new(ControllerConfig::proportional(5).unwrap(), 0.01).unwrap();
    controller.assess(0.01, 0.25).unwrap();
    let expected_next = controller.next_step();
    let mut token = Continuation {
        stepper: split,
        controller,
    };
    assert_eq!(token.controller.next_step(), expected_next);
    assert_ne!(expected_next, 0.01);
    for _ in 40..100 {
        token.stepper.attempt(0.01, &mut rhs).unwrap();
        token.stepper.accept().unwrap();
    }
    assert_eq!(token.stepper.state(), continuous.state());
    assert_eq!(token.stepper.time(), continuous.time());
    assert_eq!(token.stepper.statistics(), continuous.statistics());
    assert_eq!(
        continuous.statistics().rhs_evaluations,
        1 + 100 * (tab.stages() - 1)
    );
}
#[test]
fn cache_invalidation_after_control_change_and_restart_is_explicit() {
    let tab = Tsit5.tableau().unwrap();
    let mut s = ExplicitRungeKuttaStepper::new(tab, 0., &[0.]).unwrap();
    let mut control = 1.;
    s.attempt(0.1, &mut |_: f64, _: &[f64], d: &mut [f64]| {
        d[0] = control;
        Ok::<_, Infallible>(())
    })
    .unwrap();
    s.accept().unwrap();
    assert_eq!(s.current_derivative(), Some([1.].as_slice()));
    control = 2.;
    s.invalidate_derivative();
    s.attempt(0.1, &mut |_: f64, _: &[f64], d: &mut [f64]| {
        d[0] = control;
        Ok::<_, Infallible>(())
    })
    .unwrap();
    s.reject().unwrap();
    s.attempt(0.05, &mut |_: f64, _: &[f64], d: &mut [f64]| {
        d[0] = control;
        Ok::<_, Infallible>(())
    })
    .unwrap();
    s.accept().unwrap();
    assert!((s.state()[0] - 0.2).abs() < 1e-14);
    assert_eq!(s.statistics().rhs_evaluations, 3 * tab.stages() - 1);
    s.reset(1., &[5.]).unwrap();
    assert!(s.current_derivative().is_none());
    s.attempt(-0.1, &mut |_: f64, _: &[f64], d: &mut [f64]| {
        d[0] = control;
        Ok::<_, Infallible>(())
    })
    .unwrap();
    s.accept().unwrap();
    assert!((s.state()[0] - 4.8).abs() < 1e-14);
    assert_eq!(s.statistics().rhs_evaluations, 4 * tab.stages() - 1);
}
#[test]
fn final_clipped_interval_is_not_the_next_proposal() {
    let mut s = ExplicitRungeKuttaStepper::new(Tsit5.tableau().unwrap(), 0., &[1.]).unwrap();
    let mut config = ControllerConfig::proportional(5).unwrap();
    config.maximum_step = 0.3;
    let mut c = AdaptiveController::new(config, 0.3).unwrap();
    integrate_rk(
        &mut s,
        &mut c,
        0.31,
        &[],
        20,
        &mut rhs,
        &mut |_: &[f64], _: &[f64], _: &[f64]| Ok(0.),
        &mut |_| Ok(ObserverAction::Continue),
    )
    .unwrap();
    assert_eq!(s.time(), 0.31);
    assert!((c.next_step() - 0.1).abs() < 1e-14);
    assert_ne!(c.next_step(), 0.01);
}

#[test]
fn observer_interruption_resumes_without_repeating_requested_outputs() {
    let mut s = ExplicitRungeKuttaStepper::new(Tsit5.tableau().unwrap(), 0., &[1.]).unwrap();
    let mut c = AdaptiveController::new(ControllerConfig::proportional(5).unwrap(), 0.1).unwrap();
    let mut seen = Vec::new();
    let stopped = integrate_rk(
        &mut s,
        &mut c,
        1.,
        &[0.3, 0.6],
        100,
        &mut rhs,
        &mut |_: &[f64], _: &[f64], e: &[f64]| Ok(e[0].abs() / 1e-9),
        &mut |o: Observation<'_>| {
            if o.requested {
                seen.push(o.time);
                return Ok(ObserverAction::Stop);
            }
            Ok(ObserverAction::Continue)
        },
    )
    .unwrap();
    assert!(stopped.interrupted);
    assert_eq!(stopped.time, 0.3);
    let mut token = Continuation {
        stepper: s,
        controller: c,
    };
    integrate_rk(
        &mut token.stepper,
        &mut token.controller,
        1.,
        &[0.6],
        100,
        &mut rhs,
        &mut |_: &[f64], _: &[f64], e: &[f64]| Ok(e[0].abs() / 1e-9),
        &mut |o: Observation<'_>| {
            if o.requested {
                seen.push(o.time);
            }
            Ok(ObserverAction::Continue)
        },
    )
    .unwrap();
    assert_eq!(seen, [0.3, 0.6]);
    assert!((token.stepper.state()[0] - std::f64::consts::E).abs() < 1e-8);
}
