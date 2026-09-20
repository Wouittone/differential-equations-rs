use differential_equations::{solvers::explicit::Tsit5, stepping::*};
use std::convert::Infallible;
#[test]
fn controller_replay_exports_next_proposal_and_minimum_step_policy() {
    let config = ControllerConfig::pi(0.14, 0.08);
    let mut c = AdaptiveController::new(config, 0.2).unwrap();
    let mut h = 0.2_f64;
    let mut previous = 1_f64;
    let mut rejected = false;
    for e in [4_f64, 0.7, 0.001, 2., 0.8] {
        let mut factor = (0.9 * e.powf(-0.14) * previous.powf(0.08)).clamp(0.2, 10.);
        if e > 1. || rejected {
            factor = factor.min(1.);
        }
        let d = c.assess(h, e).unwrap();
        assert_eq!(d.accepted, e <= 1.);
        assert!((d.next_step - h * factor).abs() < 1e-14);
        if e <= 1. {
            previous = e;
            rejected = false;
        } else {
            rejected = true;
        }
        h = d.next_step;
        let mut restored = AdaptiveController::new(config, 1.).unwrap();
        restored.restore(c.state()).unwrap();
        assert_eq!(restored.state(), c.state());
    }
    let mut config = ControllerConfig::proportional(5).unwrap();
    config.minimum_step = 0.1;
    let mut strict = AdaptiveController::new(config, -0.1).unwrap();
    assert!(matches!(
        strict.assess(-0.1, 4.),
        Err(ControllerError::MinimumStep { .. })
    ));
    config.minimum_step_policy = MinimumStepPolicy::ForceAccept;
    let d = AdaptiveController::new(config, -0.1)
        .unwrap()
        .assess(-0.1, 4.)
        .unwrap();
    assert!(d.accepted && d.forced);
    assert!(d.next_step < 0.);
}
#[test]
fn output_free_driver_requested_samples_typed_observer_and_resume() {
    let tab = Tsit5.tableau().unwrap();
    let mut s = ExplicitRungeKuttaStepper::new(tab, 0., &[1.]).unwrap();
    let mut c = AdaptiveController::new(ControllerConfig::proportional(5).unwrap(), 0.1).unwrap();
    let mut rhs = |_: f64, y: &[f64], d: &mut [f64]| {
        d[0] = y[0];
        Ok::<_, Infallible>(())
    };
    let mut norm = |_: &[f64], _: &[f64], e: &[f64]| Ok::<_, Infallible>(e[0].abs() / 1e-9);
    let mut samples = Vec::new();
    let result = integrate_rk(
        &mut s,
        &mut c,
        1.,
        &[0., 0.3, 0.7, 1.],
        100,
        &mut rhs,
        &mut norm,
        &mut |o: Observation<'_>| {
            if o.requested {
                samples.push(o.time);
            }
            Ok(ObserverAction::Continue)
        },
    )
    .unwrap();
    assert_eq!(samples, vec![0., 0.3, 0.7, 1.]);
    assert_eq!(result.time, 1.);
    assert!((s.state()[0] - std::f64::consts::E).abs() < 1e-8);
    let next = c.next_step();
    assert!(next > 0.);
    let mut continuation = Continuation {
        stepper: s,
        controller: c,
    };
    integrate_rk(
        &mut continuation.stepper,
        &mut continuation.controller,
        1.1,
        &[],
        100,
        &mut rhs,
        &mut norm,
        &mut |_| Ok(ObserverAction::Continue),
    )
    .unwrap();
    let mut out = [0.];
    continuation.stepper.copy_state_into(&mut out).unwrap();
    assert!((out[0] - 1.1_f64.exp()).abs() < 1e-8);
}
#[test]
fn observer_payload_is_preserved_after_acceptance() {
    let tab = Tsit5.tableau().unwrap();
    let mut s = ExplicitRungeKuttaStepper::new(tab, 0., &[1.]).unwrap();
    let mut c = AdaptiveController::new(ControllerConfig::proportional(5).unwrap(), 0.01).unwrap();
    let result = integrate_rk(
        &mut s,
        &mut c,
        1.,
        &[],
        10,
        &mut |_: f64, y: &[f64], d: &mut [f64]| {
            d[0] = y[0];
            Ok(())
        },
        &mut |_: &[f64], _: &[f64], _: &[f64]| Ok(0.),
        &mut |_| Err("observer payload"),
    );
    assert!(matches!(
        result,
        Err(IntegrationError::Step(StepError::User("observer payload")))
    ));
    assert_eq!(s.time(), 0.01);
}
