use differential_equations::{
    solvers::{explicit::Tsit5, rosenbrock::Rodas4, second_order::FineRkn4},
    stepping::*,
};
use std::convert::Infallible;
#[test]
fn explicit_midstage_error_nonfinite_and_norm_failure_never_commit() {
    let mut s = ExplicitRungeKuttaStepper::new(Tsit5.tableau().unwrap(), 2., &[1.]).unwrap();
    let mut calls = 0;
    let result = s.attempt(0.1, &mut |_: f64, _: &[f64], d: &mut [f64]| {
        calls += 1;
        if calls == 3 {
            return Err(73);
        }
        d[0] = 2.;
        Ok(())
    });
    assert!(matches!(result, Err(StepError::User(73))));
    assert_eq!(s.time(), 2.);
    assert_eq!(s.state(), &[1.]);
    assert_eq!(s.accept(), Err(StepFailure::NoCandidate));
    // Mathematical RHS changed after failure: explicitly discard its stage-zero cache.
    s.invalidate_derivative();
    let result = s.attempt(0.1, &mut |_: f64, _: &[f64], d: &mut [f64]| {
        d[0] = f64::INFINITY;
        Ok::<_, Infallible>(())
    });
    assert!(matches!(
        result,
        Err(StepError::Solver(StepFailure::NonFinite))
    ));
    assert_eq!(s.reject(), Err(StepFailure::NoCandidate));
    assert_eq!(s.state(), &[1.]);
    let mut c = AdaptiveController::new(ControllerConfig::proportional(5).unwrap(), 0.1).unwrap();
    let before = c.state();
    let result = integrate_rk(
        &mut s,
        &mut c,
        3.,
        &[],
        10,
        &mut |_: f64, y: &[f64], d: &mut [f64]| {
            d[0] = y[0];
            Ok::<_, Infallible>(())
        },
        &mut |_: &[f64], _: &[f64], _: &[f64]| Ok(f64::NAN),
        &mut |_| Ok(ObserverAction::Continue),
    );
    assert!(matches!(
        result,
        Err(IntegrationError::Controller(ControllerError::Configuration))
    ));
    assert_eq!(s.state(), &[1.]);
    assert_eq!(s.time(), 2.);
    assert_eq!(c.state(), before);
    assert_eq!(s.accept(), Err(StepFailure::NoCandidate));
}
#[test]
fn rosenbrock_singular_retry_reuses_differentiation_transactionally() {
    let tab = Rodas4.tableau().unwrap();
    let h = 0.125;
    let rate = 1. / (h * tab.gamma());
    let mut s = RosenbrockStepper::new(tab, 0., &[1.]).unwrap();
    let mut rhs = |_: f64, y: &[f64], d: &mut [f64]| {
        d[0] = rate * y[0];
        Ok::<_, Infallible>(())
    };
    let mut jac = |_: f64, _: &[f64], j: &mut [f64]| {
        j[0] = rate;
        Ok::<_, Infallible>(())
    };
    let mut ft = |_: f64, _: &[f64], d: &mut [f64]| {
        d[0] = 0.;
        Ok::<_, Infallible>(())
    };
    assert!(matches!(
        s.attempt(h, &mut rhs, Some(&mut jac), Some(&mut ft)),
        Err(StepError::Solver(StepFailure::SingularSystem))
    ));
    assert_eq!(s.state(), &[1.]);
    assert_eq!(s.time(), 0.);
    assert_eq!(s.accept(), Err(StepFailure::NoCandidate));
    let retry = s
        .attempt(h / 2., &mut rhs, Some(&mut jac), Some(&mut ft))
        .unwrap()
        .candidate[0];
    s.accept().unwrap();
    let mut fresh = RosenbrockStepper::new(tab, 0., &[1.]).unwrap();
    let expected = fresh
        .attempt(h / 2., &mut rhs, Some(&mut jac), Some(&mut ft))
        .unwrap()
        .candidate[0];
    assert_eq!(retry, expected);
    assert_eq!(s.statistics().jacobian_evaluations, 1);
    assert_eq!(s.statistics().factorizations, 2);
}
#[test]
fn mixed_failed_auxiliary_attempt_does_not_poison_stage_zero_cache() {
    let tab = FineRkn4.tableau().unwrap();
    let mut s = MixedRknStepper::new(tab, 0., &[1.], &[0.], &[1.]).unwrap();
    let mut calls = 0;
    let result =
        s.attempt(0.1, &mut |_: f64,
                             q: &[f64],
                             _: &[f64],
                             z: &[f64],
                             a: &mut [f64],
                             dz: &mut [f64]| {
            calls += 1;
            a[0] = -q[0];
            dz[0] = -z[0];
            if calls == 3 {
                Err("later stage")
            } else {
                Ok(())
            }
        });
    assert!(matches!(result, Err(StepError::User("later stage"))));
    assert_eq!(s.position(), &[1.]);
    assert_eq!(s.auxiliary(), &[1.]);
    assert_eq!(s.accept(), Err(StepFailure::NoCandidate));
    let mut force = |_: f64, q: &[f64], _: &[f64], z: &[f64], a: &mut [f64], dz: &mut [f64]| {
        a[0] = -q[0];
        dz[0] = -z[0];
        Ok::<_, Infallible>(())
    };
    s.attempt(0.05, &mut force).unwrap();
    s.accept().unwrap();
    let mut fresh = MixedRknStepper::new(tab, 0., &[1.], &[0.], &[1.]).unwrap();
    fresh.attempt(0.05, &mut force).unwrap();
    fresh.accept().unwrap();
    assert_eq!(s.position(), fresh.position());
    assert_eq!(s.velocity(), fresh.velocity());
    assert_eq!(s.auxiliary(), fresh.auxiliary());
    assert_eq!(s.statistics().rhs_evaluations, 3 + tab.stages() - 1);
}
#[test]
fn invalid_reset_leaves_pending_candidate_available() {
    let mut s = ExplicitRungeKuttaStepper::new(Tsit5.tableau().unwrap(), 0., &[1.]).unwrap();
    s.attempt(0.1, &mut |_: f64, y: &[f64], d: &mut [f64]| {
        d[0] = y[0];
        Ok::<_, Infallible>(())
    })
    .unwrap();
    assert_eq!(s.reset(f64::NAN, &[2.]), Err(StepFailure::NonFinite));
    s.accept().unwrap();
    assert_eq!(s.time(), 0.1);
    assert!((s.state()[0] - 0.1_f64.exp()).abs() < 1e-9);
}

#[test]
fn nonfinite_stage_time_is_rejected_before_user_evaluation() {
    let tableau = differential_equations::tableau::parse_tableau(
        r#"{
       "name":"WideStage","description":"test of stage-time validation",
       "kind":"explicit-runge-kutta","order":1,
       "A":[[0,0],[1e308,0]],"b":[1,0],"c":[0,1e308]
    }"#,
        "WideStage",
    )
    .unwrap();
    let mut s = ExplicitRungeKuttaStepper::new(&tableau, 1., &[1.]).unwrap();
    let mut calls = 0;
    let result = s.attempt(10., &mut |t: f64, _: &[f64], d: &mut [f64]| {
        assert!(t.is_finite());
        calls += 1;
        d[0] = 0.;
        Ok::<_, Infallible>(())
    });
    assert!(matches!(
        result,
        Err(StepError::Solver(StepFailure::NonFinite))
    ));
    assert_eq!(calls, 1);
    assert_eq!(s.time(), 1.);
    assert_eq!(s.state(), &[1.]);
}

#[test]
fn caller_output_dimensions_are_checked_before_any_partition_is_written() {
    let s = MixedRknStepper::new(FineRkn4.tableau().unwrap(), 0., &[1.], &[2.], &[3.]).unwrap();
    let mut q = [9.];
    let mut bad_v = [8., 8.];
    let mut z = [7.];
    assert_eq!(
        s.copy_state_into(&mut q, &mut bad_v, &mut z),
        Err(StepFailure::Dimension)
    );
    assert_eq!(q, [9.]);
    assert_eq!(z, [7.]);
    let mut v = [0.];
    s.copy_state_into(&mut q, &mut v, &mut z).unwrap();
    assert_eq!((q, v, z), ([1.], [2.], [3.]));
}
