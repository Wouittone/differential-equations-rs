use differential_equations::{solvers::rosenbrock::Rodas4, stepping::*};
use std::convert::Infallible;
#[test]
fn rosenbrock_observes_solved_stages_and_exact_work_with_rejection() {
    let tab = Rodas4.tableau().unwrap();
    let mut s = RosenbrockStepper::new(tab, 0., &[1.]).unwrap();
    let mut rhs = |t: f64, y: &[f64], d: &mut [f64]| {
        d[0] = -1000. * (y[0] - t.cos()) - t.sin();
        Ok::<_, Infallible>(())
    };
    let mut jac = |_: f64, _: &[f64], j: &mut [f64]| {
        j[0] = -1000.;
        Ok::<_, Infallible>(())
    };
    let mut ft = |t: f64, _: &[f64], d: &mut [f64]| {
        d[0] = -1000. * t.sin() - t.cos();
        Ok::<_, Infallible>(())
    };
    let v = s
        .attempt(0.01, &mut rhs, Some(&mut jac), Some(&mut ft))
        .unwrap();
    assert_eq!(v.jacobian, &[-1000.]);
    assert_eq!(v.time_partial, &[-1.]);
    assert_ne!(v.stage(0), v.rhs_stage(0));
    assert_eq!(v.statistics.rhs_evaluations, tab.stages());
    assert_eq!(v.statistics.linear_solves, tab.stages());
    assert_eq!(v.statistics.factorizations, 1);
    s.reject().unwrap();
    s.attempt(0.0025, &mut rhs, Some(&mut jac), Some(&mut ft))
        .unwrap();
    s.accept().unwrap();
    assert_eq!(s.statistics().jacobian_evaluations, 1);
    assert_eq!(s.statistics().rhs_evaluations, 2 * tab.stages() - 1);
    for _ in 0..399 {
        s.attempt(0.0025, &mut rhs, Some(&mut jac), Some(&mut ft))
            .unwrap();
        s.accept().unwrap();
    }
    assert!(
        (s.state()[0] - 1_f64.cos()).abs() < 1e-9,
        "error={}",
        (s.state()[0] - 1_f64.cos()).abs()
    );
}
fn shifted(origin: f64, h: f64) -> f64 {
    let tab = Rodas4.tableau().unwrap();
    let mut s = RosenbrockStepper::new(tab, origin, &[0.]).unwrap();
    let mut rhs = |t: f64, _: &[f64], d: &mut [f64]| {
        d[0] = 3. * (t - origin).powi(2);
        Ok::<_, Infallible>(())
    };
    let mut jac = |_: f64, _: &[f64], j: &mut [f64]| {
        j[0] = 0.;
        Ok::<_, Infallible>(())
    };
    for _ in 0..8 {
        s.attempt(h, &mut rhs, Some(&mut jac), None).unwrap();
        s.accept().unwrap();
    }
    (s.state()[0] - (8. * h).powi(3)).abs()
}
#[test]
fn numerical_time_partial_is_epoch_independent_forward_and_backward() {
    for origin in [0., 1e9, 1e12] {
        for h in [0.125, -0.125] {
            assert!(shifted(origin, h) < 2e-4, "{origin} {h}");
        }
    }
}
#[test]
fn hooks_preserve_payload_and_policy_respects_domain() {
    let tab = Rodas4.tableau().unwrap();
    let mut s = RosenbrockStepper::new(tab, 0., &[1.]).unwrap();
    let e = s
        .attempt(
            0.1,
            &mut |_: f64, y: &[f64], d: &mut [f64]| {
                d[0] = y[0];
                Ok(())
            },
            Some(&mut |_, _, _| Err("jacobian")),
            None,
        )
        .unwrap_err();
    assert!(matches!(e, StepError::User("jacobian")));
    assert_eq!(s.state(), &[1.]);
    let policy = TimeDifferencePolicy {
        bounds: Some((1e12 - 0.5, 1e12)),
        ..TimeDifferencePolicy::default()
    };
    let (a, b) = policy.probes(1e12, -0.1).unwrap();
    assert!(a <= 1e12 && a >= 1e12 - 0.1);
    assert!(b.unwrap() >= 1e12 - 0.1);
    assert!(policy.probes(1e12, 0.1).is_err());
}
#[test]
fn autonomous_fourth_order_convergence_and_backward() {
    let solve = |h: f64| {
        let mut s = RosenbrockStepper::new(Rodas4.tableau().unwrap(), 0., &[1.]).unwrap();
        for _ in 0..(1. / h.abs()).round() as usize {
            s.attempt(
                h,
                &mut |_: f64, y: &[f64], d: &mut [f64]| {
                    d[0] = -y[0];
                    Ok::<_, Infallible>(())
                },
                Some(&mut |_, _, j| {
                    j[0] = -1.;
                    Ok(())
                }),
                Some(&mut |_, _, d| {
                    d[0] = 0.;
                    Ok(())
                }),
            )
            .unwrap();
            s.accept().unwrap();
        }
        (s.state()[0] - (-h.signum()).exp()).abs()
    };
    for sign in [1., -1.] {
        let e1 = solve(sign * 0.125);
        let e2 = solve(sign * 0.0625);
        assert!(e1 / e2 > 12., "{e1} {e2}");
    }
}
