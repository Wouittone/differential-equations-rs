use differential_equations::{
    solvers::{explicit::Tsit5, rosenbrock::Rodas4, second_order::FineRkn4},
    stepping::*,
};
use std::convert::Infallible;
#[test]
fn constant_rhs_matches_representable_elapsed_time_for_all_public_kernels() {
    for direction in [-1., 1.] {
        let origin = 1e12;
        let mut rk =
            ExplicitRungeKuttaStepper::new(Tsit5.tableau().unwrap(), origin, &[0.]).unwrap();
        let mut rkn = RknStepper::new(
            FineRkn4.tableau().unwrap(),
            AccelerationPolicy::VelocityDependent,
            origin,
            &[0.],
            &[1.],
        )
        .unwrap();
        let mut ros = RosenbrockStepper::new(Rodas4.tableau().unwrap(), origin, &[0.]).unwrap();
        let mut mixed =
            MixedRknStepper::new(FineRkn4.tableau().unwrap(), origin, &[0.], &[1.], &[0.]).unwrap();
        for _ in 0..20 {
            let h = direction * 0.1;
            rk.attempt(h, &mut |_: f64, _: &[f64], d: &mut [f64]| {
                d[0] = 1.;
                Ok::<_, Infallible>(())
            })
            .unwrap();
            rk.accept().unwrap();
            rkn.attempt(h, &mut |_: f64, _: &[f64], _: &[f64], a: &mut [f64]| {
                a[0] = 0.;
                Ok::<_, Infallible>(())
            })
            .unwrap();
            rkn.accept().unwrap();
            ros.attempt(
                h,
                &mut |_: f64, _: &[f64], d: &mut [f64]| {
                    d[0] = 1.;
                    Ok::<_, Infallible>(())
                },
                Some(&mut |_, _, j| {
                    j[0] = 0.;
                    Ok(())
                }),
                Some(&mut |_, _, d| {
                    d[0] = 0.;
                    Ok(())
                }),
            )
            .unwrap();
            ros.accept().unwrap();
            mixed
                .attempt(h, &mut |_: f64,
                                  _: &[f64],
                                  _: &[f64],
                                  _: &[f64],
                                  a: &mut [f64],
                                  dz: &mut [f64]| {
                    a[0] = 0.;
                    dz[0] = 1.;
                    Ok::<_, Infallible>(())
                })
                .unwrap();
            mixed.accept().unwrap();
        }
        let elapsed = rk.time() - origin;
        assert!((rk.state()[0] - elapsed).abs() < 2e-14);
        assert!((rkn.position()[0] - elapsed).abs() < 2e-14);
        assert!((ros.state()[0] - elapsed).abs() < 2e-14);
        assert!((mixed.position()[0] - elapsed).abs() < 2e-14);
        assert!((mixed.auxiliary()[0] - elapsed).abs() < 2e-14);
        assert_eq!(rk.time(), rkn.time());
        assert_eq!(rk.time(), ros.time());
        assert_eq!(rk.time(), mixed.time());
    }
}
#[test]
fn endpoint_sequence_reaches_exact_requested_times_without_state_drift() {
    for sign in [-1., 1.] {
        let origin = 1e12;
        let mut s =
            ExplicitRungeKuttaStepper::new(Tsit5.tableau().unwrap(), origin, &[0.]).unwrap();
        let mut c = AdaptiveController::new(ControllerConfig::proportional(5).unwrap(), sign * 0.1)
            .unwrap();
        let requested = [origin + sign * 0.3, origin + sign * 0.7, origin + sign * 1.];
        let mut seen = 0;
        integrate_rk(
            &mut s,
            &mut c,
            requested[2],
            &requested,
            100,
            &mut |_: f64, _: &[f64], d: &mut [f64]| {
                d[0] = 1.;
                Ok::<_, Infallible>(())
            },
            &mut |_: &[f64], _: &[f64], _: &[f64]| Ok(0.),
            &mut |o: Observation<'_>| {
                assert!((o.state[0] - (o.time - origin)).abs() < 2e-14);
                if o.requested {
                    assert_eq!(o.time, requested[seen]);
                    seen += 1;
                }
                Ok(ObserverAction::Continue)
            },
        )
        .unwrap();
        assert_eq!(seen, 3);
        assert_eq!(s.time(), requested[2]);
    }
}
