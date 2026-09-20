use super::*;
use std::convert::Infallible;
fn harmonic(_: f64, q: &[f64], _: &[f64], a: &mut [f64]) -> Result<(), Infallible> {
    for (a, q) in a.iter_mut().zip(q) {
        *a = -q;
    }
    Ok(())
}
fn propagate(h: f64, end: f64) -> GaussJackson8 {
    let mut solver =
        GaussJackson8::new(0., &[1.], &[0.], h, GaussJacksonConfig::default()).unwrap();
    while (end - solver.time()) * h > 0. {
        solver.try_step_to(end, &mut harmonic).unwrap();
    }
    solver
}
#[test]
fn harmonic_order_and_long_arc() {
    let mut errors = Vec::new();
    for h in [0.4, 0.2, 0.1] {
        let solver = propagate(h, 20.);
        let err = (solver.position()[0] - 20f64.cos())
            .abs()
            .max((solver.velocity()[0] + 20f64.sin()).abs());
        eprintln!("h={h}: error={err:.16e}, stats={:?}", solver.statistics());
        errors.push(err);
        assert!(solver.statistics().accepted_steps > solver.statistics().startup_steps);
    }
    assert!(errors[0] / errors[1] > 150., "errors={errors:?}");
    assert!(errors[1] < 1e-7);
    let solver = propagate(0.05, 200.);
    assert!((solver.position()[0] - 200f64.cos()).abs() < 1e-9);
}
#[test]
fn velocity_dependent_forces_backward_and_partial_end() {
    // q = exp(-t), q'' = -q - 2q'.
    for h in [0.1, -0.1] {
        let end = if h > 0. { 3.037 } else { -3.037 };
        let mut solver =
            GaussJackson8::new(0., &[1.], &[-1.], h, GaussJacksonConfig::default()).unwrap();
        let mut f = |_: f64, q: &[f64], v: &[f64], a: &mut [f64]| {
            a[0] = -q[0] - 2. * v[0];
            Ok::<_, Infallible>(())
        };
        while (end - solver.time()) * h > 0. {
            solver.try_step_to(end, &mut f).unwrap();
        }
        assert_eq!(solver.time(), end);
        assert!((solver.position()[0] - (-end).exp()).abs() < 2e-9);
        assert!((solver.velocity()[0] + (-end).exp()).abs() < 2e-9);
        assert_eq!(solver.history_len(), 0);
    }
}
#[test]
fn startup_short_intervals_and_domain_are_accurate() {
    let mut solver = GaussJackson8::new(
        2.,
        &[2f64.cos()],
        &[-2f64.sin()],
        0.2,
        GaussJacksonConfig::default(),
    )
    .unwrap();
    let mut f = |t: f64, q: &[f64], _: &[f64], a: &mut [f64]| {
        assert!((2.0..=2.03).contains(&t));
        a[0] = -q[0];
        Ok::<_, Infallible>(())
    };
    solver.try_step_to(2.03, &mut f).unwrap();
    assert!((solver.position()[0] - 2.03f64.cos()).abs() < 1e-14);
    assert!((solver.velocity()[0] + 2.03f64.sin()).abs() < 1e-14);
    assert_eq!(solver.statistics().startup_steps, 1);
}
#[test]
fn continuation_clone_and_restart() {
    let mut original = propagate(0.1, 2.);
    let mut clone = original.clone();
    for _ in 0..10 {
        original.try_step(&mut harmonic).unwrap();
        clone.try_step(&mut harmonic).unwrap();
    }
    assert_eq!(original.position(), clone.position());
    assert_eq!(original.statistics(), clone.statistics());
    clone.restart(0., &[2.], &[0.], 0.1).unwrap();
    assert_eq!(clone.history_len(), 0);
    clone.try_step(&mut harmonic).unwrap();
    assert!((clone.position()[0] - 2. * 0.1f64.cos()).abs() < 1e-13);
}
#[test]
fn errors_preserve_accepted_state_and_payload() {
    let mut solver = propagate(0.1, 2.);
    let before = solver.clone();
    let mut bad = |_: f64, _: &[f64], _: &[f64], _: &mut [f64]| Err::<(), _>(1234);
    assert!(matches!(
        solver.try_step(&mut bad),
        Err(GaussJacksonError::Acceleration(1234))
    ));
    assert_eq!(solver.time(), before.time());
    assert_eq!(solver.position(), before.position());
    assert_eq!(solver.history, before.history);
    solver.try_step(&mut harmonic).unwrap();
    let mut expected = before;
    expected.try_step(&mut harmonic).unwrap();
    assert_eq!(solver.position(), expected.position());
}
#[test]
fn dense_quintic_has_exact_endpoints_and_consistent_velocity() {
    let solver = propagate(0.1, 2.);
    let mut q = [0.];
    let mut v = [0.];
    solver
        .interpolate_into(solver.time(), &mut q, &mut v)
        .unwrap();
    assert_eq!(q.as_slice(), solver.position());
    assert_eq!(v.as_slice(), solver.velocity());
    let mid = (solver.dense_start + solver.time()) / 2.;
    solver.interpolate_into(mid, &mut q, &mut v).unwrap();
    assert!((q[0] - mid.cos()).abs() < 1e-9);
    assert!((v[0] + mid.sin()).abs() < 1e-8);
}
#[test]
fn validates_zero_steps_invalid_config_and_noop() {
    assert!(GaussJackson8::new(0., &[1.], &[0.], 0., GaussJacksonConfig::default()).is_err());
    let mut solver =
        GaussJackson8::new(0., &[1.], &[0.], 0.1, GaussJacksonConfig::default()).unwrap();
    solver.try_step_to(0., &mut harmonic).unwrap();
    assert_eq!(solver.statistics().acceleration_evaluations, 0);
    assert!(solver.try_step_to(-1., &mut harmonic).is_err());
}
#[test]
fn corrector_and_startup_failures_are_explicit_and_transactional() {
    let mut solver =
        GaussJackson8::new(0., &[1.], &[0.], 0.2, GaussJacksonConfig::default()).unwrap();
    for _ in 0..8 {
        solver.try_step(&mut harmonic).unwrap();
    }
    let before = solver.clone();
    solver.config.max_corrector_iterations = 1;
    assert!(matches!(
        solver.try_step(&mut harmonic),
        Err(GaussJacksonError::CorrectorConvergence { iterations: 1 })
    ));
    assert_eq!(solver.q, before.q);
    assert_eq!(solver.v, before.v);
    assert_eq!(solver.history, before.history);
    assert_eq!(solver.sum, before.sum);
    assert_eq!(solver.double_sum, before.double_sum);
    let mut q = [0.];
    let mut v = [0.];
    let mut expected_q = [0.];
    let mut expected_v = [0.];
    solver.interpolate_into(1.55, &mut q, &mut v).unwrap();
    before
        .interpolate_into(1.55, &mut expected_q, &mut expected_v)
        .unwrap();
    assert_eq!(q, expected_q);
    assert_eq!(v, expected_v);
    let config = GaussJacksonConfig {
        absolute_tolerance: 1e-30,
        relative_tolerance: 0.,
        max_startup_refinements: 0,
        ..Default::default()
    };
    let mut startup = GaussJackson8::new(0., &[1.], &[0.], 1., config).unwrap();
    assert!(matches!(
        startup.try_step(&mut harmonic),
        Err(GaussJacksonError::StartupConvergence { refinements: 0 })
    ));
    assert_eq!(startup.time(), 0.);
    assert_eq!(startup.history_len(), 0);
    assert_eq!(startup.position(), &[1.]);
}
#[test]
fn polynomial_acceleration_and_scaled_units() {
    // q=t^8 + 3t + 2, v=8t^7+3. Exactness of the ordinate formula
    // tests coefficient moments independently of oscillatory superconvergence.
    for h in [0.05, -0.05] {
        let end = if h > 0. { 2. } else { -2. };
        let mut solver =
            GaussJackson8::new(0., &[2.], &[3.], h, GaussJacksonConfig::default()).unwrap();
        let mut f = |t: f64, _: &[f64], _: &[f64], a: &mut [f64]| {
            a[0] = 56. * t.powi(6);
            Ok::<_, Infallible>(())
        };
        while (end - solver.time()) * h > 0. {
            solver.try_step_to(end, &mut f).unwrap();
        }
        assert!((solver.position()[0] - (end.powi(8) + 3. * end + 2.)).abs() < 1e-9);
        assert!((solver.velocity()[0] - (8. * end.powi(7) + 3.)).abs() < 1e-9);
    }
}
#[test]
fn two_body_circular_orbit_and_energy_long_arc() {
    let mut solver = GaussJackson8::new(
        0.,
        &[1., 0.],
        &[0., 1.],
        0.04,
        GaussJacksonConfig::default(),
    )
    .unwrap();
    let mut gravity = |_: f64, q: &[f64], _: &[f64], a: &mut [f64]| {
        let r = q[0].hypot(q[1]);
        a[0] = -q[0] / r.powi(3);
        a[1] = -q[1] / r.powi(3);
        Ok::<_, Infallible>(())
    };
    let end = 100.;
    while solver.time() < end {
        solver.try_step_to(end, &mut gravity).unwrap();
    }
    assert!((solver.position()[0] - end.cos()).abs() < 1e-8);
    assert!((solver.position()[1] - end.sin()).abs() < 1e-8);
    let q = solver.position();
    let v = solver.velocity();
    let energy = 0.5 * (v[0] * v[0] + v[1] * v[1]) - 1. / q[0].hypot(q[1]);
    assert!((energy + 0.5).abs() < 1e-10);
}
#[test]
fn dense_coefficients_round_trip_and_failed_force_preserves_dense() {
    let mut solver = propagate(0.1, 2.);
    let mut coefficients = [0.; 6];
    let (start, end) = solver.dense_coefficients_into(&mut coefficients).unwrap();
    let time = (start + end) / 2.;
    let mut q = [0.];
    let mut v = [0.];
    solver.interpolate_into(time, &mut q, &mut v).unwrap();
    let poly = coefficients.iter().rev().fold(0., |a, c| a * 0.5 + c);
    assert!((poly - q[0]).abs() < 1e-15);
    let mut calls = 0;
    let mut broken = |_: f64, q: &[f64], _: &[f64], a: &mut [f64]| {
        calls += 1;
        if calls == 2 {
            Err(42)
        } else {
            a[0] = -q[0];
            Ok(())
        }
    };
    assert!(solver.try_step(&mut broken).is_err());
    let mut after = [0.; 6];
    solver.dense_coefficients_into(&mut after).unwrap();
    assert_eq!(coefficients, after);
}
#[test]
fn agrees_with_independent_pinned_gj8_reference() {
    // BSD reference b0115763, fixture reproduced by scripts/gauss_jackson_reference.py.
    // Its centered startup differs from ours, so compare at converged step sizes.
    let solver = propagate(0.1, 20.);
    assert!((solver.position()[0] - 0.4080820618135249).abs() < 1e-12);
    assert!((solver.velocity()[0] - (-0.9129452507272791)).abs() < 1e-12);
    let mut damped =
        GaussJackson8::new(0., &[1.], &[-1.], 0.1, GaussJacksonConfig::default()).unwrap();
    let mut f = |_: f64, q: &[f64], v: &[f64], a: &mut [f64]| {
        a[0] = -q[0] - 2. * v[0];
        Ok::<_, Infallible>(())
    };
    while damped.time() < 3. {
        damped.try_step_to(3., &mut f).unwrap();
    }
    assert!((damped.position()[0] - 0.04978706836819483).abs() < 1e-12);
    assert!((damped.velocity()[0] - (-0.0497870683681613)).abs() < 1e-12);
}
#[test]
fn startup_extrapolation_has_high_order_before_refinement() {
    let config = GaussJacksonConfig {
        absolute_tolerance: 1.,
        relative_tolerance: 0.,
        max_startup_refinements: 0,
        ..Default::default()
    };
    let mut errors = Vec::new();
    for h in [1.5, 0.75] {
        let mut solver = GaussJackson8::new(0., &[1.], &[0.], h, config).unwrap();
        solver.try_step(&mut harmonic).unwrap();
        errors.push(
            (solver.position()[0] - h.cos())
                .abs()
                .max((solver.velocity()[0] + h.sin()).abs()),
        );
        assert_eq!(solver.statistics().startup_refinements, 0);
    }
    assert!(errors[0] / errors[1] > 900., "startup errors={errors:?}");
}
#[test]
fn eccentric_two_body_matches_kepler_reference() {
    let mut errors = Vec::new();
    for h in [0.02, 0.01] {
        let e = 0.6_f64;
        let mut solver = GaussJackson8::new(
            0.,
            &[1. - e, 0.],
            &[0., ((1. + e) / (1. - e)).sqrt()],
            h,
            GaussJacksonConfig::default(),
        )
        .unwrap();
        let mut gravity = |_: f64, q: &[f64], _: &[f64], a: &mut [f64]| {
            let r = q[0].hypot(q[1]);
            for i in 0..2 {
                a[i] = -q[i] / r.powi(3);
            }
            Ok::<_, Infallible>(())
        };
        let end = 30.;
        while solver.time() < end {
            solver.try_step_to(end, &mut gravity).unwrap();
        }
        let mean = end.rem_euclid(std::f64::consts::TAU);
        let mut eccentric = mean;
        for _ in 0..12 {
            eccentric -= (eccentric - e * eccentric.sin() - mean) / (1. - e * eccentric.cos());
        }
        let q = [eccentric.cos() - e, (1. - e * e).sqrt() * eccentric.sin()];
        let denominator = 1. - e * eccentric.cos();
        let v = [
            -eccentric.sin() / denominator,
            (1. - e * e).sqrt() * eccentric.cos() / denominator,
        ];
        let error = (0..2)
            .map(|i| {
                (solver.position()[i] - q[i])
                    .abs()
                    .max((solver.velocity()[i] - v[i]).abs())
            })
            .fold(0., f64::max);
        assert!(error < 1e-7, "h={h}, Kepler error={error}");
        errors.push(error);
    }
    assert!(errors[0] / errors[1] > 150., "eccentric errors={errors:?}");
}
#[test]
fn large_epoch_steps_match_accepted_time_and_restart_across_binades() {
    for h in [0.1, -0.1] {
        for start in [1e12, 2f64.powi(40) - h * 12.] {
            let mut solver =
                GaussJackson8::new(start, &[0.], &[1.], h, GaussJacksonConfig::default()).unwrap();
            assert_eq!(solver.step_size(), (start + h) - start);
            let mut force = |_: f64, _: &[f64], _: &[f64], a: &mut [f64]| {
                a[0] = 0.;
                Ok::<_, Infallible>(())
            };
            for _ in 0..40 {
                solver.try_step(&mut force).unwrap();
                assert!(
                    (solver.position()[0] - (solver.time() - start)).abs() < 2e-13,
                    "time={}, position={}",
                    solver.time(),
                    solver.position()[0]
                );
                assert!((solver.velocity()[0] - 1.).abs() < 2e-14);
            }
            let end = solver.time() + 0.037 * h.signum();
            solver.try_step_to(end, &mut force).unwrap();
            assert_eq!(solver.time(), end);
            assert!((solver.position()[0] - (end - start)).abs() < 2e-13);
        }
    }
    assert!(GaussJackson8::new(1e12, &[0.], &[1.], 1e-10, GaussJacksonConfig::default()).is_err());
}

#[test]
fn extreme_step_dense_output_remains_finite() {
    let mut solver =
        GaussJackson8::new(0., &[1.], &[0.], 1e200, GaussJacksonConfig::default()).unwrap();
    let mut force = |_: f64, _: &[f64], _: &[f64], a: &mut [f64]| {
        a[0] = 0.;
        Ok::<_, Infallible>(())
    };
    for _ in 0..12 {
        let start = solver.time();
        solver.try_step(&mut force).unwrap();
        let mut q = [0.];
        let mut v = [0.];
        solver
            .interpolate_into(start + (solver.time() - start) / 2., &mut q, &mut v)
            .unwrap();
        assert_eq!(q, [1.]);
        assert_eq!(v, [0.]);
        let mut coefficients = [0.; 6];
        solver.dense_coefficients_into(&mut coefficients).unwrap();
        assert!(coefficients.iter().all(|value| value.is_finite()));
        solver.export_dense_segment().unwrap();
    }
    // Finite endpoints do not guarantee representable Hermite coefficients.
    solver.q[0] = 1e308;
    assert!(matches!(
        solver.dense_coefficients_into(&mut [0.; 6]),
        Err(GaussJacksonError::NonFinite)
    ));
    assert!(matches!(
        solver.interpolate_into(
            solver.dense_start + (solver.time - solver.dense_start) / 2.,
            &mut [0.],
            &mut [0.]
        ),
        Err(GaussJacksonError::NonFinite)
    ));
}

#[test]
fn full_step_rejects_later_stationary_time() {
    let mut solver = GaussJackson8::new(
        2f64.powi(53) - 1.,
        &[1.],
        &[0.],
        1.,
        GaussJacksonConfig::default(),
    )
    .unwrap();
    let mut force = |_: f64, _: &[f64], _: &[f64], a: &mut [f64]| {
        a[0] = 0.;
        Ok::<_, Infallible>(())
    };
    solver.try_step(&mut force).unwrap();
    let calls = solver.statistics().acceleration_evaluations;
    assert!(solver.try_step(&mut force).is_err());
    assert_eq!(solver.statistics().acceleration_evaluations, calls);
    solver.try_step_to(solver.time(), &mut force).unwrap();
}
