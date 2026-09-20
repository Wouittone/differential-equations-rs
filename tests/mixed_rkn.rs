use differential_equations::{solvers::second_order::FineRkn4, stepping::*};
use std::convert::Infallible;
fn rhs(
    _: f64,
    q: &[f64],
    v: &[f64],
    z: &[f64],
    a: &mut [f64],
    dz: &mut [f64],
) -> Result<(), Infallible> {
    // q''=-q-0.2q'; row-major STM obeys Phi'= [[0,1],[-1,-0.2]] Phi.
    a[0] = -q[0] - 0.2 * v[0];
    dz[0] = z[2];
    dz[1] = z[3];
    dz[2] = -z[0] - 0.2 * z[2];
    dz[3] = -z[1] - 0.2 * z[3];
    Ok(())
}
#[test]
fn drag_variational_stm_matches_analytic_and_finite_difference_without_extra_forces() {
    let tab = FineRkn4.tableau().unwrap();
    let mut s = MixedRknStepper::new(tab, 0., &[1.], &[0.], &[1., 0., 0., 1.]).unwrap();
    s.attempt(0.1, &mut rhs).unwrap();
    s.reject().unwrap();
    for _ in 0..100 {
        s.attempt(0.01, &mut rhs).unwrap();
        s.accept().unwrap();
    }
    let omega = 0.99_f64.sqrt();
    let decay = (-0.1_f64).exp();
    let sin = omega.sin() / omega;
    let cos = omega.cos();
    let exact = [
        decay * (cos + 0.1 * sin),
        decay * sin,
        -decay * sin,
        decay * (cos - 0.1 * sin),
    ];
    for (actual, expected) in s.auxiliary().iter().zip(exact) {
        assert!((actual - expected).abs() < 2e-9, "{actual} {expected}");
    }
    assert!((s.position()[0] - exact[0]).abs() < 2e-9);
    assert!((s.velocity()[0] - exact[2]).abs() < 2e-9);
    assert_eq!(s.statistics().rhs_evaluations, 101 * tab.stages() - 1);
    // Independent finite differences through the physical-only stepper.
    for column in 0..2 {
        let delta = 1e-5;
        let mut q = [1.];
        let mut v = [0.];
        if column == 0 {
            q[0] += delta;
        } else {
            v[0] += delta;
        }
        let mut p =
            RknStepper::new(tab, AccelerationPolicy::VelocityDependent, 0., &q, &v).unwrap();
        for _ in 0..100 {
            p.attempt(0.01, &mut |_: f64, q: &[f64], v: &[f64], a: &mut [f64]| {
                a[0] = -q[0] - 0.2 * v[0];
                Ok::<_, Infallible>(())
            })
            .unwrap();
            p.accept().unwrap();
        }
        assert!(((p.position()[0] - s.position()[0]) / delta - s.auxiliary()[column]).abs() < 1e-8);
        assert!(
            ((p.velocity()[0] - s.velocity()[0]) / delta - s.auxiliary()[2 + column]).abs() < 1e-8
        );
    }
    for _ in 0..100 {
        s.attempt(-0.01, &mut rhs).unwrap();
        s.accept().unwrap();
    }
    assert!((s.position()[0] - 1.).abs() < 3e-9);
    assert!(s.velocity()[0].abs() < 3e-9);
    for (a, b) in s.auxiliary().iter().zip([1., 0., 0., 1.]) {
        assert!((a - b).abs() < 3e-9);
    }
}
#[test]
fn mixed_errors_and_restart_are_atomic() {
    let tab = FineRkn4.tableau().unwrap();
    let mut s = MixedRknStepper::new(tab, 0., &[1.], &[0.], &[1., 0., 0., 1.]).unwrap();
    assert_eq!(
        s.reset(2., &[1.], &[0.], &[1.]),
        Err(StepFailure::Dimension)
    );
    assert_eq!(s.time(), 0.);
    let result = s.attempt(0.1, &mut |_, _, _, _, _, _| Err::<(), _>("force"));
    assert!(matches!(result, Err(StepError::User("force"))));
    assert_eq!(s.position(), &[1.]);
    let view = s.attempt(0.1, &mut rhs).unwrap();
    assert_eq!(view.physical.previous_position, &[1.]);
    assert_eq!(view.auxiliary_error.unwrap().len(), 4);
    s.reset(0., &[1.], &[0.], &[1., 0., 0., 1.]).unwrap();
    assert_eq!(s.accept(), Err(StepFailure::NoCandidate));
}

#[test]
fn nonlinear_drag_sensitivity_matches_parameter_finite_differences() {
    // q''=-q-c*v*sqrt(v^2+eps); z=(dq/dc,dv/dc), a 2x1 rectangular sensitivity.
    let c = 0.2;
    let tab = FineRkn4.tableau().unwrap();
    let mut s = MixedRknStepper::new(tab, 0., &[1.], &[0.3], &[0., 0.]).unwrap();
    let mut calls = 0;
    let mut force = |_: f64, q: &[f64], v: &[f64], z: &[f64], a: &mut [f64], dz: &mut [f64]| {
        calls += 1;
        let speed = (v[0] * v[0] + 0.01).sqrt();
        let drag_v = -c * (speed + v[0] * v[0] / speed);
        a[0] = -q[0] - c * v[0] * speed;
        dz[0] = z[1];
        dz[1] = -z[0] + drag_v * z[1] - v[0] * speed;
        Ok::<_, Infallible>(())
    };
    for _ in 0..100 {
        s.attempt(0.01, &mut force).unwrap();
        s.accept().unwrap();
    }
    assert_eq!(calls, 100 * tab.stages());
    let physical = |parameter: f64| {
        let mut p = RknStepper::new(
            tab,
            AccelerationPolicy::VelocityDependent,
            0.,
            &[1.],
            &[0.3],
        )
        .unwrap();
        for _ in 0..100 {
            p.attempt(0.01, &mut |_: f64, q: &[f64], v: &[f64], a: &mut [f64]| {
                a[0] = -q[0] - parameter * v[0] * (v[0] * v[0] + 0.01).sqrt();
                Ok::<_, Infallible>(())
            })
            .unwrap();
            p.accept().unwrap();
        }
        [p.position()[0], p.velocity()[0]]
    };
    let delta = 1e-5;
    let plus = physical(c + delta);
    let minus = physical(c - delta);
    for k in 0..2 {
        assert!(((plus[k] - minus[k]) / (2. * delta) - s.auxiliary()[k]).abs() < 2e-9);
    }
}

#[test]
fn auxiliary_to_physical_coupling_has_fourth_order_convergence() {
    // q''=-q+z, z'=-z, q(0)=v(0)=0,z(0)=1.
    // q=(exp(-t)-cos(t)+sin(t))/2, v=(-exp(-t)+sin(t)+cos(t))/2.
    let solve = |h: f64| {
        let mut s =
            MixedRknStepper::new(FineRkn4.tableau().unwrap(), 0., &[0.], &[0.], &[1.]).unwrap();
        for _ in 0..(1. / h) as usize {
            s.attempt(h, &mut |_: f64,
                               q: &[f64],
                               _: &[f64],
                               z: &[f64],
                               a: &mut [f64],
                               dz: &mut [f64]| {
                a[0] = -q[0] + z[0];
                dz[0] = -z[0];
                Ok::<_, Infallible>(())
            })
            .unwrap();
            s.accept().unwrap();
        }
        let q = ((-1_f64).exp() - 1_f64.cos() + 1_f64.sin()) / 2.;
        let v = (-(-1_f64).exp() + 1_f64.sin() + 1_f64.cos()) / 2.;
        (s.position()[0] - q)
            .abs()
            .max((s.velocity()[0] - v).abs())
            .max((s.auxiliary()[0] - (-1_f64).exp()).abs())
    };
    let coarse = solve(0.125);
    let fine = solve(0.0625);
    assert!(coarse / fine > 12., "{coarse} {fine}");
}

#[test]
fn fully_coupled_third_order_equation_matches_independent_taylor_series() {
    // q''=z and z'=q; q'''=q, initial (q,v,z)=(1,0,0).
    let mut s = MixedRknStepper::new(FineRkn4.tableau().unwrap(), 0., &[1.], &[0.], &[0.]).unwrap();
    for _ in 0..100 {
        s.attempt(0.01, &mut |_: f64,
                              q: &[f64],
                              _: &[f64],
                              z: &[f64],
                              a: &mut [f64],
                              dz: &mut [f64]| {
            a[0] = z[0];
            dz[0] = q[0];
            Ok::<_, Infallible>(())
        })
        .unwrap();
        s.accept().unwrap();
    }
    let mut exact = [0.; 3];
    let mut term = 1.;
    for n in 0..30 {
        if n > 0 {
            term /= n as f64;
        }
        exact[n % 3] += term;
    }
    assert!((s.position()[0] - exact[0]).abs() < 1e-9);
    assert!((s.velocity()[0] - exact[2]).abs() < 1e-9);
    assert!((s.auxiliary()[0] - exact[1]).abs() < 1e-9);
}
