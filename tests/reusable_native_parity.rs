use differential_equations::solvers::{
    explicit::{Bs5, DP8, Dp5, Feagin10, Rk4, Tsit5, Vern9},
    second_order::{
        Dprkn6, Erkn5, FineRkn4, FineRkn5, Nystrom4, SecondOrderOdeProblem, solve_second_order,
    },
};
use differential_equations::stepping::*;
use differential_equations::{OdeProblem, SaveMode, SolveOptions, solve};
use std::{cell::RefCell, convert::Infallible};
fn nonlinear(t: f64, y: &[f64], d: &mut [f64]) {
    d[0] = -0.7 * y[0] + 1.3 * y[1] + 0.2 * y[0] * y[1] + 0.05 * t.cos();
    d[1] = -2.1 * y[0] + 0.4 * y[1] - 0.15 * y[0] * y[0] + 0.02 * t;
    d[2] = 0.3 * y[0] - 0.2 * y[1] - 0.5 * y[2] + 0.1 * y[1] * y[2];
}
fn close(a: &[f64], b: &[f64], tol: f64, context: &str) {
    for (&x, &y) in a.iter().zip(b) {
        assert!((x - y).abs() < tol, "{context}: {x} != {y}");
    }
}
#[test]
fn rk_native_stage_endpoint_and_both_error_formulas_match_after_rejection() {
    macro_rules! check {
        ($method:expr) => {{
            let method = $method;
            let tab = method.tableau().unwrap();
            for sign in [-1., 1.] {
                let h = sign / 64.;
                let start = 0.25;
                let initial = [0.8, -0.3, 1.2];
                let capture = RefCell::new(Vec::new());
                let problem = OdeProblem::new(
                    |d: &mut [f64], y: &[f64], _: &(), t: f64| {
                        nonlinear(t, y, d);
                        capture.borrow_mut().push(d.to_vec());
                    },
                    initial,
                    (start, start + h),
                    (),
                );
                let options = SolveOptions::new()
                    .with_adaptive(false)
                    .with_initial_step(h.abs())
                    .with_save(SaveMode::Endpoints);
                let native = solve(&problem, method, &options).unwrap();
                let mut s = ExplicitRungeKuttaStepper::new(tab, start, &initial).unwrap();
                let mut rhs = |t: f64, y: &[f64], d: &mut [f64]| {
                    nonlinear(t, y, d);
                    Ok::<_, Infallible>(())
                };
                s.attempt(2. * h, &mut rhs).unwrap();
                s.reject().unwrap();
                let view = s.attempt(h, &mut rhs).unwrap();
                let captured = capture.borrow();
                assert!(captured.len() >= tab.stages());
                for (i, stage) in captured.iter().take(tab.stages()).enumerate() {
                    close(view.stage(i).unwrap(), stage, 3e-13, tab.name());
                }
                close(view.candidate, native.last_state(), 3e-13, tab.name());
                for (weights, actual) in [
                    (tab.error(), view.component_error),
                    (tab.second_error(), view.second_error),
                ] {
                    match (weights, actual) {
                        (Some(w), Some(actual)) => {
                            let mut expected = [0.; 3];
                            for (i, stage) in captured.iter().take(tab.stages()).enumerate() {
                                for k in 0..3 {
                                    expected[k] += w[i] * stage[k];
                                }
                            }
                            for e in &mut expected {
                                *e *= h;
                            }
                            close(actual, &expected, 2e-14, tab.name());
                        }
                        (None, None) => {}
                        _ => panic!("estimator presence differs"),
                    }
                }
                s.accept().unwrap();
                for _ in 1..8 {
                    s.attempt(h, &mut rhs).unwrap();
                    s.accept().unwrap();
                }
                let p = OdeProblem::new(
                    |d: &mut [f64], y: &[f64], _: &(), t: f64| nonlinear(t, y, d),
                    initial,
                    (start, start + 8. * h),
                    (),
                );
                let full = solve(&p, method, &options).unwrap();
                close(s.state(), full.last_state(), 1e-12, tab.name());
                assert_eq!(s.statistics().rejected_steps, 1);
            }
        }};
    }
    check!(Tsit5);
    check!(Dp5);
    check!(Bs5);
    check!(DP8);
    check!(Vern9);
    check!(Feagin10);
    check!(Rk4);
}
fn acceleration(t: f64, q: &[f64], v: &[f64], a: &mut [f64], velocity_dependent: bool) {
    a[0] = -q[0] + 0.4 * q[1] + 0.1 * q[0] * q[1] + 0.03 * t.cos();
    a[1] = -0.7 * q[0] - 1.3 * q[1] + 0.2 * q[0] * q[0];
    if velocity_dependent {
        a[0] -= 0.2 * v[0] * (v[0] * v[0] + 0.1).sqrt();
        a[1] += 0.3 * v[0] - 0.1 * v[1];
    }
}
#[test]
fn rkn_native_stages_and_partition_error_weights_match_signed_retries() {
    macro_rules! check {
        ($method:expr,$velocity:expr) => {{
            let method = $method;
            let tab = method.tableau().unwrap();
            let dep = $velocity;
            let policy = if dep {
                AccelerationPolicy::VelocityDependent
            } else {
                AccelerationPolicy::VelocityIndependent
            };
            for sign in [-1., 1.] {
                let h = sign / 64.;
                let start = 0.25;
                let q = [0.8, -0.3];
                let v = [0.2, 0.7];
                let capture = RefCell::new(Vec::new());
                let p = SecondOrderOdeProblem::new(
                    |a: &mut [f64], v: &[f64], q: &[f64], _: &(), t: f64| {
                        acceleration(t, q, v, a, dep);
                        capture.borrow_mut().push(a.to_vec());
                    },
                    v,
                    q,
                    (start, start + h),
                    (),
                );
                let options = SolveOptions::new()
                    .with_adaptive(false)
                    .with_initial_step(h.abs())
                    .with_save(SaveMode::Endpoints);
                let native = solve_second_order(&p, method, &options).unwrap();
                let mut s = RknStepper::new(tab, policy, start, &q, &v).unwrap();
                let mut rhs = |t: f64, q: &[f64], v: &[f64], a: &mut [f64]| {
                    acceleration(t, q, v, a, dep);
                    Ok::<_, Infallible>(())
                };
                s.attempt(2. * h, &mut rhs).unwrap();
                s.reject().unwrap();
                let view = s.attempt(h, &mut rhs).unwrap();
                let captured = capture.borrow();
                for (i, stage) in captured.iter().take(tab.stages()).enumerate() {
                    close(view.stage(i).unwrap(), stage, 3e-13, tab.name());
                }
                close(view.position, native.last_position(), 3e-13, tab.name());
                close(view.velocity, native.last_velocity(), 3e-13, tab.name());
                for (weights, actual, scale) in [
                    (tab.error(), view.position_error, h * h),
                    (tab.velocity_error(), view.velocity_error, h),
                ] {
                    match (weights, actual) {
                        (Some(w), Some(actual)) => {
                            let mut expected = [0.; 2];
                            for (i, stage) in captured.iter().take(tab.stages()).enumerate() {
                                for k in 0..2 {
                                    expected[k] += w[i] * stage[k];
                                }
                            }
                            for e in &mut expected {
                                *e *= scale;
                            }
                            close(actual, &expected, 2e-14, tab.name());
                        }
                        (None, None) => {}
                        _ => panic!("partition error presence differs"),
                    }
                }
                s.accept().unwrap();
                for _ in 1..8 {
                    s.attempt(h, &mut rhs).unwrap();
                    s.accept().unwrap();
                }
                let p = SecondOrderOdeProblem::new(
                    |a: &mut [f64], v: &[f64], q: &[f64], _: &(), t: f64| {
                        acceleration(t, q, v, a, dep)
                    },
                    v,
                    q,
                    (start, start + 8. * h),
                    (),
                );
                let full = solve_second_order(&p, method, &options).unwrap();
                close(s.position(), full.last_position(), 1e-12, tab.name());
                close(s.velocity(), full.last_velocity(), 1e-12, tab.name());
            }
        }};
    }
    check!(FineRkn4, true);
    check!(FineRkn5, true);
    check!(Nystrom4, true);
    check!(Dprkn6, false);
    check!(Erkn5, false);
}
#[test]
fn frequency_fitted_tableau_is_explicitly_unsupported() {
    let tab = differential_equations::tableau::parse_tableau(
        include_str!("../src/tableau/resources/explicit/frk65.json"),
        "Frk65",
    )
    .unwrap();
    assert!(!tab.fitted_weights().is_empty());
    assert!(matches!(
        ExplicitRungeKuttaStepper::new(&tab, 0., &[1.]),
        Err(StepFailure::UnsupportedTableau)
    ));
}
