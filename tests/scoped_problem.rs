use differential_equations::{ScopedOdeProblem, solvers::explicit::Tsit5, stepping::*};
use std::error::Error;

#[derive(Debug)]
struct ForceError {
    code: u32,
    source: std::io::Error,
}
impl std::fmt::Display for ForceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "force {}", self.code)
    }
}
impl Error for ForceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.source)
    }
}
fn failure(code: u32) -> ForceError {
    ForceError {
        code,
        source: std::io::Error::other("original cause"),
    }
}
fn control() -> AdaptiveController {
    AdaptiveController::new(ControllerConfig::proportional(5).unwrap(), 0.1).unwrap()
}

#[test]
fn borrowed_hooks_run_in_order_and_finalize() {
    let mut calls = 0;
    let mut times = Vec::new();
    let mut final_time = None;
    let mut jac_calls = 0;
    let rate = -1.0;
    {
        let mut p = ScopedOdeProblem::new(
            |_: f64, y: &[f64], dy: &mut [f64]| {
                calls += 1;
                dy[0] = rate * y[0];
                Ok::<_, ForceError>(())
            },
            [1.0],
            (0.0, 1.0),
        )
        .with_observer(|o: Observation<'_>| {
            times.push(o.time);
            Ok(ObserverAction::Continue)
        })
        .with_jacobian(|_: f64, _: &[f64], j: &mut [f64]| {
            jac_calls += 1;
            j[0] = rate;
            Ok(())
        })
        .with_finalizer(|outcome, state: &[f64]| {
            assert!((state[0] - (-1.0f64).exp()).abs() < 1e-8);
            final_time = Some(outcome.time);
            Ok(())
        });
        let (_, jac) = p.functions_mut();
        jac.unwrap()(0.0, &[1.0], &mut [0.0]).unwrap();
        let mut stepper =
            ExplicitRungeKuttaStepper::new(Tsit5.tableau().unwrap(), 0.0, &[1.0]).unwrap();
        p.integrate(
            &mut stepper,
            &mut control(),
            &[0.0, 0.5, 1.0],
            100,
            &mut |_: &[f64], _: &[f64], e: &[f64]| Ok(e[0].abs() / 1e-10),
        )
        .unwrap();
    }
    assert!(calls > 0);
    assert_eq!(jac_calls, 1);
    assert_eq!(times[0], 0.0);
    assert!(times.contains(&0.5));
    assert_eq!(final_time, Some(1.0));
}
#[test]
fn rhs_observer_and_finalizer_preserve_original_error_and_cause() {
    for path in 0..3 {
        let mut p = ScopedOdeProblem::new(
            |_: f64, _: &[f64], dy: &mut [f64]| {
                if path == 0 {
                    return Err(failure(10));
                }
                dy[0] = 1.0;
                Ok(())
            },
            [0.0],
            (0.0, 1.0),
        )
        .with_observer(|_: Observation<'_>| {
            if path == 1 {
                Err(failure(20))
            } else {
                Ok(ObserverAction::Continue)
            }
        })
        .with_finalizer(|_, _: &[f64]| if path == 2 { Err(failure(30)) } else { Ok(()) });
        let mut stepper =
            ExplicitRungeKuttaStepper::new(Tsit5.tableau().unwrap(), 0.0, &[0.0]).unwrap();
        let error = p
            .integrate(
                &mut stepper,
                &mut control(),
                &[],
                100,
                &mut |_: &[f64], _: &[f64], _: &[f64]| Ok(0.0),
            )
            .unwrap_err();
        match error {
            IntegrationError::Step(StepError::User(original)) => {
                assert_eq!(original.code, 10 * (path + 1));
                assert_eq!(original.source().unwrap().to_string(), "original cause");
            }
            other => panic!("unexpected error {other:?}"),
        }
    }
}
#[test]
fn borrowed_fallible_jacobian_keeps_payload() {
    let code = 41;
    let mut p = ScopedOdeProblem::new(
        |_: f64, _: &[f64], dy: &mut [f64]| {
            dy[0] = 0.0;
            Ok::<_, ForceError>(())
        },
        [0.0],
        (0.0, 1.0),
    )
    .with_jacobian(|_: f64, _: &[f64], _: &mut [f64]| Err(failure(code)));
    let (_, jac) = p.functions_mut();
    assert_eq!(jac.unwrap()(0.0, &[0.0], &mut [0.0]).unwrap_err().code, 41);
}
#[cfg(feature = "parallel")]
#[test]
fn eligible_problem_moves_to_rayon_with_borrowed_context() {
    let rate = -2.0;
    let p = ScopedOdeProblem::new(
        |_: f64, y: &[f64], dy: &mut [f64]| {
            dy[0] = rate * y[0];
            Ok::<_, ForceError>(())
        },
        [1.0],
        (0.0, 1.0),
    );
    fn send_sync<T: Send + Sync>(_: &T) {}
    send_sync(&p);
    let (state, ()) = rayon::join(
        move || {
            let mut p = p;
            let mut s =
                ExplicitRungeKuttaStepper::new(Tsit5.tableau().unwrap(), 0.0, &[1.0]).unwrap();
            p.integrate(
                &mut s,
                &mut control(),
                &[],
                100,
                &mut |_: &[f64], _: &[f64], e: &[f64]| Ok(e[0].abs() / 1e-10),
            )
            .unwrap();
            s.state()[0]
        },
        || (),
    );
    assert!((state - (-2.0f64).exp()).abs() < 1e-8);
}
