use std::cell::Cell;
use std::rc::Rc;

use differential_equations::{CallbackAction, OdeProblem, SaveMode, Sdirk2, SolveOptions, solve};

#[test]
fn nonautonomous_rhs_and_backward_time() {
    let forward = OdeProblem::new(
        |du: &mut [f64], _: &[f64], _: &(), t: f64| du[0] = t,
        vec![0.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions {
        absolute_tolerance: 1.0e-9,
        relative_tolerance: 1.0e-9,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let solution = solve(&forward, Sdirk2, &options).unwrap();
    assert!((solution.last_state()[0] - 0.5).abs() < 2.0e-7);

    let backward = OdeProblem::new(
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = 1.0,
        vec![1.0],
        (1.0, 0.0),
        (),
    );
    let solution = solve(&backward, Sdirk2, &options).unwrap();
    assert!(solution.last_state()[0].abs() < 2.0e-9);
}

#[test]
fn analytic_and_finite_difference_jacobians_agree() {
    type Rhs = fn(&mut [f64], &[f64], &(), f64);
    let rhs: Rhs = |du, u, _, _| {
        du[0] = -100.0 * u[0];
        du[1] = -2.0 * u[1];
    };
    let numeric = OdeProblem::new(rhs, vec![1.0, 1.0], (0.0, 0.2), ());
    let analytic = OdeProblem::new(rhs, vec![1.0, 1.0], (0.0, 0.2), ()).with_jacobian(
        |jac: &mut [f64], _: &[f64], _: &(), _: f64| {
            jac.copy_from_slice(&[-100.0, 0.0, 0.0, -2.0]);
        },
    );
    let options = SolveOptions {
        absolute_tolerance: 1.0e-8,
        relative_tolerance: 1.0e-8,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let a = solve(&numeric, Sdirk2, &options).unwrap();
    let b = solve(&analytic, Sdirk2, &options).unwrap();
    for (x, y) in a.last_state().iter().zip(b.last_state()) {
        assert!((x - y).abs() < 2.0e-8);
    }
    assert!(b.stats().jacobian_evaluations > 0);
}

#[test]
fn adaptive_rejection_and_callback_reinitialize() {
    let rhs_calls = Rc::new(Cell::new(0usize));
    let calls = Rc::clone(&rhs_calls);
    let problem = OdeProblem::new(
        move |du: &mut [f64], u: &[f64], _: &(), _: f64| {
            calls.set(calls.get() + 1);
            du[0] = if u[0] == 42.0 {
                f64::NAN
            } else {
                -500.0 * u[0]
            };
        },
        vec![1.0],
        (0.0, 0.1),
        (),
    )
    .with_continuous_callback(
        |u: &[f64], _: &(), _: f64| u[0] - 0.8,
        |u, _, _| {
            u[0] = 42.0;
            CallbackAction::Terminate
        },
    );
    let options = SolveOptions {
        absolute_tolerance: 1.0e-7,
        relative_tolerance: 1.0e-7,
        initial_step: Some(0.05),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let solution = solve(&problem, Sdirk2, &options).unwrap();
    assert_eq!(solution.last_state(), &[42.0]);
    assert!(rhs_calls.get() > 0);
    assert!(solution.stats().rejected_steps > 0);
}
