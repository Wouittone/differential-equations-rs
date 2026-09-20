use differential_equations::solvers::explicit::Tsit5;
use differential_equations::solvers::second_order::{
    Rkn4, SecondOrderOdeProblem, solve_second_order,
};
use differential_equations::{OdeProblem, SolveError, SolveOptions, solve};

#[test]
fn mutable_first_order_borrows_and_observes_stages() {
    let mut times = Vec::new();
    {
        let problem = OdeProblem::new_mut(
            |du: &mut [f64], u: &[f64], _: &(), t| {
                times.push(t);
                du[0] = u[0];
            },
            [1.0],
            (0.0, 0.1),
            (),
        );
        let solution = solve(&problem, Tsit5, &SolveOptions::default()).unwrap();
        assert!((solution.last_state()[0] - 0.1f64.exp()).abs() < 1e-5);
        assert!(solution.stats().rhs_evaluations > 0);
    }
    assert_eq!(times[0], 0.0);
    assert!(times.len() > 5);
}
#[test]
fn first_order_error_stops_at_failing_evaluation() {
    let mut calls = 0;
    {
        let problem = OdeProblem::new_fallible(
            |du: &mut [f64], _: &[f64], _: &(), _| {
                calls += 1;
                if calls == 3 {
                    return Err(SolveError::InvalidCallbackState);
                }
                du[0] = 1.0;
                Ok(())
            },
            [0.0],
            (0.0, 1.0),
            (),
        );
        assert_eq!(
            solve(&problem, Tsit5, &SolveOptions::default()).unwrap_err(),
            SolveError::InvalidCallbackState
        );
    }
    assert_eq!(calls, 3);
}
#[test]
fn mutable_second_order_and_early_failure() {
    let mut calls = 0;
    {
        let problem = SecondOrderOdeProblem::new_fallible(
            |a: &mut [f64], _: &[f64], q: &[f64], _: &(), _| {
                calls += 1;
                if calls == 3 {
                    return Err(SolveError::InvalidCallbackState);
                }
                a[0] = -q[0];
                Ok(())
            },
            [0.0],
            [1.0],
            (0.0, 1.0),
            (),
        );
        assert!(
            solve_second_order(
                &problem,
                Rkn4,
                &SolveOptions::default()
                    .with_initial_step(0.1)
                    .with_adaptive(false)
            )
            .is_err()
        );
    }
    assert_eq!(calls, 3);
    let mut count = 0;
    {
        let problem = SecondOrderOdeProblem::new_mut(
            |a: &mut [f64], _: &[f64], _: &[f64], _: &(), _| {
                count += 1;
                a[0] = 0.0;
            },
            [1.0],
            [0.0],
            (0.0, 1.0),
            (),
        );
        solve_second_order(
            &problem,
            Rkn4,
            &SolveOptions::default()
                .with_initial_step(0.1)
                .with_adaptive(false),
        )
        .unwrap();
    }
    assert!(count > 0);
}
