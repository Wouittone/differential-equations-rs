#[cfg(feature = "parallel")]
use differential_equations::solvers::explicit::{Euler, Tsit5};
#[cfg(feature = "parallel")]
use differential_equations::{
    CaseOutcome, OdeProblem, SolveError, SolveOptions, solve_batch, solve_ensemble,
};
use differential_equations::{ExecutionPolicy, solve_batch_sequential};
use std::rc::Rc;
#[cfg(feature = "parallel")]
use std::sync::Arc;
#[cfg(feature = "parallel")]
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "parallel")]
type ExponentialProblem = OdeProblem<fn(&mut [f64], &[f64], &f64, f64), f64>;

#[cfg(feature = "parallel")]
fn exponential_rhs(du: &mut [f64], u: &[f64], rate: &f64, _: f64) {
    du[0] = *rate * u[0];
}

#[cfg(feature = "parallel")]
fn exponential_problem(initial: f64) -> ExponentialProblem {
    OdeProblem::new(
        exponential_rhs as fn(&mut [f64], &[f64], &f64, f64),
        vec![initial],
        (0.0, 1.0),
        1.0,
    )
}

#[test]
#[cfg(feature = "parallel")]
fn parallel_batch_preserves_input_order() {
    let outcomes = solve_batch(
        0..128,
        |value| {
            std::thread::yield_now();
            Ok::<_, ()>(value * value)
        },
        ExecutionPolicy::Parallel,
    );

    assert_eq!(outcomes.len(), 128);
    for (expected, outcome) in outcomes.into_iter().enumerate() {
        assert_eq!(outcome.index, expected);
        assert_eq!(outcome.result, Ok(expected * expected));
    }
}

#[test]
#[cfg(feature = "parallel")]
fn successful_ensemble_solves_every_case() {
    let options = SolveOptions::default().with_tolerances(1.0e-10, 1.0e-10);
    let outcomes = solve_ensemble(
        [1.0, 2.0, 4.0],
        exponential_problem,
        Tsit5,
        &options,
        ExecutionPolicy::Parallel,
    );

    for (initial, outcome) in [1.0, 2.0, 4.0].into_iter().zip(outcomes) {
        let solution = outcome.result.expect("case should solve");
        let final_value = solution.last_state()[0];
        assert!((final_value - initial * std::f64::consts::E).abs() < 1.0e-7);
    }
}

#[test]
#[cfg(feature = "parallel")]
fn per_case_failures_are_preserved_and_indexed() {
    let executions = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&executions);
    let outcomes = solve_batch(
        0..5,
        move |index| {
            observed.fetch_add(1, Ordering::Relaxed);
            if index == 1 || index == 3 {
                Err("expected failure")
            } else {
                Ok(index)
            }
        },
        ExecutionPolicy::Parallel,
    );

    assert_eq!(executions.load(Ordering::Relaxed), 5);
    assert_eq!(outcomes[0].result, Ok(0));
    assert_eq!(outcomes[1].index, 1);
    assert_eq!(outcomes[1].result, Err("expected failure"));
    assert_eq!(outcomes[2].result, Ok(2));
    assert_eq!(outcomes[3].index, 3);
    assert_eq!(outcomes[3].result, Err("expected failure"));
    assert_eq!(outcomes[4].result, Ok(4));
}

#[test]
#[cfg(feature = "parallel")]
fn invalid_ode_case_does_not_abort_ensemble() {
    let outcomes = solve_ensemble(
        [1.0, f64::NAN, 2.0],
        exponential_problem,
        Tsit5,
        &SolveOptions::default(),
        ExecutionPolicy::Parallel,
    );

    assert!(outcomes[0].result.is_ok());
    assert_eq!(outcomes[1].index, 1);
    assert_eq!(outcomes[1].result, Err(SolveError::NonFiniteInitialState));
    assert!(outcomes[2].result.is_ok());
}

#[test]
#[cfg(feature = "parallel")]
fn sequential_and_parallel_ensembles_are_equivalent() {
    let options = SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(0.05);
    let cases = [1.0, 2.0, 4.0];
    let sequential = solve_ensemble(
        cases,
        exponential_problem,
        Euler,
        &options,
        ExecutionPolicy::Sequential,
    );
    let parallel = solve_ensemble(
        cases,
        exponential_problem,
        Euler,
        &options,
        ExecutionPolicy::Parallel,
    );

    assert_eq!(parallel, sequential);
}

#[test]
#[cfg(feature = "parallel")]
fn empty_batches_return_no_outcomes() {
    let outcomes = solve_batch(
        std::iter::empty::<usize>(),
        Ok::<_, ()>,
        ExecutionPolicy::Parallel,
    );
    assert!(outcomes.is_empty());
}

#[test]
fn sequential_batch_accepts_thread_local_values_and_runner_state() {
    let total = Rc::new(std::cell::Cell::new(0));
    let observed = Rc::clone(&total);
    let values = [Rc::new(1), Rc::new(2), Rc::new(3)];
    let outcomes = solve_batch_sequential(values, move |value| {
        observed.set(observed.get() + *value);
        Ok::<_, Rc<str>>(value)
    });

    assert_eq!(total.get(), 6);
    assert_eq!(outcomes.len(), 3);
}

#[test]
fn execution_policy_defaults_to_sequential() {
    assert_eq!(ExecutionPolicy::default(), ExecutionPolicy::Sequential);
}

#[test]
#[cfg(feature = "parallel")]
fn public_batch_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<ExecutionPolicy>();
    assert_send_sync::<CaseOutcome<usize, SolveError>>();

    let shared = Arc::new(AtomicUsize::new(0));
    let worker_state = Arc::clone(&shared);
    let outcomes = solve_batch(
        0..32,
        move |_| {
            worker_state.fetch_add(1, Ordering::Relaxed);
            Ok::<_, ()>(())
        },
        ExecutionPolicy::Parallel,
    );
    assert!(outcomes.iter().all(|outcome| outcome.result.is_ok()));
    assert_eq!(shared.load(Ordering::Relaxed), 32);
}
