//! Independent batch and ODE ensemble execution.
//!
//! Cases may run sequentially or on Rayon's global thread pool. In either
//! mode, every case runs and the returned [`CaseOutcome`] values retain input
//! order. A failed case therefore does not discard successful sibling cases.
//!
//! # Example
//!
//! ```
//! use differential_equations::algorithms::explicit::Tsit5;
//! use differential_equations::{
//!     ExecutionPolicy, OdeProblem, SolveOptions, solve_ensemble,
//! };
//!
//! let initial_values = [1.0, 2.0, 3.0];
//! let outcomes = solve_ensemble(
//!     initial_values,
//!     |initial| {
//!         OdeProblem::new(
//!             |du: &mut [f64], u: &[f64], rate: &f64, _: f64| {
//!                 du[0] = *rate * u[0];
//!             },
//!             vec![initial],
//!             (0.0, 1.0),
//!             1.0,
//!         )
//!     },
//!     Tsit5,
//!     &SolveOptions::default(),
//!     ExecutionPolicy::Parallel,
//! );
//!
//! assert_eq!(outcomes.len(), 3);
//! assert!(outcomes.iter().all(|case| case.result.is_ok()));
//! assert_eq!(outcomes[2].index, 2);
//! ```

use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, solve};
use rayon::prelude::*;

/// Selects how independent cases are executed.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ExecutionPolicy {
    /// Execute cases in input order on the calling thread.
    Sequential,
    /// Execute cases on Rayon's global thread pool.
    #[default]
    Parallel,
}

/// The indexed result of one independent case.
///
/// Outcomes are returned in ascending index order. The explicit index makes
/// failures unambiguous even when callers later filter or partition results.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CaseOutcome<T, E> {
    /// Zero-based position of the case in the input iterator.
    pub index: usize,
    /// The case's success value or error.
    pub result: Result<T, E>,
}

impl<T, E> CaseOutcome<T, E> {
    /// Converts this outcome into its underlying result.
    pub fn into_result(self) -> Result<T, E> {
        self.result
    }
}

/// Runs independent fallible cases as an ordered batch.
///
/// All cases are attempted. Parallel execution uses Rayon's global thread
/// pool, while sequential execution is useful for reproducibility checks,
/// low-volume workloads, and environments that already provide outer
/// parallelism. Both policies produce the same ascending input-index order.
/// Parallel execution first collects the input iterator so Rayon can preserve
/// indexed order; sequential execution streams directly from the iterator.
/// Panics are not converted into case errors and follow the normal Rust/Rayon
/// propagation behavior.
///
/// The input items, result values, errors, and runner must satisfy Rayon's
/// thread-safety bounds even when `execution` is [`ExecutionPolicy::Sequential`]
/// because the policy is selected at runtime.
pub fn solve_batch<I, C, T, E, R>(
    cases: I,
    runner: R,
    execution: ExecutionPolicy,
) -> Vec<CaseOutcome<T, E>>
where
    I: IntoIterator<Item = C>,
    C: Send,
    T: Send,
    E: Send,
    R: Fn(C) -> Result<T, E> + Send + Sync,
{
    match execution {
        ExecutionPolicy::Sequential => cases
            .into_iter()
            .enumerate()
            .map(|(index, case)| CaseOutcome {
                index,
                result: runner(case),
            })
            .collect(),
        ExecutionPolicy::Parallel => cases
            .into_iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .enumerate()
            .map(|(index, case)| CaseOutcome {
                index,
                result: runner(case),
            })
            .collect(),
    }
}

/// Solves an ensemble of independently constructed ODE problems.
///
/// `problem_factory` executes in the worker handling each case. This keeps a
/// complete [`OdeProblem`] local to one thread, including callbacks that are
/// not themselves transferable between threads. The algorithm is cloned once
/// per case and options are shared read-only.
///
/// Every case is attempted. Each [`CaseOutcome`] contains that case's
/// [`Solution`] or [`SolveError`], and outcomes remain aligned with the input
/// order regardless of completion order.
pub fn solve_ensemble<I, C, F, P, A, B>(
    cases: I,
    problem_factory: B,
    algorithm: A,
    options: &SolveOptions,
    execution: ExecutionPolicy,
) -> Vec<CaseOutcome<Solution, SolveError>>
where
    I: IntoIterator<Item = C>,
    C: Send,
    A: OdeAlgorithm + Clone + Send + Sync,
    B: Fn(C) -> OdeProblem<F, P> + Send + Sync,
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    solve_batch(
        cases,
        |case| {
            let problem = problem_factory(case);
            solve(&problem, algorithm.clone(), options)
        },
        execution,
    )
}
