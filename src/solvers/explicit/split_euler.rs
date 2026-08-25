//! First-order Euler integration for typed split ODE problems.

use super::general::Euler;
use crate::{OdeProblem, Solution, SolveError, SolveOptions, SplitOdeProblem, solve};
use std::cell::RefCell;

/// Explicit Euler applied to the sum of a split problem's two right-hand sides.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SplitEuler;

/// Algorithm contract for typed split ODE problems.
pub trait SplitOdeAlgorithm {
    /// Solve a typed split problem with this algorithm.
    fn solve<FE, FI, P>(
        &self,
        problem: &SplitOdeProblem<FE, FI, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        FE: Fn(&mut [f64], &[f64], &P, f64),
        FI: Fn(&mut [f64], &[f64], &P, f64);
}

/// Solves a typed split problem with a selected split/IMEX algorithm.
pub fn solve_split<FE, FI, P, A>(
    problem: &SplitOdeProblem<FE, FI, P>,
    algorithm: A,
    options: &SolveOptions,
) -> Result<Solution, SolveError>
where
    FE: Fn(&mut [f64], &[f64], &P, f64),
    FI: Fn(&mut [f64], &[f64], &P, f64),
    A: SplitOdeAlgorithm,
{
    algorithm.solve(problem, options)
}

impl SplitOdeAlgorithm for SplitEuler {
    fn solve<FE, FI, P>(
        &self,
        problem: &SplitOdeProblem<FE, FI, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        FE: Fn(&mut [f64], &[f64], &P, f64),
        FI: Fn(&mut [f64], &[f64], &P, f64),
    {
        let implicit_derivative = RefCell::new(vec![0.0; problem.dimension()]);
        let combined = |derivative: &mut [f64], state: &[f64], _: &(), time: f64| {
            problem.evaluate_explicit(derivative, state, time);
            let mut implicit_derivative = implicit_derivative.borrow_mut();
            problem.evaluate_implicit(&mut implicit_derivative, state, time);
            for (total, implicit) in derivative.iter_mut().zip(implicit_derivative.iter()) {
                *total += implicit;
            }
        };
        let combined_problem = OdeProblem::new(
            combined,
            problem.initial_state().to_vec(),
            problem.time_span(),
            (),
        );
        solve(&combined_problem, Euler, options)
    }
}

/// Solve a typed [`SplitOdeProblem`] with [`SplitEuler`].
///
/// Each derivative evaluation invokes both split components at the same state
/// and time, matching OrdinaryDiffEq's `SplitEuler` update.
pub fn solve_split_euler<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    algorithm: SplitEuler,
    options: &SolveOptions,
) -> Result<Solution, SolveError>
where
    FE: Fn(&mut [f64], &[f64], &P, f64),
    FI: Fn(&mut [f64], &[f64], &P, f64),
{
    algorithm.solve(problem, options)
}
