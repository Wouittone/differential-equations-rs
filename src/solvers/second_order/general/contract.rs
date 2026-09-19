use super::super::function::SecondOrderFunction;
use super::driver::validate;
use super::problem::SecondOrderOdeProblem;
use super::solution::SecondOrderSolution;
use crate::{SolveError, SolveOptions};
use thiserror::Error;

/// Configuration or integration failure specific to partitioned ODE states.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
#[non_exhaustive]
pub enum SecondOrderSolveError {
    /// Position and velocity partitions do not have the same dimension.
    #[error("position and velocity dimensions must match")]
    StateDimensionMismatch,
    /// An algorithm produced malformed saved trajectory data.
    #[error("invalid saved solution: {0}")]
    InvalidSolution(#[from] crate::SolutionConstructionError),
    /// A common ODE validation or integration error.
    #[error("{0}")]
    Solve(
        #[from]
        #[source]
        SolveError,
    ),
}

/// An algorithm for `q' = v` second-order ODE problems.
///
/// This trait is a downstream extension point. Implementors can evaluate the
/// acceleration through [`SecondOrderOdeProblem::evaluate_acceleration`] and
/// construct checked output with [`SecondOrderSolution::from_saved`]. A custom
/// driver must honor the complete problem and option contract that it accepts.
/// An implementation without callback support must check
/// [`SecondOrderOdeProblem::has_callbacks`] and return
/// [`SolveError::CallbacksUnsupported`] instead of silently skipping callback
/// effects, guards, initializers, or finalizers.
pub trait SecondOrderOdeAlgorithm {
    /// Solves a problem after validating its partitioned state and options.
    fn solve<F, P>(
        &self,
        problem: &SecondOrderOdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<SecondOrderSolution, SecondOrderSolveError>
    where
        F: SecondOrderFunction<P>,
    {
        validate(problem, options)?;
        let mut solution = self.solve_validated(problem, options)?;
        solution.set_state_shape_checked(problem.state_shape())?;
        Ok(solution)
    }

    /// Executes the numerical method after common inputs have been checked.
    ///
    /// Implementors may rely on [`SecondOrderOdeAlgorithm::solve`] having
    /// validated both state partitions, the time span, solver options, and
    /// requested output times. User code should normally call
    /// [`SecondOrderOdeAlgorithm::solve`] or [`solve_second_order`]; direct
    /// callers of this lower-level hook are responsible for those invariants.
    /// Implementors remain responsible for honoring adaptive stepping, step
    /// bounds, saving, time stops, dense-output retention, and callback
    /// lifecycle semantics, or for returning the corresponding typed error
    /// when a capability is not supported.
    fn solve_validated<F, P>(
        &self,
        problem: &SecondOrderOdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<SecondOrderSolution, SecondOrderSolveError>
    where
        F: SecondOrderFunction<P>;
}

/// Solves a second-order ODE without flattening its position and velocity.
pub fn solve_second_order<F, P, A>(
    problem: &SecondOrderOdeProblem<F, P>,
    algorithm: A,
    options: &SolveOptions,
) -> Result<SecondOrderSolution, SecondOrderSolveError>
where
    F: SecondOrderFunction<P>,
    A: SecondOrderOdeAlgorithm,
{
    algorithm.solve(problem, options)
}
