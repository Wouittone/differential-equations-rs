use crate::solvers::explicit::general::Dp5;
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions};

/// Automatic low-order Dormand--Prince composite.
///
/// OrdinaryDiffEq defines `AutoDP5(stiff_alg)` as
/// `AutoAlgSwitch(DP5(), stiff_alg)`, which dynamically switches between the
/// non-stiff DP5 method and the supplied stiff method. The regular ODE driver
/// does not yet expose the state needed for an in-flight algorithm switch, so
/// this implementation uses a deterministic fallback instead: it first runs
/// DP5 and, if DP5 reports a numerical failure that can indicate stiffness,
/// restarts the problem from its initial state with the configured stiff
/// algorithm.
///
/// The fallback is attempted for [`SolveError::NonFiniteDerivative`],
/// [`SolveError::StepSizeUnderflow`], and [`SolveError::MaxStepsExceeded`].
/// Configuration and callback errors are returned directly. A fallback is a
/// full restart, so right-hand-side functions and callbacks with external side
/// effects can be evaluated again.
#[derive(Clone, Debug, PartialEq)]
pub struct AutoDp5<A> {
    /// The stiff component requested by the upstream composite constructor.
    pub stiff_algorithm: A,
}

impl<A> AutoDp5<A> {
    /// Constructs an AutoDP5 facade around a stiff component.
    pub const fn new(stiff_algorithm: A) -> Self {
        Self { stiff_algorithm }
    }

    /// Returns the configured stiff component.
    pub const fn stiff_algorithm(&self) -> &A {
        &self.stiff_algorithm
    }
}

impl<A: OdeAlgorithm> OdeAlgorithm for AutoDp5<A> {
    fn solve<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        match Dp5.solve(problem, options) {
            Err(error) if should_retry_with_stiff(error) => {
                self.stiff_algorithm.solve(problem, options)
            }
            result => result,
        }
    }
}

fn should_retry_with_stiff(error: SolveError) -> bool {
    matches!(
        error,
        SolveError::NonFiniteDerivative
            | SolveError::StepSizeUnderflow
            | SolveError::MaxStepsExceeded
    )
}

/// Uppercase acronym spelling used by the pinned Julia algorithm name.
#[allow(non_camel_case_types)]
pub type AutoDP5<A> = AutoDp5<A>;
