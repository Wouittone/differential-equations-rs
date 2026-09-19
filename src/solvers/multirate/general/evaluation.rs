//! Checked fast, slow, and combined right-hand-side evaluation helpers.

use crate::{SolveError, SolverStats, SplitOdeProblem};

pub(super) fn evaluate_fast<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    state: &[f64],
    time: f64,
    output: &mut [f64],
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    FE: crate::OdeFunction<P>,
{
    problem.evaluate_explicit(output, state, time)?;
    stats.rhs_evaluations += 1;
    finite(output)
}

pub(super) fn evaluate_slow<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    state: &[f64],
    time: f64,
    output: &mut [f64],
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    FI: crate::OdeFunction<P>,
{
    problem.evaluate_implicit(output, state, time)?;
    stats.rhs_evaluations += 1;
    finite(output)
}

pub(super) fn evaluate_total<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    state: &[f64],
    time: f64,
    output: &mut [f64],
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    FE: crate::OdeFunction<P>,
    FI: crate::OdeFunction<P>,
{
    let mut slow = vec![0.0; output.len()];
    evaluate_fast(problem, state, time, output, stats)?;
    evaluate_slow(problem, state, time, &mut slow, stats)?;
    for (value, slow) in output.iter_mut().zip(slow) {
        *value += slow;
    }
    finite(output)
}

pub(super) fn finite(values: &[f64]) -> Result<(), SolveError> {
    values
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteDerivative)
}
