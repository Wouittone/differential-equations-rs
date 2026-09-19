use crate::{OdeProblem, SolveError, SolverStats};

pub(super) fn evaluate<F, P>(
    problem: &OdeProblem<F, P>,
    output: &mut [f64],
    state: &[f64],
    time: f64,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: crate::OdeFunction<P>,
{
    problem
        .rhs
        .evaluate(output, state, problem.parameters(), time)?;
    stats.rhs_evaluations += 1;
    if output.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(SolveError::NonFiniteDerivative)
    }
}
