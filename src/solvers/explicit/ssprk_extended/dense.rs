use crate::SolverStats;
use crate::callback::CallbackOutcome;
use crate::solution::{BorrowedHermiteSegment, DenseSegment, HermiteSegment, TrajectoryRecorder};
use crate::{OdeProblem, SolveError};

#[allow(clippy::too_many_arguments)]
pub(super) fn apply_hermite_callbacks<F, P>(
    problem: &OdeProblem<F, P>,
    previous_state: &[f64],
    previous_time: f64,
    state: &mut [f64],
    time: &mut f64,
    state_before_effect: &mut [f64],
    event_tolerance: f64,
    start_derivative: &[f64],
    endpoint_state: &mut [f64],
    endpoint_derivative: &mut [f64],
    endpoint_prepared: &mut bool,
    stats: &mut SolverStats,
) -> Result<CallbackOutcome, SolveError>
where
    F: crate::OdeFunction<P>,
{
    if !problem.has_continuous_callbacks() {
        *endpoint_prepared = false;
        return problem.apply_step_callbacks(
            previous_state,
            previous_time,
            state,
            time,
            state_before_effect,
            event_tolerance,
            None,
        );
    }
    endpoint_state.copy_from_slice(state);
    problem.rhs.evaluate(
        endpoint_derivative,
        endpoint_state,
        problem.parameters(),
        *time,
    )?;
    stats.rhs_evaluations += 1;
    if !endpoint_derivative.iter().all(|value| value.is_finite()) {
        return Err(SolveError::NonFiniteDerivative);
    }
    *endpoint_prepared = true;
    let segment = BorrowedHermiteSegment::new(
        previous_time,
        *time,
        previous_state,
        endpoint_state,
        start_derivative,
        endpoint_derivative,
    )
    .map_err(|_| SolveError::NonFiniteDerivative)?;
    let mut interpolate = |sample_time: f64, output: &mut [f64]| {
        segment
            .interpolate(sample_time, output)
            .map_err(|_| SolveError::NonFiniteDerivative)
    };
    problem.apply_step_callbacks(
        previous_state,
        previous_time,
        state,
        time,
        state_before_effect,
        event_tolerance,
        Some(&mut interpolate),
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn record_hermite_step<F, P>(
    problem: &OdeProblem<F, P>,
    previous_state: &[f64],
    state: &[f64],
    start_derivative: &[f64],
    endpoint_state: &mut [f64],
    endpoint_derivative: &mut [f64],
    endpoint_prepared: &mut bool,
    previous_time: f64,
    attempted_time: f64,
    time: f64,
    final_time: bool,
    recorder: &mut TrajectoryRecorder<'_>,
    stats: &mut SolverStats,
) -> Result<bool, SolveError>
where
    F: crate::OdeFunction<P>,
{
    if !recorder.needs_dense_sampling() && !recorder.retains_dense_output() {
        *endpoint_prepared = false;
        return Ok(false);
    }
    if !*endpoint_prepared {
        endpoint_state.copy_from_slice(state);
        problem.rhs.evaluate(
            endpoint_derivative,
            endpoint_state,
            problem.parameters(),
            attempted_time,
        )?;
        stats.rhs_evaluations += 1;
        if !endpoint_derivative.iter().all(|value| value.is_finite()) {
            return Err(SolveError::NonFiniteDerivative);
        }
    }
    let segment = BorrowedHermiteSegment::new(
        previous_time,
        attempted_time,
        previous_state,
        endpoint_state,
        start_derivative,
        endpoint_derivative,
    )
    .map_err(|_| SolveError::NonFiniteDerivative)?;
    recorder
        .record_step_dense(
            previous_state,
            previous_time,
            state,
            time,
            final_time,
            &segment,
        )
        .map_err(|_| SolveError::NonFiniteDerivative)?;
    if recorder.retains_dense_output() {
        let segment = HermiteSegment::new_bounded(
            previous_time,
            attempted_time,
            time,
            previous_state.to_vec(),
            endpoint_state.to_vec(),
            start_derivative.to_vec(),
            endpoint_derivative.to_vec(),
        )
        .map_err(|_| SolveError::NonFiniteDerivative)?;
        recorder.retain_hermite_segment(segment);
    }
    *endpoint_prepared = false;
    Ok(true)
}
