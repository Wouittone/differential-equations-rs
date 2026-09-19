//! Shared evaluation, error scaling, and initial-step estimation.

use super::history::Workspace;
use crate::{OdeProblem, SolveError, SolveOptions, SolverStats};

pub(super) fn scaled_error_norm(
    error: &[f64],
    state: &[f64],
    candidate: &[f64],
    options: &SolveOptions,
) -> f64 {
    let mut squared = 0.0;
    for ((error, state), candidate) in error.iter().zip(state).zip(candidate) {
        let scale = options.absolute_tolerance
            + options.relative_tolerance * state.abs().max(candidate.abs());
        squared += (error / scale).powi(2);
    }
    (squared / state.len() as f64).sqrt()
}

#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
pub(super) fn estimate_initial_step<F, P>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
    state: &[f64],
    time: f64,
    direction: f64,
    maximum_step: f64,
    order: usize,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<f64, SolveError>
where
    F: crate::OdeFunction<P>,
{
    let dimension = state.len() as f64;
    let mut state_norm = 0.0;
    let mut derivative_norm = 0.0;
    for (value, derivative) in state.iter().zip(&workspace.derivative) {
        let scale = options.absolute_tolerance + options.relative_tolerance * value.abs();
        state_norm += (value / scale).powi(2);
        derivative_norm += (derivative / scale).powi(2);
    }
    state_norm = (state_norm / dimension).sqrt();
    derivative_norm = (derivative_norm / dimension).sqrt();
    let trial_step = if state_norm < 1.0e-5 || derivative_norm < 1.0e-5 {
        1.0e-6
    } else {
        0.01 * state_norm / derivative_norm
    }
    .min(maximum_step);
    for component in 0..state.len() {
        workspace.temporary[component] =
            state[component] + direction * trial_step * workspace.derivative[component];
    }
    evaluate(
        problem,
        &mut workspace.predicted,
        &workspace.temporary,
        time + direction * trial_step,
        stats,
    )?;
    ensure_finite(&workspace.predicted)?;
    let mut curvature_norm = 0.0;
    for component in 0..state.len() {
        let scale =
            options.absolute_tolerance + options.relative_tolerance * state[component].abs();
        curvature_norm +=
            ((workspace.predicted[component] - workspace.derivative[component]) / scale).powi(2);
    }
    curvature_norm = (curvature_norm / dimension).sqrt() / trial_step;
    let largest = derivative_norm.max(curvature_norm);
    let accuracy_step = if largest <= 1.0e-15 {
        (trial_step * 1.0e-3).max(1.0e-6)
    } else {
        (0.01 / largest).powf(1.0 / order as f64)
    };
    Ok((100.0 * trial_step).min(accuracy_step).min(maximum_step))
}

pub(super) fn evaluate<F, P>(
    problem: &OdeProblem<F, P>,
    derivative: &mut [f64],
    state: &[f64],
    time: f64,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: crate::OdeFunction<P>,
{
    problem
        .rhs
        .evaluate(derivative, state, problem.parameters(), time)?;
    stats.rhs_evaluations += 1;
    Ok(())
}

pub(super) fn ensure_finite(values: &[f64]) -> Result<(), SolveError> {
    values
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteDerivative)
}
