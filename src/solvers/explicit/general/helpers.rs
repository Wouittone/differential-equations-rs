use super::tableau_access::TableauAccess;
use super::workspace::Workspace;
use crate::{OdeProblem, SolveError, SolveOptions, SolverStats};

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

pub(super) fn evaluate_stage<F, P>(
    problem: &OdeProblem<F, P>,
    derivative: &mut [f64],
    state: &[f64],
    time: f64,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: crate::OdeFunction<P>,
{
    evaluate(problem, derivative, state, time, stats)?;
    stats.stage_evaluations += 1;
    Ok(())
}

pub(super) fn evaluate_dense_stage<F, P>(
    problem: &OdeProblem<F, P>,
    derivative: &mut [f64],
    state: &[f64],
    time: f64,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: crate::OdeFunction<P>,
{
    evaluate(problem, derivative, state, time, stats)?;
    stats.dense_stage_evaluations += 1;
    Ok(())
}

pub(super) fn ensure_finite(values: &[f64]) -> Result<(), SolveError> {
    values
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteDerivative)
}

pub(super) fn estimate_initial_step<F, P>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
    states: (&[f64], &mut [f64]),
    integration: (f64, f64, f64),
    order: usize,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<f64, SolveError>
where
    F: crate::OdeFunction<P>,
{
    let (state, scratch) = states;
    let (time, direction, maximum_step) = integration;
    let dimension = state.len() as f64;
    let mut state_norm = 0.0;
    let mut derivative_norm = 0.0;
    for (value, derivative) in state.iter().zip(workspace.stage(0)) {
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

    for ((trial, value), derivative) in workspace
        .temporary
        .iter_mut()
        .zip(state)
        .zip(&workspace.stages[..workspace.dimension])
    {
        *trial = value + direction * trial_step * derivative;
    }
    evaluate_stage(
        problem,
        scratch,
        &workspace.temporary,
        time + direction * trial_step,
        stats,
    )?;
    ensure_finite(scratch)?;

    let mut curvature_norm = 0.0;
    for ((next, initial), value) in scratch
        .iter()
        .zip(&workspace.stages[..workspace.dimension])
        .zip(state)
    {
        let scale = options.absolute_tolerance + options.relative_tolerance * value.abs();
        curvature_norm += ((next - initial) / scale).powi(2);
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

#[allow(clippy::too_many_arguments)]
pub(super) fn perform_step<F, P, T>(
    problem: &OdeProblem<F, P>,
    state: &[f64],
    time: f64,
    step: f64,
    candidate: &mut [f64],
    workspace: &mut Workspace,
    stats: &mut SolverStats,
    tableau: T,
) -> Result<(), SolveError>
where
    F: crate::OdeFunction<P>,
    T: TableauAccess,
{
    let stage_count = tableau.weights().len();
    for stage_index in 1..stage_count {
        combine(
            &mut workspace.temporary,
            state,
            step,
            &workspace.stages,
            workspace.dimension,
            stage_index,
            tableau.stage_row(stage_index),
        );
        if !workspace.stiffness_reference_state.is_empty() && stage_index + 2 == stage_count {
            workspace
                .stiffness_reference_state
                .copy_from_slice(&workspace.temporary);
        }
        let start = stage_index * workspace.dimension;
        evaluate_dense_stage(
            problem,
            &mut workspace.stages[start..start + workspace.dimension],
            &workspace.temporary,
            time + tableau.nodes()[stage_index] * step,
            stats,
        )?;
    }
    combine(
        candidate,
        state,
        step,
        &workspace.stages,
        workspace.dimension,
        stage_count,
        tableau.weights(),
    );
    Ok(())
}

/// Estimates the dominant endpoint eigenvalue from the final two stage
/// derivatives and their corresponding states.
///
/// This is Hairer's endpoint-stage secant estimate with the infinity norm, as
/// used by the pinned SciML automatic composites. An unchanged derivative at
/// an unchanged state contributes no information; treating that `0 / 0` pair
/// as stiff would spuriously switch equilibria and systems with conserved
/// components. Other non-finite quotients are retained so the switching policy
/// can classify them conservatively.
pub(super) fn endpoint_stage_stiffness_estimate(
    workspace: &Workspace,
    stage_count: usize,
) -> Option<f64> {
    if workspace.stiffness_reference_state.is_empty() || stage_count < 2 {
        return None;
    }

    let penultimate_start = (stage_count - 2) * workspace.dimension;
    let last_start = (stage_count - 1) * workspace.dimension;
    let penultimate_derivative =
        &workspace.stages[penultimate_start..penultimate_start + workspace.dimension];
    let last_derivative = &workspace.stages[last_start..last_start + workspace.dimension];
    let mut estimate = 0.0_f64;
    for (((last_derivative, penultimate_derivative), last_state), penultimate_state) in
        last_derivative
            .iter()
            .zip(penultimate_derivative)
            .zip(&workspace.temporary)
            .zip(&workspace.stiffness_reference_state)
    {
        let derivative_delta = last_derivative - penultimate_derivative;
        let state_delta = last_state - penultimate_state;
        if derivative_delta == 0.0 && state_delta == 0.0 {
            continue;
        }
        let quotient = (derivative_delta / state_delta).abs();
        if !quotient.is_finite() {
            return Some(quotient);
        }
        estimate = estimate.max(quotient);
    }
    Some(estimate)
}

pub(super) fn perform_lazy_dense_stages<F, P, T>(
    problem: &OdeProblem<F, P>,
    state: &[f64],
    time: f64,
    step: f64,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
    tableau: T,
) -> Result<(), SolveError>
where
    F: crate::OdeFunction<P>,
    T: TableauAccess,
{
    let base_stage_count = tableau.weights().len();
    for offset in 0..tableau.lazy_stage_count() {
        let (node, coefficients) = tableau.lazy_stage(offset);
        workspace.temporary.copy_from_slice(state);
        for &(source, coefficient) in coefficients {
            let source_start = source * workspace.dimension;
            for (value, derivative) in workspace
                .temporary
                .iter_mut()
                .zip(&workspace.stages[source_start..source_start + workspace.dimension])
            {
                *value += step * coefficient * derivative;
            }
        }
        let target = base_stage_count + offset;
        let target_start = target * workspace.dimension;
        evaluate(
            problem,
            &mut workspace.stages[target_start..target_start + workspace.dimension],
            &workspace.temporary,
            time + node * step,
            stats,
        )?;
        ensure_finite(&workspace.stages[target_start..target_start + workspace.dimension])?;
    }
    Ok(())
}

fn combine(
    output: &mut [f64],
    state: &[f64],
    step: f64,
    stages: &[f64],
    dimension: usize,
    stage_count: usize,
    weights: &[f64],
) {
    output.fill(0.0);
    for (stage_index, weight) in weights.iter().take(stage_count).enumerate() {
        let start = stage_index * dimension;
        let stage = &stages[start..start + dimension];
        for (increment, stage_value) in output.iter_mut().zip(stage) {
            *increment += weight * stage_value;
        }
    }
    for (output_value, state_value) in output.iter_mut().zip(state) {
        *output_value = state_value + step * *output_value;
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn error_norm(
    stages: &[f64],
    dimension: usize,
    states: (&[f64], &[f64]),
    step: f64,
    options: &SolveOptions,
    error_weights: &[f64],
    error_buffer: &mut [f64],
) -> f64 {
    let (state, candidate) = states;
    error_buffer.fill(0.0);
    for (stage_index, weight) in error_weights.iter().enumerate() {
        let start = stage_index * dimension;
        let stage = &stages[start..start + dimension];
        for (error, stage_value) in error_buffer.iter_mut().zip(stage) {
            *error += weight * stage_value;
        }
    }
    let mut squared_norm = 0.0;
    for ((error, state), candidate) in error_buffer.iter().zip(state).zip(candidate) {
        let error = step * error;
        let scale = options.absolute_tolerance
            + options.relative_tolerance * state.abs().max(candidate.abs());
        squared_norm += (error / scale).powi(2);
    }
    (squared_norm / state.len() as f64).sqrt()
}
