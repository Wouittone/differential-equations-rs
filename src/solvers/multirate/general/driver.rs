//! Shared adaptive integration and callback lifecycle for multirate kernels.

use super::evaluation::evaluate_total;
use super::kernels::{mis_step, mrab_step, mreef_step, mri_step};
use super::method::Method;
use crate::integrator::{TimeStopSchedule, callback_adjusted_step};
use crate::solution::{BorrowedHermiteSegment, DenseSegment, HermiteSegment, TrajectoryRecorder};
use crate::solver::{
    validate_preset_time_sequences, validate_state_time_options, validate_vector_callback_lengths,
};
use crate::{Solution, SolveError, SolveOptions, SolverStats, SplitOdeProblem};

pub(super) fn integrate_multirate<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    options: &SolveOptions,
    method: Method,
) -> Result<Solution, SolveError>
where
    FE: crate::OdeFunction<P>,
    FI: crate::OdeFunction<P>,
{
    validate(problem, options)?;
    if !options.adaptive && options.initial_step.is_none() {
        return Err(SolveError::InitialStepRequired);
    }
    let (start, end) = problem.time_span();
    let direction = (end - start).signum();
    let maximum_step = options.max_step.min((end - start).abs());
    let dimension = problem.dimension();
    let mut state = problem.initial_state().to_vec();
    let mut candidate = vec![0.0; dimension];
    let mut state_before_effect = vec![0.0; dimension];
    let mut start_derivative = vec![0.0; dimension];
    let mut end_derivative = vec![0.0; dimension];
    let mut dense_endpoint = vec![0.0; dimension];
    let mut stats = SolverStats::default();
    let mut recorder = TrajectoryRecorder::new(&state, start, options);
    let initial = problem.apply_initial_callbacks(&mut state, start)?;
    stats.callback_invocations += initial.invocations;
    stats.rhs_evaluations += initial.rhs_evaluations;
    if initial.state_modified {
        recorder.record_callback(start, problem.initial_state(), &state, initial, true);
    }
    if problem.domain_rejection_factor(&state, start).is_some() {
        return Err(SolveError::InitialStateOutOfDomain);
    }
    if initial.terminate {
        return finish_successful(problem, &mut state, start, recorder, stats);
    }
    evaluate_total(problem, &state, start, &mut start_derivative, &mut stats)?;
    let proposed_step = direction
        * match options.initial_step {
            Some(value) => value.min(maximum_step),
            None => estimate_initial_step(&state, &start_derivative, maximum_step),
        };
    let mut step = callback_adjusted_step(initial, proposed_step, direction, maximum_step);
    let mut time = start;
    let mut attempted = 0usize;
    let mut previous_rejected = false;
    let order = method.controller_order() as f64;
    let mut time_stops = TimeStopSchedule::new(&options.time_stops, start, end);
    while direction * (end - time) > 0.0 {
        if attempted == options.max_steps {
            return Err(SolveError::MaxStepsExceeded);
        }
        attempted += 1;
        let proposed_step = step;
        step = time_stops.clip_step_with(time, step, problem.next_preset_time(time, direction));
        if problem.has_predictive_domain() {
            step = problem.predictive_domain_adjusted_step(
                &state,
                &start_derivative,
                time,
                step,
                options.absolute_tolerance,
                &mut candidate,
            )?;
        }
        if time + step == time {
            return Err(SolveError::StepSizeUnderflow);
        }
        let error_norm = match attempt(
            method,
            problem,
            &state,
            time,
            step,
            &mut candidate,
            options,
            &mut stats,
        ) {
            Ok(error) => error,
            Err(SolveError::NonlinearSolveFailed | SolveError::SingularLinearSystem)
                if options.adaptive =>
            {
                stats.rejected_steps += 1;
                step *= 0.2;
                previous_rejected = true;
                continue;
            }
            Err(error) => return Err(error),
        };
        if !candidate.iter().all(|value| value.is_finite()) {
            return Err(SolveError::NonFiniteDerivative);
        }
        if !options.adaptive || error_norm <= 1.0 {
            let previous_time = time;
            let attempted_time = time + step;
            if let Some(reduction_factor) =
                problem.domain_rejection_factor(&candidate, attempted_time)
            {
                stats.rejected_steps += 1;
                step *= reduction_factor;
                previous_rejected = true;
                continue;
            }
            dense_endpoint.copy_from_slice(&candidate);
            evaluate_total(
                problem,
                &dense_endpoint,
                attempted_time,
                &mut end_derivative,
                &mut stats,
            )?;
            let borrowed = BorrowedHermiteSegment::new(
                previous_time,
                attempted_time,
                &state,
                &dense_endpoint,
                &start_derivative,
                &end_derivative,
            )
            .map_err(|_| SolveError::NonFiniteDerivative)?;
            let mut next_time = attempted_time;
            let mut interpolate = |target: f64, output: &mut [f64]| {
                borrowed
                    .interpolate(target, output)
                    .map_err(|_| SolveError::NonFiniteDerivative)
            };
            let callbacks = problem.apply_step_callbacks(
                &state,
                previous_time,
                &mut candidate,
                &mut next_time,
                &mut state_before_effect,
                options.event_tolerance,
                Some(&mut interpolate),
            )?;
            stats.callback_invocations += callbacks.invocations;
            stats.rhs_evaluations += callbacks.rhs_evaluations;
            stats.accepted_steps += 1;
            let dense_state = if callbacks.invocations == 0 {
                &candidate
            } else {
                &state_before_effect
            };
            recorder
                .record_step_dense(
                    &state,
                    previous_time,
                    dense_state,
                    next_time,
                    next_time == end,
                    &borrowed,
                )
                .map_err(|_| SolveError::NonFiniteDerivative)?;
            if recorder.retains_dense_output() {
                let owned = HermiteSegment::new_bounded(
                    previous_time,
                    attempted_time,
                    next_time,
                    state.clone(),
                    dense_endpoint.clone(),
                    start_derivative.clone(),
                    end_derivative.clone(),
                )
                .map_err(|_| SolveError::NonFiniteDerivative)?;
                recorder.retain_hermite_segment(owned);
            }
            if callbacks.invocations > 0 {
                recorder.record_callback(
                    next_time,
                    &state_before_effect,
                    &candidate,
                    callbacks,
                    next_time == end,
                );
            }
            if callbacks.terminate {
                return finish_successful(problem, &mut candidate, next_time, recorder, stats);
            }
            time = next_time;
            time_stops.accepted(time);
            std::mem::swap(&mut state, &mut candidate);
            if !callbacks.state_modified {
                start_derivative.copy_from_slice(&end_derivative);
            } else {
                evaluate_total(problem, &state, time, &mut start_derivative, &mut stats)?;
            }
            if callbacks.requested_step.is_some() {
                step = callback_adjusted_step(callbacks, step, direction, maximum_step);
            } else if options.adaptive {
                let factor = if error_norm == 0.0 {
                    5.0
                } else if error_norm.is_finite() {
                    (0.9 * error_norm.powf(-1.0 / (order + 1.0))).clamp(0.2, 5.0)
                } else {
                    0.2
                };
                let factor = if previous_rejected {
                    factor.min(1.0)
                } else {
                    factor
                };
                step = callback_adjusted_step(
                    callbacks,
                    direction * step.abs() * factor,
                    direction,
                    maximum_step,
                );
            } else {
                step = callback_adjusted_step(callbacks, proposed_step, direction, maximum_step);
            }
            previous_rejected = false;
        } else {
            stats.rejected_steps += 1;
            let factor = if error_norm.is_finite() {
                (0.9 * error_norm.powf(-1.0 / (order + 1.0))).clamp(0.2, 1.0)
            } else {
                0.2
            };
            step *= factor;
            previous_rejected = true;
        }
    }
    finish_successful(problem, &mut state, time, recorder, stats)
}

fn finish_successful<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    state: &mut [f64],
    time: f64,
    mut recorder: TrajectoryRecorder<'_>,
    stats: SolverStats,
) -> Result<Solution, SolveError>
where
    FE: crate::OdeFunction<P>,
    FI: crate::OdeFunction<P>,
{
    if problem.apply_finalize_callbacks(state, time)? {
        recorder.synchronize_endpoint(time, state);
    }
    let mut solution = recorder.finish(stats);
    solution.set_state_shape(problem.state_shape());
    Ok(solution)
}

fn validate<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    options: &SolveOptions,
) -> Result<(), SolveError> {
    validate_state_time_options(problem.initial_state(), problem.time_span(), options)?;
    validate_preset_time_sequences(problem.preset_time_sequences(), problem.time_span())?;
    validate_vector_callback_lengths(problem.vector_callback_lengths())
}

fn estimate_initial_step(state: &[f64], derivative: &[f64], maximum: f64) -> f64 {
    let state_scale = state
        .iter()
        .fold(1.0_f64, |acc, value| acc.max(value.abs()));
    let derivative_scale = derivative
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    if derivative_scale == 0.0 {
        (0.01 * maximum).max(f64::MIN_POSITIVE).min(maximum)
    } else {
        (0.01 * state_scale / derivative_scale)
            .max(f64::MIN_POSITIVE)
            .min(maximum)
    }
}

#[allow(clippy::too_many_arguments)]
fn attempt<FE, FI, P>(
    method: Method,
    problem: &SplitOdeProblem<FE, FI, P>,
    state: &[f64],
    time: f64,
    step: f64,
    candidate: &mut [f64],
    options: &SolveOptions,
    stats: &mut SolverStats,
) -> Result<f64, SolveError>
where
    FE: crate::OdeFunction<P>,
    FI: crate::OdeFunction<P>,
{
    let error = match method {
        Method::Mreef { m, order, sequence } => mreef_step(
            problem, state, time, step, candidate, m, order, sequence, stats,
        )?,
        Method::Mrab { m, order } => {
            mrab_step(problem, state, time, step, candidate, m, order, stats)?
        }
        Method::Mis { m, tableau } => {
            mis_step(problem, state, time, step, candidate, m, tableau, stats)?
        }
        Method::Mri { m, tableau } => {
            mri_step(problem, state, time, step, candidate, m, tableau, stats)?
        }
    };
    Ok(scaled_error(
        &error,
        state,
        candidate,
        options.absolute_tolerance,
        options.relative_tolerance,
    ))
}

fn scaled_error(error: &[f64], old: &[f64], new: &[f64], atol: f64, rtol: f64) -> f64 {
    error
        .iter()
        .zip(old)
        .zip(new)
        .fold(0.0_f64, |norm, ((error, old), new)| {
            norm.max(error.abs() / (atol + rtol * old.abs().max(new.abs())))
        })
}
