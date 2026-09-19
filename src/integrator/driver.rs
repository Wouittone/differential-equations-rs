use super::controller::ControllerState;
use super::dense::DefaultDenseState;
use super::kernel::{RejectionReason, StepKernel};
use super::schedule::{TimeStopSchedule, callback_adjusted_step};
use crate::solution::TrajectoryRecorder;
use crate::{OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

pub(crate) fn integrate<F, P, K>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
    mut kernel: K,
) -> Result<Solution, SolveError>
where
    F: crate::OdeFunction<P>,
    K: StepKernel<F, P>,
{
    crate::solver::validate_ode_problem(problem, options)?;

    if options.adaptive && !kernel.capabilities().adaptive {
        return Err(SolveError::AdaptiveStepUnsupported);
    }
    if !options.adaptive && options.initial_step.is_none() {
        return Err(SolveError::InitialStepRequired);
    }

    let dimension = problem.initial_state().len();
    let (start, end) = problem.time_span();
    let direction = (end - start).signum();
    let maximum_step = options.max_step.min((end - start).abs());
    let mut state = problem.initial_state().to_vec();
    let mut candidate = vec![0.0; dimension];
    let predictive_domain_enabled = kernel.has_predictive_domain(problem);
    let mut prediction_derivative = if predictive_domain_enabled {
        vec![0.0; dimension]
    } else {
        Vec::new()
    };
    let mut state_before_effect = if kernel.has_callbacks(problem) {
        vec![0.0; dimension]
    } else {
        Vec::new()
    };
    let mut stats = SolverStats::default();
    let custom_dense_output = kernel.has_custom_dense_output();
    let custom_callback_handling = kernel.has_custom_callback_handling();
    let default_dense_enabled = !custom_dense_output
        && (problem.has_continuous_callbacks()
            || !options.save_at.is_empty()
            || options.retain_dense_output);
    let default_callback_dense_enabled = !custom_dense_output && problem.has_continuous_callbacks();
    let mut default_dense = DefaultDenseState::new(dimension, default_dense_enabled);

    let mut recorder = TrajectoryRecorder::new(&state, start, options);
    let initial_callbacks = kernel.apply_initial_callbacks(problem, &mut state, start)?;
    stats.callback_invocations += initial_callbacks.invocations;
    stats.rhs_evaluations += initial_callbacks.rhs_evaluations;
    if initial_callbacks.state_modified {
        recorder.record_callback(
            start,
            problem.initial_state(),
            &state,
            initial_callbacks,
            true,
        );
    }
    if kernel
        .domain_rejection_factor(problem, &state, start)
        .is_some()
    {
        return Err(SolveError::InitialStateOutOfDomain);
    }
    if initial_callbacks.terminate {
        return finish_successful(&mut kernel, problem, &mut state, start, recorder, stats);
    }

    kernel.initialize(problem, &state, start, &mut stats)?;
    let proposed_initial_step = match options.initial_step {
        Some(step) => direction * step.min(maximum_step),
        None => {
            direction
                * kernel.estimate_initial_step(
                    problem,
                    &state,
                    start,
                    direction,
                    maximum_step,
                    &mut candidate,
                    options,
                    &mut stats,
                )?
        }
    };
    let mut step = kernel.modify_step(callback_adjusted_step(
        initial_callbacks,
        proposed_initial_step,
        direction,
        maximum_step,
    ));
    let mut time = start;
    let mut attempted_steps = 0;
    let mut time_stops = TimeStopSchedule::new(&options.time_stops, start, end);
    let mut previous_step_rejected = false;
    let mut controller_state = ControllerState::default();

    while direction * (end - time) > 0.0 {
        if attempted_steps == options.max_steps {
            return Err(SolveError::MaxStepsExceeded);
        }
        // A kernel may switch its numerical method after the previous
        // lifecycle hook, so capabilities belong to this attempt rather than
        // to the complete integration.
        kernel.prepare_attempt(problem, &state, time, &mut stats)?;
        let capabilities = kernel.capabilities();
        attempted_steps += 1;

        let callback_stop = kernel.next_callback_time_stop(problem, time, direction);
        let mut attempted_step = time_stops.clip_step_with(time, step, callback_stop);
        if predictive_domain_enabled {
            kernel.evaluate_dense_derivative(
                problem,
                &mut prediction_derivative,
                &state,
                time,
                &mut stats,
            )?;
            attempted_step = kernel.predictive_domain_adjusted_step(
                problem,
                &state,
                &prediction_derivative,
                time,
                attempted_step,
                options.absolute_tolerance,
                &mut candidate,
            )?;
        }
        if time + attempted_step == time {
            return Err(SolveError::StepSizeUnderflow);
        }

        let attempt = kernel.attempt_step(
            problem,
            &state,
            time,
            attempted_step,
            &mut candidate,
            options,
            &mut stats,
        );
        let attempt = attempt.and_then(|estimate| {
            candidate
                .iter()
                .all(|value| value.is_finite())
                .then_some(estimate)
                .ok_or(SolveError::NonFiniteDerivative)
        });
        let estimate = match attempt {
            Ok(estimate) => estimate,
            Err(error)
                if options.adaptive
                    && capabilities.attempt_failure_policy.is_recoverable(&error) =>
            {
                stats.rejected_steps += 1;
                kernel.reject_step_with_reason(RejectionReason::AttemptFailure(error));
                controller_state.rejected(1.0, capabilities.controller);
                let transition = kernel.take_transition();
                if transition.reset_controller {
                    controller_state.reset();
                }
                let proposed = attempted_step
                    * capabilities.controller.failed_attempt_factor
                    * transition.step_multiplier;
                step = kernel.modify_step(direction * proposed.abs().min(maximum_step));
                previous_step_rejected = true;
                continue;
            }
            Err(error) => return Err(error),
        };

        if controller_state.accepts(estimate.error_norm, capabilities.controller) {
            let previous_time = time;
            let mut next_time = time + attempted_step;
            if direction * (end - next_time) <= 0.0 {
                next_time = end;
            }
            if let Some(reduction_factor) =
                kernel.domain_rejection_factor(problem, &candidate, next_time)
            {
                stats.rejected_steps += 1;
                kernel.reject_step_with_reason(RejectionReason::Domain);
                controller_state.reset();
                let transition = kernel.take_transition();
                if transition.reset_controller {
                    controller_state.reset();
                }
                let proposed = attempted_step * reduction_factor * transition.step_multiplier;
                step = kernel.modify_step(direction * proposed.abs().min(maximum_step));
                previous_step_rejected = true;
                continue;
            }
            let attempted_time = next_time;
            if custom_callback_handling && default_dense_enabled {
                // Typed adapters dispatch callbacks against a problem other
                // than the placeholder passed to this driver. Preserve the
                // full attempted endpoint before an event truncates `candidate`
                // so retained and save-at dense output remains the accepted
                // step's left-limit interpolant.
                default_dense.prepare(
                    &mut kernel,
                    problem,
                    &state,
                    previous_time,
                    &candidate,
                    attempted_time,
                    &mut stats,
                )?;
            }
            let callbacks = if custom_dense_output || custom_callback_handling {
                kernel.apply_step_callbacks(
                    problem,
                    &state,
                    previous_time,
                    &mut candidate,
                    &mut next_time,
                    &mut state_before_effect,
                    options.event_tolerance,
                    &mut stats,
                )?
            } else {
                default_dense.apply_callbacks(
                    &mut kernel,
                    problem,
                    &state,
                    previous_time,
                    &mut candidate,
                    &mut next_time,
                    &mut state_before_effect,
                    options.event_tolerance,
                    default_callback_dense_enabled,
                    &mut stats,
                )?
            };
            stats.callback_invocations += callbacks.invocations;
            stats.rhs_evaluations += callbacks.rhs_evaluations;
            stats.accepted_steps += 1;
            kernel.note_accepted_step(&mut stats);

            let dense_recorded = if !options.save_at.is_empty() || options.retain_dense_output {
                let dense_state = if callbacks.invocations == 0 {
                    &candidate
                } else {
                    &state_before_effect
                };
                if custom_dense_output {
                    kernel.record_dense_step(
                        problem,
                        &state,
                        dense_state,
                        previous_time,
                        attempted_time,
                        next_time,
                        next_time == end,
                        &mut recorder,
                        &mut stats,
                    )?
                } else {
                    if !default_dense.prepared {
                        default_dense.prepare(
                            &mut kernel,
                            problem,
                            &state,
                            previous_time,
                            dense_state,
                            attempted_time,
                            &mut stats,
                        )?;
                    }
                    default_dense.record(
                        &state,
                        previous_time,
                        attempted_time,
                        dense_state,
                        next_time,
                        next_time == end,
                        &mut recorder,
                    )?;
                    true
                }
            } else {
                false
            };
            if !dense_recorded {
                recorder.record_step(
                    &state,
                    previous_time,
                    if callbacks.invocations == 0 {
                        &candidate
                    } else {
                        &state_before_effect
                    },
                    next_time,
                    next_time == end,
                );
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
                return finish_successful(
                    &mut kernel,
                    problem,
                    &mut candidate,
                    next_time,
                    recorder,
                    stats,
                );
            }

            time = next_time;
            time_stops.accepted(time);
            std::mem::swap(&mut state, &mut candidate);
            kernel.accept_step(
                problem,
                &candidate,
                &state,
                time,
                time - previous_time,
                callbacks.state_modified,
                &mut stats,
            )?;
            let transition = kernel.take_transition();
            if !custom_dense_output {
                default_dense.accepted(callbacks.state_modified);
            }

            if callbacks.state_modified || transition.reset_controller {
                // A callback may change the state or parameters discontinuously.
                // A kernel transition likewise changes the numerical method.
                // Do not let history from the old state or method bias the
                // next controller proposal.
                controller_state.reset();
            }
            let retain_error_history = !callbacks.state_modified && !transition.reset_controller;
            if callbacks.requested_step.is_some() {
                if options.adaptive && retain_error_history {
                    controller_state.accepted(estimate.error_norm, capabilities.controller);
                }
                step = kernel.modify_step(callback_adjusted_step(
                    callbacks,
                    step * transition.step_multiplier,
                    direction,
                    maximum_step,
                ));
            } else if options.adaptive {
                let mut factor = estimate.proposed_factor.unwrap_or_else(|| {
                    controller_state.factor(estimate.error_norm, capabilities.controller)
                });
                if retain_error_history {
                    controller_state.accepted(estimate.error_norm, capabilities.controller);
                }
                if previous_step_rejected {
                    factor = factor.min(capabilities.controller.rejected_acceptance_maximum);
                }
                let proposed =
                    direction * attempted_step.abs() * factor * transition.step_multiplier;
                step = kernel.modify_step(callback_adjusted_step(
                    callbacks,
                    proposed,
                    direction,
                    maximum_step,
                ));
            } else {
                // Fixed-step overrides and transition multipliers apply to the
                // next attempt only. After that attempt is accepted, start
                // again from the configured fixed step before applying any
                // newly requested override.
                step = kernel.modify_step(callback_adjusted_step(
                    callbacks,
                    proposed_initial_step * transition.step_multiplier,
                    direction,
                    maximum_step,
                ));
            }
            previous_step_rejected = false;
        } else {
            stats.rejected_steps += 1;
            kernel.reject_step_with_reason(RejectionReason::ErrorEstimate(estimate.error_norm));
            let factor = estimate
                .proposed_factor
                .unwrap_or_else(|| {
                    controller_state.rejection_factor(estimate.error_norm, capabilities.controller)
                })
                .min(capabilities.controller.rejection_maximum);
            controller_state.rejected(estimate.error_norm, capabilities.controller);
            let transition = kernel.take_transition();
            if transition.reset_controller {
                controller_state.reset();
            }
            let proposed = attempted_step * factor * transition.step_multiplier;
            step = kernel.modify_step(direction * proposed.abs().min(maximum_step));
            previous_step_rejected = true;
        }
    }

    finish_successful(&mut kernel, problem, &mut state, time, recorder, stats)
}

fn finish_successful<F, P, K>(
    kernel: &mut K,
    problem: &OdeProblem<F, P>,
    state: &mut [f64],
    time: f64,
    mut recorder: TrajectoryRecorder<'_>,
    mut stats: SolverStats,
) -> Result<Solution, SolveError>
where
    F: crate::OdeFunction<P>,
    K: StepKernel<F, P>,
{
    if kernel.apply_finalize_callbacks(problem, state, time)? {
        recorder.synchronize_endpoint(time, state);
    }
    kernel.finalize_stats(&mut stats);
    let mut solution = recorder.finish(stats);
    solution.set_state_shape(problem.state_shape());
    Ok(solution)
}
