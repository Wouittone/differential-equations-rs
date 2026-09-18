use super::*;

pub(crate) fn solve_newmark<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    options: &SolveOptions,
    method: StructuralParameters,
) -> Result<SecondOrderSolution, SecondOrderSolveError>
where
    F: SecondOrderFunction<P>,
{
    if !options.adaptive && options.initial_step.is_none() {
        return Err(SolveError::InitialStepRequired.into());
    }
    let dimension = problem.initial_position.len();
    let (start, end) = problem.time_span;
    let direction = (end - start).signum();
    let maximum_step = options.max_step.min((end - start).abs());
    let mut step_magnitude = options
        .initial_step
        .unwrap_or_else(|| ((end - start).abs() / 100.0).min(maximum_step))
        .min(maximum_step);
    if !step_magnitude.is_finite() || step_magnitude <= 0.0 {
        return Err(SolveError::StepSizeUnderflow.into());
    }

    let mut velocity = problem.initial_velocity.clone();
    let mut position = problem.initial_position.clone();
    let mut acceleration = vec![0.0; dimension];
    let mut workspace = StructuralWorkspace::new(dimension, !problem.callbacks.is_empty());
    let mut stats = SolverStats::default();

    let mut recorder = PartitionedRecorder::new(&velocity, &position, start, options);
    let initial = apply_initial_callbacks(problem, &mut velocity, &mut position, start)?;
    stats.callback_invocations += initial.invocations;
    stats.rhs_evaluations += initial.rhs_evaluations;
    if initial.state_modified {
        recorder.record_callback(
            start,
            &problem.initial_velocity,
            &problem.initial_position,
            &velocity,
            &position,
            initial,
            true,
        );
    }
    if problem
        .domain_rejection_factor(&velocity, &position, start)
        .is_some()
    {
        return Err(SolveError::InitialStateOutOfDomain.into());
    }
    if initial.terminate {
        return finish_successful(
            problem,
            &mut velocity,
            &mut position,
            start,
            recorder,
            stats,
        );
    }
    step_magnitude =
        callback_adjusted_step(initial, direction * step_magnitude, direction, maximum_step).abs();
    evaluate_acceleration(
        problem,
        &mut acceleration,
        &velocity,
        &position,
        start,
        &mut stats,
    )?;

    let controller = ControllerConfig::proportional(2, 0.9, 0.2, 5.0, 0.2);
    let mut controller_state = ControllerState::default();
    let mut previous_attempt_rejected = false;
    let mut time = start;
    let mut attempts = 0;
    let mut time_stops = TimeStopSchedule::new(&options.time_stops, start, end);
    while direction * (end - time) > 0.0 {
        if attempts == options.max_steps {
            return Err(SolveError::MaxStepsExceeded.into());
        }
        attempts += 1;
        let step = time_stops.clip_step_with(
            time,
            direction * step_magnitude,
            problem.next_preset_time(time, direction),
        );
        if time + step == time {
            return Err(SolveError::StepSizeUnderflow.into());
        }

        structural_substep(
            problem,
            method,
            &velocity,
            &position,
            &acceleration,
            time,
            step,
            &mut workspace.full_velocity,
            &mut workspace.full_position,
            &mut workspace.full_acceleration,
            &mut workspace.trial_acceleration,
            &mut workspace.trial_velocity,
            &mut workspace.trial_position,
            &mut workspace.evaluated_acceleration,
            &mut workspace.perturbed_acceleration,
            &mut workspace.residual,
            &mut workspace.perturbed_residual,
            &mut workspace.correction,
            &mut workspace.matrix,
            &mut workspace.pivots,
            &mut stats,
        )?;

        let error = if options.adaptive {
            structural_substep(
                problem,
                method,
                &velocity,
                &position,
                &acceleration,
                time,
                0.5 * step,
                &mut workspace.half_velocity,
                &mut workspace.half_position,
                &mut workspace.half_acceleration,
                &mut workspace.trial_acceleration,
                &mut workspace.trial_velocity,
                &mut workspace.trial_position,
                &mut workspace.evaluated_acceleration,
                &mut workspace.perturbed_acceleration,
                &mut workspace.residual,
                &mut workspace.perturbed_residual,
                &mut workspace.correction,
                &mut workspace.matrix,
                &mut workspace.pivots,
                &mut stats,
            )?;
            structural_substep(
                problem,
                method,
                &workspace.half_velocity,
                &workspace.half_position,
                &workspace.half_acceleration,
                time + 0.5 * step,
                0.5 * step,
                &mut workspace.candidate_velocity,
                &mut workspace.candidate_position,
                &mut workspace.candidate_acceleration,
                &mut workspace.trial_acceleration,
                &mut workspace.trial_velocity,
                &mut workspace.trial_position,
                &mut workspace.evaluated_acceleration,
                &mut workspace.perturbed_acceleration,
                &mut workspace.residual,
                &mut workspace.perturbed_residual,
                &mut workspace.correction,
                &mut workspace.matrix,
                &mut workspace.pivots,
                &mut stats,
            )?;
            structural_error_norm(
                &workspace.candidate_velocity,
                &workspace.candidate_position,
                &workspace.full_velocity,
                &workspace.full_position,
                &velocity,
                &position,
                options,
            )
        } else {
            workspace
                .candidate_velocity
                .copy_from_slice(&workspace.full_velocity);
            workspace
                .candidate_position
                .copy_from_slice(&workspace.full_position);
            workspace
                .candidate_acceleration
                .copy_from_slice(&workspace.full_acceleration);
            0.0
        };

        if error > 1.0 {
            stats.rejected_steps += 1;
            controller_state.rejected(error, controller);
            step_magnitude = step.abs() * controller_state.factor(error, controller).min(1.0);
            previous_attempt_rejected = true;
            continue;
        }

        let previous_time = time;
        let attempted_time = time + step;
        let mut next_time = if direction * (end - attempted_time) <= 0.0 {
            end
        } else {
            attempted_time
        };
        if let Some(reduction_factor) = problem.domain_rejection_factor(
            &workspace.candidate_velocity,
            &workspace.candidate_position,
            next_time,
        ) {
            stats.rejected_steps += 1;
            controller_state.reset();
            step_magnitude = step.abs() * reduction_factor;
            previous_attempt_rejected = true;
            continue;
        }
        let callback = apply_step_callbacks(
            problem,
            &velocity,
            &position,
            previous_time,
            &mut workspace.candidate_velocity,
            &mut workspace.candidate_position,
            &mut next_time,
            &mut workspace.previous_effect_velocity,
            &mut workspace.previous_effect_position,
            options.event_tolerance,
            None,
        )?;
        stats.callback_invocations += callback.invocations;
        stats.rhs_evaluations += callback.rhs_evaluations;
        stats.accepted_steps += 1;
        recorder.record_step(
            &velocity,
            &position,
            previous_time,
            if callback.invocations == 0 {
                &workspace.candidate_velocity
            } else {
                &workspace.previous_effect_velocity
            },
            if callback.invocations == 0 {
                &workspace.candidate_position
            } else {
                &workspace.previous_effect_position
            },
            next_time,
            next_time == end,
            None,
        )?;
        if callback.invocations > 0 {
            recorder.record_callback(
                next_time,
                &workspace.previous_effect_velocity,
                &workspace.previous_effect_position,
                &workspace.candidate_velocity,
                &workspace.candidate_position,
                callback,
                next_time == end,
            );
        }
        if callback.terminate {
            return finish_successful(
                problem,
                &mut workspace.candidate_velocity,
                &mut workspace.candidate_position,
                next_time,
                recorder,
                stats,
            );
        }

        time = next_time;
        time_stops.accepted(time);
        std::mem::swap(&mut velocity, &mut workspace.candidate_velocity);
        std::mem::swap(&mut position, &mut workspace.candidate_position);
        if callback.state_modified || next_time != attempted_time {
            evaluate_acceleration(
                problem,
                &mut acceleration,
                &velocity,
                &position,
                time,
                &mut stats,
            )?;
        } else {
            std::mem::swap(&mut acceleration, &mut workspace.candidate_acceleration);
        }

        if callback.state_modified {
            controller_state.reset();
        }
        if callback.requested_step.is_some() {
            step_magnitude = callback_adjusted_step(
                callback,
                direction * step_magnitude,
                direction,
                maximum_step,
            )
            .abs();
        } else if options.adaptive {
            controller_state.accepted(error, controller);
            let mut factor = controller_state.factor(error, controller);
            if previous_attempt_rejected {
                factor = factor.min(1.0);
            }
            step_magnitude = callback_adjusted_step(
                callback,
                direction * step.abs() * factor,
                direction,
                maximum_step,
            )
            .abs();
        } else if callback.step_limit.is_some() {
            step_magnitude = callback_adjusted_step(
                callback,
                direction * step_magnitude,
                direction,
                maximum_step,
            )
            .abs();
        }
        previous_attempt_rejected = false;
    }
    finish_successful(problem, &mut velocity, &mut position, time, recorder, stats)
}

#[allow(clippy::too_many_arguments)]
fn structural_substep<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    method: StructuralParameters,
    velocity: &[f64],
    position: &[f64],
    acceleration: &[f64],
    time: f64,
    step: f64,
    output_velocity: &mut [f64],
    output_position: &mut [f64],
    output_acceleration: &mut [f64],
    trial_acceleration: &mut [f64],
    trial_velocity: &mut [f64],
    trial_position: &mut [f64],
    evaluated_acceleration: &mut [f64],
    perturbed_acceleration: &mut [f64],
    residual: &mut [f64],
    perturbed_residual: &mut [f64],
    correction: &mut [f64],
    matrix: &mut [f64],
    pivots: &mut [usize],
    stats: &mut SolverStats,
) -> Result<(), SecondOrderSolveError>
where
    F: SecondOrderFunction<P>,
{
    const MAX_ITERATIONS: usize = 12;
    const TOLERANCE: f64 = 1.0e-12;
    trial_acceleration.copy_from_slice(acceleration);
    let dimension = acceleration.len();
    for _ in 0..MAX_ITERATIONS {
        structural_residual(
            problem,
            method,
            velocity,
            position,
            acceleration,
            time,
            step,
            trial_acceleration,
            trial_velocity,
            trial_position,
            evaluated_acceleration,
            residual,
            stats,
        )?;
        stats.nonlinear_iterations += 1;
        let residual_norm = residual
            .iter()
            .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
        let scale = 1.0
            + trial_acceleration
                .iter()
                .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
        if residual_norm <= TOLERANCE * scale {
            update_structural_state(
                method,
                velocity,
                position,
                acceleration,
                trial_acceleration,
                step,
                output_velocity,
                output_position,
            );
            evaluate_acceleration(
                problem,
                output_acceleration,
                output_velocity,
                output_position,
                time + step,
                stats,
            )?;
            return Ok(());
        }

        stats.jacobian_evaluations += 1;
        for column in 0..dimension {
            let original = trial_acceleration[column];
            let delta = f64::EPSILON.sqrt() * original.abs().max(1.0);
            trial_acceleration[column] = original + delta;
            structural_residual(
                problem,
                method,
                velocity,
                position,
                acceleration,
                time,
                step,
                trial_acceleration,
                trial_velocity,
                trial_position,
                perturbed_acceleration,
                perturbed_residual,
                stats,
            )?;
            for row in 0..dimension {
                matrix[row * dimension + column] =
                    (perturbed_residual[row] - residual[row]) / delta;
            }
            trial_acceleration[column] = original;
        }
        for (correction, residual) in correction.iter_mut().zip(residual.iter()) {
            *correction = -*residual;
        }
        factorize(matrix, pivots, dimension)?;
        stats.linear_factorizations += 1;
        solve_factorized(matrix, pivots, correction, dimension);
        stats.linear_solves += 1;
        for (value, correction) in trial_acceleration.iter_mut().zip(correction.iter()) {
            *value += correction;
        }
    }
    Err(SolveError::NonlinearSolveFailed.into())
}

#[allow(clippy::too_many_arguments)]
fn structural_residual<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    method: StructuralParameters,
    velocity: &[f64],
    position: &[f64],
    acceleration: &[f64],
    time: f64,
    step: f64,
    trial_acceleration: &[f64],
    trial_velocity: &mut [f64],
    trial_position: &mut [f64],
    evaluated_acceleration: &mut [f64],
    residual: &mut [f64],
    stats: &mut SolverStats,
) -> Result<(), SecondOrderSolveError>
where
    F: SecondOrderFunction<P>,
{
    update_structural_state(
        method,
        velocity,
        position,
        acceleration,
        trial_acceleration,
        step,
        trial_velocity,
        trial_position,
    );
    for index in 0..trial_acceleration.len() {
        trial_velocity[index] =
            (1.0 - method.alpha_f) * trial_velocity[index] + method.alpha_f * velocity[index];
        trial_position[index] =
            (1.0 - method.alpha_f) * trial_position[index] + method.alpha_f * position[index];
    }
    evaluate_acceleration(
        problem,
        evaluated_acceleration,
        trial_velocity,
        trial_position,
        time + (1.0 - method.alpha_f) * step,
        stats,
    )?;
    for index in 0..residual.len() {
        residual[index] = (1.0 - method.alpha_m) * trial_acceleration[index]
            + method.alpha_m * acceleration[index]
            - evaluated_acceleration[index];
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn update_structural_state(
    method: StructuralParameters,
    velocity: &[f64],
    position: &[f64],
    acceleration: &[f64],
    next_acceleration: &[f64],
    step: f64,
    output_velocity: &mut [f64],
    output_position: &mut [f64],
) {
    for index in 0..velocity.len() {
        output_velocity[index] = velocity[index]
            + step
                * ((1.0 - method.gamma) * acceleration[index]
                    + method.gamma * next_acceleration[index]);
        output_position[index] = position[index]
            + step * velocity[index]
            + 0.5
                * step
                * step
                * ((1.0 - 2.0 * method.beta) * acceleration[index]
                    + 2.0 * method.beta * next_acceleration[index]);
    }
}

#[allow(clippy::too_many_arguments)]
fn structural_error_norm(
    velocity: &[f64],
    position: &[f64],
    coarse_velocity: &[f64],
    coarse_position: &[f64],
    previous_velocity: &[f64],
    previous_position: &[f64],
    options: &SolveOptions,
) -> f64 {
    velocity
        .iter()
        .zip(position)
        .zip(coarse_velocity.iter().zip(coarse_position))
        .zip(previous_velocity.iter().zip(previous_position))
        .fold(
            0.0_f64,
            |maximum,
             (
                ((velocity, position), (coarse_velocity, coarse_position)),
                (previous_velocity, previous_position),
            )| {
                let velocity_scale = options.absolute_tolerance
                    + options.relative_tolerance * velocity.abs().max(previous_velocity.abs());
                let position_scale = options.absolute_tolerance
                    + options.relative_tolerance * position.abs().max(previous_position.abs());
                maximum
                    .max(((velocity - coarse_velocity) / (3.0 * velocity_scale)).abs())
                    .max(((position - coarse_position) / (3.0 * position_scale)).abs())
            },
        )
}
