use super::*;

pub(crate) fn solve_fixed<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    options: &SolveOptions,
    method: Method,
) -> Result<SecondOrderSolution, SecondOrderSolveError>
where
    F: SecondOrderFunction<P>,
{
    if options.adaptive {
        return Err(SolveError::AdaptiveStepUnsupported.into());
    }
    let fixed_step = options
        .initial_step
        .ok_or(SolveError::InitialStepRequired)?;
    let (start, end) = problem.time_span;
    let direction = (end - start).signum();
    let maximum_step = options.max_step.min((end - start).abs());
    let mut step_magnitude = fixed_step.min(maximum_step);
    let dimension = problem.initial_position.len();
    let mut velocity = problem.initial_velocity.clone();
    let mut position = problem.initial_position.clone();
    let mut workspace = Workspace::new(dimension, !problem.callbacks.is_empty());
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

    let caches_acceleration = matches!(method, Method::VelocityVerlet | Method::VerletLeapfrog);
    if caches_acceleration {
        evaluate_acceleration(
            problem,
            &mut workspace.acceleration,
            &velocity,
            &position,
            start,
            &mut stats,
        )?;
    }

    let mut time = start;
    let mut steps = 0;
    let mut time_stops = TimeStopSchedule::new(&options.time_stops, start, end);
    while direction * (end - time) > 0.0 {
        if steps == options.max_steps {
            return Err(SolveError::MaxStepsExceeded.into());
        }
        steps += 1;
        let step = time_stops.clip_step_with(
            time,
            direction * step_magnitude,
            problem.next_preset_time(time, direction),
        );
        if time + step == time {
            return Err(SolveError::StepSizeUnderflow.into());
        }
        perform_step(
            problem,
            method,
            &velocity,
            &position,
            time,
            step,
            &mut workspace,
            &mut stats,
        )?;

        let previous_time = time;
        let mut next_time = time + step;
        if direction * (end - next_time) <= 0.0 {
            next_time = end;
        }
        if let Some(reduction_factor) = problem.domain_rejection_factor(
            &workspace.candidate_velocity,
            &workspace.candidate_position,
            next_time,
        ) {
            stats.rejected_steps += 1;
            step_magnitude = step.abs() * reduction_factor;
            if caches_acceleration {
                evaluate_acceleration(
                    problem,
                    &mut workspace.acceleration,
                    &velocity,
                    &position,
                    time,
                    &mut stats,
                )?;
            }
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
        time = next_time;
        time_stops.accepted(time);
        std::mem::swap(&mut velocity, &mut workspace.candidate_velocity);
        std::mem::swap(&mut position, &mut workspace.candidate_position);
        stats.accepted_steps += 1;

        recorder.record_step(
            &workspace.candidate_velocity,
            &workspace.candidate_position,
            previous_time,
            if callback.invocations == 0 {
                &velocity
            } else {
                &workspace.previous_effect_velocity
            },
            if callback.invocations == 0 {
                &position
            } else {
                &workspace.previous_effect_position
            },
            time,
            time == end,
            None,
        )?;
        if callback.invocations > 0 {
            recorder.record_callback(
                time,
                &workspace.previous_effect_velocity,
                &workspace.previous_effect_position,
                &velocity,
                &position,
                callback,
                time == end,
            );
        }
        if callback.terminate {
            return finish_successful(problem, &mut velocity, &mut position, time, recorder, stats);
        }
        step_magnitude = callback_adjusted_step(
            callback,
            direction * step_magnitude,
            direction,
            maximum_step,
        )
        .abs();
        if callback.state_modified && caches_acceleration {
            evaluate_acceleration(
                problem,
                &mut workspace.acceleration,
                &velocity,
                &position,
                time,
                &mut stats,
            )?;
        }
    }
    finish_successful(problem, &mut velocity, &mut position, time, recorder, stats)
}

#[allow(clippy::too_many_arguments)]
fn perform_step<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    method: Method,
    velocity: &[f64],
    position: &[f64],
    time: f64,
    step: f64,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: SecondOrderFunction<P>,
{
    match method {
        Method::SymplecticEuler => {
            for ((next_position, position), velocity) in workspace
                .candidate_position
                .iter_mut()
                .zip(position)
                .zip(velocity)
            {
                *next_position = position + step * velocity;
            }
            evaluate_acceleration(
                problem,
                &mut workspace.acceleration,
                velocity,
                &workspace.candidate_position,
                time,
                stats,
            )?;
            for ((next_velocity, velocity), acceleration) in workspace
                .candidate_velocity
                .iter_mut()
                .zip(velocity)
                .zip(&workspace.acceleration)
            {
                *next_velocity = velocity + step * acceleration;
            }
        }
        Method::VelocityVerlet => {
            for (((next_position, position), velocity), acceleration) in workspace
                .candidate_position
                .iter_mut()
                .zip(position)
                .zip(velocity)
                .zip(&workspace.acceleration)
            {
                *next_position = position + step * velocity + 0.5 * step * step * acceleration;
            }
            evaluate_acceleration(
                problem,
                &mut workspace.stage_position,
                velocity,
                &workspace.candidate_position,
                time + step,
                stats,
            )?;
            for (((next_velocity, velocity), old), new) in workspace
                .candidate_velocity
                .iter_mut()
                .zip(velocity)
                .zip(&workspace.acceleration)
                .zip(&workspace.stage_position)
            {
                *next_velocity = velocity + 0.5 * step * (old + new);
            }
            workspace
                .acceleration
                .copy_from_slice(&workspace.stage_position);
        }
        Method::VerletLeapfrog => {
            for (((stage_velocity, velocity), acceleration), next_position) in workspace
                .stage_velocity
                .iter_mut()
                .zip(velocity)
                .zip(&workspace.acceleration)
                .zip(&mut workspace.candidate_position)
            {
                *stage_velocity = velocity + 0.5 * step * acceleration;
                *next_position = 0.0;
            }
            for ((next_position, position), stage_velocity) in workspace
                .candidate_position
                .iter_mut()
                .zip(position)
                .zip(&workspace.stage_velocity)
            {
                *next_position = position + step * stage_velocity;
            }
            evaluate_acceleration(
                problem,
                &mut workspace.stage_position,
                &workspace.stage_velocity,
                &workspace.candidate_position,
                time + step,
                stats,
            )?;
            for ((next_velocity, stage_velocity), acceleration) in workspace
                .candidate_velocity
                .iter_mut()
                .zip(&workspace.stage_velocity)
                .zip(&workspace.stage_position)
            {
                *next_velocity = stage_velocity + 0.5 * step * acceleration;
            }
            workspace
                .acceleration
                .copy_from_slice(&workspace.stage_position);
        }
        Method::LeapfrogDriftKickDrift => {
            for ((stage_position, position), velocity) in workspace
                .stage_position
                .iter_mut()
                .zip(position)
                .zip(velocity)
            {
                *stage_position = position + 0.5 * step * velocity;
            }
            evaluate_acceleration(
                problem,
                &mut workspace.acceleration,
                velocity,
                position,
                time,
                stats,
            )?;
            for ((stage_velocity, velocity), acceleration) in workspace
                .stage_velocity
                .iter_mut()
                .zip(velocity)
                .zip(&workspace.acceleration)
            {
                *stage_velocity = velocity + 0.5 * step * acceleration;
            }
            evaluate_acceleration(
                problem,
                &mut workspace.acceleration,
                &workspace.stage_velocity,
                &workspace.stage_position,
                time + 0.5 * step,
                stats,
            )?;
            for ((next_velocity, velocity), acceleration) in workspace
                .candidate_velocity
                .iter_mut()
                .zip(velocity)
                .zip(&workspace.acceleration)
            {
                *next_velocity = velocity + step * acceleration;
            }
            for ((next_position, stage_position), next_velocity) in workspace
                .candidate_position
                .iter_mut()
                .zip(&workspace.stage_position)
                .zip(&workspace.candidate_velocity)
            {
                *next_position = stage_position + 0.5 * step * next_velocity;
            }
        }
    }
    Ok(())
}
