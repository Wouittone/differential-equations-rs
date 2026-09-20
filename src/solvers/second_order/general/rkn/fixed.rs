use super::super::*;

pub(crate) fn solve_rkn_fixed<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    options: &SolveOptions,
    tableau: &RungeKuttaNystromTableau,
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
    let nodes = tableau.c();
    let position_coefficients = tableau.a();
    let velocity_coefficients = tableau.a_velocity();
    let position_weights = tableau.b();
    let velocity_weights = tableau.b_velocity();
    let stages = tableau.stages();

    let mut velocity = problem.initial_velocity.clone();
    let mut position = problem.initial_position.clone();
    let mut workspace = RknWorkspace::new(dimension, stages, !problem.callbacks.is_empty());
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
        // Integrate over the interval represented by the returned time.
        let step = (time + step) - time;
        if time + step == time {
            return Err(SolveError::StepSizeUnderflow.into());
        }

        for stage in 0..stages {
            let node = nodes[stage];
            workspace.stage_position.copy_from_slice(&position);
            workspace.stage_velocity.copy_from_slice(&velocity);
            for (stage_position, velocity) in workspace.stage_position.iter_mut().zip(&velocity) {
                *stage_position += step * node * velocity;
            }
            for previous_stage in 0..stage {
                let acceleration = workspace.stage_accelerations
                    [previous_stage * dimension..(previous_stage + 1) * dimension]
                    .iter();
                let position_coefficient = position_coefficients[stage][previous_stage];
                for (value, acceleration) in workspace.stage_position.iter_mut().zip(acceleration) {
                    *value += step * step * position_coefficient * acceleration;
                }
                if let Some(velocity_coefficients) = velocity_coefficients {
                    let acceleration = &workspace.stage_accelerations
                        [previous_stage * dimension..(previous_stage + 1) * dimension];
                    let coefficient = velocity_coefficients[stage][previous_stage];
                    for (value, acceleration) in
                        workspace.stage_velocity.iter_mut().zip(acceleration)
                    {
                        *value += step * coefficient * acceleration;
                    }
                }
            }
            let stage_velocity = if velocity_coefficients.is_some() {
                &workspace.stage_velocity
            } else {
                &velocity
            };
            let acceleration =
                &mut workspace.stage_accelerations[stage * dimension..(stage + 1) * dimension];
            evaluate_acceleration(
                problem,
                acceleration,
                stage_velocity,
                &workspace.stage_position,
                time + node * step,
                &mut stats,
            )?;
        }

        workspace.candidate_position.copy_from_slice(&position);
        workspace.candidate_velocity.copy_from_slice(&velocity);
        for (candidate_position, velocity) in workspace.candidate_position.iter_mut().zip(&velocity)
        {
            *candidate_position += step * velocity;
        }
        for stage in 0..stages {
            let acceleration =
                &workspace.stage_accelerations[stage * dimension..(stage + 1) * dimension];
            for ((candidate_position, candidate_velocity), acceleration) in workspace
                .candidate_position
                .iter_mut()
                .zip(&mut workspace.candidate_velocity)
                .zip(acceleration)
            {
                *candidate_position += step * step * position_weights[stage] * acceleration;
                *candidate_velocity += step * velocity_weights[stage] * acceleration;
            }
        }
        ensure_finite_state(&workspace.candidate_velocity, &workspace.candidate_position)?;

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
    }
    finish_successful(problem, &mut velocity, &mut position, time, recorder, stats)
}
