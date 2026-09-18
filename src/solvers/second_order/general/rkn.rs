use super::*;

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

pub(crate) fn solve_rkn_adaptive<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    options: &SolveOptions,
    tableau: &RungeKuttaNystromTableau,
) -> Result<SecondOrderSolution, SecondOrderSolveError>
where
    F: SecondOrderFunction<P>,
{
    if !options.adaptive && options.initial_step.is_none() {
        return Err(SolveError::InitialStepRequired.into());
    }
    let dimension = problem.initial_position.len();
    let nodes = tableau.c();
    let position_coefficients = tableau.a();
    let velocity_coefficients = tableau.a_velocity();
    let position_weights = tableau.b();
    let velocity_weights = tableau.b_velocity();
    let stages = tableau.stages();

    let (start, end) = problem.time_span;
    let direction = (end - start).signum();
    let span = (end - start).abs();
    let maximum_step = options.max_step.min(span);
    let mut step_magnitude = options
        .initial_step
        .unwrap_or(span / 100.0)
        .min(maximum_step);
    if !step_magnitude.is_finite() || step_magnitude <= 0.0 {
        return Err(SolveError::StepSizeUnderflow.into());
    }

    let mut velocity = problem.initial_velocity.clone();
    let mut position = problem.initial_position.clone();
    let mut workspace = RknWorkspace::new(dimension, stages, !problem.callbacks.is_empty());
    let mut stats = SolverStats::default();
    let controller = ControllerConfig::proportional(tableau.order(), 0.9, 0.2, 10.0, 0.2);
    let mut controller_state = ControllerState::default();
    let mut previous_attempt_rejected = false;

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

        for stage in 0..stages {
            workspace.stage_position.copy_from_slice(&position);
            workspace.stage_velocity.copy_from_slice(&velocity);
            for (stage_position, velocity) in workspace.stage_position.iter_mut().zip(&velocity) {
                *stage_position += step * nodes[stage] * velocity;
            }
            for previous_stage in 0..stage {
                let coefficient = position_coefficients[stage][previous_stage];
                let acceleration = &workspace.stage_accelerations
                    [previous_stage * dimension..(previous_stage + 1) * dimension];
                for (stage_position, acceleration) in
                    workspace.stage_position.iter_mut().zip(acceleration)
                {
                    *stage_position += step * step * coefficient * acceleration;
                }
                if let Some(velocity_coefficients) = velocity_coefficients {
                    let coefficient = velocity_coefficients[stage][previous_stage];
                    for (stage_velocity, acceleration) in
                        workspace.stage_velocity.iter_mut().zip(acceleration)
                    {
                        *stage_velocity += step * coefficient * acceleration;
                    }
                }
            }
            let acceleration =
                &mut workspace.stage_accelerations[stage * dimension..(stage + 1) * dimension];
            evaluate_acceleration(
                problem,
                acceleration,
                if velocity_coefficients.is_some() {
                    &workspace.stage_velocity
                } else {
                    &velocity
                },
                &workspace.stage_position,
                time + nodes[stage] * step,
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

        let error = if options.adaptive {
            rkn_error_norm(
                &velocity,
                &position,
                &workspace.candidate_velocity,
                &workspace.candidate_position,
                &workspace.stage_accelerations,
                step,
                tableau,
                options,
            )?
        } else {
            0.0
        };

        if error <= 1.0 {
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
            let callback = if tableau.dense().is_some() {
                let stage_accelerations = &workspace.stage_accelerations;
                let mut interpolate =
                    |fraction: f64, output_velocity: &mut [f64], output_position: &mut [f64]| {
                        interpolate_dprkn6(
                            tableau,
                            &velocity,
                            &position,
                            stage_accelerations,
                            step,
                            fraction,
                            output_velocity,
                            output_position,
                        )
                    };
                apply_step_callbacks(
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
                    Some(&mut interpolate),
                )?
            } else {
                apply_step_callbacks(
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
                )?
            };
            stats.callback_invocations += callback.invocations;
            stats.rhs_evaluations += callback.rhs_evaluations;
            time = next_time;
            time_stops.accepted(time);
            std::mem::swap(&mut velocity, &mut workspace.candidate_velocity);
            std::mem::swap(&mut position, &mut workspace.candidate_position);
            stats.accepted_steps += 1;

            let recorded_velocity = if callback.invocations == 0 {
                &velocity
            } else {
                &workspace.previous_effect_velocity
            };
            let recorded_position = if callback.invocations == 0 {
                &position
            } else {
                &workspace.previous_effect_position
            };
            if tableau.dense().is_some() {
                let previous_velocity = &workspace.candidate_velocity;
                let previous_position = &workspace.candidate_position;
                let stage_accelerations = &workspace.stage_accelerations;
                let mut interpolate =
                    |target: f64, output_velocity: &mut [f64], output_position: &mut [f64]| {
                        let fraction = (target - previous_time) / step;
                        interpolate_dprkn6(
                            tableau,
                            previous_velocity,
                            previous_position,
                            stage_accelerations,
                            step,
                            fraction,
                            output_velocity,
                            output_position,
                        )
                    };
                recorder.record_step(
                    previous_velocity,
                    previous_position,
                    previous_time,
                    recorded_velocity,
                    recorded_position,
                    time,
                    time == end,
                    Some(&mut interpolate),
                )?;
            } else {
                recorder.record_step(
                    &workspace.candidate_velocity,
                    &workspace.candidate_position,
                    previous_time,
                    recorded_velocity,
                    recorded_position,
                    time,
                    time == end,
                    None,
                )?;
            }
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
            if callback.state_modified {
                controller_state.reset();
            }
            if callback.terminate {
                return finish_successful(
                    problem,
                    &mut velocity,
                    &mut position,
                    time,
                    recorder,
                    stats,
                );
            }

            if callback.requested_step.is_some() {
                step_magnitude = callback_adjusted_step(
                    callback,
                    direction * step_magnitude,
                    direction,
                    maximum_step,
                )
                .abs();
                previous_attempt_rejected = false;
            } else if options.adaptive {
                let factor = controller_state.factor(error, controller);
                controller_state.accepted(error, controller);
                let factor = if previous_attempt_rejected {
                    factor.min(1.0)
                } else {
                    factor
                };
                step_magnitude = callback_adjusted_step(
                    callback,
                    direction * step.abs() * factor,
                    direction,
                    maximum_step,
                )
                .abs();
                previous_attempt_rejected = false;
            } else if callback.step_limit.is_some() {
                step_magnitude = callback_adjusted_step(
                    callback,
                    direction * step_magnitude,
                    direction,
                    maximum_step,
                )
                .abs();
            }
        } else {
            stats.rejected_steps += 1;
            controller_state.rejected(error, controller);
            let factor = controller_state.factor(error, controller).min(1.0);
            step_magnitude = (step.abs() * factor).min(maximum_step);
            previous_attempt_rejected = true;
            if time + direction * step_magnitude == time {
                return Err(SolveError::StepSizeUnderflow.into());
            }
        }
    }
    finish_successful(problem, &mut velocity, &mut position, time, recorder, stats)
}

#[allow(clippy::too_many_arguments)]
fn rkn_error_norm(
    velocity: &[f64],
    position: &[f64],
    candidate_velocity: &[f64],
    candidate_position: &[f64],
    stage_accelerations: &[f64],
    step: f64,
    tableau: &RungeKuttaNystromTableau,
    options: &SolveOptions,
) -> Result<f64, SolveError> {
    let dimension = position.len();
    let stages = tableau.stages();
    let position_error_weights = tableau.error().ok_or(SolveError::InvalidTableau)?;
    let velocity_error_weights = tableau.velocity_error();
    let mut sum = 0.0;
    for component in 0..dimension {
        let mut position_error = 0.0;
        let mut velocity_error = 0.0;
        for stage in 0..stages {
            let acceleration = stage_accelerations[stage * dimension + component];
            position_error += position_error_weights[stage] * acceleration;
            if let Some(velocity_error_weights) = velocity_error_weights {
                velocity_error += velocity_error_weights[stage] * acceleration;
            }
        }
        position_error *= step * step;
        let position_scale = options.absolute_tolerance
            + options.relative_tolerance
                * position[component]
                    .abs()
                    .max(candidate_position[component].abs());
        sum += (position_error / position_scale).powi(2);
        if !tableau.position_only_error() {
            velocity_error *= step;
            let velocity_scale = options.absolute_tolerance
                + options.relative_tolerance
                    * velocity[component]
                        .abs()
                        .max(candidate_velocity[component].abs());
            sum += (velocity_error / velocity_scale).powi(2);
        }
    }
    let components = if tableau.position_only_error() {
        dimension
    } else {
        2 * dimension
    };
    Ok((sum / components as f64).sqrt())
}

#[allow(clippy::too_many_arguments)]
fn interpolate_dprkn6(
    tableau: &RungeKuttaNystromTableau,
    previous_velocity: &[f64],
    previous_position: &[f64],
    stage_accelerations: &[f64],
    step: f64,
    fraction: f64,
    output_velocity: &mut [f64],
    output_position: &mut [f64],
) -> Result<(), SolveError> {
    let position_coefficients = tableau.dense().ok_or(SolveError::InvalidTableau)?;
    let velocity_coefficients = tableau.velocity_dense().ok_or(SolveError::InvalidTableau)?;
    let dimension = previous_position.len();
    let stages = tableau.stages();
    for component in 0..dimension {
        let mut position_sum = 0.0;
        let mut velocity_sum = 0.0;
        for stage in 0..stages {
            let position_row = &position_coefficients[stage];
            let velocity_row = &velocity_coefficients[stage];
            let position_weight = position_row
                .iter()
                .rev()
                .fold(0.0, |value, coefficient| value * fraction + coefficient);
            let velocity_weight = velocity_row
                .iter()
                .rev()
                .fold(0.0, |value, coefficient| value * fraction + coefficient);
            let acceleration = stage_accelerations[stage * dimension + component];
            position_sum += position_weight * acceleration;
            velocity_sum += velocity_weight * acceleration;
        }
        output_velocity[component] = previous_velocity[component] + step * fraction * velocity_sum;
        output_position[component] = previous_position[component]
            + step * fraction * (previous_velocity[component] + step * fraction * position_sum);
    }
    ensure_finite_state(output_velocity, output_position)
}

struct IrknWorkspace {
    candidate_velocity: Vec<f64>,
    candidate_position: Vec<f64>,
    k2: Vec<f64>,
    k3: Vec<f64>,
    next_acceleration: Vec<f64>,
    old_acceleration: Vec<f64>,
    old_internal_first: Vec<f64>,
    old_internal_second: Vec<f64>,
    internal_first: Vec<f64>,
    internal_second: Vec<f64>,
    previous_velocity: Vec<f64>,
    previous_effect_velocity: Vec<f64>,
    previous_effect_position: Vec<f64>,
}

impl IrknWorkspace {
    fn new(dimension: usize, callbacks: bool) -> Self {
        Self {
            candidate_velocity: vec![0.0; dimension],
            candidate_position: vec![0.0; dimension],
            k2: vec![0.0; dimension],
            k3: vec![0.0; dimension],
            next_acceleration: vec![0.0; dimension],
            old_acceleration: vec![0.0; dimension],
            old_internal_first: vec![0.0; dimension],
            old_internal_second: vec![0.0; dimension],
            internal_first: vec![0.0; dimension],
            internal_second: vec![0.0; dimension],
            previous_velocity: vec![0.0; dimension],
            previous_effect_velocity: if callbacks {
                vec![0.0; dimension]
            } else {
                Vec::new()
            },
            previous_effect_position: if callbacks {
                vec![0.0; dimension]
            } else {
                Vec::new()
            },
        }
    }
}

pub(crate) fn solve_irkn<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    options: &SolveOptions,
    tableau: &IrknTableau,
    bootstrap_tableau: &RungeKuttaNystromTableau,
) -> Result<SecondOrderSolution, SecondOrderSolveError>
where
    F: SecondOrderFunction<P>,
{
    let stages = tableau.stages();
    if !(1..=2).contains(&stages)
        || bootstrap_tableau.kind() != RungeKuttaNystromKind::Fixed
        || bootstrap_tableau.order() < tableau.bootstrap_order()
        || bootstrap_tableau.stages() != 3
        || bootstrap_tableau.a_velocity().is_some()
    {
        return Err(SolveError::InvalidTableau.into());
    }
    let nodes = tableau.c();
    let stage_coefficients = tableau.a();
    let velocity_weights = tableau.velocity_weights();
    let history_weights = tableau.history_weights();
    let velocity_history = tableau.velocity_history();
    let bootstrap_nodes = bootstrap_tableau.c();
    let bootstrap_coefficients = bootstrap_tableau.a();
    let bootstrap_position_weights = bootstrap_tableau.b();
    let bootstrap_velocity_weights = bootstrap_tableau.b_velocity();
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
    let mut acceleration = vec![0.0; dimension];
    let mut workspace = IrknWorkspace::new(dimension, !problem.callbacks.is_empty());
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

    let mut time = start;
    let mut attempts = 0;
    let mut time_stops = TimeStopSchedule::new(&options.time_stops, start, end);
    let mut history_valid = false;
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
        let constant_step = step.abs() == step_magnitude;
        let bootstrap = !history_valid || !constant_step;

        if bootstrap {
            // Exact pinned Nyström4VelocityIndependent startup, read from its
            // independently validated RKN resource.
            for component in 0..dimension {
                workspace.candidate_position[component] = position[component]
                    + bootstrap_nodes[1] * step * velocity[component]
                    + step * step * bootstrap_coefficients[1][0] * acceleration[component];
            }
            evaluate_acceleration(
                problem,
                &mut workspace.k2,
                &velocity,
                &workspace.candidate_position,
                time + bootstrap_nodes[1] * step,
                &mut stats,
            )?;
            for component in 0..dimension {
                workspace.candidate_position[component] = position[component]
                    + bootstrap_nodes[2] * step * velocity[component]
                    + step
                        * step
                        * (bootstrap_coefficients[2][0] * acceleration[component]
                            + bootstrap_coefficients[2][1] * workspace.k2[component]);
            }
            evaluate_acceleration(
                problem,
                &mut workspace.k3,
                &velocity,
                &workspace.candidate_position,
                time + bootstrap_nodes[2] * step,
                &mut stats,
            )?;
            for component in 0..dimension {
                workspace.candidate_position[component] = position[component]
                    + step * velocity[component]
                    + step
                        * step
                        * (bootstrap_position_weights[0] * acceleration[component]
                            + bootstrap_position_weights[1] * workspace.k2[component]
                            + bootstrap_position_weights[2] * workspace.k3[component]);
                workspace.candidate_velocity[component] = velocity[component]
                    + step
                        * (bootstrap_velocity_weights[0] * acceleration[component]
                            + bootstrap_velocity_weights[1] * workspace.k2[component]
                            + bootstrap_velocity_weights[2] * workspace.k3[component]);
            }
            evaluate_acceleration(
                problem,
                &mut workspace.next_acceleration,
                &workspace.candidate_velocity,
                &workspace.candidate_position,
                time + step,
                &mut stats,
            )?;

            let c1 = nodes[0];
            let a21 = stage_coefficients[0];
            // Preserve the pinned in-place cache seeds, including their time arguments.
            evaluate_acceleration(
                problem,
                &mut workspace.old_acceleration,
                &velocity,
                &position,
                time + c1 * step,
                &mut stats,
            )?;
            for component in 0..dimension {
                let seed_acceleration = match tableau.bootstrap_seed() {
                    IrknBootstrapSeed::PreviousEndpoint => workspace.old_acceleration[component],
                    IrknBootstrapSeed::NewEndpoint => workspace.next_acceleration[component],
                };
                workspace.k2[component] = position[component]
                    + step * (c1 * velocity[component] + step * a21 * seed_acceleration);
            }
            evaluate_acceleration(
                problem,
                &mut workspace.old_internal_first,
                &velocity,
                &workspace.k2,
                time + c1 * step,
                &mut stats,
            )?;
            if stages == 2 {
                for component in 0..dimension {
                    workspace.k2[component] = position[component]
                        + step
                            * (nodes[1] * velocity[component]
                                + step
                                    * stage_coefficients[1]
                                    * workspace.old_acceleration[component]);
                }
                evaluate_acceleration(
                    problem,
                    &mut workspace.old_internal_second,
                    &velocity,
                    &workspace.k2,
                    time + nodes[0] * step,
                    &mut stats,
                )?;
            }
        } else {
            if stages == 1 {
                for component in 0..dimension {
                    workspace.k2[component] = position[component]
                        + step
                            * (nodes[0] * velocity[component]
                                + step
                                    * stage_coefficients[0]
                                    * workspace.old_acceleration[component]);
                }
                evaluate_acceleration(
                    problem,
                    &mut workspace.internal_first,
                    &velocity,
                    &workspace.k2,
                    time + nodes[0] * step,
                    &mut stats,
                )?;
                for component in 0..dimension {
                    let difference = workspace.internal_first[component]
                        - workspace.old_internal_first[component];
                    workspace.candidate_velocity[component] = velocity[component]
                        + step
                            * (velocity_weights[0] * acceleration[component]
                                + history_weights[0] * workspace.old_acceleration[component]
                                + velocity_weights[1] * difference);
                    workspace.candidate_position[component] = position[component]
                        + step
                            * (velocity_history[0] * velocity[component]
                                + velocity_history[1] * workspace.previous_velocity[component])
                        + step * step * history_weights[1] * difference;
                }
            } else {
                for component in 0..dimension {
                    workspace.k2[component] = position[component]
                        + step
                            * (nodes[0] * velocity[component]
                                + step * stage_coefficients[0] * acceleration[component]);
                }
                evaluate_acceleration(
                    problem,
                    &mut workspace.internal_first,
                    &velocity,
                    &workspace.k2,
                    time + nodes[0] * step,
                    &mut stats,
                )?;
                for component in 0..dimension {
                    workspace.k2[component] = position[component]
                        + step
                            * (nodes[1] * velocity[component]
                                + step
                                    * stage_coefficients[1]
                                    * workspace.internal_first[component]);
                }
                evaluate_acceleration(
                    problem,
                    &mut workspace.internal_second,
                    &velocity,
                    &workspace.k2,
                    time + nodes[1] * step,
                    &mut stats,
                )?;
                for component in 0..dimension {
                    let first_difference = workspace.internal_first[component]
                        - workspace.old_internal_first[component];
                    let second_difference = workspace.internal_second[component]
                        - workspace.old_internal_second[component];
                    workspace.candidate_velocity[component] = velocity[component]
                        + step
                            * (velocity_weights[0] * acceleration[component]
                                + history_weights[0] * workspace.old_acceleration[component]
                                + velocity_weights[1] * first_difference
                                + velocity_weights[2] * second_difference);
                    workspace.candidate_position[component] = position[component]
                        + step
                            * (velocity_history[0] * velocity[component]
                                + velocity_history[1] * workspace.previous_velocity[component])
                        + step
                            * step
                            * (history_weights[1] * first_difference
                                + history_weights[2] * second_difference);
                }
            }
            evaluate_acceleration(
                problem,
                &mut workspace.next_acceleration,
                &workspace.candidate_velocity,
                &workspace.candidate_position,
                time + step,
                &mut stats,
            )?;
        }
        ensure_finite_state(&workspace.candidate_velocity, &workspace.candidate_position)?;

        let previous_time = time;
        let mut next_time = if direction * (end - (time + step)) <= 0.0 {
            end
        } else {
            time + step
        };
        if let Some(reduction_factor) = problem.domain_rejection_factor(
            &workspace.candidate_velocity,
            &workspace.candidate_position,
            next_time,
        ) {
            stats.rejected_steps += 1;
            step_magnitude = step.abs() * reduction_factor;
            history_valid = false;
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

        if callback.state_modified {
            evaluate_acceleration(
                problem,
                &mut acceleration,
                &velocity,
                &position,
                time,
                &mut stats,
            )?;
            history_valid = false;
        } else {
            workspace
                .previous_velocity
                .copy_from_slice(&workspace.candidate_velocity);
            if !bootstrap {
                workspace.old_acceleration.copy_from_slice(&acceleration);
                workspace
                    .old_internal_first
                    .copy_from_slice(&workspace.internal_first);
                if stages == 2 {
                    workspace
                        .old_internal_second
                        .copy_from_slice(&workspace.internal_second);
                }
            }
            acceleration.copy_from_slice(&workspace.next_acceleration);
            history_valid = constant_step;
        }
        if callback.requested_step.is_some() || callback.step_limit.is_some() {
            let adjusted = callback_adjusted_step(
                callback,
                direction * step_magnitude,
                direction,
                maximum_step,
            )
            .abs();
            history_valid &= adjusted == step_magnitude;
            step_magnitude = adjusted;
        }
    }
    finish_successful(problem, &mut velocity, &mut position, time, recorder, stats)
}
