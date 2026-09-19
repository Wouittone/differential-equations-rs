use super::super::*;

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
