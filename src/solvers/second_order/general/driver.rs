use super::*;

#[path = "fixed.rs"]
mod fixed;
#[path = "rkn.rs"]
mod rkn;
#[path = "structural.rs"]
mod structural;

pub(crate) use fixed::solve_fixed;
pub(crate) use rkn::{solve_irkn, solve_rkn_adaptive, solve_rkn_fixed};
pub(crate) use structural::solve_newmark;

pub(crate) fn validate<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    options: &SolveOptions,
) -> Result<(), SecondOrderSolveError> {
    if problem.initial_position.is_empty() {
        return Err(SolveError::EmptyState.into());
    }
    if problem.initial_position.len() != problem.initial_velocity.len() {
        return Err(SecondOrderSolveError::StateDimensionMismatch);
    }
    validate_state_time_options(&problem.initial_position, problem.time_span, options)?;
    validate_preset_time_sequences(problem.preset_time_sequences(), problem.time_span)?;
    validate_vector_callback_lengths(problem.vector_callback_lengths())?;
    if !problem
        .initial_velocity
        .iter()
        .all(|value| value.is_finite())
    {
        return Err(SolveError::NonFiniteInitialState.into());
    }
    Ok(())
}

struct Workspace {
    candidate_velocity: Vec<f64>,
    candidate_position: Vec<f64>,
    acceleration: Vec<f64>,
    stage_velocity: Vec<f64>,
    stage_position: Vec<f64>,
    previous_effect_velocity: Vec<f64>,
    previous_effect_position: Vec<f64>,
}

impl Workspace {
    fn new(dimension: usize, callbacks: bool) -> Self {
        Self {
            candidate_velocity: vec![0.0; dimension],
            candidate_position: vec![0.0; dimension],
            acceleration: vec![0.0; dimension],
            stage_velocity: vec![0.0; dimension],
            stage_position: vec![0.0; dimension],
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

struct RknWorkspace {
    candidate_velocity: Vec<f64>,
    candidate_position: Vec<f64>,
    stage_velocity: Vec<f64>,
    stage_position: Vec<f64>,
    stage_accelerations: Vec<f64>,
    previous_effect_velocity: Vec<f64>,
    previous_effect_position: Vec<f64>,
}

impl RknWorkspace {
    fn new(dimension: usize, stages: usize, callbacks: bool) -> Self {
        Self {
            candidate_velocity: vec![0.0; dimension],
            candidate_position: vec![0.0; dimension],
            stage_velocity: vec![0.0; dimension],
            stage_position: vec![0.0; dimension],
            stage_accelerations: vec![0.0; dimension * stages],
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

#[derive(Clone, Copy)]
pub(crate) struct StructuralParameters {
    pub(crate) alpha_m: f64,
    pub(crate) alpha_f: f64,
    pub(crate) beta: f64,
    pub(crate) gamma: f64,
}

struct StructuralWorkspace {
    full_velocity: Vec<f64>,
    full_position: Vec<f64>,
    full_acceleration: Vec<f64>,
    candidate_velocity: Vec<f64>,
    candidate_position: Vec<f64>,
    candidate_acceleration: Vec<f64>,
    half_velocity: Vec<f64>,
    half_position: Vec<f64>,
    half_acceleration: Vec<f64>,
    trial_acceleration: Vec<f64>,
    trial_velocity: Vec<f64>,
    trial_position: Vec<f64>,
    evaluated_acceleration: Vec<f64>,
    perturbed_acceleration: Vec<f64>,
    residual: Vec<f64>,
    perturbed_residual: Vec<f64>,
    correction: Vec<f64>,
    matrix: Vec<f64>,
    pivots: Vec<usize>,
    previous_effect_velocity: Vec<f64>,
    previous_effect_position: Vec<f64>,
}

impl StructuralWorkspace {
    fn new(dimension: usize, callbacks: bool) -> Self {
        Self {
            full_velocity: vec![0.0; dimension],
            full_position: vec![0.0; dimension],
            full_acceleration: vec![0.0; dimension],
            candidate_velocity: vec![0.0; dimension],
            candidate_position: vec![0.0; dimension],
            candidate_acceleration: vec![0.0; dimension],
            half_velocity: vec![0.0; dimension],
            half_position: vec![0.0; dimension],
            half_acceleration: vec![0.0; dimension],
            trial_acceleration: vec![0.0; dimension],
            trial_velocity: vec![0.0; dimension],
            trial_position: vec![0.0; dimension],
            evaluated_acceleration: vec![0.0; dimension],
            perturbed_acceleration: vec![0.0; dimension],
            residual: vec![0.0; dimension],
            perturbed_residual: vec![0.0; dimension],
            correction: vec![0.0; dimension],
            matrix: vec![0.0; dimension * dimension],
            pivots: vec![0; dimension],
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

fn finish_successful<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    velocity: &mut [f64],
    position: &mut [f64],
    time: f64,
    mut recorder: PartitionedRecorder<'_>,
    stats: SolverStats,
) -> Result<SecondOrderSolution, SecondOrderSolveError>
where
    F: SecondOrderFunction<P>,
{
    if apply_finalize_callbacks(problem, velocity, position, time)? {
        recorder.synchronize_endpoint(time, velocity, position);
    }
    Ok(recorder.finish(stats, problem.state_shape.clone()))
}

fn evaluate_acceleration<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    output: &mut [f64],
    velocity: &[f64],
    position: &[f64],
    time: f64,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: SecondOrderFunction<P>,
{
    problem
        .acceleration
        .evaluate(output, velocity, position, &problem.parameters, time)?;
    stats.rhs_evaluations += 1;
    output
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteDerivative)
}

pub(crate) fn apply_initial_callbacks<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    velocity: &mut [f64],
    position: &mut [f64],
    time: f64,
) -> Result<CallbackOutcome, SolveError>
where
    F: SecondOrderFunction<P>,
{
    let mut outcome = CallbackOutcome::default();
    for initialization in &problem.initializers {
        (initialization.hook)(velocity, position, &problem.parameters, time);
        ensure_finite_state(velocity, position)?;
        outcome.register_initialization(initialization.save);
    }
    for callback in &problem.callbacks {
        let PartitionedCallback::Discrete(callback) = callback else {
            continue;
        };
        callback
            .trigger
            .initialize(velocity, position, &problem.parameters, time)?;
        if callback.trigger.is_triggered(
            velocity,
            position,
            &problem.parameters,
            time,
            |du, _| {
                let (acceleration, rate) = du.split_at_mut(velocity.len());
                problem.acceleration.evaluate(
                    acceleration,
                    velocity,
                    position,
                    &problem.parameters,
                    time,
                )?;
                rate.copy_from_slice(velocity);
                outcome.rhs_evaluations += 1;
                Ok(())
            },
        )? {
            outcome.register(callback.save);
            outcome.apply_action((callback.affect)(
                velocity,
                position,
                &problem.parameters,
                time,
            )?)?;
            ensure_finite_state(velocity, position)?;
            if outcome.terminate {
                break;
            }
        }
    }
    Ok(outcome)
}

pub(crate) fn apply_finalize_callbacks<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    velocity: &mut [f64],
    position: &mut [f64],
    time: f64,
) -> Result<bool, SolveError> {
    for finalize in &problem.finalizers {
        finalize(velocity, position, &problem.parameters, time);
        ensure_finite_state(velocity, position)?;
    }
    Ok(!problem.finalizers.is_empty())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_step_callbacks<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    previous_velocity: &[f64],
    previous_position: &[f64],
    previous_time: f64,
    velocity: &mut [f64],
    position: &mut [f64],
    time: &mut f64,
    state_before_velocity: &mut [f64],
    state_before_position: &mut [f64],
    event_tolerance: f64,
    mut interpolator: Option<&mut PartitionedInterpolator<'_>>,
) -> Result<CallbackOutcome, SolveError>
where
    F: SecondOrderFunction<P>,
{
    if problem.callbacks.is_empty() {
        return Ok(CallbackOutcome::default());
    }
    let mut outcome = CallbackOutcome::default();
    let mut root = None;
    for (index, callback) in problem.callbacks.iter().enumerate() {
        match callback {
            PartitionedCallback::Continuous(callback) => {
                let before = (callback.condition)(
                    previous_velocity,
                    previous_position,
                    &problem.parameters,
                    previous_time,
                );
                let after = (callback.condition)(velocity, position, &problem.parameters, *time);
                if !before.is_finite() || !after.is_finite() {
                    return Err(SolveError::NonFiniteCallbackCondition);
                }
                if callback.direction.accepts(before, after) {
                    let fraction = locate_root(
                        callback,
                        previous_velocity,
                        previous_position,
                        previous_time,
                        velocity,
                        position,
                        *time,
                        before,
                        state_before_velocity,
                        state_before_position,
                        &problem.parameters,
                        event_tolerance,
                        interpolator.as_deref_mut(),
                    )?;
                    if root.is_none_or(|(_, earliest)| fraction < earliest) {
                        root = Some((index, fraction));
                    }
                }
            }
            PartitionedCallback::VectorContinuous(callback) => {
                let mut scratch = callback.scratch.borrow_mut();
                evaluate_partitioned_vector_condition(
                    callback,
                    &mut scratch.before,
                    previous_velocity,
                    previous_position,
                    &problem.parameters,
                    previous_time,
                )?;
                evaluate_partitioned_vector_condition(
                    callback,
                    &mut scratch.after,
                    velocity,
                    position,
                    &problem.parameters,
                    *time,
                )?;
                scratch.root_fractions.fill(f64::INFINITY);
                scratch.crossings.fill(EventCrossing::None);
                for event_index in 0..callback.event_count {
                    let before = scratch.before[event_index];
                    let crossing = EventDirection::Any.crossing(before, scratch.after[event_index]);
                    if crossing == EventCrossing::None {
                        continue;
                    }
                    let fraction = locate_partitioned_vector_root(
                        callback,
                        event_index,
                        previous_velocity,
                        previous_position,
                        previous_time,
                        velocity,
                        position,
                        *time,
                        before,
                        state_before_velocity,
                        state_before_position,
                        &problem.parameters,
                        event_tolerance,
                        interpolator.as_deref_mut(),
                        &mut scratch.middle,
                    )?;
                    scratch.root_fractions[event_index] = fraction;
                    scratch.crossings[event_index] = crossing;
                    if root.is_none_or(|(_, earliest)| fraction < earliest) {
                        root = Some((index, fraction));
                    }
                }
            }
            PartitionedCallback::Discrete(_) => {}
        }
    }
    if let Some((index, fraction)) = root {
        let end_time = *time;
        if let Some(interpolator) = interpolator {
            interpolator(fraction, state_before_velocity, state_before_position)?;
        } else {
            interpolate_partitioned(
                previous_velocity,
                previous_position,
                velocity,
                position,
                end_time - previous_time,
                fraction,
                state_before_velocity,
                state_before_position,
            );
        }
        velocity.copy_from_slice(state_before_velocity);
        position.copy_from_slice(state_before_position);
        *time = previous_time + fraction * (end_time - previous_time);
        match &problem.callbacks[index] {
            PartitionedCallback::Continuous(callback) => {
                outcome.register(callback.save);
                outcome.apply_action((callback.affect)(
                    velocity,
                    position,
                    &problem.parameters,
                    *time,
                ))?;
            }
            PartitionedCallback::VectorContinuous(callback) => {
                let root_time = *time;
                let mut scratch = callback.scratch.borrow_mut();
                for event_index in 0..callback.event_count {
                    let event_time = previous_time
                        + scratch.root_fractions[event_index] * (end_time - previous_time);
                    let tolerance =
                        effective_event_tolerance(event_tolerance, root_time, event_time);
                    let crossing = scratch.crossings[event_index];
                    scratch.simultaneous_events[event_index] =
                        if (event_time - root_time).abs() <= tolerance {
                            crossing
                        } else {
                            EventCrossing::None
                        };
                }
                outcome.register(callback.save);
                outcome.apply_action((callback.affect)(
                    velocity,
                    position,
                    &problem.parameters,
                    *time,
                    &scratch.simultaneous_events,
                ))?;
            }
            PartitionedCallback::Discrete(_) => return Err(SolveError::InvalidCallbackState),
        }
        // The localized root truncates the attempted step even when its
        // effect is observation-only, so endpoint-dependent caches cannot be
        // reused for the next step.
        outcome.state_modified = true;
        ensure_finite_state(velocity, position)?;
    }
    if !outcome.terminate {
        for callback in &problem.callbacks {
            let PartitionedCallback::Discrete(callback) = callback else {
                continue;
            };
            if callback.trigger.is_triggered(
                velocity,
                position,
                &problem.parameters,
                *time,
                |du, _| {
                    let (acceleration, rate) = du.split_at_mut(velocity.len());
                    problem.acceleration.evaluate(
                        acceleration,
                        velocity,
                        position,
                        &problem.parameters,
                        *time,
                    )?;
                    rate.copy_from_slice(velocity);
                    outcome.rhs_evaluations += 1;
                    Ok(())
                },
            )? {
                if outcome.invocations == 0 {
                    state_before_velocity.copy_from_slice(velocity);
                    state_before_position.copy_from_slice(position);
                }
                outcome.register(callback.save);
                outcome.apply_action((callback.affect)(
                    velocity,
                    position,
                    &problem.parameters,
                    *time,
                )?)?;
                ensure_finite_state(velocity, position)?;
                if outcome.terminate {
                    break;
                }
            }
        }
    }
    Ok(outcome)
}

#[allow(clippy::too_many_arguments)]
fn locate_root<P>(
    callback: &ContinuousCallback<P>,
    previous_velocity: &[f64],
    previous_position: &[f64],
    previous_time: f64,
    velocity: &[f64],
    position: &[f64],
    time: f64,
    before: f64,
    interpolation_velocity: &mut [f64],
    interpolation_position: &mut [f64],
    parameters: &P,
    event_tolerance: f64,
    mut interpolator: Option<&mut PartitionedInterpolator<'_>>,
) -> Result<f64, SolveError> {
    let mut left = 0.0;
    let mut right = 1.0;
    let mut left_value = before;
    for _ in 0..MAX_EVENT_ROOT_ITERATIONS {
        let middle = 0.5 * (left + right);
        if middle == left || middle == right {
            break;
        }
        if let Some(interpolator) = interpolator.as_deref_mut() {
            interpolator(middle, interpolation_velocity, interpolation_position)?;
        } else {
            interpolate_partitioned(
                previous_velocity,
                previous_position,
                velocity,
                position,
                time - previous_time,
                middle,
                interpolation_velocity,
                interpolation_position,
            );
        }
        let middle_time = previous_time + middle * (time - previous_time);
        let value = (callback.condition)(
            interpolation_velocity,
            interpolation_position,
            parameters,
            middle_time,
        );
        if !value.is_finite() {
            return Err(SolveError::NonFiniteCallbackCondition);
        }
        if value == 0.0 {
            return Ok(middle);
        }
        if left_value.signum() == value.signum() {
            left = middle;
            left_value = value;
        } else {
            right = middle;
        }
        if event_interval_converged(event_tolerance, previous_time, time, left, right) {
            break;
        }
    }
    // Keep the accepted state on the post-crossing side so a continuing
    // callback cannot immediately retrigger the same root.
    Ok(right)
}

fn evaluate_partitioned_vector_condition<P>(
    callback: &VectorContinuousCallback<P>,
    output: &mut [f64],
    velocity: &[f64],
    position: &[f64],
    parameters: &P,
    time: f64,
) -> Result<(), SolveError> {
    output.fill(f64::NAN);
    (callback.condition)(output, velocity, position, parameters, time);
    output
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteCallbackCondition)
}

#[allow(clippy::too_many_arguments)]
fn locate_partitioned_vector_root<P>(
    callback: &VectorContinuousCallback<P>,
    event_index: usize,
    previous_velocity: &[f64],
    previous_position: &[f64],
    previous_time: f64,
    velocity: &[f64],
    position: &[f64],
    time: f64,
    before: f64,
    interpolation_velocity: &mut [f64],
    interpolation_position: &mut [f64],
    parameters: &P,
    event_tolerance: f64,
    mut interpolator: Option<&mut PartitionedInterpolator<'_>>,
    condition_values: &mut [f64],
) -> Result<f64, SolveError> {
    let mut left = 0.0;
    let mut right = 1.0;
    let mut left_value = before;
    for _ in 0..MAX_EVENT_ROOT_ITERATIONS {
        let middle = 0.5 * (left + right);
        if middle == left || middle == right {
            break;
        }
        if let Some(interpolator) = interpolator.as_deref_mut() {
            interpolator(middle, interpolation_velocity, interpolation_position)?;
        } else {
            interpolate_partitioned(
                previous_velocity,
                previous_position,
                velocity,
                position,
                time - previous_time,
                middle,
                interpolation_velocity,
                interpolation_position,
            );
        }
        let middle_time = previous_time + middle * (time - previous_time);
        evaluate_partitioned_vector_condition(
            callback,
            condition_values,
            interpolation_velocity,
            interpolation_position,
            parameters,
            middle_time,
        )?;
        let value = condition_values[event_index];
        if value == 0.0 {
            return Ok(middle);
        }
        if left_value.signum() == value.signum() {
            left = middle;
            left_value = value;
        } else {
            right = middle;
        }
        if event_interval_converged(event_tolerance, previous_time, time, left, right) {
            break;
        }
    }
    Ok(right)
}

pub(crate) fn interpolate(current: &[f64], previous: &[f64], fraction: f64, output: &mut [f64]) {
    for ((output, previous), current) in output.iter_mut().zip(previous).zip(current) {
        *output = interpolate_value(*previous, *current, fraction);
    }
}

#[allow(clippy::too_many_arguments)]
fn interpolate_partitioned(
    start_velocity: &[f64],
    start_position: &[f64],
    end_velocity: &[f64],
    end_position: &[f64],
    step: f64,
    theta: f64,
    velocity: &mut [f64],
    position: &mut [f64],
) {
    let theta2 = theta * theta;
    let theta3 = theta2 * theta;
    let h00 = 2.0 * theta3 - 3.0 * theta2 + 1.0;
    let h10 = theta3 - 2.0 * theta2 + theta;
    let h01 = -2.0 * theta3 + 3.0 * theta2;
    let h11 = theta3 - theta2;
    for index in 0..velocity.len() {
        velocity[index] = interpolate_value(start_velocity[index], end_velocity[index], theta);
        position[index] = h00 * start_position[index]
            + h10 * step * start_velocity[index]
            + h01 * end_position[index]
            + h11 * step * end_velocity[index];
    }
}

fn ensure_finite_state(velocity: &[f64], position: &[f64]) -> Result<(), SolveError> {
    velocity
        .iter()
        .chain(position)
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteCallbackState)
}

struct PartitionedRecorder<'a> {
    times: Vec<f64>,
    velocities: Vec<f64>,
    positions: Vec<f64>,
    dimension: usize,
    save_at: &'a [f64],
    next_save: usize,
    save_mode: SaveMode,
    interpolation_velocity: Vec<f64>,
    interpolation_position: Vec<f64>,
    dense_segments: Vec<PartitionedDenseSegment>,
    retain_dense_output: bool,
}

impl<'a> PartitionedRecorder<'a> {
    fn new(velocity: &[f64], position: &[f64], time: f64, options: &'a SolveOptions) -> Self {
        let save_initial = options.save_at.is_empty() || options.save_at.first() == Some(&time);
        let capacity = if options.save_at.is_empty() {
            2
        } else {
            options.save_at.len()
        };
        let mut recorder = Self {
            times: Vec::with_capacity(capacity),
            velocities: Vec::with_capacity(capacity * velocity.len()),
            positions: Vec::with_capacity(capacity * position.len()),
            dimension: position.len(),
            save_at: &options.save_at,
            next_save: usize::from(!options.save_at.is_empty() && save_initial),
            save_mode: options.save,
            interpolation_velocity: if options.save_at.is_empty() {
                Vec::new()
            } else {
                vec![0.0; velocity.len()]
            },
            interpolation_position: if options.save_at.is_empty() {
                Vec::new()
            } else {
                vec![0.0; position.len()]
            },
            dense_segments: Vec::new(),
            retain_dense_output: options.retain_dense_output,
        };
        if save_initial {
            recorder.push_unique(time, velocity, position);
        }
        recorder
    }

    #[allow(clippy::too_many_arguments)]
    fn record_step(
        &mut self,
        previous_velocity: &[f64],
        previous_position: &[f64],
        previous_time: f64,
        velocity: &[f64],
        position: &[f64],
        time: f64,
        final_time: bool,
        mut interpolator: Option<&mut PartitionedInterpolator<'_>>,
    ) -> Result<(), SolveError> {
        let generic_segment = PartitionedDenseSegment::new(
            previous_time,
            time,
            previous_velocity,
            velocity,
            previous_position,
            position,
        );
        if self.retain_dense_output {
            self.dense_segments.push(generic_segment.clone());
        }
        if self.save_at.is_empty() {
            if self.save_mode == SaveMode::EveryStep || final_time {
                self.push_unique(time, velocity, position);
            }
            return Ok(());
        }
        let direction = (time - previous_time).signum();
        while let Some(&target) = self.save_at.get(self.next_save) {
            if direction * (target - previous_time) <= 0.0 {
                self.next_save += 1;
                continue;
            }
            if direction * (time - target) < 0.0 {
                break;
            }
            if let Some(interpolator) = interpolator.as_deref_mut() {
                interpolator(
                    target,
                    &mut self.interpolation_velocity,
                    &mut self.interpolation_position,
                )
                .map_err(|_| SolveError::DenseOutputFailed)?;
            } else {
                generic_segment
                    .interpolate(
                        target,
                        &mut self.interpolation_velocity,
                        &mut self.interpolation_position,
                    )
                    .ok_or(SolveError::DenseOutputFailed)?;
            }
            self.times.push(target);
            self.velocities
                .extend_from_slice(&self.interpolation_velocity);
            self.positions
                .extend_from_slice(&self.interpolation_position);
            self.next_save += 1;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn record_callback(
        &mut self,
        time: f64,
        before_velocity: &[f64],
        before_position: &[f64],
        after_velocity: &[f64],
        after_position: &[f64],
        outcome: CallbackOutcome,
        boundary: bool,
    ) {
        let canonical_time = self
            .save_at
            .iter()
            .copied()
            .find(|target| times_are_numerically_equal(*target, time))
            .unwrap_or(time);
        let requested_at = self
            .save_at
            .iter()
            .any(|target| times_are_numerically_equal(*target, time));
        let globally_saved_after = (self.save_at.is_empty()
            && (self.save_mode == SaveMode::EveryStep || boundary))
            || outcome.terminate;
        let save_before = outcome.save_before || requested_at;
        let save_after = outcome.save_after || globally_saved_after;

        if save_before {
            self.push_unique(canonical_time, before_velocity, before_position);
        }
        if save_after {
            if save_before {
                self.push(canonical_time, after_velocity, after_position);
            } else {
                self.push_unique(canonical_time, after_velocity, after_position);
            }
        }
    }

    fn synchronize_endpoint(&mut self, time: f64, velocity: &[f64], position: &[f64]) {
        if self
            .times
            .last()
            .is_some_and(|saved| times_are_representably_equal(*saved, time))
        {
            let start = self.velocities.len() - self.dimension;
            self.velocities[start..].copy_from_slice(velocity);
            self.positions[start..].copy_from_slice(position);
        }
    }

    fn push_unique(&mut self, time: f64, velocity: &[f64], position: &[f64]) {
        if self
            .times
            .last()
            .is_some_and(|saved| times_are_representably_equal(*saved, time))
        {
            let start = self.velocities.len() - self.dimension;
            self.velocities[start..].copy_from_slice(velocity);
            self.positions[start..].copy_from_slice(position);
        } else {
            self.push(time, velocity, position);
        }
    }

    fn push(&mut self, time: f64, velocity: &[f64], position: &[f64]) {
        debug_assert_eq!(velocity.len(), self.dimension);
        debug_assert_eq!(position.len(), self.dimension);
        self.times.push(time);
        self.velocities.extend_from_slice(velocity);
        self.positions.extend_from_slice(position);
    }

    fn finish(self, stats: SolverStats, state_shape: IxDyn) -> SecondOrderSolution {
        SecondOrderSolution {
            times: self.times,
            velocities: self.velocities,
            positions: self.positions,
            dimension: self.dimension,
            state_shape,
            stats,
            dense_segments: self.dense_segments,
        }
    }
}

#[cfg(test)]
mod interpolation_tests {
    use super::*;
    use crate::InterpolationError;

    fn saved_solution(velocities: Vec<f64>, positions: Vec<f64>) -> SecondOrderSolution {
        SecondOrderSolution {
            times: vec![0.0, 1.0],
            velocities,
            positions,
            dimension: 1,
            state_shape: IxDyn(&[1]),
            stats: SolverStats::default(),
            dense_segments: Vec::new(),
        }
    }

    #[test]
    fn saved_interpolation_is_bounded_for_opposite_sign_finite_endpoints() {
        let mut solution = saved_solution(vec![f64::MAX, -f64::MAX], vec![-f64::MAX, f64::MAX]);

        let (velocity, position) = solution.try_interpolate(0.25).unwrap();

        assert!((velocity[0] / f64::MAX - 0.5).abs() <= f64::EPSILON);
        assert!((position[0] / f64::MAX + 0.5).abs() <= f64::EPSILON);

        solution.times = vec![-f64::MAX, f64::MAX];
        assert_eq!(solution.try_interpolate(0.0), Ok((vec![0.0], vec![0.0])));
    }

    #[test]
    fn exact_saved_interpolation_rejects_non_finite_partitions() {
        let solution = saved_solution(vec![1.0, 2.0], vec![f64::INFINITY, 3.0]);

        assert_eq!(
            solution.try_interpolate(0.0),
            Err(InterpolationError::NonFiniteResult {
                context: "saved second-order state",
            })
        );
    }

    #[test]
    fn retained_dense_interpolation_rejects_non_finite_output() {
        let mut solution = saved_solution(vec![1.0, 1.0], vec![1.0, 1.0]);
        solution.times[1] = 2.0;
        solution.dense_segments.push(PartitionedDenseSegment::new(
            0.0,
            2.0,
            &[f64::MAX],
            &[-f64::MAX],
            &[f64::MAX],
            &[f64::MAX],
        ));

        assert_eq!(
            solution.try_interpolate(1.0),
            Err(InterpolationError::NonFiniteResult {
                context: "second-order dense segment",
            })
        );
    }
}
