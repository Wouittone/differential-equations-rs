//! Explicit symplectic composition methods for partitioned second-order problems.
//!
//! The coefficient vectors are pinned copies of the `SymplecticTableau` data in
//! OrdinaryDiffEqSymplecticRK.  A stage is a drift of the position by `bᵢ`
//! followed by a kick of the velocity by `aᵢ`.

use super::function::SecondOrderFunction;
use crate::callback::CallbackOutcome;
use crate::event::{times_are_numerically_equal, times_are_representably_equal};
use crate::integrator::{TimeStopSchedule, callback_adjusted_step};
use crate::solver::{
    validate_preset_time_sequences, validate_state_time_options, validate_vector_callback_lengths,
};
use crate::{InterpolationError, SaveMode, SolveError, SolveOptions};
use ndarray::{ArrayD, ArrayViewD, Dimension, IxDyn};
use thiserror::Error;

use super::general::{
    SecondOrderOdeProblem, apply_finalize_callbacks, apply_initial_callbacks, apply_step_callbacks,
};

pub use crate::tableau::SymplecticTableau;
use crate::tableau::{TableauError, define_symplectic_from_file};

/// A named explicit symplectic composition.
pub trait SymplecticAlgorithm: Copy {
    /// Returns the validated, lazily initialized composition tableau.
    fn tableau() -> Result<&'static SymplecticTableau, TableauError>;
}

define_symplectic_from_file!(pub PseudoVerletLeapfrog, "src/tableau/resources/symplectic/pseudoverletleapfrog.json", crate = crate);
define_symplectic_from_file!(pub McAte2, "src/tableau/resources/symplectic/mcate2.json", crate = crate);
define_symplectic_from_file!(pub Ruth3, "src/tableau/resources/symplectic/ruth3.json", crate = crate);
define_symplectic_from_file!(pub McAte3, "src/tableau/resources/symplectic/mcate3.json", crate = crate);
define_symplectic_from_file!(pub CandyRoz4, "src/tableau/resources/symplectic/candyroz4.json", crate = crate);
define_symplectic_from_file!(pub McAte4, "src/tableau/resources/symplectic/mcate4.json", crate = crate);
define_symplectic_from_file!(pub CalvoSanz4, "src/tableau/resources/symplectic/calvosanz4.json", crate = crate);
define_symplectic_from_file!(pub McAte42, "src/tableau/resources/symplectic/mcate42.json", crate = crate);
define_symplectic_from_file!(pub McAte5, "src/tableau/resources/symplectic/mcate5.json", crate = crate);
define_symplectic_from_file!(pub Yoshida6, "src/tableau/resources/symplectic/yoshida6.json", crate = crate);
define_symplectic_from_file!(pub KahanLi6, "src/tableau/resources/symplectic/kahanli6.json", crate = crate);
define_symplectic_from_file!(pub McAte8, "src/tableau/resources/symplectic/mcate8.json", crate = crate);
define_symplectic_from_file!(pub KahanLi8, "src/tableau/resources/symplectic/kahanli8.json", crate = crate);
define_symplectic_from_file!(pub SofSpa10, "src/tableau/resources/symplectic/sofspa10.json", crate = crate);

/// A trajectory returned by [`solve_symplectic`].
///
/// Callbacks configured with [`crate::CallbackSave::Both`] produce adjacent
/// states at the same time, ordered before-effect then after-effect. Exact
/// interpolation at that time returns the latter state.
#[derive(Clone, Debug, PartialEq)]
pub struct SymplecticSolution {
    times: Vec<f64>,
    positions: Vec<f64>,
    velocities: Vec<f64>,
    dimension: usize,
    state_shape: IxDyn,
    rhs_evaluations: usize,
    dense_segments: Vec<SymplecticDenseSegment>,
}

impl SymplecticSolution {
    /// Shape of each partition; an empty slice denotes an ndarray scalar.
    pub fn state_shape(&self) -> &[usize] {
        self.state_shape.slice()
    }

    /// Saved position as a shape-preserving ndarray view.
    pub fn position_array(&self, index: usize) -> Option<ArrayViewD<'_, f64>> {
        ArrayViewD::from_shape(self.state_shape.clone(), self.position(index)?).ok()
    }

    /// Saved velocity as a shape-preserving ndarray view.
    pub fn velocity_array(&self, index: usize) -> Option<ArrayViewD<'_, f64>> {
        ArrayViewD::from_shape(self.state_shape.clone(), self.velocity(index)?).ok()
    }

    /// Last saved position as a shape-preserving ndarray view.
    pub fn last_position_array(&self) -> ArrayViewD<'_, f64> {
        ArrayViewD::from_shape(self.state_shape.clone(), self.last_position())
            .expect("partition shape must match its validated storage")
    }

    /// Last saved velocity as a shape-preserving ndarray view.
    pub fn last_velocity_array(&self) -> ArrayViewD<'_, f64> {
        ArrayViewD::from_shape(self.state_shape.clone(), self.last_velocity())
            .expect("partition shape must match its validated storage")
    }

    /// Interpolates shape-preserving `(position, velocity)` arrays.
    ///
    /// This follows the existing symplectic interpolation order, which differs
    /// from the `(velocity, position)` order of `SecondOrderSolution`.
    pub fn interpolate_array(
        &self,
        time: f64,
    ) -> Result<(ArrayD<f64>, ArrayD<f64>), InterpolationError> {
        let (position, velocity) = self.try_interpolate(time)?;
        let reshape = |values| {
            ArrayD::from_shape_vec(self.state_shape.clone(), values)
                .map_err(|_| InterpolationError::DimensionMismatch)
        };
        Ok((reshape(position)?, reshape(velocity)?))
    }

    /// Saved times in integration order.
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    /// Number of scalar components in each partition.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// All saved positions in contiguous row-major storage.
    pub fn position_values(&self) -> &[f64] {
        &self.positions
    }

    /// All saved velocities in contiguous row-major storage.
    pub fn velocity_values(&self) -> &[f64] {
        &self.velocities
    }

    /// Last position partition.
    pub fn last_position(&self) -> &[f64] {
        let start = self.positions.len() - self.dimension;
        &self.positions[start..]
    }

    /// Last velocity partition.
    pub fn last_velocity(&self) -> &[f64] {
        let start = self.velocities.len() - self.dimension;
        &self.velocities[start..]
    }

    /// Position partition at a saved index.
    pub fn position(&self, index: usize) -> Option<&[f64]> {
        partition(&self.positions, self.dimension, index)
    }

    /// Velocity partition at a saved index.
    pub fn velocity(&self, index: usize) -> Option<&[f64]> {
        partition(&self.velocities, self.dimension, index)
    }

    /// Number of acceleration evaluations.
    pub fn rhs_evaluations(&self) -> usize {
        self.rhs_evaluations
    }

    /// Interpolates `(position, velocity)` at a covered time.
    ///
    /// Retained segments use cubic-Hermite position interpolation consistent
    /// with `q' = v` and linear velocity interpolation. Saved-only solutions
    /// retain the stable linear compatibility fallback.
    pub fn interpolate(&self, time: f64) -> Option<(Vec<f64>, Vec<f64>)> {
        self.try_interpolate(time).ok()
    }

    /// Interpolates `(position, velocity)` and reports why the query fails.
    pub fn try_interpolate(&self, time: f64) -> Result<(Vec<f64>, Vec<f64>), InterpolationError> {
        if !time.is_finite() {
            return Err(InterpolationError::NonFiniteTime);
        }
        if self.times.is_empty() {
            return Err(InterpolationError::EmptySolution);
        }
        for (index, &saved_time) in self.times.iter().enumerate().rev() {
            if time == saved_time {
                return Ok((
                    self.position(index)
                        .ok_or(InterpolationError::InvalidSegmentData {
                            context: "saved symplectic position",
                        })?
                        .to_vec(),
                    self.velocity(index)
                        .ok_or(InterpolationError::InvalidSegmentData {
                            context: "saved symplectic velocity",
                        })?
                        .to_vec(),
                ));
            }
        }
        for segment in &self.dense_segments {
            if segment.contains(time) {
                let mut position = vec![0.0; self.dimension];
                let mut velocity = vec![0.0; self.dimension];
                segment
                    .interpolate(time, &mut position, &mut velocity)
                    .ok_or(InterpolationError::InvalidSegmentData {
                        context: "symplectic dense segment",
                    })?;
                return Ok((position, velocity));
            }
        }
        for index in 1..self.times.len() {
            let left = self.times[index - 1];
            let right = self.times[index];
            if between(time, left, right) && left != right {
                let fraction = (time - left) / (right - left);
                let mut position = vec![0.0; self.dimension];
                let mut velocity = vec![0.0; self.dimension];
                interpolate(
                    self.position(index)
                        .ok_or(InterpolationError::InvalidSegmentData {
                            context: "saved symplectic position",
                        })?,
                    self.position(index - 1)
                        .ok_or(InterpolationError::InvalidSegmentData {
                            context: "saved symplectic position",
                        })?,
                    fraction,
                    &mut position,
                );
                interpolate(
                    self.velocity(index)
                        .ok_or(InterpolationError::InvalidSegmentData {
                            context: "saved symplectic velocity",
                        })?,
                    self.velocity(index - 1)
                        .ok_or(InterpolationError::InvalidSegmentData {
                            context: "saved symplectic velocity",
                        })?,
                    fraction,
                    &mut velocity,
                );
                return Ok((position, velocity));
            }
        }
        Err(InterpolationError::OutsideTimeSpan)
    }
}

fn between(time: f64, left: f64, right: f64) -> bool {
    (left <= time && time <= right) || (right <= time && time <= left)
}

#[derive(Clone, Debug, PartialEq)]
struct SymplecticDenseSegment {
    start_time: f64,
    end_time: f64,
    start_position: Vec<f64>,
    end_position: Vec<f64>,
    start_velocity: Vec<f64>,
    end_velocity: Vec<f64>,
}

impl SymplecticDenseSegment {
    fn new(
        start_time: f64,
        end_time: f64,
        start_position: &[f64],
        end_position: &[f64],
        start_velocity: &[f64],
        end_velocity: &[f64],
    ) -> Self {
        Self {
            start_time,
            end_time,
            start_position: start_position.to_vec(),
            end_position: end_position.to_vec(),
            start_velocity: start_velocity.to_vec(),
            end_velocity: end_velocity.to_vec(),
        }
    }

    fn contains(&self, time: f64) -> bool {
        between(time, self.start_time, self.end_time)
    }

    fn interpolate(&self, time: f64, position: &mut [f64], velocity: &mut [f64]) -> Option<()> {
        if !self.contains(time)
            || position.len() != self.start_position.len()
            || velocity.len() != self.start_velocity.len()
        {
            return None;
        }
        if time == self.start_time {
            position.copy_from_slice(&self.start_position);
            velocity.copy_from_slice(&self.start_velocity);
            return Some(());
        }
        if time == self.end_time {
            position.copy_from_slice(&self.end_position);
            velocity.copy_from_slice(&self.end_velocity);
            return Some(());
        }
        let step = self.end_time - self.start_time;
        let theta = (time - self.start_time) / step;
        let theta2 = theta * theta;
        let theta3 = theta2 * theta;
        let h00 = 2.0 * theta3 - 3.0 * theta2 + 1.0;
        let h10 = theta3 - 2.0 * theta2 + theta;
        let h01 = -2.0 * theta3 + 3.0 * theta2;
        let h11 = theta3 - theta2;
        for index in 0..position.len() {
            position[index] = h00 * self.start_position[index]
                + h10 * step * self.start_velocity[index]
                + h01 * self.end_position[index]
                + h11 * step * self.end_velocity[index];
            velocity[index] = self.start_velocity[index]
                + theta * (self.end_velocity[index] - self.start_velocity[index]);
        }
        Some(())
    }
}

fn partition(values: &[f64], dimension: usize, index: usize) -> Option<&[f64]> {
    let start = index.checked_mul(dimension)?;
    let end = start.checked_add(dimension)?;
    values.get(start..end)
}

/// Failure specific to a fixed-step symplectic composition.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
#[non_exhaustive]
pub enum SymplecticSolveError {
    /// Position and velocity partitions differ in size.
    #[error("position and velocity dimensions must match")]
    StateDimensionMismatch,
    /// A common solver validation or execution error.
    #[error("{0}")]
    Solve(
        #[from]
        #[source]
        SolveError,
    ),
}

/// Solves a second-order problem with a pinned alternating drift/kick method.
///
pub fn solve_symplectic<F, P, A>(
    problem: &SecondOrderOdeProblem<F, P>,
    _algorithm: A,
    options: &SolveOptions,
) -> Result<SymplecticSolution, SymplecticSolveError>
where
    F: SecondOrderFunction<P>,
    A: SymplecticAlgorithm,
{
    validate(problem, options)?;
    if options.adaptive {
        return Err(SolveError::AdaptiveStepUnsupported.into());
    }
    let fixed_step = options
        .initial_step
        .ok_or(SolveError::InitialStepRequired)?;
    let (start, end) = problem.time_span();
    let tableau = A::tableau().map_err(|_| SolveError::InvalidTableau)?;

    let direction = (end - start).signum();
    let maximum_step = options.max_step.min((end - start).abs());
    let mut step_size = fixed_step.min(maximum_step);
    let dimension = problem.initial_position().len();
    let mut position = problem.initial_position().to_vec();
    let mut velocity = problem.initial_velocity().to_vec();
    let mut recorder = SymplecticRecorder::new(&position, &velocity, start, options);
    let initial_callbacks = apply_initial_callbacks(problem, &mut velocity, &mut position, start)?;
    if initial_callbacks.state_modified {
        recorder.record_callback(
            start,
            problem.initial_position(),
            problem.initial_velocity(),
            &position,
            &velocity,
            initial_callbacks,
            true,
        );
    }
    if problem
        .domain_rejection_factor(&velocity, &position, start)
        .is_some()
    {
        return Err(SolveError::InitialStateOutOfDomain.into());
    }
    let mut candidate_position = position.clone();
    let mut candidate_velocity = velocity.clone();
    let mut acceleration = vec![0.0; dimension];
    let mut state_before_position = if problem.has_callbacks() {
        vec![0.0; dimension]
    } else {
        Vec::new()
    };
    let mut state_before_velocity = if problem.has_callbacks() {
        vec![0.0; dimension]
    } else {
        Vec::new()
    };
    if initial_callbacks.terminate {
        return finish_successful(problem, &mut velocity, &mut position, start, recorder, 0);
    }
    step_size = callback_adjusted_step(
        initial_callbacks,
        direction * step_size,
        direction,
        maximum_step,
    )
    .abs();
    let mut time = start;
    let mut steps = 0usize;
    let mut rhs_evaluations = 0usize;
    let mut time_stops = TimeStopSchedule::new(&options.time_stops, start, end);

    while direction * (end - time) > 0.0 {
        if steps >= options.max_steps {
            return Err(SolveError::MaxStepsExceeded.into());
        }
        steps += 1;
        let step = time_stops.clip_step_with(
            time,
            direction * step_size,
            problem.next_preset_time(time, direction),
        );
        if time + step == time {
            return Err(SolveError::StepSizeUnderflow.into());
        }
        candidate_position.copy_from_slice(&position);
        candidate_velocity.copy_from_slice(&velocity);
        let previous_time = time;
        rhs_evaluations += perform_step(
            problem,
            tableau,
            &mut candidate_position,
            &mut candidate_velocity,
            &mut acceleration,
            time,
            step,
        )?;
        let mut next_time = time + step;
        if direction * (end - next_time) <= 0.0 {
            next_time = end;
        }
        if let Some(reduction_factor) =
            problem.domain_rejection_factor(&candidate_velocity, &candidate_position, next_time)
        {
            step_size = step.abs() * reduction_factor;
            continue;
        }
        let callback = apply_step_callbacks(
            problem,
            &velocity,
            &position,
            previous_time,
            &mut candidate_velocity,
            &mut candidate_position,
            &mut next_time,
            &mut state_before_velocity,
            &mut state_before_position,
            options.event_tolerance,
            None,
        )?;
        recorder.record_step(
            &position,
            &velocity,
            previous_time,
            if callback.invocations == 0 {
                &candidate_position
            } else {
                &state_before_position
            },
            if callback.invocations == 0 {
                &candidate_velocity
            } else {
                &state_before_velocity
            },
            next_time,
            next_time == end,
        )?;
        if callback.invocations > 0 {
            recorder.record_callback(
                next_time,
                &state_before_position,
                &state_before_velocity,
                &candidate_position,
                &candidate_velocity,
                callback,
                next_time == end,
            );
        }
        time = next_time;
        time_stops.accepted(time);
        std::mem::swap(&mut position, &mut candidate_position);
        std::mem::swap(&mut velocity, &mut candidate_velocity);
        if callback.terminate {
            return finish_successful(
                problem,
                &mut velocity,
                &mut position,
                time,
                recorder,
                rhs_evaluations,
            );
        }
        step_size =
            callback_adjusted_step(callback, direction * step_size, direction, maximum_step).abs();
    }

    finish_successful(
        problem,
        &mut velocity,
        &mut position,
        time,
        recorder,
        rhs_evaluations,
    )
}

fn finish_successful<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    velocity: &mut [f64],
    position: &mut [f64],
    time: f64,
    mut recorder: SymplecticRecorder<'_>,
    rhs_evaluations: usize,
) -> Result<SymplecticSolution, SymplecticSolveError>
where
    F: SecondOrderFunction<P>,
{
    if apply_finalize_callbacks(problem, velocity, position, time)? {
        recorder.synchronize_endpoint(time, position, velocity);
    }
    Ok(recorder.finish(rhs_evaluations, IxDyn(problem.state_shape())))
}

fn validate<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    options: &SolveOptions,
) -> Result<(), SymplecticSolveError> {
    let position = problem.initial_position();
    let velocity = problem.initial_velocity();
    if position.is_empty() {
        return Err(SolveError::EmptyState.into());
    }
    if position.len() != velocity.len() {
        return Err(SymplecticSolveError::StateDimensionMismatch);
    }
    validate_state_time_options(position, problem.time_span(), options)?;
    validate_preset_time_sequences(problem.preset_time_sequences(), problem.time_span())?;
    validate_vector_callback_lengths(problem.vector_callback_lengths())?;
    if !velocity.iter().all(|value| value.is_finite()) {
        return Err(SolveError::NonFiniteInitialState.into());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn perform_step<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    tableau: &SymplecticTableau,
    position: &mut [f64],
    velocity: &mut [f64],
    acceleration: &mut [f64],
    time: f64,
    step: f64,
) -> Result<usize, SolveError>
where
    F: SecondOrderFunction<P>,
{
    let mut stage_time = time;
    for (stage, (&kick, &drift)) in tableau.a().iter().zip(tableau.b()).enumerate() {
        for (position, &velocity) in position.iter_mut().zip(&*velocity) {
            *position += drift * step * velocity;
        }
        problem.evaluate_acceleration(acceleration, velocity, position, stage_time)?;
        if !acceleration.iter().all(|value| value.is_finite()) {
            return Err(SolveError::NonFiniteDerivative);
        }
        for (velocity, &acceleration) in velocity.iter_mut().zip(&*acceleration) {
            *velocity += kick * step * acceleration;
        }
        if stage + 1 < tableau.stages() {
            stage_time += kick * step;
        }
    }
    Ok(tableau.stages())
}

struct SymplecticRecorder<'a> {
    times: Vec<f64>,
    positions: Vec<f64>,
    velocities: Vec<f64>,
    dimension: usize,
    save_at: &'a [f64],
    next_save: usize,
    save_mode: SaveMode,
    interpolation_position: Vec<f64>,
    interpolation_velocity: Vec<f64>,
    dense_segments: Vec<SymplecticDenseSegment>,
    retain_dense_output: bool,
}

impl<'a> SymplecticRecorder<'a> {
    fn new(position: &[f64], velocity: &[f64], time: f64, options: &'a SolveOptions) -> Self {
        let save_initial = options.save_at.is_empty() || options.save_at.first() == Some(&time);
        let capacity = options.save_at.len().max(2);
        let mut recorder = Self {
            times: Vec::with_capacity(capacity),
            positions: Vec::with_capacity(capacity * position.len()),
            velocities: Vec::with_capacity(capacity * velocity.len()),
            dimension: position.len(),
            save_at: &options.save_at,
            next_save: usize::from(!options.save_at.is_empty() && save_initial),
            save_mode: options.save,
            interpolation_position: if options.save_at.is_empty() {
                Vec::new()
            } else {
                vec![0.0; position.len()]
            },
            interpolation_velocity: if options.save_at.is_empty() {
                Vec::new()
            } else {
                vec![0.0; velocity.len()]
            },
            dense_segments: Vec::new(),
            retain_dense_output: options.retain_dense_output,
        };
        if save_initial {
            recorder.push_unique(time, position, velocity);
        }
        recorder
    }

    #[allow(clippy::too_many_arguments)]
    fn record_step(
        &mut self,
        previous_position: &[f64],
        previous_velocity: &[f64],
        previous_time: f64,
        position: &[f64],
        velocity: &[f64],
        time: f64,
        final_time: bool,
    ) -> Result<(), SolveError> {
        let segment = SymplecticDenseSegment::new(
            previous_time,
            time,
            previous_position,
            position,
            previous_velocity,
            velocity,
        );
        if self.retain_dense_output {
            self.dense_segments.push(segment.clone());
        }
        if self.save_at.is_empty() {
            if self.save_mode == SaveMode::EveryStep || final_time {
                self.push_unique(time, position, velocity);
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
            segment
                .interpolate(
                    target,
                    &mut self.interpolation_position,
                    &mut self.interpolation_velocity,
                )
                .ok_or(SolveError::DenseOutputFailed)?;
            self.times.push(target);
            self.positions
                .extend_from_slice(&self.interpolation_position);
            self.velocities
                .extend_from_slice(&self.interpolation_velocity);
            self.next_save += 1;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn record_callback(
        &mut self,
        time: f64,
        before_position: &[f64],
        before_velocity: &[f64],
        after_position: &[f64],
        after_velocity: &[f64],
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
            self.push_unique(canonical_time, before_position, before_velocity);
        }
        if save_after {
            if save_before {
                self.push(canonical_time, after_position, after_velocity);
            } else {
                self.push_unique(canonical_time, after_position, after_velocity);
            }
        }
    }

    fn synchronize_endpoint(&mut self, time: f64, position: &[f64], velocity: &[f64]) {
        if self
            .times
            .last()
            .is_some_and(|saved| times_are_representably_equal(*saved, time))
        {
            let start = self.positions.len() - self.dimension;
            self.positions[start..].copy_from_slice(position);
            self.velocities[start..].copy_from_slice(velocity);
        }
    }

    fn push_unique(&mut self, time: f64, position: &[f64], velocity: &[f64]) {
        if self
            .times
            .last()
            .is_some_and(|saved| times_are_representably_equal(*saved, time))
        {
            let start = self.positions.len() - self.dimension;
            self.positions[start..].copy_from_slice(position);
            self.velocities[start..].copy_from_slice(velocity);
        } else {
            self.push(time, position, velocity);
        }
    }

    fn push(&mut self, time: f64, position: &[f64], velocity: &[f64]) {
        debug_assert_eq!(position.len(), self.dimension);
        debug_assert_eq!(velocity.len(), self.dimension);
        self.times.push(time);
        self.positions.extend_from_slice(position);
        self.velocities.extend_from_slice(velocity);
    }

    fn finish(self, rhs_evaluations: usize, state_shape: IxDyn) -> SymplecticSolution {
        SymplecticSolution {
            times: self.times,
            positions: self.positions,
            velocities: self.velocities,
            dimension: self.dimension,
            state_shape,
            rhs_evaluations,
            dense_segments: self.dense_segments,
        }
    }
}

fn interpolate(current: &[f64], previous: &[f64], fraction: f64, output: &mut [f64]) {
    for ((output, previous), current) in output.iter_mut().zip(previous).zip(current) {
        *output = previous + fraction * (current - previous);
    }
}
