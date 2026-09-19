use crate::callback::CallbackOutcome;
use crate::event::{times_are_numerically_equal, times_are_representably_equal};
use crate::solution::{
    interpolate_value, interpolation_fraction, validate_finite_partitioned_interpolation,
};
use crate::{InterpolationError, SaveMode, SolveError, SolveOptions, SolverStats};
use ndarray::{ArrayD, ArrayViewD, Dimension, IxDyn};

/// A trajectory returned by [`super::solve_symplectic`].
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
    stats: SolverStats,
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

    /// Interpolates shape-preserving `(velocity, position)` arrays.
    pub fn interpolate_array(&self, time: f64) -> Option<(ArrayD<f64>, ArrayD<f64>)> {
        self.try_interpolate_array(time).ok()
    }

    /// Interpolates shape-preserving `(velocity, position)` arrays and retains
    /// interpolation errors.
    pub fn try_interpolate_array(
        &self,
        time: f64,
    ) -> Result<(ArrayD<f64>, ArrayD<f64>), InterpolationError> {
        let (velocity, position) = self.try_interpolate(time)?;
        let reshape = |values| {
            ArrayD::from_shape_vec(self.state_shape.clone(), values).map_err(|_| {
                InterpolationError::InvalidSegmentData {
                    context: "symplectic solution state shape",
                }
            })
        };
        Ok((reshape(velocity)?, reshape(position)?))
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

    /// Solver work counters. Acceleration evaluations contribute to
    /// `rhs_evaluations`; the identity position rate `q' = v` is not evaluated
    /// as a user function.
    pub fn stats(&self) -> SolverStats {
        self.stats
    }

    /// Number of acceleration evaluations.
    ///
    /// This is a convenience accessor for [`Self::stats`].
    pub fn rhs_evaluations(&self) -> usize {
        self.stats.rhs_evaluations
    }

    /// Interpolates `(velocity, position)` at a covered time.
    ///
    /// Retained segments use cubic-Hermite position interpolation consistent
    /// with `q' = v` and linear velocity interpolation. Saved-only solutions
    /// retain the stable linear compatibility fallback.
    pub fn interpolate(&self, time: f64) -> Option<(Vec<f64>, Vec<f64>)> {
        self.try_interpolate(time).ok()
    }

    /// Interpolates `(velocity, position)` and reports why the query fails.
    pub fn try_interpolate(&self, time: f64) -> Result<(Vec<f64>, Vec<f64>), InterpolationError> {
        let mut velocity = vec![0.0; self.dimension];
        let mut position = vec![0.0; self.dimension];
        self.try_interpolate_into(time, &mut velocity, &mut position)?;
        Ok((velocity, position))
    }

    /// Interpolates into caller-owned velocity and position buffers.
    ///
    /// Both buffers must contain exactly [`Self::dimension`] entries. On
    /// success they are overwritten with `(velocity, position)` at `time`; on
    /// failure their contents are unspecified.
    pub fn try_interpolate_into(
        &self,
        time: f64,
        velocity: &mut [f64],
        position: &mut [f64],
    ) -> Result<(), InterpolationError> {
        if velocity.len() != self.dimension || position.len() != self.dimension {
            return Err(InterpolationError::DimensionMismatch);
        }
        if !time.is_finite() {
            return Err(InterpolationError::NonFiniteTime);
        }
        if self.times.is_empty() {
            return Err(InterpolationError::EmptySolution);
        }
        for (index, &saved_time) in self.times.iter().enumerate().rev() {
            if time == saved_time {
                let saved_velocity =
                    self.velocity(index)
                        .ok_or(InterpolationError::InvalidSegmentData {
                            context: "saved symplectic velocity",
                        })?;
                let saved_position =
                    self.position(index)
                        .ok_or(InterpolationError::InvalidSegmentData {
                            context: "saved symplectic position",
                        })?;
                velocity.copy_from_slice(saved_velocity);
                position.copy_from_slice(saved_position);
                return validate_finite_partitioned_interpolation(
                    velocity,
                    position,
                    "saved symplectic state",
                );
            }
        }
        for segment in &self.dense_segments {
            if segment.contains(time) {
                segment.interpolate(time, position, velocity).ok_or(
                    InterpolationError::InvalidSegmentData {
                        context: "symplectic dense segment",
                    },
                )?;
                return validate_finite_partitioned_interpolation(
                    velocity,
                    position,
                    "symplectic dense segment",
                );
            }
        }
        for index in 1..self.times.len() {
            let left = self.times[index - 1];
            let right = self.times[index];
            if between(time, left, right) && left != right {
                let fraction = interpolation_fraction(time, left, right).clamp(0.0, 1.0);
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
                    position,
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
                    velocity,
                );
                return validate_finite_partitioned_interpolation(
                    velocity,
                    position,
                    "saved symplectic interpolation",
                );
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
            velocity[index] =
                interpolate_value(self.start_velocity[index], self.end_velocity[index], theta);
        }
        Some(())
    }
}

fn partition(values: &[f64], dimension: usize, index: usize) -> Option<&[f64]> {
    let start = index.checked_mul(dimension)?;
    let end = start.checked_add(dimension)?;
    values.get(start..end)
}

pub(super) struct SymplecticRecorder<'a> {
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
    pub(super) fn new(
        position: &[f64],
        velocity: &[f64],
        time: f64,
        options: &'a SolveOptions,
    ) -> Self {
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
    pub(super) fn record_step(
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
    pub(super) fn record_callback(
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

    pub(super) fn synchronize_endpoint(&mut self, time: f64, position: &[f64], velocity: &[f64]) {
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

    pub(super) fn finish(self, stats: SolverStats, state_shape: IxDyn) -> SymplecticSolution {
        SymplecticSolution {
            times: self.times,
            positions: self.positions,
            velocities: self.velocities,
            dimension: self.dimension,
            state_shape,
            stats,
            dense_segments: self.dense_segments,
        }
    }
}

fn interpolate(current: &[f64], previous: &[f64], fraction: f64, output: &mut [f64]) {
    for ((output, previous), current) in output.iter_mut().zip(previous).zip(current) {
        *output = interpolate_value(*previous, *current, fraction);
    }
}

#[cfg(test)]
mod tests;
