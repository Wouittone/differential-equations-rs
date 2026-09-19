use super::driver::interpolate;
use crate::solution::{
    interpolate_value, interpolation_fraction, validate_finite_partitioned_interpolation,
};
use crate::{InterpolationError, SolverStats};
use ndarray::IxDyn;

/// A saved trajectory for a second-order ODE.
///
/// Callbacks configured with [`crate::callback::CallbackSave::Both`] produce adjacent states
/// at the same time, ordered before-effect then after-effect. Exact
/// interpolation at that time returns the latter state.
#[derive(Clone, Debug, PartialEq)]
pub struct SecondOrderSolution {
    pub(super) times: Vec<f64>,
    pub(super) velocities: Vec<f64>,
    pub(super) positions: Vec<f64>,
    pub(super) dimension: usize,
    pub(super) state_shape: IxDyn,
    pub(super) stats: SolverStats,
    pub(super) dense_segments: Vec<PartitionedDenseSegment>,
}

impl SecondOrderSolution {
    /// Constructs a solution from already-saved velocity and position states.
    ///
    /// Each partition contains one flattened row-major state for every entry
    /// in `times`. `state_shape` is the shared logical ndarray shape; an empty
    /// shape denotes a scalar. The constructor checks shape arithmetic, buffer
    /// lengths, finiteness, and monotonic time order. The resulting solution
    /// has no retained method-specific dense segments, so interpolation
    /// between saved states uses the saved-state fallback.
    ///
    /// This is the construction seam for downstream
    /// [`crate::solvers::second_order::SecondOrderOdeAlgorithm`]
    /// implementations.
    pub fn from_saved(
        times: Vec<f64>,
        velocities: Vec<f64>,
        positions: Vec<f64>,
        state_shape: &[usize],
        stats: SolverStats,
    ) -> Result<Self, crate::SolutionConstructionError> {
        let dimension = crate::solution::validate_saved_solution(
            &times,
            state_shape,
            &[&velocities, &positions],
        )?;
        Ok(Self {
            times,
            velocities,
            positions,
            dimension,
            state_shape: IxDyn(state_shape),
            stats,
            dense_segments: Vec::new(),
        })
    }

    /// Saved times in integration order.
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    /// Number of scalar components in each position or velocity partition.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// All saved velocities in contiguous row-major storage.
    pub fn velocity_values(&self) -> &[f64] {
        &self.velocities
    }

    /// All saved positions in contiguous row-major storage.
    pub fn position_values(&self) -> &[f64] {
        &self.positions
    }

    /// Saved velocity at a time index.
    pub fn velocity(&self, index: usize) -> Option<&[f64]> {
        partition(&self.velocities, self.dimension, index)
    }

    /// Saved position at a time index.
    pub fn position(&self, index: usize) -> Option<&[f64]> {
        partition(&self.positions, self.dimension, index)
    }

    /// Last saved velocity.
    pub fn last_velocity(&self) -> &[f64] {
        let start = self.velocities.len() - self.dimension;
        &self.velocities[start..]
    }

    /// Last saved position.
    pub fn last_position(&self) -> &[f64] {
        let start = self.positions.len() - self.dimension;
        &self.positions[start..]
    }

    /// Solver work counters. Acceleration evaluations contribute to
    /// `rhs_evaluations`; the identity position rate `q' = v` is not evaluated
    /// as a user function.
    pub fn stats(&self) -> SolverStats {
        self.stats
    }

    pub(super) fn set_state_shape_checked(
        &mut self,
        state_shape: &[usize],
    ) -> Result<(), crate::SolutionConstructionError> {
        if crate::solution::checked_state_dimension(state_shape)? != self.dimension {
            return Err(crate::SolutionConstructionError::DimensionMismatch);
        }
        self.state_shape = IxDyn(state_shape);
        Ok(())
    }

    /// Interpolates `(velocity, position)` at a time covered by the solution.
    ///
    /// When dense output was retained, positions use a cubic Hermite segment
    /// consistent with `q' = v` and velocities use a stable linear segment.
    /// Without retained segments, saved states are linearly interpolated.
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
                            context: "saved second-order velocity",
                        })?;
                let saved_position =
                    self.position(index)
                        .ok_or(InterpolationError::InvalidSegmentData {
                            context: "saved second-order position",
                        })?;
                velocity.copy_from_slice(saved_velocity);
                position.copy_from_slice(saved_position);
                return validate_finite_partitioned_interpolation(
                    velocity,
                    position,
                    "saved second-order state",
                );
            }
        }
        for segment in &self.dense_segments {
            if segment.contains(time) {
                segment.interpolate(time, velocity, position).ok_or(
                    InterpolationError::InvalidSegmentData {
                        context: "second-order dense segment",
                    },
                )?;
                return validate_finite_partitioned_interpolation(
                    velocity,
                    position,
                    "second-order dense segment",
                );
            }
        }
        for index in 1..self.times.len() {
            let left = self.times[index - 1];
            let right = self.times[index];
            if between(time, left, right) && left != right {
                let fraction = interpolation_fraction(time, left, right).clamp(0.0, 1.0);
                interpolate(
                    self.velocity(index)
                        .ok_or(InterpolationError::InvalidSegmentData {
                            context: "saved second-order velocity",
                        })?,
                    self.velocity(index - 1)
                        .ok_or(InterpolationError::InvalidSegmentData {
                            context: "saved second-order velocity",
                        })?,
                    fraction,
                    velocity,
                );
                interpolate(
                    self.position(index)
                        .ok_or(InterpolationError::InvalidSegmentData {
                            context: "saved second-order position",
                        })?,
                    self.position(index - 1)
                        .ok_or(InterpolationError::InvalidSegmentData {
                            context: "saved second-order position",
                        })?,
                    fraction,
                    position,
                );
                return validate_finite_partitioned_interpolation(
                    velocity,
                    position,
                    "saved second-order interpolation",
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
pub(super) struct PartitionedDenseSegment {
    start_time: f64,
    end_time: f64,
    start_velocity: Vec<f64>,
    end_velocity: Vec<f64>,
    start_position: Vec<f64>,
    end_position: Vec<f64>,
}

impl PartitionedDenseSegment {
    pub(super) fn new(
        start_time: f64,
        end_time: f64,
        start_velocity: &[f64],
        end_velocity: &[f64],
        start_position: &[f64],
        end_position: &[f64],
    ) -> Self {
        Self {
            start_time,
            end_time,
            start_velocity: start_velocity.to_vec(),
            end_velocity: end_velocity.to_vec(),
            start_position: start_position.to_vec(),
            end_position: end_position.to_vec(),
        }
    }

    pub(super) fn contains(&self, time: f64) -> bool {
        between(time, self.start_time, self.end_time)
    }

    pub(super) fn interpolate(
        &self,
        time: f64,
        velocity: &mut [f64],
        position: &mut [f64],
    ) -> Option<()> {
        if !self.contains(time)
            || velocity.len() != self.start_velocity.len()
            || position.len() != self.start_position.len()
        {
            return None;
        }
        if time == self.start_time {
            velocity.copy_from_slice(&self.start_velocity);
            position.copy_from_slice(&self.start_position);
            return Some(());
        }
        if time == self.end_time {
            velocity.copy_from_slice(&self.end_velocity);
            position.copy_from_slice(&self.end_position);
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
        for index in 0..velocity.len() {
            velocity[index] =
                interpolate_value(self.start_velocity[index], self.end_velocity[index], theta);
            position[index] = h00 * self.start_position[index]
                + h10 * step * self.start_velocity[index]
                + h01 * self.end_position[index]
                + h11 * step * self.end_velocity[index];
        }
        Some(())
    }
}

fn partition(values: &[f64], dimension: usize, index: usize) -> Option<&[f64]> {
    let start = index.checked_mul(dimension)?;
    let end = start.checked_add(dimension)?;
    values.get(start..end)
}
