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
    pub(super) portable_segments: Vec<(crate::PortableDenseSegment, crate::PortableDenseSegment)>,
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
            portable_segments: Vec::new(),
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
        let after = self.saved_partition(time);
        if after > 0 && self.times[after - 1] == time {
            velocity.copy_from_slice(self.velocity(after - 1).ok_or(
                InterpolationError::InvalidSegmentData {
                    context: "saved second-order velocity",
                },
            )?);
            position.copy_from_slice(self.position(after - 1).ok_or(
                InterpolationError::InvalidSegmentData {
                    context: "saved second-order position",
                },
            )?);
            return validate_finite_partitioned_interpolation(
                velocity,
                position,
                "saved second-order state",
            );
        }
        if let Some(pair) = self.portable_segment_at(time) {
            pair.0.interpolate_into(time, velocity)?;
            pair.1.interpolate_into(time, position)?;
            return Ok(());
        }
        if let Some(segment) = self.dense_segment_at(time) {
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
        if after > 0 && after < self.times.len() {
            let fraction = interpolation_fraction(time, self.times[after - 1], self.times[after])
                .clamp(0.0, 1.0);
            interpolate(
                self.velocity(after)
                    .ok_or(InterpolationError::DimensionMismatch)?,
                self.velocity(after - 1)
                    .ok_or(InterpolationError::DimensionMismatch)?,
                fraction,
                velocity,
            );
            interpolate(
                self.position(after)
                    .ok_or(InterpolationError::DimensionMismatch)?,
                self.position(after - 1)
                    .ok_or(InterpolationError::DimensionMismatch)?,
                fraction,
                position,
            );
            return validate_finite_partitioned_interpolation(
                velocity,
                position,
                "linear second-order state",
            );
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

/// Portable second-order trajectory with separate velocity and position data.
/// The two partition trajectories must have identical times and logical shapes.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SecondOrderSolutionData {
    /// Schema version, currently one.
    pub version: u32,
    /// Velocity trajectory and its actual interpolation quality.
    pub velocity: crate::SolutionData,
    /// Position trajectory and its actual interpolation quality.
    pub position: crate::SolutionData,
}

impl SecondOrderSolution {
    /// Exports both partitions, preserving dense coefficients and quality.
    /// Native RKN retained velocity interpolation is explicitly linear while
    /// its position interpolation uses cubic Hermite data.
    pub fn export_data(&self) -> Result<SecondOrderSolutionData, InterpolationError> {
        let shape = ndarray::Dimension::slice(&self.state_shape).to_vec();
        let mut velocity = crate::SolutionData {
            version: 1,
            times: self.times.clone(),
            values: self.velocities.clone(),
            state_shape: shape.clone(),
            segments: Vec::new(),
            statistics: self.stats,
        };
        let mut position = crate::SolutionData {
            version: 1,
            times: self.times.clone(),
            values: self.positions.clone(),
            state_shape: shape,
            segments: Vec::new(),
            statistics: self.stats,
        };
        for segment in &self.dense_segments {
            let (v, q) = segment.portable()?;
            velocity.segments.push(v);
            position.segments.push(q);
        }
        for (v, q) in &self.portable_segments {
            velocity.segments.push(v.clone());
            position.segments.push(q.clone());
        }
        Ok(SecondOrderSolutionData {
            version: 1,
            velocity,
            position,
        })
    }
    /// Imports ordered matching partition trajectories with validated dense data.
    pub fn from_data(data: SecondOrderSolutionData) -> Result<Self, InterpolationError> {
        if data.version != 1
            || data.velocity.times != data.position.times
            || data.velocity.state_shape != data.position.state_shape
            || data.velocity.segments.len() != data.position.segments.len()
            || data.velocity.statistics != data.position.statistics
        {
            return Err(InterpolationError::InvalidSegmentData {
                context: "second-order portable partition mismatch or version",
            });
        }
        // The ordinary import applies monotonicity, dimension and segment-domain
        // checks identically to both partitions, including sparse requested saves.
        let velocity = crate::Solution::from_data(data.velocity)?;
        let position = crate::Solution::from_data(data.position)?;
        let mut solution = Self::from_saved(
            velocity.times().to_vec(),
            velocity.values().to_vec(),
            position.values().to_vec(),
            velocity.state_shape(),
            velocity.stats(),
        )
        .map_err(|_| InterpolationError::InvalidSegmentData {
            context: "second-order portable saved states",
        })?;
        for (v, q) in velocity
            .export_dense_segments()?
            .into_iter()
            .zip(position.export_dense_segments()?)
        {
            if v.time_bounds() != q.time_bounds() {
                return Err(InterpolationError::InvalidSegmentData {
                    context: "second-order portable segment boundaries differ",
                });
            }
            solution.portable_segments.push((v, q));
        }
        Ok(solution)
    }
    /// Returns `(velocity quality, position quality)` at a query time.
    /// Exact callback times report exact saved states for both partitions.
    pub fn interpolation_quality(
        &self,
        time: f64,
    ) -> Result<(crate::InterpolationQuality, crate::InterpolationQuality), InterpolationError>
    {
        use crate::InterpolationQuality as Q;
        if !time.is_finite() {
            return Err(InterpolationError::NonFiniteTime);
        }
        let after = self.saved_partition(time);
        if after > 0 && self.times[after - 1] == time {
            return Ok((Q::ExactSavedState, Q::ExactSavedState));
        }
        if let Some(pair) = self.portable_segment_at(time) {
            return Ok((pair.0.quality(), pair.1.quality()));
        }
        if self.dense_segment_at(time).is_some() {
            return Ok((Q::Linear, Q::MethodSpecific));
        }
        if after > 0 && after < self.times.len() {
            Ok((Q::Linear, Q::Linear))
        } else {
            Err(InterpolationError::OutsideTimeSpan)
        }
    }
    /// Requires retained method-quality interpolation for both partitions.
    /// Current native RKN velocity interpolation is linear and therefore fails
    /// this requirement between saved states instead of silently downgrading.
    pub fn try_interpolate_method_into(
        &self,
        time: f64,
        velocity: &mut [f64],
        position: &mut [f64],
    ) -> Result<(), InterpolationError> {
        let (v, q) = self.interpolation_quality(time)?;
        if v == crate::InterpolationQuality::Linear || q == crate::InterpolationQuality::Linear {
            return Err(InterpolationError::InvalidSegmentData {
                context: "method-specific second-order dense output unavailable for a partition",
            });
        }
        self.try_interpolate_into(time, velocity, position)
    }
    fn portable_segment_at(
        &self,
        time: f64,
    ) -> Option<&(crate::PortableDenseSegment, crate::PortableDenseSegment)> {
        let forward = self.portable_segments.first()?.0.time_bounds().0
            <= self.portable_segments.last()?.0.time_bounds().1;
        let index = self.portable_segments.partition_point(|pair| {
            let end = pair.0.time_bounds().1;
            if forward { end < time } else { end > time }
        });
        self.portable_segments
            .get(index)
            .filter(|pair| pair.0.contains(time))
    }
    fn saved_partition(&self, time: f64) -> usize {
        let forward = self.times.first() <= self.times.last();
        self.times
            .partition_point(|&t| if forward { t <= time } else { t >= time })
    }
    fn dense_segment_at(&self, time: f64) -> Option<&PartitionedDenseSegment> {
        let forward =
            self.dense_segments.first()?.start_time <= self.dense_segments.last()?.end_time;
        let index = self.dense_segments.partition_point(|s| {
            if forward {
                s.end_time < time
            } else {
                s.end_time > time
            }
        });
        self.dense_segments.get(index).filter(|s| s.contains(time))
    }
}
impl PartitionedDenseSegment {
    fn portable(
        &self,
    ) -> Result<(crate::PortableDenseSegment, crate::PortableDenseSegment), InterpolationError>
    {
        let n = self.start_position.len();
        let h = self.end_time - self.start_time;
        let mut vc = vec![0.0; 2 * n];
        let mut qc = vec![0.0; 4 * n];
        for i in 0..n {
            vc[i] = self.start_velocity[i];
            vc[n + i] = self.end_velocity[i] - self.start_velocity[i];
            qc[i] = self.start_position[i];
            qc[n + i] = h * self.start_velocity[i];
            qc[2 * n + i] = 3.0 * (self.end_position[i] - self.start_position[i])
                - h * (2.0 * self.start_velocity[i] + self.end_velocity[i]);
            qc[3 * n + i] = 2.0 * (self.start_position[i] - self.end_position[i])
                + h * (self.start_velocity[i] + self.end_velocity[i]);
        }
        let data = |coefficients, end_state, quality| crate::DenseSegmentData {
            version: 1,
            start_time: self.start_time,
            end_time: self.end_time,
            bound_time: self.end_time,
            dimension: n,
            coefficients,
            end_state,
            bound_state: None,
            quality,
        };
        Ok((
            crate::PortableDenseSegment::from_data(data(
                vc,
                self.end_velocity.clone(),
                crate::InterpolationQuality::Linear,
            ))?,
            crate::PortableDenseSegment::from_data(data(
                qc,
                self.end_position.clone(),
                crate::InterpolationQuality::MethodSpecific,
            ))?,
        ))
    }
}
#[cfg(feature = "serde")]
impl serde::Serialize for SecondOrderSolution {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serde::Serialize::serialize(
            &self.export_data().map_err(serde::ser::Error::custom)?,
            serializer,
        )
    }
}
#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for SecondOrderSolution {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::from_data(SecondOrderSolutionData::deserialize(deserializer)?)
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod storage_tests {
    use super::PartitionedDenseSegment;
    #[test]
    fn native_rkn_segment_keeps_its_original_storage_size() {
        // Original layout: two times and four owned endpoint vectors. Portable
        // imports must not add a pointer/tag to every retained native segment.
        assert_eq!(
            std::mem::size_of::<PartitionedDenseSegment>(),
            2 * std::mem::size_of::<f64>() + 4 * std::mem::size_of::<Vec<f64>>()
        );
    }
}
