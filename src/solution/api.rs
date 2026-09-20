use super::{
    DenseSegment, InterpolationError, OwnedDenseSegment, SolutionConstructionError, SolverStats,
};

/// A saved ODE trajectory.
///
/// States are kept in one row-major allocation. The state at saved time `i`
/// occupies `values[i * dimension..(i + 1) * dimension]`.
/// Callbacks configured with [`crate::CallbackSave::Both`] produce adjacent
/// states with the same time, ordered before-effect then after-effect. Exact
/// interpolation at that time returns the latter state.
#[derive(Clone, Debug, PartialEq)]
pub struct Solution {
    times: Vec<f64>,
    values: Vec<f64>,
    dimension: usize,
    state_shape: ndarray::IxDyn,
    stats: SolverStats,
    dense_segments: Vec<OwnedDenseSegment>,
}

impl Solution {
    /// Constructs a solution from already-saved states.
    ///
    /// `values` contains one flattened row-major state for every entry in
    /// `times`. `state_shape` is the logical ndarray shape of each state; an
    /// empty shape denotes a scalar. The constructor checks shape arithmetic,
    /// buffer lengths, finiteness, and monotonic time order. The resulting
    /// solution has no method-specific dense segments, so interpolation
    /// between saved states is linear.
    ///
    /// This is the construction seam for downstream [`crate::OdeAlgorithm`]
    /// implementations.
    ///
    /// # Examples
    ///
    /// ```
    /// use differential_equations::{Solution, SolverStats};
    ///
    /// let solution = Solution::from_saved(
    ///     vec![0.0, 1.0],
    ///     vec![1.0, 2.0, 3.0, 4.0],
    ///     &[2],
    ///     SolverStats::default(),
    /// )?;
    /// assert_eq!(solution.state(1), Some([3.0, 4.0].as_slice()));
    /// # Ok::<(), differential_equations::SolutionConstructionError>(())
    /// ```
    pub fn from_saved(
        times: Vec<f64>,
        values: Vec<f64>,
        state_shape: &[usize],
        stats: SolverStats,
    ) -> Result<Self, SolutionConstructionError> {
        let dimension = validate_saved_solution(&times, state_shape, &[&values])?;
        Ok(Self {
            times,
            values,
            dimension,
            state_shape: ndarray::IxDyn(state_shape),
            stats,
            dense_segments: Vec::new(),
        })
    }

    pub(crate) fn new(
        times: Vec<f64>,
        values: Vec<f64>,
        dimension: usize,
        stats: SolverStats,
    ) -> Self {
        debug_assert_eq!(values.len(), times.len() * dimension);
        Self {
            times,
            values,
            dimension,
            state_shape: ndarray::IxDyn(&[dimension]),
            stats,
            dense_segments: Vec::new(),
        }
    }

    pub(super) fn new_with_dense(
        times: Vec<f64>,
        values: Vec<f64>,
        dimension: usize,
        stats: SolverStats,
        dense_segments: Vec<OwnedDenseSegment>,
    ) -> Self {
        debug_assert_eq!(values.len(), times.len() * dimension);
        Self {
            times,
            values,
            dimension,
            state_shape: ndarray::IxDyn(&[dimension]),
            stats,
            dense_segments,
        }
    }

    /// Saved times in integration order.
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    /// All saved states in contiguous row-major storage.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Number of scalar components in each state.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the logical ndarray shape of each state.
    pub fn state_shape(&self) -> &[usize] {
        ndarray::Dimension::slice(&self.state_shape)
    }

    /// Returns a saved state by its time index.
    pub fn state(&self, index: usize) -> Option<&[f64]> {
        let start = index.checked_mul(self.dimension)?;
        let end = start.checked_add(self.dimension)?;
        self.values.get(start..end)
    }

    /// Returns a saved state as an ndarray view with its original shape.
    pub fn state_array(&self, index: usize) -> Option<ndarray::ArrayViewD<'_, f64>> {
        let state = self.state(index)?;
        ndarray::ArrayViewD::from_shape(self.state_shape.clone(), state).ok()
    }

    /// Returns the last saved state.
    pub fn last_state(&self) -> &[f64] {
        let start = self.values.len() - self.dimension;
        &self.values[start..]
    }

    /// Returns the last saved state as an ndarray view with its original shape.
    pub fn last_state_array(&self) -> ndarray::ArrayViewD<'_, f64> {
        ndarray::ArrayViewD::from_shape(self.state_shape.clone(), self.last_state())
            .expect("solution state shape must match its contiguous storage")
    }

    /// Interpolates the saved trajectory at `time`.
    ///
    /// Retained method-specific dense output is preferred. When none covers the
    /// requested time, the query uses a stable linear interpolation between
    /// adjacent saved states. Use [`try_interpolate`](Self::try_interpolate) to
    /// retain the reason a query fails.
    pub fn interpolate(&self, time: f64) -> Option<Vec<f64>> {
        self.try_interpolate(time).ok()
    }

    /// Interpolates the saved trajectory and reports why a query cannot be served.
    pub fn try_interpolate(&self, time: f64) -> Result<Vec<f64>, InterpolationError> {
        let mut output = vec![0.0; self.dimension];
        self.try_interpolate_into(time, &mut output)?;
        Ok(output)
    }

    /// Interpolates into caller-owned contiguous storage without allocating.
    ///
    /// `output` must contain exactly [`Self::dimension`] entries. On success it
    /// is overwritten with the state at `time`; on failure its contents are
    /// unspecified.
    pub fn try_interpolate_into(
        &self,
        time: f64,
        output: &mut [f64],
    ) -> Result<(), InterpolationError> {
        if output.len() != self.dimension {
            return Err(InterpolationError::DimensionMismatch);
        }
        if !time.is_finite() {
            return Err(InterpolationError::NonFiniteTime);
        }
        if self.times.is_empty() {
            return Err(InterpolationError::EmptySolution);
        }
        let forward = self.times.first() <= self.times.last();
        let after = self.times.partition_point(|&saved| {
            if forward {
                saved <= time
            } else {
                saved >= time
            }
        });
        if after > 0 && self.times[after - 1] == time {
            output.copy_from_slice(self.state(after - 1).ok_or(
                InterpolationError::InvalidSegmentData {
                    context: "saved solution state",
                },
            )?);
            return finite_interpolation(output, "saved solution state");
        }
        let dense_index = self.dense_segments.partition_point(|segment| {
            let (start, end) = segment.time_bounds();
            if start <= end { end < time } else { end > time }
        });
        if let Some(segment) = self
            .dense_segments
            .get(dense_index)
            .filter(|s| s.contains(time))
        {
            segment.interpolate(time, output)?;
            return finite_interpolation(output, "dense output");
        }
        if after > 0 && after < self.times.len() {
            let left = self.times[after - 1];
            let right = self.times[after];
            let fraction = interpolation_fraction(time, left, right).clamp(0.0, 1.0);
            let previous = self
                .state(after - 1)
                .ok_or(InterpolationError::InvalidSegmentData {
                    context: "saved solution state",
                })?;
            let current = self
                .state(after)
                .ok_or(InterpolationError::InvalidSegmentData {
                    context: "saved solution state",
                })?;
            for ((output, &previous), &current) in output.iter_mut().zip(previous).zip(current) {
                *output = interpolate_value(previous, current, fraction);
            }
            return finite_interpolation(output, "linear");
        }
        Err(InterpolationError::OutsideTimeSpan)
    }

    /// Interpolates the trajectory into an ndarray with the original state shape.
    pub fn interpolate_array(&self, time: f64) -> Option<ndarray::ArrayD<f64>> {
        self.try_interpolate_array(time).ok()
    }

    /// Interpolates into a shaped ndarray and retains interpolation errors.
    pub fn try_interpolate_array(
        &self,
        time: f64,
    ) -> Result<ndarray::ArrayD<f64>, InterpolationError> {
        let values = self.try_interpolate(time)?;
        ndarray::ArrayD::from_shape_vec(self.state_shape.clone(), values).map_err(|_| {
            InterpolationError::InvalidSegmentData {
                context: "solution state shape",
            }
        })
    }

    /// Solver work counters.
    pub fn stats(&self) -> SolverStats {
        self.stats
    }

    pub(crate) fn set_state_shape(&mut self, state_shape: &[usize]) {
        debug_assert_eq!(state_shape.iter().product::<usize>(), self.dimension);
        self.state_shape = ndarray::IxDyn(state_shape);
    }

    pub(crate) fn set_state_shape_checked(
        &mut self,
        state_shape: &[usize],
    ) -> Result<(), SolutionConstructionError> {
        if checked_state_dimension(state_shape)? != self.dimension {
            return Err(SolutionConstructionError::DimensionMismatch);
        }
        self.state_shape = ndarray::IxDyn(state_shape);
        Ok(())
    }
}

pub(crate) fn checked_state_dimension(
    state_shape: &[usize],
) -> Result<usize, SolutionConstructionError> {
    let dimension = state_shape
        .iter()
        .try_fold(1_usize, |dimension, &extent| dimension.checked_mul(extent));
    let dimension = dimension.ok_or(SolutionConstructionError::DimensionOverflow)?;
    if dimension == 0 {
        return Err(SolutionConstructionError::EmptyState);
    }
    Ok(dimension)
}

pub(crate) fn validate_saved_solution(
    times: &[f64],
    state_shape: &[usize],
    partitions: &[&[f64]],
) -> Result<usize, SolutionConstructionError> {
    if times.is_empty() {
        return Err(SolutionConstructionError::EmptyTrajectory);
    }
    if !times.iter().all(|time| time.is_finite()) {
        return Err(SolutionConstructionError::NonFiniteTime);
    }

    let mut ordering = None;
    for pair in times.windows(2) {
        let current = pair[1]
            .partial_cmp(&pair[0])
            .expect("finite times must be comparable");
        if current == std::cmp::Ordering::Equal {
            continue;
        }
        if ordering.is_some_and(|ordering| ordering != current) {
            return Err(SolutionConstructionError::NonMonotonicTimes);
        }
        ordering = Some(current);
    }

    let dimension = checked_state_dimension(state_shape)?;
    let expected = times
        .len()
        .checked_mul(dimension)
        .ok_or(SolutionConstructionError::DimensionOverflow)?;
    if partitions
        .iter()
        .any(|partition| partition.len() != expected)
    {
        return Err(SolutionConstructionError::DimensionMismatch);
    }
    if !partitions
        .iter()
        .flat_map(|partition| partition.iter())
        .all(|value| value.is_finite())
    {
        return Err(SolutionConstructionError::NonFiniteState);
    }
    Ok(dimension)
}

pub(crate) fn interpolation_fraction(time: f64, left: f64, right: f64) -> f64 {
    if left.is_sign_negative() != right.is_sign_negative() {
        let scale = left.abs().max(right.abs());
        (time / scale - left / scale) / (right / scale - left / scale)
    } else {
        (time - left) / (right - left)
    }
}

pub(crate) fn interpolate_value(previous: f64, current: f64, fraction: f64) -> f64 {
    if previous.is_finite()
        && current.is_finite()
        && previous.is_sign_negative() != current.is_sign_negative()
    {
        previous * (1.0 - fraction) + current * fraction
    } else {
        previous + fraction * (current - previous)
    }
}

fn finite_interpolation(output: &[f64], context: &'static str) -> Result<(), InterpolationError> {
    output
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(InterpolationError::NonFiniteResult { context })
}

pub(crate) fn validate_finite_partitioned_interpolation(
    velocity: &[f64],
    position: &[f64],
    context: &'static str,
) -> Result<(), InterpolationError> {
    velocity
        .iter()
        .chain(position.iter())
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(InterpolationError::NonFiniteResult { context })
}

impl Solution {
    /// Exports method-specific dense segments as validated portable polynomials.
    ///
    /// Export allocates; interpolation of the resulting segments does not.
    /// Polynomial coefficient regrouping can change final rounding by a few ulps.
    pub fn export_dense_segments(
        &self,
    ) -> Result<Vec<crate::PortableDenseSegment>, InterpolationError> {
        let mut segments = Vec::new();
        for segment in &self.dense_segments {
            segments.extend(segment.portable()?);
        }
        Ok(segments)
    }
    /// Reports the actual interpolation quality at the requested time.
    pub fn interpolation_quality(
        &self,
        time: f64,
    ) -> Result<crate::InterpolationQuality, InterpolationError> {
        use crate::InterpolationQuality;
        if !time.is_finite() {
            return Err(InterpolationError::NonFiniteTime);
        }
        let forward = self.times.first() <= self.times.last();
        let after = self.times.partition_point(|&saved| {
            if forward {
                saved <= time
            } else {
                saved >= time
            }
        });
        if after > 0 && self.times[after - 1] == time {
            return Ok(InterpolationQuality::ExactSavedState);
        }
        let index = self.dense_segments.partition_point(|s| {
            let (start, end) = s.time_bounds();
            if start <= end { end < time } else { end > time }
        });
        if let Some(segment) = self.dense_segments.get(index).filter(|s| s.contains(time)) {
            return Ok(segment.quality());
        }
        if after > 0 && after < self.times.len() {
            Ok(InterpolationQuality::Linear)
        } else {
            Err(InterpolationError::OutsideTimeSpan)
        }
    }
    /// Interpolates only if the query has method-specific dense output or an exact saved state.
    pub fn try_interpolate_method_into(
        &self,
        time: f64,
        output: &mut [f64],
    ) -> Result<(), InterpolationError> {
        if self.interpolation_quality(time)? == crate::InterpolationQuality::Linear {
            return Err(InterpolationError::InvalidSegmentData {
                context: "method-specific dense output is unavailable",
            });
        }
        self.try_interpolate_into(time, output)
    }
}

impl Solution {
    /// Exports a versioned trajectory and portable dense output.
    pub fn export_data(&self) -> Result<crate::SolutionData, InterpolationError> {
        Ok(crate::SolutionData {
            version: 1,
            times: self.times.clone(),
            values: self.values.clone(),
            state_shape: self.state_shape().to_vec(),
            segments: self.export_dense_segments()?,
        })
    }
    /// Imports a validated trajectory; work counters start at their defaults.
    pub fn from_data(data: crate::SolutionData) -> Result<Self, InterpolationError> {
        if data.version != 1 {
            return Err(InterpolationError::InvalidSegmentData {
                context: "unsupported solution schema version",
            });
        }
        let mut solution = Self::from_saved(
            data.times,
            data.values,
            &data.state_shape,
            SolverStats::default(),
        )
        .map_err(|_| InterpolationError::InvalidSegmentData {
            context: "portable solution saved states",
        })?;
        let start = solution.times[0];
        let end = *solution.times.last().unwrap();
        let direction = if start < end {
            1.0
        } else if start > end {
            -1.0
        } else {
            data.segments
                .first()
                .map_or(1.0, |s| (s.data().end_time - s.data().start_time).signum())
        };
        let mut previous_end: Option<f64> = None;
        for segment in data.segments {
            let (left, right) = segment.time_bounds();
            if segment.dimension() != solution.dimension
                || previous_end.is_some_and(|previous| direction * (left - previous) < 0.0)
                || direction * (right - left) < 0.0
                || direction * (segment.data().end_time - segment.data().start_time) <= 0.0
            {
                return Err(InterpolationError::InvalidSegmentData {
                    context: "portable solution segment order, bounds, or dimensions",
                });
            }
            previous_end = Some(right);
            solution
                .dense_segments
                .push(OwnedDenseSegment::Portable(segment));
        }
        Ok(solution)
    }
}
