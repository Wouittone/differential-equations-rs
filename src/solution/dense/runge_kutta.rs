use super::{DenseSegment, validate_dense_query};
use crate::solution::InterpolationError;

/// Borrowed Runge--Kutta continuous extension for one accepted step.
///
/// Each coefficient row describes one stage weight as
/// `theta * (r0 + r1*theta + r2*theta^2 + ...)`. This matches the continuous
/// extension representation used by OrdinaryDiffEq's explicit RK methods.
pub(crate) struct BorrowedRungeKuttaSegment<'a> {
    start_time: f64,
    end_time: f64,
    start_state: &'a [f64],
    end_state: &'a [f64],
    stages: &'a [f64],
    dimension: usize,
    coefficients: RungeKuttaCoefficients,
}

/// Process-lifetime continuous-extension rows from a lazily materialized
/// resource tableau.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RungeKuttaCoefficients(&'static [Vec<f64>]);

impl RungeKuttaCoefficients {
    fn len(self) -> usize {
        self.0.len()
    }

    fn row(self, index: usize) -> &'static [f64] {
        &self.0[index]
    }

    fn rows_are_valid(self) -> bool {
        (0..self.len()).all(|index| {
            let row = self.row(index);
            !row.is_empty() && row.iter().all(|coefficient| coefficient.is_finite())
        })
    }
}

impl From<&'static [Vec<f64>]> for RungeKuttaCoefficients {
    fn from(rows: &'static [Vec<f64>]) -> Self {
        Self(rows)
    }
}

/// Owning Runge--Kutta continuous extension retained after a solve.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RungeKuttaSegment {
    start_time: f64,
    end_time: f64,
    bound_time: f64,
    start_state: Vec<f64>,
    end_state: Vec<f64>,
    stages: Vec<f64>,
    dimension: usize,
    coefficients: RungeKuttaCoefficients,
}

impl<'a> BorrowedRungeKuttaSegment<'a> {
    pub(crate) fn new(
        start_time: f64,
        end_time: f64,
        start_state: &'a [f64],
        end_state: &'a [f64],
        stages: &'a [f64],
        coefficients: impl Into<RungeKuttaCoefficients>,
    ) -> Result<Self, InterpolationError> {
        let coefficients = coefficients.into();
        let dimension = start_state.len();
        if !start_time.is_finite()
            || !end_time.is_finite()
            || end_time == start_time
            || dimension == 0
            || end_state.len() != dimension
            || coefficients.len() == 0
            || stages.len() != coefficients.len() * dimension
            || !coefficients.rows_are_valid()
            || !start_state.iter().all(|value| value.is_finite())
            || !end_state.iter().all(|value| value.is_finite())
            || !stages.iter().all(|value| value.is_finite())
        {
            return Err(InterpolationError::InvalidSegmentData {
                context: "Runge--Kutta segment",
            });
        }
        Ok(Self {
            start_time,
            end_time,
            start_state,
            end_state,
            stages,
            dimension,
            coefficients,
        })
    }

    pub(super) fn contains(&self, time: f64) -> bool {
        if !time.is_finite() {
            return false;
        }
        if self.start_time < self.end_time {
            (self.start_time..=self.end_time).contains(&time)
        } else {
            (self.end_time..=self.start_time).contains(&time)
        }
    }
}

impl RungeKuttaSegment {
    pub(super) fn time_bounds(&self) -> (f64, f64) {
        (self.start_time, self.bound_time)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        start_time: f64,
        end_time: f64,
        bound_time: f64,
        start_state: &[f64],
        end_state: &[f64],
        stages: &[f64],
        coefficients: impl Into<RungeKuttaCoefficients>,
    ) -> Result<Self, InterpolationError> {
        let coefficients = coefficients.into();
        let borrowed = BorrowedRungeKuttaSegment::new(
            start_time,
            end_time,
            start_state,
            end_state,
            stages,
            coefficients,
        )?;
        if !borrowed.contains(bound_time) {
            return Err(InterpolationError::InvalidSegmentData {
                context: "Runge--Kutta segment bound",
            });
        }
        Ok(Self {
            start_time,
            end_time,
            bound_time,
            start_state: start_state.to_vec(),
            end_state: end_state.to_vec(),
            stages: stages.to_vec(),
            dimension: start_state.len(),
            coefficients,
        })
    }

    pub(super) fn contains(&self, time: f64) -> bool {
        if !time.is_finite() {
            return false;
        }
        if self.start_time < self.bound_time {
            (self.start_time..=self.bound_time).contains(&time)
        } else {
            (self.bound_time..=self.start_time).contains(&time)
        }
    }
}

impl DenseSegment for BorrowedRungeKuttaSegment<'_> {
    fn interpolate(&self, time: f64, output: &mut [f64]) -> Result<(), InterpolationError> {
        validate_dense_query(self.contains(time), output.len(), self.dimension)?;
        if time == self.start_time {
            output.copy_from_slice(self.start_state);
            return Ok(());
        }
        if time == self.end_time {
            output.copy_from_slice(self.end_state);
            return Ok(());
        }

        interpolate_runge_kutta(
            self.start_time,
            self.end_time,
            self.start_state,
            self.stages,
            self.coefficients,
            time,
            output,
        )
    }
}

impl DenseSegment for RungeKuttaSegment {
    fn interpolate(&self, time: f64, output: &mut [f64]) -> Result<(), InterpolationError> {
        validate_dense_query(self.contains(time), output.len(), self.dimension)?;
        if time == self.start_time {
            output.copy_from_slice(&self.start_state);
            return Ok(());
        }
        if time == self.end_time {
            output.copy_from_slice(&self.end_state);
            return Ok(());
        }
        interpolate_runge_kutta(
            self.start_time,
            self.end_time,
            &self.start_state,
            &self.stages,
            self.coefficients,
            time,
            output,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn interpolate_runge_kutta(
    start_time: f64,
    end_time: f64,
    start_state: &[f64],
    stages: &[f64],
    coefficients: impl Into<RungeKuttaCoefficients>,
    time: f64,
    output: &mut [f64],
) -> Result<(), InterpolationError> {
    let coefficients = coefficients.into();
    let dimension = start_state.len();
    if output.len() != dimension {
        return Err(InterpolationError::DimensionMismatch);
    }
    if !time.is_finite() {
        return Err(InterpolationError::NonFiniteTime);
    }
    if end_time == start_time || stages.len() != coefficients.len() * dimension {
        return Err(InterpolationError::InvalidSegmentData {
            context: "Runge--Kutta interpolation stages",
        });
    }
    let step = end_time - start_time;
    let theta = (time - start_time) / step;
    output.copy_from_slice(start_state);
    for stage_index in 0..coefficients.len() {
        let polynomial = coefficients
            .row(stage_index)
            .iter()
            .rev()
            .fold(0.0, |value, coefficient| value * theta + coefficient);
        let weight = step * theta * polynomial;
        let stage_start = stage_index * dimension;
        for (value, derivative) in output
            .iter_mut()
            .zip(&stages[stage_start..stage_start + dimension])
        {
            *value += weight * derivative;
        }
    }
    Ok(())
}
