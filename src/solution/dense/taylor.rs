use super::{DenseSegment, validate_dense_query};
use crate::solution::InterpolationError;

/// Borrowed Taylor polynomial with normalized full-step coefficients.
pub(crate) struct BorrowedTaylorSegment<'a> {
    start_time: f64,
    end_time: f64,
    start_state: &'a [f64],
    end_state: &'a [f64],
    coefficients: &'a [f64],
    dimension: usize,
    order: usize,
}

/// Owning Taylor polynomial retained after a solve.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TaylorSegment {
    start_time: f64,
    end_time: f64,
    bound_time: f64,
    start_state: Vec<f64>,
    end_state: Vec<f64>,
    coefficients: Vec<f64>,
    dimension: usize,
    order: usize,
}

impl<'a> BorrowedTaylorSegment<'a> {
    pub(crate) fn new(
        start_time: f64,
        end_time: f64,
        start_state: &'a [f64],
        end_state: &'a [f64],
        coefficients: &'a [f64],
        order: usize,
    ) -> Result<Self, InterpolationError> {
        let dimension = start_state.len();
        if !start_time.is_finite()
            || !end_time.is_finite()
            || start_time == end_time
            || dimension == 0
            || end_state.len() != dimension
            || order == 0
            || coefficients.len() < (order + 1) * dimension
        {
            return Err(InterpolationError::InvalidSegmentData {
                context: "Taylor segment",
            });
        }
        Ok(Self {
            start_time,
            end_time,
            start_state,
            end_state,
            coefficients,
            dimension,
            order,
        })
    }

    pub(super) fn contains(&self, time: f64) -> bool {
        time.is_finite()
            && if self.start_time < self.end_time {
                (self.start_time..=self.end_time).contains(&time)
            } else {
                (self.end_time..=self.start_time).contains(&time)
            }
    }
}

impl TaylorSegment {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_bounded(
        start_time: f64,
        end_time: f64,
        bound_time: f64,
        start_state: &[f64],
        end_state: &[f64],
        coefficients: &[f64],
        order: usize,
    ) -> Result<Self, InterpolationError> {
        let borrowed = BorrowedTaylorSegment::new(
            start_time,
            end_time,
            start_state,
            end_state,
            coefficients,
            order,
        )?;
        if !borrowed.contains(bound_time) {
            return Err(InterpolationError::InvalidSegmentData {
                context: "Taylor segment bound",
            });
        }
        Ok(Self {
            start_time,
            end_time,
            bound_time,
            start_state: start_state.to_vec(),
            end_state: end_state.to_vec(),
            coefficients: coefficients[..(order + 1) * borrowed.dimension].to_vec(),
            dimension: borrowed.dimension,
            order,
        })
    }

    pub(super) fn contains(&self, time: f64) -> bool {
        time.is_finite()
            && if self.start_time < self.bound_time {
                (self.start_time..=self.bound_time).contains(&time)
            } else {
                (self.bound_time..=self.start_time).contains(&time)
            }
    }
}

impl DenseSegment for BorrowedTaylorSegment<'_> {
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
        interpolate_taylor(
            self.start_time,
            self.end_time,
            self.start_state,
            self.coefficients,
            self.dimension,
            self.order,
            time,
            output,
        )
    }
}

impl DenseSegment for TaylorSegment {
    fn interpolate(&self, time: f64, output: &mut [f64]) -> Result<(), InterpolationError> {
        validate_dense_query(self.contains(time), output.len(), self.dimension)?;
        if time == self.start_time {
            output.copy_from_slice(&self.start_state);
            return Ok(());
        }
        if time == self.bound_time {
            output.copy_from_slice(&self.end_state);
            return Ok(());
        }
        interpolate_taylor(
            self.start_time,
            self.end_time,
            &self.start_state,
            &self.coefficients,
            self.dimension,
            self.order,
            time,
            output,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn interpolate_taylor(
    start_time: f64,
    end_time: f64,
    start_state: &[f64],
    coefficients: &[f64],
    dimension: usize,
    order: usize,
    time: f64,
    output: &mut [f64],
) -> Result<(), InterpolationError> {
    if output.len() != dimension {
        return Err(InterpolationError::DimensionMismatch);
    }
    if !time.is_finite() {
        return Err(InterpolationError::NonFiniteTime);
    }
    if coefficients.len() < (order + 1) * dimension {
        return Err(InterpolationError::InvalidSegmentData {
            context: "Taylor coefficients",
        });
    }
    let theta = (time - start_time) / (end_time - start_time);
    output.copy_from_slice(start_state);
    for component in 0..dimension {
        let mut value = coefficients[order * dimension + component];
        for power in (1..order).rev() {
            value = coefficients[power * dimension + component] + theta * value;
        }
        output[component] += theta * value;
    }
    output
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(InterpolationError::NonFiniteResult { context: "Taylor" })
}
