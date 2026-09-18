use super::{DenseSegment, validate_dense_query};
use crate::solution::InterpolationError;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct HermiteSegment {
    start_time: f64,
    end_time: f64,
    bound_time: f64,
    start_state: Vec<f64>,
    end_state: Vec<f64>,
    start_derivative: Vec<f64>,
    end_derivative: Vec<f64>,
}

/// Borrowed accepted-step Hermite data for allocation-free recorder calls.
///
/// Solver workspaces retain endpoint states and derivatives, so dense sampling
/// can borrow those slices for the duration of one accepted step instead of
/// allocating an owning segment on every step.
pub(crate) struct BorrowedHermiteSegment<'a> {
    start_time: f64,
    end_time: f64,
    start_state: &'a [f64],
    end_state: &'a [f64],
    start_derivative: &'a [f64],
    end_derivative: &'a [f64],
}

impl HermiteSegment {
    #[allow(dead_code)]
    pub(crate) fn new(
        start_time: f64,
        end_time: f64,
        start_state: Vec<f64>,
        end_state: Vec<f64>,
        start_derivative: Vec<f64>,
        end_derivative: Vec<f64>,
    ) -> Result<Self, InterpolationError> {
        Self::new_bounded(
            start_time,
            end_time,
            end_time,
            start_state,
            end_state,
            start_derivative,
            end_derivative,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_bounded(
        start_time: f64,
        end_time: f64,
        bound_time: f64,
        start_state: Vec<f64>,
        end_state: Vec<f64>,
        start_derivative: Vec<f64>,
        end_derivative: Vec<f64>,
    ) -> Result<Self, InterpolationError> {
        if !start_time.is_finite()
            || !end_time.is_finite()
            || !bound_time.is_finite()
            || end_time == start_time
            || start_state.is_empty()
            || start_state.len() != end_state.len()
            || start_state.len() != start_derivative.len()
            || start_state.len() != end_derivative.len()
            || !start_state.iter().all(|value| value.is_finite())
            || !end_state.iter().all(|value| value.is_finite())
            || !start_derivative.iter().all(|value| value.is_finite())
            || !end_derivative.iter().all(|value| value.is_finite())
        {
            return Err(InterpolationError::InvalidSegmentData {
                context: "Hermite segment dimensions or times",
            });
        }
        let within_step = if start_time < end_time {
            (start_time..=end_time).contains(&bound_time)
        } else {
            (end_time..=start_time).contains(&bound_time)
        };
        if !within_step {
            return Err(InterpolationError::InvalidSegmentData {
                context: "Hermite segment bound",
            });
        }
        Ok(Self {
            start_time,
            end_time,
            bound_time,
            start_state,
            end_state,
            start_derivative,
            end_derivative,
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

impl<'a> BorrowedHermiteSegment<'a> {
    pub(crate) fn new(
        start_time: f64,
        end_time: f64,
        start_state: &'a [f64],
        end_state: &'a [f64],
        start_derivative: &'a [f64],
        end_derivative: &'a [f64],
    ) -> Result<Self, InterpolationError> {
        if !start_time.is_finite()
            || !end_time.is_finite()
            || end_time == start_time
            || start_state.is_empty()
            || start_state.len() != end_state.len()
            || start_state.len() != start_derivative.len()
            || start_state.len() != end_derivative.len()
            || !start_state.iter().all(|value| value.is_finite())
            || !end_state.iter().all(|value| value.is_finite())
            || !start_derivative.iter().all(|value| value.is_finite())
            || !end_derivative.iter().all(|value| value.is_finite())
        {
            return Err(InterpolationError::InvalidSegmentData {
                context: "borrowed Hermite segment dimensions or times",
            });
        }
        Ok(Self {
            start_time,
            end_time,
            start_state,
            end_state,
            start_derivative,
            end_derivative,
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

#[allow(dead_code)]
impl DenseSegment for HermiteSegment {
    fn interpolate(&self, time: f64, output: &mut [f64]) -> Result<(), InterpolationError> {
        validate_dense_query(self.contains(time), output.len(), self.start_state.len())?;
        interpolate_hermite(
            self.start_time,
            self.end_time,
            &self.start_state,
            &self.end_state,
            &self.start_derivative,
            &self.end_derivative,
            time,
            output,
        )
    }
}

impl DenseSegment for BorrowedHermiteSegment<'_> {
    fn interpolate(&self, time: f64, output: &mut [f64]) -> Result<(), InterpolationError> {
        validate_dense_query(self.contains(time), output.len(), self.start_state.len())?;
        interpolate_hermite(
            self.start_time,
            self.end_time,
            self.start_state,
            self.end_state,
            self.start_derivative,
            self.end_derivative,
            time,
            output,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn interpolate_hermite(
    start_time: f64,
    end_time: f64,
    start_state: &[f64],
    end_state: &[f64],
    start_derivative: &[f64],
    end_derivative: &[f64],
    time: f64,
    output: &mut [f64],
) -> Result<(), InterpolationError> {
    if output.len() != start_state.len() {
        return Err(InterpolationError::DimensionMismatch);
    }
    if !time.is_finite() {
        return Err(InterpolationError::NonFiniteTime);
    }
    if time == start_time {
        output.copy_from_slice(start_state);
        return Ok(());
    }
    if time == end_time {
        output.copy_from_slice(end_state);
        return Ok(());
    }
    let h = end_time - start_time;
    let theta = (time - start_time) / h;
    for (((output, start), end), (start_derivative, end_derivative)) in output
        .iter_mut()
        .zip(start_state)
        .zip(end_state)
        .zip(start_derivative.iter().zip(end_derivative))
    {
        let delta = end - start;
        let quadratic = 3.0 * delta - h * (2.0 * start_derivative + end_derivative);
        let cubic = -2.0 * delta + h * (start_derivative + end_derivative);
        *output = start + theta * (h * start_derivative + theta * (quadratic + theta * cubic));
    }
    Ok(())
}
