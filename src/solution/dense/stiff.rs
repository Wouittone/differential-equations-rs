use super::{DenseSegment, validate_dense_query};
use crate::solution::InterpolationError;

/// Borrowed Rosenbrock/Rodas continuous extension for one accepted step.
///
/// `corrections` stores the already-combined rows `H * k`. Unlike explicit
/// Runge--Kutta stages these values are state increments, so interpolation
/// does not multiply them by the step size.
pub(crate) struct BorrowedStiffSegment<'a> {
    start_time: f64,
    end_time: f64,
    start_state: &'a [f64],
    end_state: &'a [f64],
    corrections: &'a [f64],
    order: usize,
}

/// Owning Rosenbrock/Rodas continuous extension retained after a solve.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct StiffSegment {
    start_time: f64,
    end_time: f64,
    bound_time: f64,
    start_state: Vec<f64>,
    end_state: Vec<f64>,
    corrections: Vec<f64>,
    order: usize,
}

impl<'a> BorrowedStiffSegment<'a> {
    pub(crate) fn new(
        start_time: f64,
        end_time: f64,
        start_state: &'a [f64],
        end_state: &'a [f64],
        corrections: &'a [f64],
        order: usize,
    ) -> Result<Self, InterpolationError> {
        let dimension = start_state.len();
        if !start_time.is_finite()
            || !end_time.is_finite()
            || end_time == start_time
            || dimension == 0
            || end_state.len() != dimension
            || !(2..=4).contains(&order)
            || corrections.len() != order * dimension
            || !start_state.iter().all(|value| value.is_finite())
            || !end_state.iter().all(|value| value.is_finite())
            || !corrections.iter().all(|value| value.is_finite())
        {
            return Err(InterpolationError::InvalidSegmentData {
                context: "stiff segment",
            });
        }
        Ok(Self {
            start_time,
            end_time,
            start_state,
            end_state,
            corrections,
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

impl StiffSegment {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        start_time: f64,
        end_time: f64,
        bound_time: f64,
        start_state: &[f64],
        end_state: &[f64],
        corrections: &[f64],
        order: usize,
    ) -> Result<Self, InterpolationError> {
        let borrowed = BorrowedStiffSegment::new(
            start_time,
            end_time,
            start_state,
            end_state,
            corrections,
            order,
        )?;
        if !borrowed.contains(bound_time) {
            return Err(InterpolationError::InvalidSegmentData {
                context: "stiff segment bound",
            });
        }
        Ok(Self {
            start_time,
            end_time,
            bound_time,
            start_state: start_state.to_vec(),
            end_state: end_state.to_vec(),
            corrections: corrections.to_vec(),
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

impl DenseSegment for BorrowedStiffSegment<'_> {
    fn interpolate(&self, time: f64, output: &mut [f64]) -> Result<(), InterpolationError> {
        validate_dense_query(self.contains(time), output.len(), self.start_state.len())?;
        interpolate_stiff(
            self.start_time,
            self.end_time,
            self.start_state,
            self.end_state,
            self.corrections,
            self.order,
            time,
            output,
        )
    }
}

impl DenseSegment for StiffSegment {
    fn interpolate(&self, time: f64, output: &mut [f64]) -> Result<(), InterpolationError> {
        validate_dense_query(self.contains(time), output.len(), self.start_state.len())?;
        interpolate_stiff(
            self.start_time,
            self.end_time,
            &self.start_state,
            &self.end_state,
            &self.corrections,
            self.order,
            time,
            output,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn interpolate_stiff(
    start_time: f64,
    end_time: f64,
    start_state: &[f64],
    end_state: &[f64],
    corrections: &[f64],
    order: usize,
    time: f64,
    output: &mut [f64],
) -> Result<(), InterpolationError> {
    let dimension = start_state.len();
    if output.len() != dimension {
        return Err(InterpolationError::DimensionMismatch);
    }
    if !time.is_finite() {
        return Err(InterpolationError::NonFiniteTime);
    }
    if order == 0 || corrections.len() != order * dimension {
        return Err(InterpolationError::InvalidSegmentData {
            context: "stiff interpolation corrections",
        });
    }
    if time == start_time {
        output.copy_from_slice(start_state);
        return Ok(());
    }
    if time == end_time {
        output.copy_from_slice(end_state);
        return Ok(());
    }
    let theta = (time - start_time) / (end_time - start_time);
    let theta1 = 1.0 - theta;
    for component in 0..dimension {
        let mut polynomial = corrections[(order - 1) * dimension + component];
        for row in (0..order - 1).rev() {
            polynomial = corrections[row * dimension + component] + theta * polynomial;
        }
        output[component] =
            theta1 * start_state[component] + theta * (end_state[component] + theta1 * polynomial);
    }
    Ok(())
}
