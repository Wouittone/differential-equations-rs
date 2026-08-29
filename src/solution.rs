use thiserror::Error;

use crate::event::times_are_numerically_equal;

/// A dense-output query or retained interpolation segment is invalid.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
#[non_exhaustive]
pub enum InterpolationError {
    /// The requested time is NaN or infinite.
    #[error("interpolation time must be finite")]
    NonFiniteTime,
    /// The solution contains no saved state.
    #[error("cannot interpolate an empty solution")]
    EmptySolution,
    /// The requested time is outside the saved trajectory.
    #[error("interpolation time is outside the saved trajectory")]
    OutsideTimeSpan,
    /// An output buffer has the wrong state dimension.
    #[error("interpolation output dimension does not match the solution")]
    DimensionMismatch,
    /// Retained dense data violates a solver invariant.
    #[error("invalid dense-output data: {context}")]
    InvalidSegmentData {
        /// The failed dense-output representation.
        context: &'static str,
    },
    /// Interpolation produced a NaN or infinity.
    #[error("{context} interpolation produced a non-finite value")]
    NonFiniteResult {
        /// The dense-output representation that failed.
        context: &'static str,
    },
}

/// Work performed by an ODE solver.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub struct SolverStats {
    /// Number of right-hand-side evaluations.
    pub rhs_evaluations: usize,
    /// Number of accepted time steps.
    pub accepted_steps: usize,
    /// Number of rejected time steps.
    pub rejected_steps: usize,
    /// Number of nonlinear iterations performed by implicit methods.
    pub nonlinear_iterations: usize,
    /// Number of Jacobian evaluations.
    pub jacobian_evaluations: usize,
    /// Number of linear systems solved.
    pub linear_solves: usize,
    /// Number of dense linear factorizations built.
    pub linear_factorizations: usize,
    /// Number of discrete or continuous callback effects applied.
    pub callback_invocations: usize,
}

use crate::{SaveMode, SolveOptions};

/// Method-specific dense interpolation seam. Segments own their endpoint
/// data and can be evaluated without mutating the solver kernel.
#[allow(dead_code)]
pub(crate) trait DenseSegment {
    fn interpolate(&self, time: f64, output: &mut [f64]) -> Result<(), InterpolationError>;
}

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

/// Process-lifetime continuous-extension rows from either legacy static
/// tableaus or lazily materialized resource tableaus.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum RungeKuttaCoefficients {
    Static(&'static [&'static [f64]]),
    Resource(&'static [Vec<f64>]),
}

impl RungeKuttaCoefficients {
    fn len(self) -> usize {
        match self {
            Self::Static(rows) => rows.len(),
            Self::Resource(rows) => rows.len(),
        }
    }

    fn row(self, index: usize) -> &'static [f64] {
        match self {
            Self::Static(rows) => rows[index],
            Self::Resource(rows) => &rows[index],
        }
    }

    fn rows_are_valid(self) -> bool {
        (0..self.len()).all(|index| {
            let row = self.row(index);
            !row.is_empty() && row.iter().all(|coefficient| coefficient.is_finite())
        })
    }
}

impl From<&'static [&'static [f64]]> for RungeKuttaCoefficients {
    fn from(rows: &'static [&'static [f64]]) -> Self {
        Self::Static(rows)
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

/// Owning dynamic collocation extension used by variable-stage FIRK methods.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CollocationSegment {
    start_time: f64,
    attempted_time: f64,
    bound_time: f64,
    start_state: Vec<f64>,
    midpoint_state: Vec<f64>,
    endpoint_state: Vec<f64>,
    stages: Vec<f64>,
    first_half_stages: Vec<f64>,
    second_half_stages: Vec<f64>,
    lagrange: Vec<f64>,
    dimension: usize,
    stage_count: usize,
    adaptive: bool,
}

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

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum OwnedDenseSegment {
    Hermite(HermiteSegment),
    RungeKutta(RungeKuttaSegment),
    Stiff(StiffSegment),
    Collocation(CollocationSegment),
    Taylor(TaylorSegment),
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

    fn contains(&self, time: f64) -> bool {
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

    fn contains(&self, time: f64) -> bool {
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

    fn contains(&self, time: f64) -> bool {
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

    fn contains(&self, time: f64) -> bool {
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

    fn contains(&self, time: f64) -> bool {
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

    fn contains(&self, time: f64) -> bool {
        time.is_finite()
            && if self.start_time < self.bound_time {
                (self.start_time..=self.bound_time).contains(&time)
            } else {
                (self.bound_time..=self.start_time).contains(&time)
            }
    }
}

impl CollocationSegment {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        start_time: f64,
        attempted_time: f64,
        bound_time: f64,
        start_state: &[f64],
        midpoint_state: &[f64],
        endpoint_state: &[f64],
        stages: &[f64],
        first_half_stages: &[f64],
        second_half_stages: &[f64],
        lagrange: &[f64],
        stage_count: usize,
        adaptive: bool,
    ) -> Result<Self, InterpolationError> {
        let dimension = start_state.len();
        let within_step = if start_time < attempted_time {
            (start_time..=attempted_time).contains(&bound_time)
        } else {
            (attempted_time..=start_time).contains(&bound_time)
        };
        if !start_time.is_finite()
            || !attempted_time.is_finite()
            || !bound_time.is_finite()
            || attempted_time == start_time
            || !within_step
            || dimension == 0
            || endpoint_state.len() != dimension
            || midpoint_state.len() != dimension
            || stage_count == 0
            || lagrange.len() != stage_count * stage_count
            || stages.len() != stage_count * dimension
            || first_half_stages.len() != stage_count * dimension
            || second_half_stages.len() != stage_count * dimension
        {
            return Err(InterpolationError::InvalidSegmentData {
                context: "collocation segment",
            });
        }
        Ok(Self {
            start_time,
            attempted_time,
            bound_time,
            start_state: start_state.to_vec(),
            midpoint_state: midpoint_state.to_vec(),
            endpoint_state: endpoint_state.to_vec(),
            stages: stages.to_vec(),
            first_half_stages: first_half_stages.to_vec(),
            second_half_stages: second_half_stages.to_vec(),
            lagrange: lagrange.to_vec(),
            dimension,
            stage_count,
            adaptive,
        })
    }

    fn contains(&self, time: f64) -> bool {
        time.is_finite()
            && if self.start_time < self.bound_time {
                (self.start_time..=self.bound_time).contains(&time)
            } else {
                (self.bound_time..=self.start_time).contains(&time)
            }
    }

    fn interpolate_piece(
        &self,
        start_time: f64,
        step: f64,
        start_state: &[f64],
        stages: &[f64],
        time: f64,
        output: &mut [f64],
    ) {
        let theta = ((time - start_time) / step).clamp(0.0, 1.0);
        output.copy_from_slice(start_state);
        for stage in 0..self.stage_count {
            let mut power = theta;
            let mut weight = 0.0;
            for degree in 0..self.stage_count {
                weight +=
                    self.lagrange[stage * self.stage_count + degree] * power / (degree + 1) as f64;
                power *= theta;
            }
            for component in 0..self.dimension {
                output[component] += step * weight * stages[stage * self.dimension + component];
            }
        }
    }
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

    fn contains(&self, time: f64) -> bool {
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

    fn contains(&self, time: f64) -> bool {
        time.is_finite()
            && if self.start_time < self.bound_time {
                (self.start_time..=self.bound_time).contains(&time)
            } else {
                (self.bound_time..=self.start_time).contains(&time)
            }
    }
}

fn validate_dense_query(
    contains_time: bool,
    output_dimension: usize,
    state_dimension: usize,
) -> Result<(), InterpolationError> {
    if output_dimension != state_dimension {
        return Err(InterpolationError::DimensionMismatch);
    }
    if !contains_time {
        return Err(InterpolationError::OutsideTimeSpan);
    }
    Ok(())
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

impl DenseSegment for CollocationSegment {
    fn interpolate(&self, time: f64, output: &mut [f64]) -> Result<(), InterpolationError> {
        validate_dense_query(self.contains(time), output.len(), self.dimension)?;
        if time == self.bound_time {
            output.copy_from_slice(&self.endpoint_state);
            return Ok(());
        }
        let step = self.attempted_time - self.start_time;
        if self.adaptive {
            let half = 0.5 * step;
            if step.signum() * (time - (self.start_time + half)) <= 0.0 {
                self.interpolate_piece(
                    self.start_time,
                    half,
                    &self.start_state,
                    &self.first_half_stages,
                    time,
                    output,
                );
            } else {
                self.interpolate_piece(
                    self.start_time + half,
                    half,
                    &self.midpoint_state,
                    &self.second_half_stages,
                    time,
                    output,
                );
            }
        } else {
            self.interpolate_piece(
                self.start_time,
                step,
                &self.start_state,
                &self.stages,
                time,
                output,
            );
        }
        output
            .iter()
            .all(|value| value.is_finite())
            .then_some(())
            .ok_or(InterpolationError::NonFiniteResult {
                context: "collocation",
            })
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

impl OwnedDenseSegment {
    fn contains(&self, time: f64) -> bool {
        match self {
            Self::Hermite(segment) => segment.contains(time),
            Self::RungeKutta(segment) => segment.contains(time),
            Self::Stiff(segment) => segment.contains(time),
            Self::Collocation(segment) => segment.contains(time),
            Self::Taylor(segment) => segment.contains(time),
        }
    }
}

impl DenseSegment for OwnedDenseSegment {
    fn interpolate(&self, time: f64, output: &mut [f64]) -> Result<(), InterpolationError> {
        match self {
            Self::Hermite(segment) => segment.interpolate(time, output),
            Self::RungeKutta(segment) => segment.interpolate(time, output),
            Self::Stiff(segment) => segment.interpolate(time, output),
            Self::Collocation(segment) => segment.interpolate(time, output),
            Self::Taylor(segment) => segment.interpolate(time, output),
        }
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

pub(crate) struct TrajectoryRecorder<'a> {
    times: Vec<f64>,
    values: Vec<f64>,
    dimension: usize,
    save_at: &'a [f64],
    next_save: usize,
    save_mode: SaveMode,
    interpolation: Vec<f64>,
    dense_segments: Vec<OwnedDenseSegment>,
    retain_dense_output: bool,
}

impl<'a> TrajectoryRecorder<'a> {
    pub(crate) fn new(state: &[f64], time: f64, options: &'a SolveOptions) -> Self {
        let save_initial = options.save_at.is_empty() || options.save_at.first() == Some(&time);
        let capacity = if options.save_at.is_empty() {
            2
        } else {
            options.save_at.len()
        };
        let mut times = Vec::with_capacity(capacity);
        let mut values = Vec::with_capacity(capacity * state.len());
        if save_initial {
            times.push(time);
            values.extend_from_slice(state);
        }
        Self {
            times,
            values,
            dimension: state.len(),
            save_at: &options.save_at,
            next_save: usize::from(!options.save_at.is_empty() && save_initial),
            save_mode: options.save,
            interpolation: if options.save_at.is_empty() {
                Vec::new()
            } else {
                vec![0.0; state.len()]
            },
            dense_segments: Vec::new(),
            retain_dense_output: options.retain_dense_output,
        }
    }

    pub(crate) fn record_step(
        &mut self,
        previous_state: &[f64],
        previous_time: f64,
        state: &[f64],
        time: f64,
        final_time: bool,
    ) {
        if self.save_at.is_empty() {
            if self.save_mode == SaveMode::EveryStep || final_time {
                self.push_unique(time, state);
            }
            return;
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
            let fraction = (target - previous_time) / (time - previous_time);
            for ((output, previous), current) in
                self.interpolation.iter_mut().zip(previous_state).zip(state)
            {
                *output = previous + fraction * (current - previous);
            }
            self.push_interpolation_target(target);
            self.next_save += 1;
        }
    }

    /// Records an accepted step using its method-provided dense interpolant.
    ///
    /// This is deliberately separate from [`record_step`]: existing kernels
    /// still use the endpoint fallback until they expose their accepted-step
    /// derivative/stage data. The helper shares the recorder's preallocated
    /// scratch buffer and never evaluates the interpolant more than once per
    /// requested save point.
    #[allow(dead_code)]
    pub(crate) fn record_step_dense(
        &mut self,
        previous_state: &[f64],
        previous_time: f64,
        state: &[f64],
        time: f64,
        final_time: bool,
        segment: &dyn DenseSegment,
    ) -> Result<(), InterpolationError> {
        let _ = previous_state;
        if self.save_at.is_empty() {
            if self.save_mode == SaveMode::EveryStep || final_time {
                self.push_unique(time, state);
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
            segment.interpolate(target, &mut self.interpolation)?;
            self.push_interpolation_target(target);
            self.next_save += 1;
        }
        Ok(())
    }

    pub(crate) fn finish(self, stats: SolverStats) -> Solution {
        if self.dense_segments.is_empty() {
            Solution::new(self.times, self.values, self.dimension, stats)
        } else {
            Solution::new_with_dense(
                self.times,
                self.values,
                self.dimension,
                stats,
                self.dense_segments,
            )
        }
    }

    pub(crate) fn retains_dense_output(&self) -> bool {
        self.retain_dense_output
    }

    pub(crate) fn needs_dense_sampling(&self) -> bool {
        !self.save_at.is_empty()
    }

    pub(crate) fn retain_runge_kutta_segment(&mut self, segment: RungeKuttaSegment) {
        debug_assert!(self.retain_dense_output);
        self.dense_segments
            .push(OwnedDenseSegment::RungeKutta(segment));
    }

    pub(crate) fn retain_hermite_segment(&mut self, segment: HermiteSegment) {
        debug_assert!(self.retain_dense_output);
        self.dense_segments
            .push(OwnedDenseSegment::Hermite(segment));
    }

    pub(crate) fn retain_stiff_segment(&mut self, segment: StiffSegment) {
        debug_assert!(self.retain_dense_output);
        self.dense_segments.push(OwnedDenseSegment::Stiff(segment));
    }

    pub(crate) fn retain_collocation_segment(&mut self, segment: CollocationSegment) {
        debug_assert!(self.retain_dense_output);
        self.dense_segments
            .push(OwnedDenseSegment::Collocation(segment));
    }

    pub(crate) fn retain_taylor_segment(&mut self, segment: TaylorSegment) {
        debug_assert!(self.retain_dense_output);
        self.dense_segments.push(OwnedDenseSegment::Taylor(segment));
    }

    pub(crate) fn force_state(&mut self, time: f64, state: &[f64]) {
        let canonical_time = self
            .save_at
            .iter()
            .copied()
            .find(|target| times_are_numerically_equal(*target, time))
            .unwrap_or(time);
        self.push_unique(canonical_time, state);
    }

    fn push_interpolation_target(&mut self, time: f64) {
        if let Some(saved) = self
            .times
            .last_mut()
            .filter(|saved| times_are_numerically_equal(**saved, time))
        {
            *saved = time;
            let start = self.values.len() - self.dimension;
            self.values[start..].copy_from_slice(&self.interpolation);
        } else {
            self.times.push(time);
            self.values.extend_from_slice(&self.interpolation);
        }
    }

    fn push_unique(&mut self, time: f64, state: &[f64]) {
        if self
            .times
            .last()
            .is_some_and(|saved| times_are_numerically_equal(*saved, time))
        {
            let start = self.values.len() - self.dimension;
            self.values[start..].copy_from_slice(state);
        } else {
            self.times.push(time);
            self.values.extend_from_slice(state);
        }
    }
}

/// A saved ODE trajectory.
///
/// States are kept in one row-major allocation. The state at saved time `i`
/// occupies `values[i * dimension..(i + 1) * dimension]`.
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

    fn new_with_dense(
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
        self.values.get(start..start + self.dimension)
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
        if !time.is_finite() {
            return Err(InterpolationError::NonFiniteTime);
        }
        if self.times.is_empty() {
            return Err(InterpolationError::EmptySolution);
        }
        for (index, &saved_time) in self.times.iter().enumerate() {
            if time == saved_time {
                return self.state(index).map(<[f64]>::to_vec).ok_or(
                    InterpolationError::InvalidSegmentData {
                        context: "saved solution state",
                    },
                );
            }
        }
        for segment in &self.dense_segments {
            if segment.contains(time) {
                let mut output = vec![0.0; self.dimension];
                segment.interpolate(time, &mut output)?;
                return Ok(output);
            }
        }
        for index in 1..self.times.len() {
            let left = self.times[index - 1];
            let right = self.times[index];
            if (left <= right && time <= right && time >= left)
                || (left >= right && time >= right && time <= left)
            {
                let fraction = (time - left) / (right - left);
                let previous =
                    self.state(index - 1)
                        .ok_or(InterpolationError::InvalidSegmentData {
                            context: "saved solution state",
                        })?;
                let current = self
                    .state(index)
                    .ok_or(InterpolationError::InvalidSegmentData {
                        context: "saved solution state",
                    })?;
                return Ok(previous
                    .iter()
                    .zip(current)
                    .map(|(previous, current)| previous + fraction * (current - previous))
                    .collect());
            }
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
}

#[cfg(test)]
mod tests {
    use super::{DenseSegment, HermiteSegment, Solution, SolverStats, TrajectoryRecorder};

    #[test]
    fn exposes_flat_states_as_slices() {
        let solution = Solution::new(
            vec![0.0, 0.5, 1.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            2,
            SolverStats::default(),
        );

        assert_eq!(solution.state(0), Some([1.0, 2.0].as_slice()));
        assert_eq!(solution.state(2), Some([5.0, 6.0].as_slice()));
        assert_eq!(solution.state(3), None);
        assert_eq!(solution.last_state(), &[5.0, 6.0]);
        assert_eq!(solution.interpolate(0.25), Some(vec![2.0, 3.0]));
        assert_eq!(solution.interpolate(2.0), None);
    }

    #[test]
    fn hermite_segment_matches_endpoints_and_midpoint() {
        let segment =
            HermiteSegment::new(0.0, 1.0, vec![0.0], vec![1.0], vec![0.0], vec![2.0]).unwrap();
        let mut output = [0.0];
        segment.interpolate(0.0, &mut output).unwrap();
        assert_eq!(output, [0.0]);
        segment.interpolate(1.0, &mut output).unwrap();
        assert_eq!(output, [1.0]);
        segment.interpolate(0.5, &mut output).unwrap();
        assert!((output[0] - 0.25).abs() < 1.0e-14);
    }

    #[test]
    fn hermite_segment_is_checked_and_exact_at_endpoints() {
        let segment =
            HermiteSegment::new(1.0, 0.0, vec![1.0], vec![0.0], vec![2.0], vec![0.0]).unwrap();
        let mut output = [f64::NAN];
        segment.interpolate(1.0, &mut output).unwrap();
        assert_eq!(output, [1.0]);
        segment.interpolate(0.0, &mut output).unwrap();
        assert_eq!(output, [0.0]);
        assert!(segment.interpolate(1.1, &mut output).is_err());
        assert!(segment.interpolate(0.5, &mut []).is_err());
        assert!(
            HermiteSegment::new(0.0, 1.0, vec![f64::NAN], vec![1.0], vec![0.0], vec![1.0],)
                .is_err()
        );
    }

    #[test]
    fn recorder_uses_accepted_hermite_segment_for_save_at() {
        let options = crate::SolveOptions {
            save_at: vec![0.25, 0.75],
            ..crate::SolveOptions::default()
        };
        let mut recorder = TrajectoryRecorder::new(&[0.0], 0.0, &options);
        let segment =
            HermiteSegment::new(0.0, 1.0, vec![0.0], vec![1.0], vec![0.0], vec![3.0]).unwrap();
        recorder
            .record_step_dense(&[0.0], 0.0, &[1.0], 1.0, true, &segment)
            .unwrap();
        let solution = recorder.finish(SolverStats::default());
        assert_eq!(solution.times(), &[0.25, 0.75]);
        assert!((solution.values()[0] - 0.015625).abs() < 1.0e-14);
        assert!((solution.values()[1] - 0.421875).abs() < 1.0e-14);
    }
}
