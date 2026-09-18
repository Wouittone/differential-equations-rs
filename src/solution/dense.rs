use super::InterpolationError;

mod collocation;
mod hermite;
mod runge_kutta;
mod stiff;
mod taylor;

pub(crate) use collocation::CollocationSegment;
pub(crate) use hermite::{BorrowedHermiteSegment, HermiteSegment};
pub(crate) use runge_kutta::{
    BorrowedRungeKuttaSegment, RungeKuttaCoefficients, RungeKuttaSegment, interpolate_runge_kutta,
};
pub(crate) use stiff::{BorrowedStiffSegment, StiffSegment};
pub(crate) use taylor::{BorrowedTaylorSegment, TaylorSegment};

/// Method-specific dense interpolation seam. Segments own their endpoint
/// data and can be evaluated without mutating the solver kernel.
#[allow(dead_code)]
pub(crate) trait DenseSegment {
    fn interpolate(&self, time: f64, output: &mut [f64]) -> Result<(), InterpolationError>;
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum OwnedDenseSegment {
    Hermite(HermiteSegment),
    RungeKutta(RungeKuttaSegment),
    Stiff(StiffSegment),
    Collocation(CollocationSegment),
    Taylor(TaylorSegment),
}

impl OwnedDenseSegment {
    pub(super) fn contains(&self, time: f64) -> bool {
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
