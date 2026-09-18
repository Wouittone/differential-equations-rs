use thiserror::Error;

use crate::solvers::automatic::AutomaticBranch;

mod api;
mod dense;
mod recorder;

pub use api::Solution;
pub(crate) use api::{
    checked_state_dimension, finite_partitioned_interpolation, interpolate_value,
    interpolation_fraction, validate_saved_solution,
};
pub(crate) use dense::{
    BorrowedHermiteSegment, BorrowedRungeKuttaSegment, BorrowedStiffSegment, BorrowedTaylorSegment,
    CollocationSegment, DenseSegment, HermiteSegment, OwnedDenseSegment, RungeKuttaCoefficients,
    RungeKuttaSegment, StiffSegment, TaylorSegment, interpolate_runge_kutta,
};
pub(crate) use recorder::TrajectoryRecorder;

/// Invalid saved trajectory data supplied when constructing a [`Solution`].
///
/// This error is shared by ordinary and partitioned second-order solutions so
/// downstream algorithm implementations can use `?` from their
/// `solve_validated` methods.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
#[non_exhaustive]
pub enum SolutionConstructionError {
    /// At least one saved state is required.
    #[error("a saved solution must contain at least one time and state")]
    EmptyTrajectory,
    /// The logical state shape contains no scalar components.
    #[error("the saved solution state shape must contain at least one component")]
    EmptyState,
    /// Multiplying the logical state extents overflowed `usize`.
    #[error("the saved solution state dimension overflowed")]
    DimensionOverflow,
    /// A flattened state partition does not match the times and logical shape.
    #[error("saved solution values do not match the times and state shape")]
    DimensionMismatch,
    /// At least one saved time is NaN or infinite.
    #[error("saved solution times must be finite")]
    NonFiniteTime,
    /// Saved times change integration direction.
    #[error("saved solution times must be monotonic in one integration direction")]
    NonMonotonicTimes,
    /// At least one saved state component is NaN or infinite.
    #[error("saved solution states must contain only finite values")]
    NonFiniteState,
}

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
    /// Number of right-hand-side evaluations, including callback checks.
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
    /// Number of in-flight transitions between automatic solver branches.
    pub algorithm_switches: usize,
    /// Number of steps accepted by an automatic solver's non-stiff branch.
    pub nonstiff_accepted_steps: usize,
    /// Number of steps accepted by an automatic solver's stiff branch.
    pub stiff_accepted_steps: usize,
    /// Branch active when an automatic solve finished.
    ///
    /// Ordinary, non-composite algorithms leave this as `None`.
    pub final_automatic_branch: Option<AutomaticBranch>,
}

#[cfg(test)]
mod tests;
