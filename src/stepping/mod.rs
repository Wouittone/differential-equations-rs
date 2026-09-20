//! Reusable, output-free numerical stepping with explicit commit/reject lifecycle.
//!
//! A stepper owns accepted state and scratch storage. An attempt borrows scratch
//! results; only `accept` changes the accepted state. User closures may borrow
//! their environment and mutate it. Changing their mathematical RHS requires
//! invalidating any cached derivative. See `docs/STEPPING_CONTRACT.md`.

mod explicit;
mod state_buffer;
pub use explicit::{ExplicitRungeKuttaStepper, StepView};

/// A lifecycle or numerical input violation, independent of application errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum StepFailure {
    /// The Rosenbrock stage matrix could not be factorized.
    #[error("singular stage matrix")]
    SingularSystem,
    /// State or output dimensions differ from the constructed workspace.
    #[error("stepper dimension mismatch")]
    Dimension,
    /// A state, time, step, derivative, or candidate is nonfinite.
    #[error("nonfinite stepping value")]
    NonFinite,
    /// The supplied tableau requires an unsupported specialized kernel.
    #[error("unsupported stepping tableau")]
    UnsupportedTableau,
    /// Commit or reject the previous candidate first.
    #[error("a candidate is already pending")]
    PendingCandidate,
    /// An accept/reject operation requires a successful attempt.
    #[error("no candidate is pending")]
    NoCandidate,
    /// A nonzero step cannot advance the floating-point time.
    #[error("step does not advance time")]
    TimeResolution,
    /// A proposal points away from the endpoint.
    #[error("step points away from endpoint")]
    Direction,
}

/// Preserves an application error directly, without side channels.
#[derive(Debug, thiserror::Error)]
pub enum StepError<E> {
    /// Invalid lifecycle, input, or numerical output.
    #[error(transparent)]
    Solver(#[from] StepFailure),
    /// The original RHS/Jacobian/observer error.
    #[error("application evaluation failed: {0}")]
    User(#[source] E),
}

/// Cumulative work performed by a reusable stepper.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepStatistics {
    /// All attempts, including zero steps and failed evaluations.
    pub attempts: usize,
    /// Calls to the right-hand side (including failed calls).
    pub rhs_evaluations: usize,
    /// Explicitly committed candidates.
    pub accepted_steps: usize,
    /// Explicitly discarded candidates.
    pub rejected_steps: usize,
    /// Calls to analytic or numerical Jacobian construction.
    pub jacobian_evaluations: usize,
    /// Matrix factorizations.
    pub factorizations: usize,
    /// Solved linear systems; these are not RHS evaluations.
    pub linear_solves: usize,
}

pub(crate) fn finite(values: &[f64]) -> Result<(), StepFailure> {
    if values.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(StepFailure::NonFinite)
    }
}
pub(crate) fn checked_time(time: f64, step: f64) -> Result<f64, StepFailure> {
    let end = time + step;
    if !time.is_finite() || !step.is_finite() || !end.is_finite() {
        return Err(StepFailure::NonFinite);
    }
    if step != 0.0 && end == time {
        return Err(StepFailure::TimeResolution);
    }
    Ok(end)
}
/// Clamp a signed proposal to a finite endpoint without overshooting it.
pub fn endpoint_step(time: f64, endpoint: f64, proposal: f64) -> Result<f64, StepFailure> {
    finite(&[time, endpoint, proposal])?;
    let remaining = endpoint - time;
    if remaining == 0.0 {
        return Ok(0.0);
    }
    if proposal == 0.0 || remaining.signum() != proposal.signum() {
        return Err(StepFailure::Direction);
    }
    let step = if proposal.abs() > remaining.abs() {
        remaining
    } else {
        proposal
    };
    checked_time(time, step)?;
    Ok(step)
}

mod rkn;
pub use rkn::{AccelerationPolicy, RknStepView, RknStepper};

mod controller;
pub use controller::{
    AdaptiveController, Continuation, ControllerConfig, ControllerError, ControllerState,
    MinimumStepPolicy, StepDecision, initial_step,
};

mod driver;
pub use driver::{IntegrationError, IntegrationOutcome, Observation, ObserverAction, integrate_rk};

pub(crate) mod time_difference;
pub use time_difference::TimeDifferencePolicy;
mod events;
pub use events::{RootDirection, RootError, RootOptions, RootOutcome, integrate_rk_until_event};

mod rosenbrock;
pub use rosenbrock::{DerivativeHook, RosenbrockStepView, RosenbrockStepper};
