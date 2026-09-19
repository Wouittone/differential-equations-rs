mod controller;
mod dense;
mod driver;
mod kernel;
mod schedule;

pub(crate) use controller::{ControllerConfig, ControllerState};
pub(crate) use driver::integrate;
pub(crate) use kernel::{
    KernelCapabilities, KernelTransition, RejectionReason, StepEstimate, StepKernel,
};
pub(crate) use schedule::{TimeStopSchedule, callback_adjusted_step};

#[cfg(test)]
use controller::{step_factor, step_factor_with_history};

#[cfg(test)]
mod tests;
