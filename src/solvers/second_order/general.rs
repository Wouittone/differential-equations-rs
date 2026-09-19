mod algorithms;
mod arrays;
mod callbacks;
mod contract;
mod driver;
mod problem;
mod solution;

pub use algorithms::*;
pub use callbacks::SecondOrderCallbackSet;
pub use contract::{SecondOrderOdeAlgorithm, SecondOrderSolveError, solve_second_order};
pub use problem::SecondOrderOdeProblem;
pub use solution::SecondOrderSolution;

use algorithms::Method;
use callbacks::{
    ContinuousCallback, PartitionedCallback, PartitionedInterpolator, VectorContinuousCallback,
};
pub(super) use driver::{apply_finalize_callbacks, apply_initial_callbacks, apply_step_callbacks};
use solution::PartitionedDenseSegment;

use super::function::SecondOrderFunction;
use crate::callback::CallbackOutcome;
use crate::event::{
    MAX_EVENT_ROOT_ITERATIONS, effective_event_tolerance, event_interval_converged,
    times_are_numerically_equal, times_are_representably_equal,
};
use crate::integrator::{
    ControllerConfig, ControllerState, TimeStopSchedule, callback_adjusted_step,
};
use crate::linear::{factorize, solve_factorized};
use crate::solution::interpolate_value;
use crate::solver::{
    validate_preset_time_sequences, validate_state_time_options, validate_vector_callback_lengths,
};
use crate::tableau::{
    IrknBootstrapSeed, IrknTableau, RungeKuttaNystromKind, RungeKuttaNystromTableau,
};
use crate::{EventCrossing, EventDirection, SaveMode, SolveError, SolveOptions, SolverStats};
use ndarray::IxDyn;
