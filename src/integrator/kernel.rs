use super::controller::ControllerConfig;
use crate::callback::CallbackOutcome;
use crate::solution::TrajectoryRecorder;
use crate::{OdeProblem, SolveError, SolveOptions, SolverStats};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum AttemptFailurePolicy {
    Terminal,
    NonlinearOrSingular,
    NonFiniteDerivative,
}

impl AttemptFailurePolicy {
    pub(super) const fn is_recoverable(self, error: SolveError) -> bool {
        matches!(
            (self, error),
            (
                Self::NonlinearOrSingular,
                SolveError::NonlinearSolveFailed | SolveError::SingularLinearSystem
            ) | (Self::NonFiniteDerivative, SolveError::NonFiniteDerivative)
        )
    }
}

/// Properties the common driver needs without knowing a kernel's internals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct KernelCapabilities {
    pub(super) adaptive: bool,
    pub(super) controller: ControllerConfig,
    pub(super) attempt_failure_policy: AttemptFailurePolicy,
}

/// One-shot changes requested by a kernel after an attempt lifecycle hook.
///
/// This keeps algorithm transitions out of the common driver's numerical
/// interface: kernels can change their capabilities internally, discard the
/// old controller history, and scale the next proposal without changing the
/// signature of every step hook.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct KernelTransition {
    pub(crate) step_multiplier: f64,
    pub(crate) reset_controller: bool,
}

/// Why the shared driver rejected the most recent candidate or attempt.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum RejectionReason {
    /// A recoverable numerical attempt failed before producing a candidate.
    AttemptFailure(SolveError),
    /// The candidate violated a user-supplied domain policy.
    Domain,
    /// The adaptive error estimate exceeded the active controller threshold.
    ErrorEstimate(f64),
}

impl KernelTransition {
    pub(crate) const fn identity() -> Self {
        Self {
            step_multiplier: 1.0,
            reset_controller: false,
        }
    }
}

impl Default for KernelTransition {
    fn default() -> Self {
        Self::identity()
    }
}

impl KernelCapabilities {
    pub(crate) const fn new(adaptive: bool, controller_order: usize) -> Self {
        Self {
            adaptive,
            controller: ControllerConfig::default_for_order(controller_order),
            attempt_failure_policy: AttemptFailurePolicy::Terminal,
        }
    }

    pub(crate) const fn with_controller(adaptive: bool, controller: ControllerConfig) -> Self {
        Self {
            adaptive,
            controller,
            attempt_failure_policy: AttemptFailurePolicy::Terminal,
        }
    }

    pub(crate) const fn recover_nonlinear_and_singular_failures(mut self) -> Self {
        self.attempt_failure_policy = AttemptFailurePolicy::NonlinearOrSingular;
        self
    }

    /// Lets a composite retry an explicit-stage numerical blow-up with a
    /// different kernel. Standalone explicit solvers keep such failures
    /// terminal, so invalid user right-hand sides retain their existing error
    /// behavior.
    pub(crate) const fn recover_non_finite_derivatives(mut self) -> Self {
        self.attempt_failure_policy = AttemptFailurePolicy::NonFiniteDerivative;
        self
    }
}

/// The result of one numerical attempt. The candidate state is written into
/// the driver-owned buffer passed to [`StepKernel::attempt_step`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct StepEstimate {
    pub(crate) error_norm: f64,
    pub(super) proposed_factor: Option<f64>,
}

impl StepEstimate {
    pub(crate) const fn new(error_norm: f64) -> Self {
        Self {
            error_norm,
            proposed_factor: None,
        }
    }

    /// Supplies an algorithm-owned next-step ratio while retaining the shared
    /// driver's acceptance, rejection, and lifecycle machinery.
    pub(crate) const fn with_factor(error_norm: f64, proposed_factor: f64) -> Self {
        Self {
            error_norm,
            proposed_factor: Some(proposed_factor),
        }
    }
}

/// Static-dispatch boundary between integration lifecycle and numerical work.
///
/// Kernels own numerical caches. The driver owns time/state progression,
/// callbacks, saving, attempt accounting, controller policy, and termination.
#[allow(clippy::too_many_arguments)]
pub(crate) trait StepKernel<F, P>
where
    F: crate::OdeFunction<P>,
{
    fn capabilities(&self) -> KernelCapabilities;

    /// Prepares the active numerical cache before the driver queries its
    /// capabilities for the next attempt.
    ///
    /// Automatic composites use this hook to allocate and initialize a newly
    /// selected branch at the current accepted state. Ordinary kernels are
    /// already initialized before entering the loop and keep the default.
    fn prepare_attempt(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: f64,
        _: &mut SolverStats,
    ) -> Result<(), SolveError> {
        Ok(())
    }

    /// Drains a one-shot transition requested by the most recent lifecycle
    /// hook. Kernels without automatic switching keep the identity default.
    fn take_transition(&mut self) -> KernelTransition {
        KernelTransition::default()
    }

    /// Records algorithm-specific accepted-step statistics before an early
    /// callback termination can leave the numerical cache uncommitted.
    fn note_accepted_step(&mut self, _: &mut SolverStats) {}

    /// Adds kernel-specific terminal statistics to a successful solution.
    fn finalize_stats(&self, _: &mut SolverStats) {}

    /// Reports whether the effective problem representation has callbacks.
    /// Typed adapters that drive through a callback-free placeholder problem
    /// override this so the driver allocates its callback state buffer.
    fn has_callbacks(&self, problem: &OdeProblem<F, P>) -> bool {
        problem.has_callbacks()
    }

    /// Returns the next callback-owned time that the driver must hit exactly.
    fn next_callback_time_stop(
        &self,
        problem: &OdeProblem<F, P>,
        time: f64,
        direction: f64,
    ) -> Option<f64> {
        problem.next_preset_time(time, direction)
    }

    /// Applies callbacks at the initial state for the effective problem.
    fn apply_initial_callbacks(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &mut [f64],
        time: f64,
    ) -> Result<CallbackOutcome, SolveError> {
        problem.apply_initial_callbacks(state, time)
    }

    /// Applies end-of-solve hooks for the effective problem representation.
    fn apply_finalize_callbacks(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &mut [f64],
        time: f64,
    ) -> Result<bool, SolveError> {
        problem.apply_finalize_callbacks(state, time)
    }

    /// Returns the retry factor when the effective problem rejects a state.
    ///
    /// Typed adapters override this when the shared driver receives a
    /// callback-free placeholder problem.
    fn domain_rejection_factor(
        &self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
    ) -> Option<f64> {
        problem.domain_rejection_factor(state, time)
    }

    /// Reports whether the effective problem checks extrapolated states.
    fn has_predictive_domain(&self, problem: &OdeProblem<F, P>) -> bool {
        problem.has_predictive_domain()
    }

    /// Restricts an upcoming step using the effective problem's domain policy.
    fn predictive_domain_adjusted_step(
        &self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        derivative: &[f64],
        time: f64,
        proposed_step: f64,
        default_tolerance: f64,
        prediction: &mut [f64],
    ) -> Result<f64, SolveError> {
        problem.predictive_domain_adjusted_step(
            state,
            derivative,
            time,
            proposed_step,
            default_tolerance,
            prediction,
        )
    }

    /// Adjusts a controller proposal before it becomes the next attempted
    /// step. Most methods keep the proposal unchanged; interval-prediction
    /// methods can snap it to a precomputed exponential grid.
    fn modify_step(&mut self, proposed_step: f64) -> f64 {
        proposed_step
    }

    /// Reports whether this kernel supplies a complete accepted-step dense
    /// lifecycle. The shared driver otherwise provides cubic Hermite output.
    fn has_custom_dense_output(&self) -> bool {
        false
    }

    /// Reports whether callback dispatch is implemented by the kernel rather
    /// than by the [`OdeProblem`] passed to the shared driver.
    ///
    /// Typed adapters use this when the driver receives a placeholder problem
    /// but callbacks belong to another problem representation.
    fn has_custom_callback_handling(&self) -> bool {
        false
    }

    /// Evaluates the derivative used by the shared Hermite dense lifecycle.
    /// Typed problem adapters that drive through a placeholder `OdeProblem`
    /// override this hook with their real derivative representation.
    fn evaluate_dense_derivative(
        &mut self,
        problem: &OdeProblem<F, P>,
        output: &mut [f64],
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        problem
            .rhs
            .evaluate(output, state, problem.parameters(), time)?;
        stats.rhs_evaluations += 1;
        output
            .iter()
            .all(|value| value.is_finite())
            .then_some(())
            .ok_or(SolveError::NonFiniteDerivative)
    }

    fn initialize(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError>;

    fn estimate_initial_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        direction: f64,
        maximum_step: f64,
        candidate: &mut [f64],
        options: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<f64, SolveError>;

    fn attempt_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        options: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError>;

    /// Applies callbacks using the kernel's accepted-step interpolant when one
    /// is available. The default preserves endpoint-linear localization.
    #[allow(clippy::too_many_arguments)]
    fn apply_step_callbacks(
        &mut self,
        problem: &OdeProblem<F, P>,
        previous_state: &[f64],
        previous_time: f64,
        state: &mut [f64],
        time: &mut f64,
        state_before_effect: &mut [f64],
        event_tolerance: f64,
        _: &mut SolverStats,
    ) -> Result<CallbackOutcome, SolveError> {
        problem.apply_step_callbacks(
            previous_state,
            previous_time,
            state,
            time,
            state_before_effect,
            event_tolerance,
            None,
        )
    }

    /// Samples `save_at` through an accepted method-specific dense segment.
    ///
    /// The hook runs after callbacks have identified any truncated endpoint,
    /// but receives the pre-effect state so endpoint callbacks cannot corrupt
    /// the left-limit interpolant. Returning `false` keeps the compatibility
    /// endpoint recorder path.
    fn record_dense_step(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: &[f64],
        _: f64,
        _: f64,
        _: f64,
        _: bool,
        _: &mut TrajectoryRecorder<'_>,
        _: &mut SolverStats,
    ) -> Result<bool, SolveError> {
        Ok(false)
    }

    fn accept_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        previous_state: &[f64],
        state: &[f64],
        time: f64,
        accepted_step: f64,
        callback_applied: bool,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError>;

    fn reject_step(&mut self);

    /// Rejects the current attempt while preserving its reason for composite
    /// policies. Existing kernels delegate to their ordinary rejection hook.
    fn reject_step_with_reason(&mut self, _: RejectionReason) {
        self.reject_step();
    }
}
