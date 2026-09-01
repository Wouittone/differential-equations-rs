use crate::callback::CallbackOutcome;
use crate::solution::{BorrowedHermiteSegment, HermiteSegment, TrajectoryRecorder};
use crate::{OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

const DEFAULT_SAFETY: f64 = 0.9;
const DEFAULT_MIN_FACTOR: f64 = 0.2;
const DEFAULT_MAX_FACTOR: f64 = 10.0;

/// Tracks directionally ordered integration times that a driver must hit.
///
/// Specialized drivers share this small scheduler with the common first-order
/// driver so `SolveOptions::time_stops` has identical clipping semantics
/// without copying stop-search logic into every integration loop.
pub(crate) struct TimeStopSchedule<'a> {
    stops: &'a [f64],
    next: usize,
    end: f64,
    direction: f64,
}

impl<'a> TimeStopSchedule<'a> {
    pub(crate) fn new(stops: &'a [f64], start: f64, end: f64) -> Self {
        let direction = (end - start).signum();
        let mut schedule = Self {
            stops,
            next: 0,
            end,
            direction,
        };
        schedule.accepted(start);
        schedule
    }

    pub(crate) fn clip_step_with(&self, time: f64, step: f64, additional_stop: Option<f64>) -> f64 {
        let scheduled = self.stops.get(self.next).copied().unwrap_or(self.end);
        let target = additional_stop.map_or(scheduled, |additional| {
            if self.direction * (additional - scheduled) < 0.0 {
                additional
            } else {
                scheduled
            }
        });
        if self.direction * (time + step - target) > 0.0 {
            target - time
        } else {
            step
        }
    }

    pub(crate) fn accepted(&mut self, time: f64) {
        while self
            .stops
            .get(self.next)
            .is_some_and(|stop| self.direction * (*stop - time) <= 0.0)
        {
            self.next += 1;
        }
    }
}

/// Per-family metadata for the proportional step-size controller.
///
/// Keeping the complete policy on the kernel capability lets solver families
/// preserve their existing constants while sharing the integration lifecycle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ControllerConfig {
    error_order: usize,
    safety: f64,
    minimum_factor: f64,
    maximum_factor: f64,
    rejected_acceptance_maximum: f64,
    rejection_maximum: f64,
    failed_attempt_factor: f64,
    integral_exponent: f64,
}

impl ControllerConfig {
    pub(crate) const fn proportional(
        error_order: usize,
        safety: f64,
        minimum_factor: f64,
        maximum_factor: f64,
        failed_attempt_factor: f64,
    ) -> Self {
        Self {
            error_order,
            safety,
            minimum_factor,
            maximum_factor,
            rejected_acceptance_maximum: 1.0,
            rejection_maximum: 1.0,
            failed_attempt_factor,
            integral_exponent: 0.0,
        }
    }

    /// Adds an optional integral-history exponent for PI controller metadata.
    /// Zero preserves the existing proportional controller exactly.
    #[allow(dead_code)]
    pub(crate) const fn with_integral_exponent(mut self, integral_exponent: f64) -> Self {
        self.integral_exponent = integral_exponent;
        self
    }

    const fn default_for_order(error_order: usize) -> Self {
        Self::proportional(
            error_order,
            DEFAULT_SAFETY,
            DEFAULT_MIN_FACTOR,
            DEFAULT_MAX_FACTOR,
            DEFAULT_MIN_FACTOR,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AttemptFailurePolicy {
    Terminal,
    NonlinearOrSingular,
}

impl AttemptFailurePolicy {
    const fn is_recoverable(self, error: SolveError) -> bool {
        matches!(
            (self, error),
            (
                Self::NonlinearOrSingular,
                SolveError::NonlinearSolveFailed | SolveError::SingularLinearSystem
            )
        )
    }
}

/// Properties the common driver needs without knowing a kernel's internals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct KernelCapabilities {
    adaptive: bool,
    controller: ControllerConfig,
    attempt_failure_policy: AttemptFailurePolicy,
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
}

/// The result of one numerical attempt. The candidate state is written into
/// the driver-owned buffer passed to [`StepKernel::attempt_step`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct StepEstimate {
    pub(crate) error_norm: f64,
    proposed_factor: Option<f64>,
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
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    fn capabilities(&self) -> KernelCapabilities;

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
        (problem.rhs)(output, state, problem.parameters(), time);
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
}

struct DefaultDenseState {
    start_derivative: Vec<f64>,
    endpoint_state: Vec<f64>,
    endpoint_derivative: Vec<f64>,
    start_derivative_valid: bool,
    prepared: bool,
}

impl DefaultDenseState {
    fn new(dimension: usize, enabled: bool) -> Self {
        let size = if enabled { dimension } else { 0 };
        Self {
            start_derivative: vec![0.0; size],
            endpoint_state: vec![0.0; size],
            endpoint_derivative: vec![0.0; size],
            start_derivative_valid: false,
            prepared: false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare<F, P, K>(
        &mut self,
        kernel: &mut K,
        problem: &OdeProblem<F, P>,
        previous_state: &[f64],
        previous_time: f64,
        endpoint_state: &[f64],
        endpoint_time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
        K: StepKernel<F, P>,
    {
        self.endpoint_state.copy_from_slice(endpoint_state);
        if !self.start_derivative_valid {
            kernel.evaluate_dense_derivative(
                problem,
                &mut self.start_derivative,
                previous_state,
                previous_time,
                stats,
            )?;
        }
        kernel.evaluate_dense_derivative(
            problem,
            &mut self.endpoint_derivative,
            &self.endpoint_state,
            endpoint_time,
            stats,
        )?;
        self.prepared = true;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_callbacks<F, P, K>(
        &mut self,
        kernel: &mut K,
        problem: &OdeProblem<F, P>,
        previous_state: &[f64],
        previous_time: f64,
        state: &mut [f64],
        time: &mut f64,
        state_before_effect: &mut [f64],
        event_tolerance: f64,
        enabled: bool,
        stats: &mut SolverStats,
    ) -> Result<CallbackOutcome, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
        K: StepKernel<F, P>,
    {
        if !enabled {
            return problem.apply_step_callbacks(
                previous_state,
                previous_time,
                state,
                time,
                state_before_effect,
                event_tolerance,
                None,
            );
        }
        let endpoint_time = *time;
        self.prepare(
            kernel,
            problem,
            previous_state,
            previous_time,
            state,
            endpoint_time,
            stats,
        )?;
        let segment = BorrowedHermiteSegment::new(
            previous_time,
            endpoint_time,
            previous_state,
            &self.endpoint_state,
            &self.start_derivative,
            &self.endpoint_derivative,
        )
        .map_err(|_| SolveError::NonFiniteDerivative)?;
        let mut interpolate = |sample_time: f64, output: &mut [f64]| {
            crate::solution::DenseSegment::interpolate(&segment, sample_time, output)
                .map_err(|_| SolveError::NonFiniteDerivative)
        };
        problem.apply_step_callbacks(
            previous_state,
            previous_time,
            state,
            time,
            state_before_effect,
            event_tolerance,
            Some(&mut interpolate),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn record(
        &self,
        previous_state: &[f64],
        previous_time: f64,
        attempted_time: f64,
        state: &[f64],
        time: f64,
        final_time: bool,
        recorder: &mut TrajectoryRecorder<'_>,
    ) -> Result<(), SolveError> {
        debug_assert!(self.prepared);
        let segment = BorrowedHermiteSegment::new(
            previous_time,
            attempted_time,
            previous_state,
            &self.endpoint_state,
            &self.start_derivative,
            &self.endpoint_derivative,
        )
        .map_err(|_| SolveError::NonFiniteDerivative)?;
        recorder
            .record_step_dense(
                previous_state,
                previous_time,
                state,
                time,
                final_time,
                &segment,
            )
            .map_err(|_| SolveError::NonFiniteDerivative)?;
        if recorder.retains_dense_output() {
            recorder.retain_hermite_segment(
                HermiteSegment::new_bounded(
                    previous_time,
                    attempted_time,
                    time,
                    previous_state.to_vec(),
                    self.endpoint_state.clone(),
                    self.start_derivative.clone(),
                    self.endpoint_derivative.clone(),
                )
                .map_err(|_| SolveError::NonFiniteDerivative)?,
            );
        }
        Ok(())
    }

    fn accepted(&mut self, callback_applied: bool) {
        if callback_applied || !self.prepared {
            self.start_derivative_valid = false;
        } else {
            std::mem::swap(&mut self.start_derivative, &mut self.endpoint_derivative);
            self.start_derivative_valid = true;
        }
        self.prepared = false;
    }
}

pub(crate) fn integrate<F, P, K>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
    mut kernel: K,
) -> Result<Solution, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
    K: StepKernel<F, P>,
{
    crate::solver::validate_ode_problem(problem, options)?;

    let capabilities = kernel.capabilities();
    if options.adaptive && !capabilities.adaptive {
        return Err(SolveError::AdaptiveStepUnsupported);
    }
    if !options.adaptive && options.initial_step.is_none() {
        return Err(SolveError::InitialStepRequired);
    }

    let dimension = problem.initial_state().len();
    let (start, end) = problem.time_span();
    let direction = (end - start).signum();
    let maximum_step = options.max_step.min((end - start).abs());
    let mut state = problem.initial_state().to_vec();
    let mut candidate = vec![0.0; dimension];
    let mut state_before_effect = if kernel.has_callbacks(problem) {
        vec![0.0; dimension]
    } else {
        Vec::new()
    };
    let mut stats = SolverStats::default();
    let custom_dense_output = kernel.has_custom_dense_output();
    let custom_callback_handling = kernel.has_custom_callback_handling();
    let default_dense_enabled = !custom_dense_output
        && (problem.has_continuous_callbacks()
            || !options.save_at.is_empty()
            || options.retain_dense_output);
    let default_callback_dense_enabled = !custom_dense_output && problem.has_continuous_callbacks();
    let mut default_dense = DefaultDenseState::new(dimension, default_dense_enabled);

    let mut recorder = TrajectoryRecorder::new(&state, start, options);
    let initial_callbacks = kernel.apply_initial_callbacks(problem, &mut state, start)?;
    stats.callback_invocations += initial_callbacks.invocations;
    if initial_callbacks.state_modified {
        recorder.record_callback(
            start,
            problem.initial_state(),
            &state,
            initial_callbacks,
            true,
        );
    }
    if initial_callbacks.terminate {
        return finish_successful(&mut kernel, problem, &mut state, start, recorder, stats);
    }

    kernel.initialize(problem, &state, start, &mut stats)?;
    let step_magnitude = match options.initial_step {
        Some(step) => step.min(maximum_step),
        None => kernel.estimate_initial_step(
            problem,
            &state,
            start,
            direction,
            maximum_step,
            &mut candidate,
            options,
            &mut stats,
        )?,
    };
    let mut step = kernel.modify_step(direction * step_magnitude);
    let mut time = start;
    let mut attempted_steps = 0;
    let mut time_stops = TimeStopSchedule::new(&options.time_stops, start, end);
    let mut previous_step_rejected = false;
    let mut controller_state = ControllerState::default();

    while direction * (end - time) > 0.0 {
        if attempted_steps == options.max_steps {
            return Err(SolveError::MaxStepsExceeded);
        }
        attempted_steps += 1;

        let callback_stop = kernel.next_callback_time_stop(problem, time, direction);
        let attempted_step = time_stops.clip_step_with(time, step, callback_stop);
        if time + attempted_step == time {
            return Err(SolveError::StepSizeUnderflow);
        }

        let estimate = match kernel.attempt_step(
            problem,
            &state,
            time,
            attempted_step,
            &mut candidate,
            options,
            &mut stats,
        ) {
            Ok(estimate) => estimate,
            Err(error)
                if options.adaptive
                    && capabilities.attempt_failure_policy.is_recoverable(error) =>
            {
                stats.rejected_steps += 1;
                kernel.reject_step();
                controller_state.rejected(1.0);
                step = kernel
                    .modify_step(attempted_step * capabilities.controller.failed_attempt_factor);
                previous_step_rejected = true;
                continue;
            }
            Err(error) => return Err(error),
        };
        if !candidate.iter().all(|value| value.is_finite()) {
            return Err(SolveError::NonFiniteDerivative);
        }

        if estimate.error_norm <= 1.0 {
            let previous_time = time;
            let mut next_time = time + attempted_step;
            if direction * (end - next_time) <= 0.0 {
                next_time = end;
            }
            let attempted_time = next_time;
            if custom_callback_handling && default_dense_enabled {
                // Typed adapters dispatch callbacks against a problem other
                // than the placeholder passed to this driver. Preserve the
                // full attempted endpoint before an event truncates `candidate`
                // so retained and save-at dense output remains the accepted
                // step's left-limit interpolant.
                default_dense.prepare(
                    &mut kernel,
                    problem,
                    &state,
                    previous_time,
                    &candidate,
                    attempted_time,
                    &mut stats,
                )?;
            }
            let callbacks = if custom_dense_output || custom_callback_handling {
                kernel.apply_step_callbacks(
                    problem,
                    &state,
                    previous_time,
                    &mut candidate,
                    &mut next_time,
                    &mut state_before_effect,
                    options.event_tolerance,
                    &mut stats,
                )?
            } else {
                default_dense.apply_callbacks(
                    &mut kernel,
                    problem,
                    &state,
                    previous_time,
                    &mut candidate,
                    &mut next_time,
                    &mut state_before_effect,
                    options.event_tolerance,
                    default_callback_dense_enabled,
                    &mut stats,
                )?
            };
            stats.callback_invocations += callbacks.invocations;
            stats.accepted_steps += 1;

            let dense_recorded = if !options.save_at.is_empty() || options.retain_dense_output {
                let dense_state = if callbacks.invocations == 0 {
                    &candidate
                } else {
                    &state_before_effect
                };
                if custom_dense_output {
                    kernel.record_dense_step(
                        problem,
                        &state,
                        dense_state,
                        previous_time,
                        attempted_time,
                        next_time,
                        next_time == end,
                        &mut recorder,
                        &mut stats,
                    )?
                } else {
                    if !default_dense.prepared {
                        default_dense.prepare(
                            &mut kernel,
                            problem,
                            &state,
                            previous_time,
                            dense_state,
                            attempted_time,
                            &mut stats,
                        )?;
                    }
                    default_dense.record(
                        &state,
                        previous_time,
                        attempted_time,
                        dense_state,
                        next_time,
                        next_time == end,
                        &mut recorder,
                    )?;
                    true
                }
            } else {
                false
            };
            if !dense_recorded {
                recorder.record_step(
                    &state,
                    previous_time,
                    if callbacks.invocations == 0 {
                        &candidate
                    } else {
                        &state_before_effect
                    },
                    next_time,
                    next_time == end,
                );
            }
            if callbacks.invocations > 0 {
                recorder.record_callback(
                    next_time,
                    &state_before_effect,
                    &candidate,
                    callbacks,
                    next_time == end,
                );
            }
            if callbacks.terminate {
                return finish_successful(
                    &mut kernel,
                    problem,
                    &mut candidate,
                    next_time,
                    recorder,
                    stats,
                );
            }

            time = next_time;
            time_stops.accepted(time);
            std::mem::swap(&mut state, &mut candidate);
            kernel.accept_step(
                problem,
                &candidate,
                &state,
                time,
                time - previous_time,
                callbacks.invocations > 0,
                &mut stats,
            )?;
            if !custom_dense_output {
                default_dense.accepted(callbacks.invocations > 0);
            }

            if options.adaptive {
                if callbacks.invocations > 0 {
                    // A callback may change the accepted state discontinuously.
                    // Do not let an error measured before that mutation bias the
                    // next PI proposal.
                    controller_state.reset();
                }
                controller_state.accepted(estimate.error_norm);
                let mut factor = estimate.proposed_factor.unwrap_or_else(|| {
                    controller_state.factor(estimate.error_norm, capabilities.controller)
                });
                if previous_step_rejected {
                    factor = factor.min(capabilities.controller.rejected_acceptance_maximum);
                }
                step = kernel
                    .modify_step(direction * (attempted_step.abs() * factor).min(maximum_step));
            }
            previous_step_rejected = false;
        } else {
            stats.rejected_steps += 1;
            kernel.reject_step();
            controller_state.rejected(estimate.error_norm);
            let factor = estimate
                .proposed_factor
                .unwrap_or_else(|| {
                    controller_state.factor(estimate.error_norm, capabilities.controller)
                })
                .min(capabilities.controller.rejection_maximum);
            step = kernel.modify_step(attempted_step * factor);
            previous_step_rejected = true;
        }
    }

    finish_successful(&mut kernel, problem, &mut state, time, recorder, stats)
}

fn finish_successful<F, P, K>(
    kernel: &mut K,
    problem: &OdeProblem<F, P>,
    state: &mut [f64],
    time: f64,
    mut recorder: TrajectoryRecorder<'_>,
    stats: SolverStats,
) -> Result<Solution, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
    K: StepKernel<F, P>,
{
    if kernel.apply_finalize_callbacks(problem, state, time)? {
        recorder.synchronize_endpoint(time, state);
    }
    Ok(recorder.finish(stats))
}

fn step_factor(error: f64, controller: ControllerConfig) -> f64 {
    if error == 0.0 {
        controller.maximum_factor
    } else if error.is_finite() {
        (controller.safety * error.powf(-1.0 / controller.error_order as f64))
            .clamp(controller.minimum_factor, controller.maximum_factor)
    } else {
        controller.minimum_factor
    }
}

#[allow(dead_code)]
fn step_factor_with_history(
    error: f64,
    previous_error: Option<f64>,
    controller: ControllerConfig,
) -> f64 {
    let proportional = step_factor(error, controller);
    if controller.integral_exponent == 0.0 {
        return proportional;
    }
    let Some(previous_error) = previous_error.filter(|value| value.is_finite() && *value > 0.0)
    else {
        return proportional;
    };
    if !error.is_finite() || error <= 0.0 {
        return proportional;
    }
    (proportional * previous_error.powf(controller.integral_exponent))
        .clamp(controller.minimum_factor, controller.maximum_factor)
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ControllerState {
    previous_error: Option<f64>,
}

#[allow(dead_code)]
impl ControllerState {
    pub(crate) fn factor(&self, error: f64, controller: ControllerConfig) -> f64 {
        step_factor_with_history(error, self.previous_error, controller)
    }

    pub(crate) fn accepted(&mut self, error: f64) {
        self.previous_error = error.is_finite().then_some(error.max(f64::MIN_POSITIVE));
    }

    pub(crate) fn rejected(&mut self, error: f64) {
        self.previous_error = error.is_finite().then_some(error.max(f64::MIN_POSITIVE));
    }

    pub(crate) fn reset(&mut self) {
        self.previous_error = None;
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use super::{
        ControllerConfig, ControllerState, KernelCapabilities, StepEstimate, StepKernel, integrate,
        step_factor, step_factor_with_history,
    };
    use crate::{CallbackAction, OdeProblem, SaveMode, SolveError, SolveOptions, SolverStats};

    struct MockKernel {
        errors: Vec<f64>,
        failures: Vec<Option<SolveError>>,
        recover_failures: bool,
        failed_attempt_factor: f64,
        attempts: usize,
        initialize_calls: usize,
        accept_calls: usize,
        reject_calls: usize,
        first_candidate: Option<*const f64>,
        second_candidate: Option<*const f64>,
        unexpected_candidate: bool,
    }

    impl MockKernel {
        fn fixed() -> Self {
            Self::with_errors(vec![0.0])
        }

        fn with_errors(errors: Vec<f64>) -> Self {
            Self {
                errors,
                failures: Vec::new(),
                recover_failures: false,
                failed_attempt_factor: 0.2,
                attempts: 0,
                initialize_calls: 0,
                accept_calls: 0,
                reject_calls: 0,
                first_candidate: None,
                second_candidate: None,
                unexpected_candidate: false,
            }
        }

        fn with_failures(failures: Vec<Option<SolveError>>) -> Self {
            let mut kernel = Self::with_errors(vec![0.0]);
            kernel.failures = failures;
            kernel.recover_failures = true;
            kernel
        }

        fn observe_candidate(&mut self, pointer: *const f64) {
            if self.first_candidate.is_none() {
                self.first_candidate = Some(pointer);
            } else if self.first_candidate != Some(pointer) && self.second_candidate.is_none() {
                self.second_candidate = Some(pointer);
            } else if self.first_candidate != Some(pointer)
                && self.second_candidate != Some(pointer)
            {
                self.unexpected_candidate = true;
            }
        }
    }

    impl<F, P> StepKernel<F, P> for &mut MockKernel
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        fn capabilities(&self) -> KernelCapabilities {
            let capabilities = KernelCapabilities::with_controller(
                true,
                ControllerConfig::proportional(1, 0.9, 0.2, 10.0, self.failed_attempt_factor),
            );
            if self.recover_failures {
                capabilities.recover_nonlinear_and_singular_failures()
            } else {
                capabilities
            }
        }

        fn initialize(
            &mut self,
            _: &OdeProblem<F, P>,
            _: &[f64],
            _: f64,
            _: &mut SolverStats,
        ) -> Result<(), SolveError> {
            self.initialize_calls += 1;
            Ok(())
        }

        fn estimate_initial_step(
            &mut self,
            _: &OdeProblem<F, P>,
            _: &[f64],
            _: f64,
            _: f64,
            maximum_step: f64,
            _: &mut [f64],
            _: &SolveOptions,
            _: &mut SolverStats,
        ) -> Result<f64, SolveError> {
            Ok(maximum_step.min(0.25))
        }

        fn attempt_step(
            &mut self,
            _: &OdeProblem<F, P>,
            state: &[f64],
            _: f64,
            step: f64,
            candidate: &mut [f64],
            _: &SolveOptions,
            _: &mut SolverStats,
        ) -> Result<StepEstimate, SolveError> {
            self.observe_candidate(candidate.as_ptr());
            let attempt = self.attempts;
            self.attempts += 1;
            if let Some(error) = self.failures.get(attempt).copied().flatten() {
                candidate.fill(f64::NAN);
                return Err(error);
            }
            for (candidate, state) in candidate.iter_mut().zip(state) {
                *candidate = state + step;
            }
            let error = self
                .errors
                .get(attempt)
                .copied()
                .unwrap_or_else(|| *self.errors.last().unwrap());
            Ok(StepEstimate::new(error))
        }

        fn accept_step(
            &mut self,
            _: &OdeProblem<F, P>,
            _: &[f64],
            _: &[f64],
            _: f64,
            _: f64,
            _: bool,
            _: &mut SolverStats,
        ) -> Result<(), SolveError> {
            self.accept_calls += 1;
            Ok(())
        }

        fn reject_step(&mut self) {
            self.reject_calls += 1;
        }
    }

    type TestRhs = fn(&mut [f64], &[f64], &(), f64);

    fn unit_problem(span: (f64, f64), initial: f64) -> OdeProblem<TestRhs, ()> {
        fn unit_rate(du: &mut [f64], _: &[f64], _: &(), _: f64) {
            du[0] = 1.0;
        }
        OdeProblem::new(unit_rate, vec![initial], span, ())
    }

    fn fixed_options(step: f64) -> SolveOptions {
        SolveOptions {
            adaptive: false,
            initial_step: Some(step),
            save: SaveMode::EveryStep,
            ..SolveOptions::default()
        }
    }

    #[test]
    fn rejection_has_no_callback_or_save_side_effects() {
        let effects = Rc::new(Cell::new(0));
        let effect_count = Rc::clone(&effects);
        let problem = unit_problem((0.0, 0.5), 0.0).with_discrete_callback(
            |_, _, time| time > 0.0,
            move |_, _, _| {
                effect_count.set(effect_count.get() + 1);
                CallbackAction::Continue
            },
        );
        let mut kernel = MockKernel::with_errors(vec![4.0, 0.0]);
        let options = SolveOptions {
            initial_step: Some(0.5),
            save: SaveMode::EveryStep,
            ..SolveOptions::default()
        };

        let solution = integrate(&problem, &options, &mut kernel).unwrap();

        assert_eq!(kernel.reject_calls, 1);
        assert_eq!(effects.get(), solution.stats().accepted_steps);
        assert_eq!(solution.times().len(), solution.stats().accepted_steps + 1);
        assert_eq!(solution.stats().rejected_steps, 1);
    }

    #[test]
    fn integrates_backward_and_clips_the_endpoint() {
        let problem = unit_problem((1.0, 0.0), 1.0);
        let mut kernel = MockKernel::fixed();
        let solution = integrate(&problem, &fixed_options(0.3), &mut kernel).unwrap();

        assert_eq!(solution.times().last(), Some(&0.0));
        assert!((solution.last_state()[0]).abs() < 1.0e-15);
        assert_eq!(solution.stats().accepted_steps, 4);
    }

    #[test]
    fn fixed_steps_hit_time_stops_then_resume_the_configured_step() {
        let problem = unit_problem((0.0, 1.0), 0.0);
        let options = fixed_options(0.4).with_time_stops([0.25, 0.5]);
        let mut kernel = MockKernel::fixed();

        let solution = integrate(&problem, &options, &mut kernel).unwrap();

        assert_eq!(solution.times(), &[0.0, 0.25, 0.5, 0.9, 1.0]);
        assert_eq!(solution.values(), solution.times());
    }

    #[test]
    fn backward_time_stops_follow_the_integration_direction() {
        let problem = unit_problem((1.0, 0.0), 1.0);
        let options = fixed_options(0.4).with_time_stops([0.75, 0.2]);
        let mut kernel = MockKernel::fixed();

        let solution = integrate(&problem, &options, &mut kernel).unwrap();

        assert_eq!(solution.times(), &[1.0, 0.75, 0.35, 0.2, 0.0]);
        assert!(solution.last_state()[0].abs() < 1.0e-15);
    }

    #[test]
    fn time_stops_do_not_force_solution_output() {
        let problem = unit_problem((0.0, 1.0), 0.0);
        let options = fixed_options(0.4)
            .with_save(SaveMode::Endpoints)
            .with_time_stops([0.25, 0.5]);
        let mut kernel = MockKernel::fixed();

        let solution = integrate(&problem, &options, &mut kernel).unwrap();

        assert_eq!(solution.times(), &[0.0, 1.0]);
        assert_eq!(solution.stats().accepted_steps, 4);
    }

    #[test]
    fn discrete_callbacks_can_act_at_exact_time_stops() {
        let effects = Rc::new(Cell::new(0));
        let effect_count = Rc::clone(&effects);
        let problem = unit_problem((0.0, 1.0), 0.0).with_discrete_callback(
            |_, _, time| time == 0.3,
            move |state, _, _| {
                effect_count.set(effect_count.get() + 1);
                state[0] += 1.0;
                CallbackAction::Continue
            },
        );
        let options = fixed_options(0.4).with_time_stops([0.3]);
        let mut kernel = MockKernel::fixed();

        let solution = integrate(&problem, &options, &mut kernel).unwrap();

        assert_eq!(effects.get(), 1);
        assert_eq!(solution.stats().callback_invocations, 1);
        assert!((solution.last_state()[0] - 2.0).abs() < 1.0e-15);
    }

    #[test]
    fn callbacks_record_pre_effect_samples_and_force_the_effect_state() {
        let problem = unit_problem((0.0, 1.0), 0.0).with_discrete_callback(
            |_, _, time| time >= 0.6,
            |state, _, _| {
                state[0] = 10.0;
                CallbackAction::Continue
            },
        );
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.6),
            save: SaveMode::Endpoints,
            save_at: vec![0.2, 0.5],
            ..SolveOptions::default()
        };
        let mut kernel = MockKernel::fixed();

        let solution = integrate(&problem, &options, &mut kernel).unwrap();

        assert_eq!(solution.times()[..3], [0.2, 0.5, 0.6]);
        assert!((solution.state(0).unwrap()[0] - 0.2).abs() < 1.0e-15);
        assert!((solution.state(1).unwrap()[0] - 0.5).abs() < 1.0e-15);
        assert_eq!(solution.state(2), Some([10.0].as_slice()));
    }

    #[test]
    fn terminating_effect_returns_before_the_kernel_accept_hook() {
        let problem = unit_problem((0.0, 1.0), 0.0).with_continuous_callback(
            |state, _, _| state[0] - 0.5,
            |state, _, _| {
                state[0] = 42.0;
                CallbackAction::Terminate
            },
        );
        let mut kernel = MockKernel::fixed();

        let solution = integrate(&problem, &fixed_options(1.0), &mut kernel).unwrap();

        assert_eq!(solution.last_state(), &[42.0]);
        assert_eq!(kernel.accept_calls, 0);
        assert_eq!(kernel.attempts, 1);
    }

    #[test]
    fn initial_termination_never_initializes_the_kernel() {
        let problem = unit_problem((0.0, 1.0), 0.0).with_discrete_callback(
            |_, _, time| time == 0.0,
            |state, _, _| {
                state[0] = 7.0;
                CallbackAction::Terminate
            },
        );
        let mut kernel = MockKernel::fixed();

        let solution = integrate(&problem, &fixed_options(0.25), &mut kernel).unwrap();

        assert_eq!(solution.last_state(), &[7.0]);
        assert_eq!(kernel.initialize_calls, 0);
        assert_eq!(kernel.attempts, 0);
    }

    #[test]
    fn save_at_uses_the_accepted_segment() {
        let problem = unit_problem((0.0, 1.0), 0.0);
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.4),
            save_at: vec![0.1, 0.7, 1.0],
            ..SolveOptions::default()
        };
        let mut kernel = MockKernel::fixed();
        let solution = integrate(&problem, &options, &mut kernel).unwrap();

        assert_eq!(solution.times(), &[0.1, 0.7, 1.0]);
        assert_eq!(solution.values(), &[0.1, 0.7, 1.0]);
    }

    #[test]
    fn reports_step_underflow_before_attempting_the_kernel() {
        let problem = unit_problem((1.0, 2.0), 0.0);
        let mut kernel = MockKernel::fixed();
        let result = integrate(&problem, &fixed_options(f64::MIN_POSITIVE), &mut kernel);

        assert_eq!(result, Err(SolveError::StepSizeUnderflow));
        assert_eq!(kernel.attempts, 0);
    }

    #[test]
    fn max_steps_counts_rejected_and_accepted_attempts() {
        let problem = unit_problem((0.0, 1.0), 0.0);
        let options = SolveOptions {
            initial_step: Some(0.5),
            max_steps: 2,
            ..SolveOptions::default()
        };
        let mut kernel = MockKernel::with_errors(vec![4.0, 0.0]);

        assert_eq!(
            integrate(&problem, &options, &mut kernel),
            Err(SolveError::MaxStepsExceeded)
        );
        assert_eq!(kernel.attempts, 2);
    }

    #[test]
    fn recoverable_attempt_failures_reject_without_using_the_candidate() {
        let effects = Rc::new(Cell::new(0));
        let effect_count = Rc::clone(&effects);
        let problem = unit_problem((0.0, 0.5), 0.0).with_discrete_callback(
            |_, _, time| time > 0.0,
            move |_, _, _| {
                effect_count.set(effect_count.get() + 1);
                CallbackAction::Continue
            },
        );
        let options = SolveOptions {
            initial_step: Some(0.5),
            save: SaveMode::EveryStep,
            ..SolveOptions::default()
        };
        let mut kernel = MockKernel::with_failures(vec![
            Some(SolveError::NonlinearSolveFailed),
            Some(SolveError::SingularLinearSystem),
            None,
        ]);

        let solution = integrate(&problem, &options, &mut kernel).unwrap();

        assert_eq!(kernel.reject_calls, 2);
        assert_eq!(solution.stats().rejected_steps, 2);
        assert_eq!(effects.get(), solution.stats().accepted_steps);
        assert_eq!(solution.times().len(), solution.stats().accepted_steps + 1);
        assert_eq!(solution.last_state(), &[0.5]);
    }

    #[test]
    fn failed_attempt_shrink_is_checked_for_underflow_before_retry() {
        let problem = unit_problem((1.0, 2.0), 0.0);
        let options = SolveOptions {
            initial_step: Some(f64::EPSILON),
            ..SolveOptions::default()
        };
        let mut kernel = MockKernel::with_failures(vec![Some(SolveError::NonlinearSolveFailed)]);

        assert_eq!(
            integrate(&problem, &options, &mut kernel),
            Err(SolveError::StepSizeUnderflow)
        );
        assert_eq!(kernel.attempts, 1);
        assert_eq!(kernel.reject_calls, 1);
        assert_eq!(kernel.accept_calls, 0);
    }

    #[test]
    fn recoverable_failure_policy_is_terminal_in_fixed_step_mode() {
        let problem = unit_problem((0.0, 1.0), 0.0);
        let mut kernel = MockKernel::with_failures(vec![Some(SolveError::SingularLinearSystem)]);

        assert_eq!(
            integrate(&problem, &fixed_options(0.25), &mut kernel),
            Err(SolveError::SingularLinearSystem)
        );
        assert_eq!(kernel.attempts, 1);
        assert_eq!(kernel.reject_calls, 0);
        assert_eq!(kernel.accept_calls, 0);
    }

    #[test]
    fn pi_controller_metadata_uses_previous_error_without_changing_defaults() {
        let proportional = ControllerConfig::proportional(5, 0.9, 0.2, 10.0, 0.2);
        assert_eq!(
            step_factor_with_history(0.25, Some(0.5), proportional),
            step_factor_with_history(0.25, Some(0.5), proportional.with_integral_exponent(0.0))
        );
        let pi = proportional.with_integral_exponent(0.2);
        assert!(step_factor_with_history(0.25, Some(0.5), pi) < step_factor(0.25, pi));
        assert_eq!(
            step_factor_with_history(0.25, None, pi),
            step_factor(0.25, pi)
        );
        let mut state = ControllerState::default();
        state.accepted(0.5);
        assert!(state.factor(0.25, pi) < step_factor(0.25, pi));
        state.reset();
        assert_eq!(state.factor(0.25, pi), step_factor(0.25, pi));
    }

    #[test]
    fn reuses_exactly_two_driver_state_buffers_across_steps() {
        let problem = unit_problem((0.0, 1.0), 0.0);
        let mut kernel = MockKernel::fixed();
        let solution = integrate(&problem, &fixed_options(0.01), &mut kernel).unwrap();

        assert_eq!(solution.stats().accepted_steps, 100);
        assert!(kernel.first_candidate.is_some());
        assert!(kernel.second_candidate.is_some());
        assert!(!kernel.unexpected_candidate);
    }
}
