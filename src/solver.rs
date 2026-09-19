use crate::tableau::TableauError;
use crate::{DEFAULT_EVENT_TOLERANCE, OdeProblem, Solution, SolutionConstructionError};
use thiserror::Error;

/// Controls which accepted states are retained in a [`Solution`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum SaveMode {
    /// Save the initial state and every accepted step.
    #[default]
    EveryStep,
    /// Save only the initial and final states.
    Endpoints,
}

/// Common options for adaptive ODE solvers.
///
/// Construct options with [`SolveOptions::new`] or [`SolveOptions::default`]
/// and the `with_*` methods. The fields are intentionally private so new
/// options can be added without breaking downstream code.
///
/// ```
/// use differential_equations::{SaveMode, SolveOptions};
///
/// let options = SolveOptions::new()
///     .with_tolerances(1.0e-9, 1.0e-7)
///     .with_initial_step(0.01)
///     .with_max_steps(10_000)
///     .with_save(SaveMode::Endpoints);
///
/// assert_eq!(options.absolute_tolerance(), 1.0e-9);
/// assert_eq!(options.initial_step(), Some(0.01));
/// ```
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct SolveOptions {
    /// Absolute local-error tolerance.
    pub(crate) absolute_tolerance: f64,
    /// Relative local-error tolerance.
    pub(crate) relative_tolerance: f64,
    /// Initial step-size magnitude. The solver estimates it when absent.
    pub(crate) initial_step: Option<f64>,
    /// Whether an algorithm should use adaptive step-size control.
    ///
    /// Fixed-step-only algorithms require this to be `false` and
    /// [`initial_step`](Self::initial_step) to be present.
    pub(crate) adaptive: bool,
    /// Maximum allowed step-size magnitude.
    pub(crate) max_step: f64,
    /// Maximum number of attempted steps.
    pub(crate) max_steps: usize,
    /// Requested absolute time tolerance for continuous callback roots.
    ///
    /// Localization applies a scale-aware representability floor when this is
    /// smaller than the spacing between floating-point times near the root.
    pub(crate) event_tolerance: f64,
    /// Accepted states retained in the solution.
    pub(crate) save: SaveMode,
    /// Requested output times. Empty means to follow [`save`](Self::save).
    ///
    /// Values must be finite, lie inside the time span, and be ordered in the
    /// integration direction. As in SciML, supplying values overrides the
    /// ordinary start/end/every-step saving controlled by [`save`](Self::save).
    pub(crate) save_at: Vec<f64>,
    /// Extra times that the integrator must hit exactly.
    ///
    /// This is the Rust-facing equivalent of SciML's `tstops`. Values must be
    /// finite, lie inside the time span, and be strictly ordered in the
    /// integration direction. Reaching a time stop does not itself save the
    /// state; combine this with [`save_at`](Self::save_at) when both behaviors
    /// are wanted.
    pub(crate) time_stops: Vec<f64>,
    /// Retain accepted-step method-specific dense segments for post-solve queries.
    ///
    /// This is opt-in because retaining stage data allocates per accepted step.
    /// When disabled, [`Solution::interpolate`](crate::Solution::interpolate)
    /// keeps its stable linear fallback between saved states.
    pub(crate) retain_dense_output: bool,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            absolute_tolerance: 1.0e-6,
            relative_tolerance: 1.0e-3,
            initial_step: None,
            adaptive: true,
            max_step: f64::INFINITY,
            max_steps: 100_000,
            event_tolerance: DEFAULT_EVENT_TOLERANCE,
            save: SaveMode::EveryStep,
            save_at: Vec::new(),
            time_stops: Vec::new(),
            retain_dense_output: false,
        }
    }
}

impl SolveOptions {
    /// Creates the default solver configuration for builder-style customization.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the absolute local-error tolerance.
    pub fn absolute_tolerance(&self) -> f64 {
        self.absolute_tolerance
    }

    /// Returns the relative local-error tolerance.
    pub fn relative_tolerance(&self) -> f64 {
        self.relative_tolerance
    }

    /// Returns the requested initial step-size magnitude, if one was set.
    pub fn initial_step(&self) -> Option<f64> {
        self.initial_step
    }

    /// Returns whether adaptive step-size control is enabled.
    pub fn adaptive(&self) -> bool {
        self.adaptive
    }

    /// Returns the maximum allowed step-size magnitude.
    pub fn max_step(&self) -> f64 {
        self.max_step
    }

    /// Returns the maximum number of attempted steps.
    pub fn max_steps(&self) -> usize {
        self.max_steps
    }

    /// Returns the requested callback root-localization tolerance.
    pub fn event_tolerance(&self) -> f64 {
        self.event_tolerance
    }

    /// Returns the accepted-state saving mode.
    pub fn save(&self) -> SaveMode {
        self.save
    }

    /// Returns the requested output times.
    pub fn save_at(&self) -> &[f64] {
        &self.save_at
    }

    /// Returns the extra times that the integrator must hit exactly.
    pub fn time_stops(&self) -> &[f64] {
        &self.time_stops
    }

    /// Returns whether method-specific dense-output segments are retained.
    pub fn retain_dense_output(&self) -> bool {
        self.retain_dense_output
    }

    /// Sets the absolute local-error tolerance.
    #[must_use]
    pub fn with_absolute_tolerance(mut self, absolute: f64) -> Self {
        self.absolute_tolerance = absolute;
        self
    }

    /// Sets the relative local-error tolerance.
    #[must_use]
    pub fn with_relative_tolerance(mut self, relative: f64) -> Self {
        self.relative_tolerance = relative;
        self
    }

    /// Sets the absolute and relative local-error tolerances.
    #[must_use]
    pub fn with_tolerances(mut self, absolute: f64, relative: f64) -> Self {
        self.absolute_tolerance = absolute;
        self.relative_tolerance = relative;
        self
    }

    /// Sets or clears the initial step-size magnitude.
    ///
    /// Passing a number requests that magnitude; passing `None` restores
    /// automatic initial-step selection.
    #[must_use]
    pub fn with_initial_step(mut self, step: impl Into<Option<f64>>) -> Self {
        self.initial_step = step.into();
        self
    }

    /// Enables or disables adaptive step-size control.
    #[must_use]
    pub fn with_adaptive(mut self, adaptive: bool) -> Self {
        self.adaptive = adaptive;
        self
    }

    /// Sets the maximum step-size magnitude.
    #[must_use]
    pub fn with_max_step(mut self, max_step: f64) -> Self {
        self.max_step = max_step;
        self
    }

    /// Sets the maximum number of attempted steps.
    #[must_use]
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Sets the absolute time tolerance for continuous callback root localization.
    #[must_use]
    pub fn with_event_tolerance(mut self, tolerance: f64) -> Self {
        self.event_tolerance = tolerance;
        self
    }

    /// Sets the accepted-state saving mode.
    #[must_use]
    pub fn with_save(mut self, save: SaveMode) -> Self {
        self.save = save;
        self
    }

    /// Replaces the requested output times.
    #[must_use]
    pub fn with_save_at(mut self, times: impl IntoIterator<Item = f64>) -> Self {
        self.save_at = times.into_iter().collect();
        self
    }

    /// Replaces the extra times that the integrator must hit exactly.
    #[must_use]
    pub fn with_time_stops(mut self, times: impl IntoIterator<Item = f64>) -> Self {
        self.time_stops = times.into_iter().collect();
        self
    }

    /// Enables or disables retention of method-specific accepted-step segments.
    #[must_use]
    pub fn with_dense_output(mut self, retain: bool) -> Self {
        self.retain_dense_output = retain;
        self
    }
}

/// Explains why two components cannot be used as an automatic solver pair.
///
/// Automatic solvers require both components to support adaptive stepping.
/// Pairs that allow switching back to the explicit component also require a
/// stiffness diagnostic from the implicit component. This reason is carried
/// by [`SolveError::IncompatibleAutomaticPair`] so callers can diagnose pair
/// construction without parsing an error message.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
#[non_exhaustive]
pub enum AutomaticPairIncompatibility {
    /// A component does not support adaptive step-size control.
    #[error("a component does not support adaptive stepping")]
    NonAdaptiveComponent,
    /// A component cannot provide the diagnostic required for switch-back.
    #[error("a component cannot estimate stiffness for switch-back")]
    MissingStiffnessDiagnostic,
}

/// A failure to configure or complete an ODE solve.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
#[non_exhaustive]
pub enum SolveError {
    /// An algorithm produced malformed saved trajectory data.
    #[error("invalid saved solution: {0}")]
    InvalidSolution(#[from] SolutionConstructionError),
    /// The initial state contains no components.
    #[error("the initial state is empty")]
    EmptyState,
    /// At least one initial-state component is NaN or infinite.
    #[error("the initial state contains a non-finite value")]
    NonFiniteInitialState,
    /// The initialized state lies outside an attached domain guard.
    #[error("the initialized state lies outside an attached domain guard")]
    InitialStateOutOfDomain,
    /// The time span is degenerate or contains a non-finite endpoint.
    #[error("the time span must contain distinct finite values")]
    InvalidTimeSpan,
    /// An absolute or relative tolerance is non-positive or non-finite.
    #[error("absolute and relative tolerances must be finite and positive")]
    InvalidTolerance,
    /// The configured initial step is non-positive or non-finite.
    #[error("the initial step must be finite and positive")]
    InvalidInitialStep,
    /// A fixed-step algorithm was used without an initial step.
    #[error("fixed-step integration requires an initial step")]
    InitialStepRequired,
    /// Adaptive stepping was requested from a fixed-step algorithm.
    #[error("the selected algorithm does not support adaptive stepping")]
    AdaptiveStepUnsupported,
    /// The selected algorithm cannot execute the problem's callback lifecycle.
    ///
    /// This is primarily useful to downstream algorithm implementations. The
    /// built-in drivers support callbacks, while a custom driver that does not
    /// must reject callback-bearing problems instead of silently ignoring
    /// their effects, guards, or lifecycle hooks.
    #[error("the selected algorithm does not support callbacks")]
    CallbacksUnsupported,
    /// The selected algorithm cannot act on the problem's state representation.
    #[error("the selected algorithm does not support this problem representation")]
    UnsupportedProblemRepresentation,
    /// The selected components cannot participate in in-flight automatic switching.
    #[error("the automatic solver pair is incompatible: {reason}")]
    IncompatibleAutomaticPair {
        /// The capability or invariant that the pair does not satisfy.
        reason: AutomaticPairIncompatibility,
    },
    /// The requested multistep order is not supported.
    #[error("the configured multistep order is unsupported")]
    InvalidMultistepOrder,
    /// The supplied multistep history is incomplete or inconsistent.
    #[error("the multistep solver history is incomplete or inconsistent")]
    InvalidMultistepHistory,
    /// The maximum step is non-positive or NaN.
    #[error("the maximum step must be positive and not NaN")]
    InvalidMaxStep,
    /// The maximum attempted-step count is zero.
    #[error("the maximum step count must be positive")]
    InvalidMaxSteps,
    /// The callback root-localization tolerance is invalid.
    #[error("the event-localization tolerance must be finite and positive")]
    InvalidEventTolerance,
    /// An explicit Runge–Kutta tableau violates its structural invariants.
    #[error("the explicit Runge–Kutta tableau is malformed")]
    InvalidTableau,
    /// An embedded tableau resource could not be decoded or validated.
    #[error("tableau resource error: {0}")]
    TableauResource(
        #[from]
        #[source]
        TableauError,
    ),
    /// An accepted-step dense interpolant could not be evaluated.
    #[error("dense-output interpolation failed for an accepted step")]
    DenseOutputFailed,
    /// A right-hand side returned a NaN or infinite derivative.
    #[error("the right-hand side produced a non-finite derivative")]
    NonFiniteDerivative,
    /// Requested output times are invalid for the integration direction.
    #[error("save-at times must be finite, ordered, and inside the time span")]
    InvalidSaveAt,
    /// Requested integration time stops are invalid for the integration direction.
    #[error("time stops must be finite, strictly ordered, and inside the time span")]
    InvalidTimeStops,
    /// A preset-time callback contains invalid trigger times.
    #[error("preset callback times must be finite, strictly ordered, and inside the time span")]
    InvalidPresetTimes,
    /// A vector continuous callback contains no event conditions.
    #[error("a vector continuous callback must contain at least one condition")]
    InvalidVectorCallbackLength,
    /// A continuous callback condition returned a non-finite value.
    #[error("a continuous callback condition produced a non-finite value")]
    NonFiniteCallbackCondition,
    /// A callback effect produced a non-finite state.
    #[error("a callback produced a non-finite state")]
    NonFiniteCallbackState,
    /// A manifold residual or its Jacobian produced a non-finite value.
    #[error("a manifold projection function produced a non-finite value")]
    NonFiniteManifoldProjection,
    /// A predictive domain residual produced a non-finite value.
    #[error("a predictive domain residual produced a non-finite value")]
    NonFiniteDomainResidual,
    /// An iterative callback returned a non-finite or non-advancing time,
    /// or was initialized at a different start than its configured time span.
    #[error(
        "an iterative callback must use its configured start and request finite, advancing times"
    )]
    InvalidIterativeCallbackTime,
    /// Steady-state tolerance arrays do not match the problem's state dimension.
    #[error("steady-state tolerances must contain one value or one per state component")]
    InvalidSteadyStateDimension,
    /// An out-of-place derivative has a different shape than the state.
    #[error("the derivative array shape must match the state shape")]
    DerivativeShapeMismatch,
    /// A direct function evaluation received buffers with the wrong dimension.
    #[error("function evaluation buffers must match the problem state dimension")]
    EvaluationDimensionMismatch,
    /// The manifold has more constraints than state components.
    #[error("the manifold residual dimension exceeds the state dimension")]
    InvalidManifoldDimension,
    /// A manifold projection iteration was singular or did not converge.
    #[error("the manifold projection did not converge")]
    ManifoldProjectionFailed,
    /// A callback requested a non-positive or non-finite next step.
    #[error("a callback-requested step size must be finite and positive")]
    InvalidCallbackStepSize,
    /// Callback state changed inconsistently during event localization.
    #[error("the callback selected during event localization is no longer available")]
    InvalidCallbackState,
    /// An implicit nonlinear iteration failed to converge.
    #[error("the implicit nonlinear solve did not converge")]
    NonlinearSolveFailed,
    /// An implicit linear system could not be factorized.
    #[error("the implicit linear system is singular")]
    SingularLinearSystem,
    /// Adaptive control requested an unrepresentably small step.
    #[error("the adaptive step size underflowed")]
    StepSizeUnderflow,
    /// Integration exhausted the configured attempted-step budget.
    #[error("the solver exceeded its maximum attempted step count")]
    MaxStepsExceeded,
}

/// An ODE integration algorithm.
///
/// This trait is a downstream extension point. Implementors can evaluate the
/// right-hand side through [`OdeProblem::evaluate`] and construct a checked
/// trajectory with [`Solution::from_saved`]. [`Self::solve_validated`] must
/// honor every problem policy and [`SolveOptions`] value that the algorithm
/// accepts. In particular, an implementation without a callback driver must
/// check [`OdeProblem::has_callbacks`] and return
/// [`SolveError::CallbacksUnsupported`] rather than silently skipping callback
/// effects, domain guards, initializers, or finalizers.
pub trait OdeAlgorithm {
    /// Solves a problem after validating its state, time span, and options.
    fn solve<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        validate_ode_problem(problem, options)?;
        let mut solution = self.solve_validated(problem, options)?;
        solution.set_state_shape_checked(problem.state_shape())?;
        Ok(solution)
    }

    /// Executes the numerical method after common inputs have been checked.
    ///
    /// Implementors must provide the algorithm-specific integration here and
    /// may rely on [`OdeAlgorithm::solve`] having validated the initial state,
    /// time span, tolerances, step bounds, callback tolerance, and requested
    /// output times. User code should normally call [`OdeAlgorithm::solve`] or
    /// the crate-level [`solve`] function; calling this lower-level hook
    /// directly makes the caller responsible for those common invariants.
    ///
    /// Common validation does not execute algorithm behavior. Implementors
    /// remain responsible for honoring adaptive stepping, step bounds, saving,
    /// time stops, dense-output retention, and callback lifecycle semantics,
    /// or for returning the corresponding typed error when a capability is not
    /// supported.
    fn solve_validated<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: crate::OdeFunction<P>;
}

/// Solves an ODE problem with a selected algorithm.
pub fn solve<F, P, A>(
    problem: &OdeProblem<F, P>,
    algorithm: A,
    options: &SolveOptions,
) -> Result<Solution, SolveError>
where
    F: crate::OdeFunction<P>,
    A: OdeAlgorithm,
{
    algorithm.solve(problem, options)
}

pub(crate) fn validate_ode_problem<F, P>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
) -> Result<(), SolveError> {
    validate_state_time_options(problem.initial_state(), problem.time_span(), options)?;
    validate_preset_time_sequences(problem.preset_time_sequences(), problem.time_span())?;
    validate_vector_callback_lengths(problem.vector_callback_lengths())
}

pub(crate) fn validate_vector_callback_lengths(
    lengths: impl IntoIterator<Item = usize>,
) -> Result<(), SolveError> {
    lengths
        .into_iter()
        .all(|length| length > 0)
        .then_some(())
        .ok_or(SolveError::InvalidVectorCallbackLength)
}

pub(crate) fn time_sequence_is_valid(times: &[f64], time_span: (f64, f64)) -> bool {
    let (start, end) = time_span;
    let direction = (end - start).signum();
    times.iter().all(|time| {
        time.is_finite() && direction * (*time - start) >= 0.0 && direction * (end - *time) >= 0.0
    }) && !times
        .windows(2)
        .any(|pair| direction * (pair[1] - pair[0]) <= 0.0)
}

pub(crate) fn validate_preset_time_sequences<'a>(
    sequences: impl IntoIterator<Item = &'a [f64]>,
    time_span: (f64, f64),
) -> Result<(), SolveError> {
    sequences
        .into_iter()
        .all(|times| time_sequence_is_valid(times, time_span))
        .then_some(())
        .ok_or(SolveError::InvalidPresetTimes)
}

pub(crate) fn validate_state_time_options(
    initial_state: &[f64],
    time_span: (f64, f64),
    options: &SolveOptions,
) -> Result<(), SolveError> {
    if initial_state.is_empty() {
        return Err(SolveError::EmptyState);
    }
    if !initial_state.iter().all(|value| value.is_finite()) {
        return Err(SolveError::NonFiniteInitialState);
    }

    let (start, end) = time_span;
    if !start.is_finite() || !end.is_finite() || start == end {
        return Err(SolveError::InvalidTimeSpan);
    }

    if !options.absolute_tolerance.is_finite()
        || options.absolute_tolerance <= 0.0
        || !options.relative_tolerance.is_finite()
        || options.relative_tolerance <= 0.0
    {
        return Err(SolveError::InvalidTolerance);
    }
    if options
        .initial_step
        .is_some_and(|step| !step.is_finite() || step <= 0.0)
    {
        return Err(SolveError::InvalidInitialStep);
    }
    if options.max_step.is_nan() || options.max_step <= 0.0 {
        return Err(SolveError::InvalidMaxStep);
    }
    if options.max_steps == 0 {
        return Err(SolveError::InvalidMaxSteps);
    }
    if !options.event_tolerance.is_finite() || options.event_tolerance <= 0.0 {
        return Err(SolveError::InvalidEventTolerance);
    }
    if !time_sequence_is_valid(&options.save_at, time_span) {
        return Err(SolveError::InvalidSaveAt);
    }
    if !time_sequence_is_valid(&options.time_stops, time_span) {
        return Err(SolveError::InvalidTimeStops);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{DEFAULT_EVENT_TOLERANCE, OdeAlgorithm, OdeProblem, Solution, SolverStats};

    use super::{AutomaticPairIncompatibility, SaveMode, SolveError, SolveOptions, solve};

    struct Noop;
    type TestRhs = fn(&mut [f64], &[f64], &(), f64);

    impl OdeAlgorithm for Noop {
        fn solve_validated<F, P>(
            &self,
            problem: &OdeProblem<F, P>,
            _: &SolveOptions,
        ) -> Result<Solution, SolveError>
        where
            F: crate::OdeFunction<P>,
        {
            let state = problem.initial_state().to_vec();
            Ok(Solution::new(
                vec![problem.time_span().0],
                state,
                problem.initial_state().len(),
                SolverStats::default(),
            ))
        }
    }

    fn problem(initial_state: Vec<f64>, time_span: (f64, f64)) -> OdeProblem<TestRhs, ()> {
        fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
            du.copy_from_slice(u);
        }

        OdeProblem::new(rhs, initial_state, time_span, ())
    }

    #[test]
    fn defaults_match_sciml_tolerances() {
        let options = SolveOptions::default();

        assert_eq!(options.absolute_tolerance(), 1.0e-6);
        assert_eq!(options.relative_tolerance(), 1.0e-3);
        assert_eq!(options.event_tolerance(), DEFAULT_EVENT_TOLERANCE);
        assert!(options.adaptive());
        assert_eq!(options.save(), SaveMode::EveryStep);
        assert_eq!(options.initial_step(), None);
        assert_eq!(options.max_step(), f64::INFINITY);
        assert_eq!(options.max_steps(), 100_000);
        assert!(options.save_at().is_empty());
        assert!(options.time_stops().is_empty());
        assert!(!options.retain_dense_output());
    }

    #[test]
    fn builder_style_options_cover_the_public_configuration() {
        let options = SolveOptions::new()
            .with_tolerances(1.0e-9, 1.0e-7)
            .with_initial_step(0.01)
            .with_adaptive(false)
            .with_max_step(0.1)
            .with_max_steps(42)
            .with_event_tolerance(1.0e-8)
            .with_save(SaveMode::Endpoints)
            .with_save_at([0.25, 0.5])
            .with_time_stops([0.3, 0.7])
            .with_dense_output(true);

        assert_eq!(options.absolute_tolerance(), 1.0e-9);
        assert_eq!(options.relative_tolerance(), 1.0e-7);
        assert_eq!(options.initial_step(), Some(0.01));
        assert!(!options.adaptive());
        assert_eq!(options.max_step(), 0.1);
        assert_eq!(options.max_steps(), 42);
        assert_eq!(options.event_tolerance(), 1.0e-8);
        assert_eq!(options.save(), SaveMode::Endpoints);
        assert_eq!(options.save_at(), [0.25, 0.5]);
        assert_eq!(options.time_stops(), [0.3, 0.7]);
        assert!(options.retain_dense_output());

        let cleared = options.with_initial_step(None);
        assert_eq!(cleared.initial_step(), None);
    }

    #[test]
    fn validates_problem_before_dispatch() {
        assert_eq!(
            solve(
                &problem(Vec::new(), (0.0, 1.0)),
                Noop,
                &SolveOptions::default()
            ),
            Err(SolveError::EmptyState)
        );
        assert_eq!(
            solve(
                &problem(vec![f64::NAN], (0.0, 1.0)),
                Noop,
                &SolveOptions::default()
            ),
            Err(SolveError::NonFiniteInitialState)
        );
        assert_eq!(
            solve(
                &problem(vec![1.0], (0.0, 0.0)),
                Noop,
                &SolveOptions::default()
            ),
            Err(SolveError::InvalidTimeSpan)
        );

        assert_eq!(
            Noop.solve(&problem(Vec::new(), (0.0, 1.0)), &SolveOptions::default()),
            Err(SolveError::EmptyState)
        );
    }

    #[test]
    fn validates_solver_options_before_dispatch() {
        let mut options = SolveOptions {
            absolute_tolerance: 0.0,
            ..SolveOptions::default()
        };
        assert_eq!(
            solve(&problem(vec![1.0], (0.0, 1.0)), Noop, &options),
            Err(SolveError::InvalidTolerance)
        );

        options = SolveOptions {
            initial_step: Some(f64::INFINITY),
            ..SolveOptions::default()
        };
        assert_eq!(
            solve(&problem(vec![1.0], (0.0, 1.0)), Noop, &options),
            Err(SolveError::InvalidInitialStep)
        );

        options = SolveOptions {
            event_tolerance: f64::NAN,
            ..SolveOptions::default()
        };
        assert_eq!(
            solve(&problem(vec![1.0], (0.0, 1.0)), Noop, &options),
            Err(SolveError::InvalidEventTolerance)
        );

        options = SolveOptions::default().with_time_stops([0.75, 0.25]);
        assert_eq!(
            solve(&problem(vec![1.0], (0.0, 1.0)), Noop, &options),
            Err(SolveError::InvalidTimeStops)
        );
    }

    #[test]
    fn errors_have_stable_human_readable_messages() {
        assert_eq!(
            SolveError::NonlinearSolveFailed.to_string(),
            "the implicit nonlinear solve did not converge"
        );
        assert_eq!(
            SolveError::MaxStepsExceeded.to_string(),
            "the solver exceeded its maximum attempted step count"
        );
        assert_eq!(
            SolveError::InvalidPresetTimes.to_string(),
            "preset callback times must be finite, strictly ordered, and inside the time span"
        );
        assert_eq!(
            SolveError::IncompatibleAutomaticPair {
                reason: AutomaticPairIncompatibility::MissingStiffnessDiagnostic,
            }
            .to_string(),
            "the automatic solver pair is incompatible: a component cannot estimate stiffness for switch-back"
        );
    }
}
