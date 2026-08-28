use crate::{DEFAULT_EVENT_TOLERANCE, OdeProblem, Solution};
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
#[derive(Clone, Debug, PartialEq)]
pub struct SolveOptions {
    /// Absolute local-error tolerance.
    pub absolute_tolerance: f64,
    /// Relative local-error tolerance.
    pub relative_tolerance: f64,
    /// Initial step-size magnitude. The solver estimates it when absent.
    pub initial_step: Option<f64>,
    /// Whether an algorithm should use adaptive step-size control.
    ///
    /// Fixed-step-only algorithms require this to be `false` and
    /// [`initial_step`](Self::initial_step) to be present.
    pub adaptive: bool,
    /// Maximum allowed step-size magnitude.
    pub max_step: f64,
    /// Maximum number of attempted steps.
    pub max_steps: usize,
    /// Requested absolute time tolerance for continuous callback roots.
    ///
    /// Localization applies a scale-aware representability floor when this is
    /// smaller than the spacing between floating-point times near the root.
    pub event_tolerance: f64,
    /// Accepted states retained in the solution.
    pub save: SaveMode,
    /// Requested output times. Empty means to follow [`save`](Self::save).
    ///
    /// Values must be finite, lie inside the time span, and be ordered in the
    /// integration direction. As in SciML, supplying values overrides the
    /// ordinary start/end/every-step saving controlled by [`save`](Self::save).
    pub save_at: Vec<f64>,
    /// Retain accepted-step method-specific dense segments for post-solve queries.
    ///
    /// This is opt-in because retaining stage data allocates per accepted step.
    /// When disabled, [`Solution::interpolate`](crate::Solution::interpolate)
    /// keeps its stable linear fallback between saved states.
    pub retain_dense_output: bool,
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
            retain_dense_output: false,
        }
    }
}

impl SolveOptions {
    /// Creates the default solver configuration for builder-style customization.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the absolute and relative local-error tolerances.
    #[must_use]
    pub fn with_tolerances(mut self, absolute: f64, relative: f64) -> Self {
        self.absolute_tolerance = absolute;
        self.relative_tolerance = relative;
        self
    }

    /// Sets the initial step-size magnitude.
    #[must_use]
    pub fn with_initial_step(mut self, step: f64) -> Self {
        self.initial_step = Some(step);
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

    /// Enables or disables retention of method-specific accepted-step segments.
    #[must_use]
    pub fn with_dense_output(mut self, retain: bool) -> Self {
        self.retain_dense_output = retain;
        self
    }
}

/// A failure to configure or complete an ODE solve.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
#[non_exhaustive]
pub enum SolveError {
    /// The initial state contains no components.
    #[error("the initial state is empty")]
    EmptyState,
    /// At least one initial-state component is NaN or infinite.
    #[error("the initial state contains a non-finite value")]
    NonFiniteInitialState,
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
    /// An accepted-step dense interpolant could not be evaluated.
    #[error("dense-output interpolation failed for an accepted step")]
    DenseOutputFailed,
    /// A right-hand side returned a NaN or infinite derivative.
    #[error("the right-hand side produced a non-finite derivative")]
    NonFiniteDerivative,
    /// Requested output times are invalid for the integration direction.
    #[error("save-at times must be finite, ordered, and inside the time span")]
    InvalidSaveAt,
    /// A continuous callback condition returned a non-finite value.
    #[error("a continuous callback condition produced a non-finite value")]
    NonFiniteCallbackCondition,
    /// A callback effect produced a non-finite state.
    #[error("a callback produced a non-finite state")]
    NonFiniteCallbackState,
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
pub trait OdeAlgorithm {
    /// Solves a problem after validating its state, time span, and options.
    fn solve<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        validate_ode_problem(problem, options)?;
        self.solve_validated(problem, options)
    }

    /// Executes the numerical method after common inputs have been checked.
    ///
    /// Implementors must provide the algorithm-specific integration here and
    /// may rely on [`OdeAlgorithm::solve`] having validated the initial state,
    /// time span, tolerances, step bounds, callback tolerance, and requested
    /// output times. User code should normally call [`OdeAlgorithm::solve`] or
    /// the crate-level [`solve`] function; calling this lower-level hook
    /// directly makes the caller responsible for those common invariants.
    fn solve_validated<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64);
}

/// Solves an ODE problem with a selected algorithm.
pub fn solve<F, P, A>(
    problem: &OdeProblem<F, P>,
    algorithm: A,
    options: &SolveOptions,
) -> Result<Solution, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
    A: OdeAlgorithm,
{
    algorithm.solve(problem, options)
}

pub(crate) fn validate_ode_problem<F, P>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
) -> Result<(), SolveError> {
    validate_state_time_options(problem.initial_state(), problem.time_span(), options)
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
    let direction = (end - start).signum();
    if !options.save_at.iter().all(|time| {
        time.is_finite() && direction * (*time - start) >= 0.0 && direction * (end - *time) >= 0.0
    }) || options
        .save_at
        .windows(2)
        .any(|pair| direction * (pair[1] - pair[0]) <= 0.0)
    {
        return Err(SolveError::InvalidSaveAt);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{DEFAULT_EVENT_TOLERANCE, OdeAlgorithm, OdeProblem, Solution, SolverStats};

    use super::{SaveMode, SolveError, SolveOptions, solve};

    struct Noop;
    type TestRhs = fn(&mut [f64], &[f64], &(), f64);

    impl OdeAlgorithm for Noop {
        fn solve_validated<F, P>(
            &self,
            problem: &OdeProblem<F, P>,
            _: &SolveOptions,
        ) -> Result<Solution, SolveError>
        where
            F: Fn(&mut [f64], &[f64], &P, f64),
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

        assert_eq!(options.absolute_tolerance, 1.0e-6);
        assert_eq!(options.relative_tolerance, 1.0e-3);
        assert_eq!(options.event_tolerance, DEFAULT_EVENT_TOLERANCE);
        assert!(options.adaptive);
        assert_eq!(options.save, SaveMode::EveryStep);
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
            .with_save_at([0.25, 0.5]);

        assert_eq!(options.absolute_tolerance, 1.0e-9);
        assert_eq!(options.relative_tolerance, 1.0e-7);
        assert_eq!(options.initial_step, Some(0.01));
        assert!(!options.adaptive);
        assert_eq!(options.max_step, 0.1);
        assert_eq!(options.max_steps, 42);
        assert_eq!(options.event_tolerance, 1.0e-8);
        assert_eq!(options.save, SaveMode::Endpoints);
        assert_eq!(options.save_at, [0.25, 0.5]);
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
    }
}
