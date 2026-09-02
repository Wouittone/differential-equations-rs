use std::cell::RefCell;

use crate::callback::{Callback, DiscreteCallback, DiscreteTrigger};
use crate::solvers::second_order::SecondOrderCallbackSet;
use crate::{CallbackAction, CallbackSave, CallbackSet, ConfigurationError, SolveError};

/// Stops integration when every derivative is small relative to its state.
///
/// At initialization and after accepted steps, the problem's right-hand side
/// is evaluated at the current state. Each component must satisfy
/// `abs(du[i]) <= max(absolute[i], relative[i] * abs(u[i]))`. A small derivative
/// at one instant is not proof of an asymptotic steady state, particularly for
/// non-autonomous problems; use [`Self::with_min_time`] to exclude early times.
///
/// Checks follow discrete callback insertion order and see earlier effects.
/// Unsuccessful checks do not save states, count as callback invocations, or
/// invalidate solver caches. Additional derivative evaluations are included
/// in solution statistics. Scratch storage is reused between checks.
///
/// # Example
///
/// ```
/// use differential_equations::callbacks::TerminateSteadyState;
/// use differential_equations::solvers::explicit::Tsit5;
/// use differential_equations::{OdeProblem, SolveOptions, solve};
///
/// let problem = OdeProblem::new(
///     |du: &mut [f64], u: &[f64], _: &(), _| du[0] = 1.0 - u[0],
///     [0.0], (0.0, 100.0), (),
/// ).with_callback_set(TerminateSteadyState::new().into_callback_set()?);
/// let solution = solve(&problem, Tsit5, &SolveOptions::new()
///     .with_tolerances(1.0e-10, 1.0e-10))?;
/// assert!(*solution.times().last().unwrap() < 100.0);
/// assert!((solution.last_state()[0] - 1.0).abs() < 1.0e-6);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
#[must_use]
pub struct TerminateSteadyState {
    absolute: Vec<f64>,
    relative: Vec<f64>,
    min_time: Option<f64>,
    save: CallbackSave,
}

impl TerminateSteadyState {
    /// Uses absolute tolerance `1e-8` and relative tolerance `1e-6`.
    pub fn new() -> Self {
        Self {
            absolute: vec![1.0e-8],
            relative: vec![1.0e-6],
            min_time: None,
            save: CallbackSave::After,
        }
    }

    /// Sets scalar tolerances, independently of the integration tolerances.
    ///
    /// Both must be finite and non-negative. Two zero tolerances require an
    /// exactly zero derivative.
    pub fn with_tolerances(mut self, absolute: f64, relative: f64) -> Self {
        self.absolute = vec![absolute];
        self.relative = vec![relative];
        self
    }

    /// Sets componentwise tolerances in contiguous state order.
    ///
    /// Each array must contain one value (broadcast) or exactly one per state
    /// component. For second-order problems the order is velocity, then
    /// position. Dimensions are checked when the solve initializes.
    pub fn with_component_tolerances(
        mut self,
        absolute: impl IntoIterator<Item = f64>,
        relative: impl IntoIterator<Item = f64>,
    ) -> Self {
        self.absolute = absolute.into_iter().collect();
        self.relative = relative.into_iter().collect();
        self
    }

    /// Disables termination checks while `time < minimum`.
    ///
    /// This is an absolute time bound, also for backward solves, not an elapsed
    /// duration. It does not force an exact integration stop at that time.
    pub const fn with_min_time(mut self, minimum: f64) -> Self {
        self.min_time = Some(minimum);
        self
    }

    /// Selects extra callback snapshots (the current state by default).
    ///
    /// As for any terminating callback, the final state is always retained,
    /// even with [`CallbackSave::None`]. Saving the before-state also records
    /// a separate after-state at the same time under the callback contract.
    pub const fn with_save(mut self, save: CallbackSave) -> Self {
        self.save = save;
        self
    }

    /// Builds a policy using the ordinary or total split problem derivative.
    ///
    /// Scalar, vector, and matrix ndarray states use the same criterion.
    pub fn into_callback_set<P>(self) -> Result<CallbackSet<P>, ConfigurationError> {
        self.validate()?;
        let save = self.save;
        let mut callbacks = CallbackSet::new();
        callbacks
            .callbacks
            .push(Callback::Discrete(DiscreteCallback {
                trigger: DiscreteTrigger::SteadyState(SteadyStateCondition::new(self)),
                affect: Box::new(|_, _, _| Ok(CallbackAction::Terminate)),
                save,
            }));
        Ok(callbacks)
    }

    /// Builds a policy checking both acceleration and velocity (`q' = v`).
    ///
    /// Zero acceleration alone does not establish a steady second-order state.
    pub fn into_second_order_callback_set<P>(
        self,
    ) -> Result<SecondOrderCallbackSet<P>, ConfigurationError> {
        self.validate()?;
        let save = self.save;
        Ok(SecondOrderCallbackSet::new().with_steady_state(SteadyStateCondition::new(self), save))
    }

    fn validate(&self) -> Result<(), ConfigurationError> {
        for values in [&self.absolute, &self.relative] {
            if values.is_empty() || values.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return Err(ConfigurationError::InvalidParameter {
                    parameter: "steady-state tolerances",
                    reason: "must be non-empty, finite, and non-negative",
                });
            }
        }
        if self.min_time.is_some_and(|time| !time.is_finite()) {
            return Err(ConfigurationError::InvalidParameter {
                parameter: "steady-state minimum time",
                reason: "must be finite",
            });
        }
        Ok(())
    }
}

impl Default for TerminateSteadyState {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Default)]
struct Scratch {
    derivative: Vec<f64>,
    work: Vec<f64>,
}

pub(crate) struct SteadyStateCondition {
    policy: TerminateSteadyState,
    scratch: RefCell<Scratch>,
}

impl SteadyStateCondition {
    fn new(policy: TerminateSteadyState) -> Self {
        Self {
            policy,
            scratch: RefCell::new(Scratch::default()),
        }
    }

    pub(crate) fn test(
        &self,
        state: &[f64],
        second_state: &[f64],
        time: f64,
        evaluate: impl FnOnce(&mut [f64], &mut [f64]),
    ) -> Result<bool, SolveError> {
        let dimension = state.len() + second_state.len();
        if [&self.policy.absolute, &self.policy.relative]
            .iter()
            .any(|values| values.len() != 1 && values.len() != dimension)
        {
            return Err(SolveError::InvalidSteadyStateDimension);
        }
        if self.policy.min_time.is_some_and(|minimum| time < minimum) {
            return Ok(false);
        }
        let mut scratch = self.scratch.borrow_mut();
        let Scratch { derivative, work } = &mut *scratch;
        derivative.resize(dimension, f64::NAN);
        work.resize(dimension, f64::NAN);
        derivative.fill(f64::NAN);
        work.fill(f64::NAN);
        evaluate(derivative, work);
        if derivative.iter().any(|value| !value.is_finite()) {
            return Err(SolveError::NonFiniteDerivative);
        }
        Ok(derivative
            .iter()
            .zip(state.iter().chain(second_state))
            .enumerate()
            .all(|(i, (du, u))| {
                let absolute = self.policy.absolute[i % self.policy.absolute.len()];
                let relative = self.policy.relative[i % self.policy.relative.len()];
                du.abs() <= absolute.max(relative * u.abs())
            }))
    }
}
