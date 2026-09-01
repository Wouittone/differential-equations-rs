//! Reusable callback policies built on [`crate::CallbackSet`].
//!
//! These policies cover common timing and observation tasks without requiring
//! callers to reproduce scheduling conditions. They can be combined with
//! custom callbacks using [`crate::CallbackSet::append`] before attaching the
//! resulting set to a problem.

use std::rc::Rc;

use crate::ConfigurationError;
use crate::callback::{
    Callback, CallbackAction, CallbackSave, CallbackSet, DiscreteCallback, DiscreteTrigger,
    PeriodicTimes,
};
use crate::event::times_are_representably_equal;
use crate::solver::time_sequence_is_valid;
use crate::solvers::second_order::SecondOrderCallbackSet;

/// Configuration for an integration-time periodic callback.
///
/// With the default phase, effects occur every `period` units after the start
/// of the time span. The initial and final times are not forced by default.
/// A nonzero phase shifts the first periodic effect by that amount from the
/// start. For backward solves, both period and phase remain positive
/// magnitudes and the schedule follows the integration direction.
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use]
pub struct PeriodicCallback {
    period: f64,
    phase: f64,
    initial_affect: bool,
    final_affect: bool,
    save: CallbackSave,
}

impl PeriodicCallback {
    /// Creates a periodic policy with no phase or forced boundary effects.
    pub const fn new(period: f64) -> Self {
        Self {
            period,
            phase: 0.0,
            initial_affect: false,
            final_affect: false,
            save: CallbackSave::After,
        }
    }

    /// Shifts the periodic schedule from the initial time by `phase`.
    pub const fn with_phase(mut self, phase: f64) -> Self {
        self.phase = phase;
        self
    }

    /// Selects whether the effect is also applied at the initial time.
    pub const fn with_initial_affect(mut self, enabled: bool) -> Self {
        self.initial_affect = enabled;
        self
    }

    /// Selects whether the effect is also applied at the final time.
    pub const fn with_final_affect(mut self, enabled: bool) -> Self {
        self.final_affect = enabled;
        self
    }

    /// Selects which states are saved around each effect.
    pub const fn with_save(mut self, save: CallbackSave) -> Self {
        self.save = save;
        self
    }

    /// Builds a callback set for `time_span` using the supplied effect.
    ///
    /// The schedule is evaluated mathematically as integration advances, so
    /// memory use does not grow with the number of periods.
    pub fn into_callback_set<P, A>(
        self,
        time_span: (f64, f64),
        affect: A,
    ) -> Result<CallbackSet<P>, ConfigurationError>
    where
        A: Fn(&mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        let times = self.schedule(time_span)?;

        Ok(CallbackSet {
            callbacks: vec![Callback::Discrete(DiscreteCallback {
                trigger: DiscreteTrigger::Periodic(times),
                affect: Box::new(affect),
                save: self.save,
            })],
            initializers: Vec::new(),
            finalizers: Vec::new(),
            step_guards: Vec::new(),
        })
    }

    /// Builds a partitioned callback set for a second-order problem.
    pub fn into_second_order_callback_set<P, A>(
        self,
        time_span: (f64, f64),
        affect: A,
    ) -> Result<SecondOrderCallbackSet<P>, ConfigurationError>
    where
        A: Fn(&mut [f64], &mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        let times = self.schedule(time_span)?;

        Ok(SecondOrderCallbackSet::new().with_periodic_callback_saving(times, self.save, affect))
    }

    fn schedule(self, time_span: (f64, f64)) -> Result<PeriodicTimes, ConfigurationError> {
        validate_time_span(time_span)?;
        if !self.period.is_finite() || self.period <= 0.0 {
            return Err(ConfigurationError::InvalidParameter {
                parameter: "callback period",
                reason: "must be finite and positive",
            });
        }
        if !self.phase.is_finite() || self.phase < 0.0 {
            return Err(ConfigurationError::InvalidParameter {
                parameter: "callback phase",
                reason: "must be finite and non-negative",
            });
        }
        validate_representable_period(time_span, self.period, self.phase)?;

        Ok(PeriodicTimes::new(
            time_span,
            self.period,
            self.phase,
            self.initial_affect,
            self.final_affect,
        ))
    }
}

/// Rejects candidate steps whose state lies outside an application-defined domain.
///
/// The predicate is evaluated on the initialized state and on each finite
/// candidate state before callbacks run or output is saved. Returning `true`
/// for a candidate rejects that attempt and retries it with a smaller step.
/// Rejecting the initialized state returns
/// [`crate::SolveError::InitialStateOutOfDomain`]. When several guards reject
/// the same candidate, the solver uses their smallest reduction factor.
/// This checks the state produced by the numerical method; unlike SciML's
/// `PositiveDomain`, it does not extrapolate or project state components.
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct DomainGuard<G> {
    is_out_of_domain: G,
    reduction_factor: f64,
}

impl<G> DomainGuard<G> {
    /// Creates a guard that halves the attempted step after a rejection.
    pub const fn new(is_out_of_domain: G) -> Self {
        Self {
            is_out_of_domain,
            reduction_factor: 0.5,
        }
    }

    /// Sets the factor applied to the rejected attempted-step magnitude.
    ///
    /// The factor is validated when a callback set is built and must lie
    /// strictly between zero and one.
    pub const fn with_reduction_factor(mut self, reduction_factor: f64) -> Self {
        self.reduction_factor = reduction_factor;
        self
    }

    /// Builds a guard policy for an ordinary or split ODE problem.
    pub fn into_callback_set<P>(self) -> Result<CallbackSet<P>, ConfigurationError>
    where
        G: Fn(&[f64], &P, f64) -> bool + 'static,
    {
        validate_reduction_factor(self.reduction_factor)?;
        Ok(CallbackSet::new().with_step_guard(self.reduction_factor, self.is_out_of_domain))
    }

    /// Builds a guard policy for a partitioned second-order ODE problem.
    ///
    /// The predicate receives velocity before position, matching the
    /// partitioned solver API.
    pub fn into_second_order_callback_set<P>(
        self,
    ) -> Result<SecondOrderCallbackSet<P>, ConfigurationError>
    where
        G: Fn(&[f64], &[f64], &P, f64) -> bool + 'static,
    {
        validate_reduction_factor(self.reduction_factor)?;
        Ok(SecondOrderCallbackSet::new()
            .with_step_guard(self.reduction_factor, self.is_out_of_domain))
    }
}

/// A state-dependent upper bound for the next integration step.
///
/// This policy is useful for stability restrictions such as a CFL condition.
/// The supplied function returns a positive step-size magnitude from the
/// current state, parameters, and time. By default, the solver may choose a
/// smaller step; [`Self::with_max_step`] instead makes a fixed-step solve track
/// the scaled bound exactly.
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct StepsizeLimiter<F> {
    limit: F,
    safety_factor: f64,
    max_step: bool,
}

impl<F> StepsizeLimiter<F> {
    /// Creates a limiter with a safety factor of `0.9`.
    pub const fn new(limit: F) -> Self {
        Self {
            limit,
            safety_factor: 0.9,
            max_step: false,
        }
    }

    /// Sets the factor applied below the returned stability limit.
    ///
    /// The value is validated when a callback set is built and must lie in
    /// `(0, 1]`.
    pub const fn with_safety_factor(mut self, safety_factor: f64) -> Self {
        self.safety_factor = safety_factor;
        self
    }

    /// Selects whether each next step is set to, rather than capped by, the
    /// scaled limit.
    ///
    /// This mirrors the fixed-step `max_step` mode of SciML's policy. Adaptive
    /// solves should normally keep this disabled so their error controller can
    /// select a smaller step.
    pub const fn with_max_step(mut self, enabled: bool) -> Self {
        self.max_step = enabled;
        self
    }

    /// Builds a callback set for an ordinary or split ODE problem.
    pub fn into_callback_set<P>(self) -> Result<CallbackSet<P>, ConfigurationError>
    where
        F: Fn(&[f64], &P, f64) -> f64 + 'static,
    {
        validate_safety_factor(self.safety_factor)?;
        let safety_factor = self.safety_factor;
        let max_step = self.max_step;
        let limit = self.limit;
        Ok(CallbackSet::new().with_discrete_callback_saving(
            CallbackSave::None,
            |_, _, _| true,
            move |state, parameters, time| {
                step_limit_action(safety_factor * limit(state, parameters, time), max_step)
            },
        ))
    }

    /// Builds a callback set for a partitioned second-order ODE problem.
    ///
    /// The limit function receives velocity before position, matching the
    /// partitioned solver API.
    pub fn into_second_order_callback_set<P>(
        self,
    ) -> Result<SecondOrderCallbackSet<P>, ConfigurationError>
    where
        F: Fn(&[f64], &[f64], &P, f64) -> f64 + 'static,
    {
        validate_safety_factor(self.safety_factor)?;
        let safety_factor = self.safety_factor;
        let max_step = self.max_step;
        let limit = self.limit;
        Ok(SecondOrderCallbackSet::new().with_discrete_callback_saving(
            CallbackSave::None,
            |_, _, _, _| true,
            move |velocity, position, parameters, time| {
                step_limit_action(
                    safety_factor * limit(velocity, position, parameters, time),
                    max_step,
                )
            },
        ))
    }
}

/// Configuration for an observation-only function callback.
///
/// The default policy calls the function at the initial condition and after
/// every accepted step. Use [`Self::at_times`] to instead start with an exact
/// list of integration times. Explicit times are automatically exposed to the
/// solver as exact stops.
#[derive(Clone, Debug, PartialEq)]
#[must_use]
pub struct FunctionCallingCallback {
    times: Vec<f64>,
    every_step: bool,
    call_at_start: bool,
}

impl FunctionCallingCallback {
    /// Calls the function at the start and after every accepted step.
    pub const fn every_step() -> Self {
        Self {
            times: Vec::new(),
            every_step: true,
            call_at_start: true,
        }
    }

    /// Calls the function at the listed exact times and at the initial state.
    ///
    /// Times must be finite, lie within the eventual time span, and be strictly
    /// ordered in its integration direction. They are validated by
    /// [`Self::into_callback_set`].
    pub fn at_times(times: impl IntoIterator<Item = f64>) -> Self {
        Self {
            times: times.into_iter().collect(),
            every_step: false,
            call_at_start: true,
        }
    }

    /// Selects whether the function is called after every accepted step.
    ///
    /// Explicit times remain active and are called only once when they coincide
    /// with an accepted step.
    pub const fn with_every_step(mut self, enabled: bool) -> Self {
        self.every_step = enabled;
        self
    }

    /// Selects whether the function is called at the initial state.
    pub const fn with_start(mut self, enabled: bool) -> Self {
        self.call_at_start = enabled;
        self
    }

    /// Builds a callback set for `time_span` using the supplied observer.
    ///
    /// The observer receives read-only state and parameter references. It must
    /// not use parameter interior mutability to change values observed by the
    /// differential equation; the solver deliberately preserves its caches
    /// after this observation-only callback.
    pub fn into_callback_set<P, F>(
        self,
        time_span: (f64, f64),
        function: F,
    ) -> Result<CallbackSet<P>, ConfigurationError>
    where
        F: Fn(&[f64], &P, f64) + 'static,
    {
        validate_time_span(time_span)?;
        if !time_sequence_is_valid(&self.times, time_span) {
            return Err(ConfigurationError::InvalidParameter {
                parameter: "function-calling times",
                reason: "must be finite, strictly ordered, and inside the time span",
            });
        }

        let function = Rc::new(function);
        let mut callbacks = CallbackSet::new();
        if !self.times.is_empty() {
            let function = Rc::clone(&function);
            callbacks = callbacks.with_preset_time_callback_saving(
                self.times.iter().copied(),
                CallbackSave::None,
                move |state, parameters, time| {
                    function(state, parameters, time);
                    CallbackAction::ContinueUnmodified
                },
            );
        }

        if self.every_step || self.call_at_start {
            let start = time_span.0;
            let explicit_times = self.times;
            let every_step = self.every_step;
            let call_at_start = self.call_at_start;
            callbacks = callbacks.with_discrete_callback_saving(
                CallbackSave::None,
                move |_, _, time| {
                    !explicit_times.contains(&time)
                        && ((time == start && call_at_start) || (time != start && every_step))
                },
                move |state, parameters, time| {
                    function(state, parameters, time);
                    CallbackAction::ContinueUnmodified
                },
            );
        }

        Ok(callbacks)
    }

    /// Builds an observation-only callback set for a second-order problem.
    ///
    /// The observer receives velocity before position, matching the partitioned
    /// solver API. As with [`Self::into_callback_set`], it must not mutate
    /// right-hand-side inputs through parameter interior mutability.
    pub fn into_second_order_callback_set<P, F>(
        self,
        time_span: (f64, f64),
        function: F,
    ) -> Result<SecondOrderCallbackSet<P>, ConfigurationError>
    where
        F: Fn(&[f64], &[f64], &P, f64) + 'static,
    {
        validate_time_span(time_span)?;
        if !time_sequence_is_valid(&self.times, time_span) {
            return Err(ConfigurationError::InvalidParameter {
                parameter: "function-calling times",
                reason: "must be finite, strictly ordered, and inside the time span",
            });
        }

        let function = Rc::new(function);
        let mut callbacks = SecondOrderCallbackSet::new();
        if !self.times.is_empty() {
            let function = Rc::clone(&function);
            callbacks = callbacks.with_preset_time_callback_saving(
                self.times.iter().copied(),
                CallbackSave::None,
                move |velocity, position, parameters, time| {
                    function(velocity, position, parameters, time);
                    CallbackAction::ContinueUnmodified
                },
            );
        }

        if self.every_step || self.call_at_start {
            let start = time_span.0;
            let explicit_times = self.times;
            let every_step = self.every_step;
            let call_at_start = self.call_at_start;
            callbacks = callbacks.with_discrete_callback_saving(
                CallbackSave::None,
                move |_, _, _, time| {
                    !explicit_times.contains(&time)
                        && ((time == start && call_at_start) || (time != start && every_step))
                },
                move |velocity, position, parameters, time| {
                    function(velocity, position, parameters, time);
                    CallbackAction::ContinueUnmodified
                },
            );
        }

        Ok(callbacks)
    }
}

impl Default for FunctionCallingCallback {
    fn default() -> Self {
        Self::every_step()
    }
}

fn validate_time_span(time_span: (f64, f64)) -> Result<(), ConfigurationError> {
    let (start, end) = time_span;
    if !start.is_finite() || !end.is_finite() || start == end || !(end - start).is_finite() {
        return Err(ConfigurationError::InvalidBounds {
            context: "callback time span",
            reason: "endpoints and span length must be finite and distinct",
        });
    }
    Ok(())
}

fn validate_representable_period(
    time_span: (f64, f64),
    period: f64,
    phase: f64,
) -> Result<(), ConfigurationError> {
    let (start, end) = time_span;
    let direction = (end - start).signum();
    let span = (end - start).abs();
    let first_offset = if phase == 0.0 { period } else { phase };
    let first_is_scheduled = first_offset <= span;
    let multiple_are_scheduled = first_offset + period <= span;
    if (first_is_scheduled
        && times_are_representably_equal(start + direction * first_offset, start))
        || (multiple_are_scheduled && times_are_representably_equal(end - direction * period, end))
    {
        return Err(ConfigurationError::InvalidParameter {
            parameter: "callback period",
            reason: "must advance representable integration times across the time span",
        });
    }
    Ok(())
}

fn validate_safety_factor(safety_factor: f64) -> Result<(), ConfigurationError> {
    if !safety_factor.is_finite() || safety_factor <= 0.0 || safety_factor > 1.0 {
        return Err(ConfigurationError::InvalidParameter {
            parameter: "stepsize safety factor",
            reason: "must be finite and lie in (0, 1]",
        });
    }
    Ok(())
}

fn validate_reduction_factor(reduction_factor: f64) -> Result<(), ConfigurationError> {
    if !reduction_factor.is_finite() || reduction_factor <= 0.0 || reduction_factor >= 1.0 {
        return Err(ConfigurationError::InvalidParameter {
            parameter: "domain-guard reduction factor",
            reason: "must be finite and lie in (0, 1)",
        });
    }
    Ok(())
}

fn step_limit_action(step: f64, max_step: bool) -> CallbackAction {
    if max_step {
        CallbackAction::ContinueUnmodifiedWithStepSize(step)
    } else {
        CallbackAction::LimitStepSize(step)
    }
}
