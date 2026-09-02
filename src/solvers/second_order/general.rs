use crate::callbacks::SteadyStateCondition;
use std::cell::RefCell;
use std::rc::Rc;

use super::coefficient_data::*;
use crate::callback::{
    CallbackOutcome, CallbackSave, IterativeTimes, PeriodicTimes, PresetTimes,
    VectorCallbackScratch,
};
use crate::event::{
    MAX_EVENT_ROOT_ITERATIONS, effective_event_tolerance, event_interval_converged,
    times_are_numerically_equal, times_are_representably_equal,
};
use crate::integrator::{
    ControllerConfig, ControllerState, TimeStopSchedule, callback_adjusted_step,
};
use crate::linear::{factorize, solve_factorized};
use crate::solver::{
    validate_preset_time_sequences, validate_state_time_options, validate_vector_callback_lengths,
};
use crate::{
    CallbackAction, ConfigurationError, EventCrossing, EventDirection, InterpolationError,
    SaveMode, SolveError, SolveOptions, SolverStats,
};
use thiserror::Error;

type DiscreteCondition<P> = dyn Fn(&[f64], &[f64], &P, f64) -> bool;
type ContinuousCondition<P> = dyn Fn(&[f64], &[f64], &P, f64) -> f64;
type Affect<P> = dyn Fn(&mut [f64], &mut [f64], &P, f64) -> CallbackAction;
type FallibleAffect<P> =
    dyn Fn(&mut [f64], &mut [f64], &P, f64) -> Result<CallbackAction, SolveError>;
type IterativeInitialization<P> = dyn Fn(&[f64], &[f64], &P, f64) -> Result<(), SolveError>;
type VectorContinuousCondition<P> = dyn Fn(&mut [f64], &[f64], &[f64], &P, f64);
type VectorAffect<P> = dyn Fn(&mut [f64], &mut [f64], &P, f64, &[EventCrossing]) -> CallbackAction;
type LifecycleHook<P> = dyn Fn(&mut [f64], &mut [f64], &P, f64);
type DomainCondition<P> = dyn Fn(&[f64], &[f64], &P, f64) -> bool;
type PartitionedInterpolator<'a> =
    dyn FnMut(f64, &mut [f64], &mut [f64]) -> Result<(), SolveError> + 'a;

enum DiscreteTrigger<P> {
    Condition(Box<DiscreteCondition<P>>),
    SteadyState(SteadyStateCondition),
    PresetTimes(PresetTimes),
    Periodic(PeriodicTimes),
    Iterative {
        times: Rc<IterativeTimes>,
        initialize: Box<IterativeInitialization<P>>,
    },
}

struct DiscreteCallback<P> {
    trigger: DiscreteTrigger<P>,
    affect: Box<FallibleAffect<P>>,
    save: CallbackSave,
}

impl<P> DiscreteTrigger<P> {
    fn initialize(
        &self,
        velocity: &[f64],
        position: &[f64],
        parameters: &P,
        time: f64,
    ) -> Result<(), SolveError> {
        if let Self::Iterative { initialize, .. } = self {
            initialize(velocity, position, parameters, time)?;
        }
        Ok(())
    }

    fn is_triggered(
        &self,
        velocity: &[f64],
        position: &[f64],
        parameters: &P,
        time: f64,
        evaluate: impl FnOnce(&mut [f64], &mut [f64]) -> Result<(), SolveError>,
    ) -> Result<bool, SolveError> {
        Ok(match self {
            Self::SteadyState(condition) => {
                return condition.test(velocity, position, time, evaluate);
            }
            Self::Condition(condition) => condition(velocity, position, parameters, time),
            Self::PresetTimes(times) => times.contains(time),
            Self::Periodic(times) => times.contains(time),
            Self::Iterative { times, .. } => times.contains(time),
        })
    }

    fn preset_times(&self) -> Option<&[f64]> {
        match self {
            Self::Condition(_) | Self::SteadyState(_) => None,
            Self::PresetTimes(times) => Some(times.as_slice()),
            Self::Periodic(_) => None,
            Self::Iterative { .. } => None,
        }
    }

    fn next_preset_time(&self, time: f64, direction: f64) -> Option<f64> {
        match self {
            Self::Condition(_) | Self::SteadyState(_) => None,
            Self::PresetTimes(times) => times.next(time, direction),
            Self::Periodic(times) => times.next(time, direction),
            Self::Iterative { times, .. } => times.next(time, direction),
        }
    }
}

struct ContinuousCallback<P> {
    condition: Box<ContinuousCondition<P>>,
    affect: Box<Affect<P>>,
    direction: EventDirection,
    save: CallbackSave,
}

struct VectorContinuousCallback<P> {
    condition: Box<VectorContinuousCondition<P>>,
    affect: Box<VectorAffect<P>>,
    event_count: usize,
    save: CallbackSave,
    scratch: RefCell<VectorCallbackScratch>,
}

impl<P> VectorContinuousCallback<P> {
    fn new<C, A>(event_count: usize, save: CallbackSave, condition: C, affect: A) -> Self
    where
        C: Fn(&mut [f64], &[f64], &[f64], &P, f64) + 'static,
        A: Fn(&mut [f64], &mut [f64], &P, f64, &[EventCrossing]) -> CallbackAction + 'static,
    {
        Self {
            condition: Box::new(condition),
            affect: Box::new(affect),
            event_count,
            save,
            scratch: RefCell::new(VectorCallbackScratch::new(event_count)),
        }
    }
}

enum PartitionedCallback<P> {
    Discrete(DiscreteCallback<P>),
    Continuous(ContinuousCallback<P>),
    VectorContinuous(VectorContinuousCallback<P>),
}

struct InitializationHook<P> {
    hook: Box<LifecycleHook<P>>,
    save: CallbackSave,
}

struct StepGuard<P> {
    is_out_of_domain: Box<DomainCondition<P>>,
    reduction_factor: f64,
}

/// An ordered collection of callbacks for a second-order ODE problem.
///
/// Conditions receive velocity before position, and effects receive mutable
/// velocity and position partitions in the same order.
#[must_use]
pub struct SecondOrderCallbackSet<P> {
    callbacks: Vec<PartitionedCallback<P>>,
    initializers: Vec<InitializationHook<P>>,
    finalizers: Vec<Box<LifecycleHook<P>>>,
    step_guards: Vec<StepGuard<P>>,
}

impl<P> SecondOrderCallbackSet<P> {
    /// Creates an empty callback set.
    pub const fn new() -> Self {
        Self {
            callbacks: Vec::new(),
            initializers: Vec::new(),
            finalizers: Vec::new(),
            step_guards: Vec::new(),
        }
    }

    /// Returns the number of event callbacks in the set.
    ///
    /// Lifecycle hooks and candidate-state guards are not included.
    pub fn len(&self) -> usize {
        self.callbacks.len()
    }

    /// Returns whether the set contains no callbacks, hooks, or guards.
    pub fn is_empty(&self) -> bool {
        self.callbacks.is_empty()
            && self.initializers.is_empty()
            && self.finalizers.is_empty()
            && self.step_guards.is_empty()
    }

    /// Adds a partitioned initialization hook that saves the initialized state.
    pub fn with_initialize<I>(self, initialize: I) -> Self
    where
        I: Fn(&mut [f64], &mut [f64], &P, f64) + 'static,
    {
        self.with_initialize_saving(CallbackSave::After, initialize)
    }

    /// Adds an initialization hook with explicit initial-state saving behavior.
    pub fn with_initialize_saving<I>(mut self, save: CallbackSave, initialize: I) -> Self
    where
        I: Fn(&mut [f64], &mut [f64], &P, f64) + 'static,
    {
        self.initializers.push(InitializationHook {
            hook: Box::new(initialize),
            save,
        });
        self
    }

    /// Adds an end-of-solve partitioned state finalization hook.
    pub fn with_finalize<F>(mut self, finalize: F) -> Self
    where
        F: Fn(&mut [f64], &mut [f64], &P, f64) + 'static,
    {
        self.finalizers.push(Box::new(finalize));
        self
    }

    pub(crate) fn with_step_guard<G>(mut self, reduction_factor: f64, guard: G) -> Self
    where
        G: Fn(&[f64], &[f64], &P, f64) -> bool + 'static,
    {
        self.step_guards.push(StepGuard {
            is_out_of_domain: Box::new(guard),
            reduction_factor,
        });
        self
    }

    /// Adds a callback evaluated at initialization and after accepted steps.
    pub fn with_discrete_callback<C, A>(self, condition: C, affect: A) -> Self
    where
        C: Fn(&[f64], &[f64], &P, f64) -> bool + 'static,
        A: Fn(&mut [f64], &mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        self.with_discrete_callback_saving(CallbackSave::After, condition, affect)
    }

    /// Adds a discrete callback with explicit callback-time saving behavior.
    pub fn with_discrete_callback_saving<C, A>(
        mut self,
        save: CallbackSave,
        condition: C,
        affect: A,
    ) -> Self
    where
        C: Fn(&[f64], &[f64], &P, f64) -> bool + 'static,
        A: Fn(&mut [f64], &mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        self.callbacks
            .push(PartitionedCallback::Discrete(DiscreteCallback {
                trigger: DiscreteTrigger::Condition(Box::new(condition)),
                affect: Box::new(move |v, q, p, t| Ok(affect(v, q, p, t))),
                save,
            }));
        self
    }

    /// Adds a callback that runs at each listed integration time.
    pub fn with_preset_time_callback<A>(
        self,
        times: impl IntoIterator<Item = f64>,
        affect: A,
    ) -> Self
    where
        A: Fn(&mut [f64], &mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        self.with_preset_time_callback_saving(times, CallbackSave::After, affect)
    }

    /// Adds a preset-time callback with explicit callback-time saving behavior.
    pub fn with_preset_time_callback_saving<A>(
        mut self,
        times: impl IntoIterator<Item = f64>,
        save: CallbackSave,
        affect: A,
    ) -> Self
    where
        A: Fn(&mut [f64], &mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        self.callbacks
            .push(PartitionedCallback::Discrete(DiscreteCallback {
                trigger: DiscreteTrigger::PresetTimes(PresetTimes::new(times)),
                affect: Box::new(move |v, q, p, t| Ok(affect(v, q, p, t))),
                save,
            }));
        self
    }

    pub(crate) fn with_periodic_callback_saving<A>(
        mut self,
        times: PeriodicTimes,
        save: CallbackSave,
        affect: A,
    ) -> Self
    where
        A: Fn(&mut [f64], &mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        self.callbacks
            .push(PartitionedCallback::Discrete(DiscreteCallback {
                trigger: DiscreteTrigger::Periodic(times),
                affect: Box::new(move |v, q, p, t| Ok(affect(v, q, p, t))),
                save,
            }));
        self
    }

    pub(crate) fn with_iterative_callback<I, A>(
        mut self,
        times: Rc<IterativeTimes>,
        save: CallbackSave,
        initialize: I,
        affect: A,
    ) -> Self
    where
        I: Fn(&[f64], &[f64], &P, f64) -> Result<(), SolveError> + 'static,
        A: Fn(&mut [f64], &mut [f64], &P, f64) -> Result<CallbackAction, SolveError> + 'static,
    {
        self.callbacks
            .push(PartitionedCallback::Discrete(DiscreteCallback {
                trigger: DiscreteTrigger::Iterative {
                    times,
                    initialize: Box::new(initialize),
                },
                affect: Box::new(affect),
                save,
            }));
        self
    }

    pub(crate) fn with_steady_state(
        mut self,
        condition: SteadyStateCondition,
        save: CallbackSave,
    ) -> Self {
        self.callbacks
            .push(PartitionedCallback::Discrete(DiscreteCallback {
                trigger: DiscreteTrigger::SteadyState(condition),
                affect: Box::new(|_, _, _, _| Ok(CallbackAction::Terminate)),
                save,
            }));
        self
    }

    /// Adds a zero-crossing callback that triggers in either direction.
    pub fn with_continuous_callback<C, A>(self, condition: C, affect: A) -> Self
    where
        C: Fn(&[f64], &[f64], &P, f64) -> f64 + 'static,
        A: Fn(&mut [f64], &mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        self.with_continuous_callback_saving(CallbackSave::Both, condition, affect)
    }

    /// Adds a zero-crossing callback with explicit callback-time saving behavior.
    pub fn with_continuous_callback_saving<C, A>(
        self,
        save: CallbackSave,
        condition: C,
        affect: A,
    ) -> Self
    where
        C: Fn(&[f64], &[f64], &P, f64) -> f64 + 'static,
        A: Fn(&mut [f64], &mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        self.with_continuous_callback_direction_saving(EventDirection::Any, save, condition, affect)
    }

    /// Adds a direction-filtered zero-crossing callback.
    pub fn with_continuous_callback_direction<C, A>(
        self,
        direction: EventDirection,
        condition: C,
        affect: A,
    ) -> Self
    where
        C: Fn(&[f64], &[f64], &P, f64) -> f64 + 'static,
        A: Fn(&mut [f64], &mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        self.with_continuous_callback_direction_saving(
            direction,
            CallbackSave::Both,
            condition,
            affect,
        )
    }

    /// Adds a direction-filtered callback with explicit saving behavior.
    pub fn with_continuous_callback_direction_saving<C, A>(
        mut self,
        direction: EventDirection,
        save: CallbackSave,
        condition: C,
        affect: A,
    ) -> Self
    where
        C: Fn(&[f64], &[f64], &P, f64) -> f64 + 'static,
        A: Fn(&mut [f64], &mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        self.callbacks
            .push(PartitionedCallback::Continuous(ContinuousCallback {
                condition: Box::new(condition),
                affect: Box::new(affect),
                direction,
                save,
            }));
        self
    }

    /// Adds a vector-valued partitioned zero-crossing callback.
    ///
    /// The effect runs once at the earliest root and receives all simultaneous
    /// crossing directions in condition index order.
    pub fn with_vector_continuous_callback<C, A>(
        self,
        event_count: usize,
        condition: C,
        affect: A,
    ) -> Self
    where
        C: Fn(&mut [f64], &[f64], &[f64], &P, f64) + 'static,
        A: Fn(&mut [f64], &mut [f64], &P, f64, &[EventCrossing]) -> CallbackAction + 'static,
    {
        self.with_vector_continuous_callback_saving(
            event_count,
            CallbackSave::Both,
            condition,
            affect,
        )
    }

    /// Adds a partitioned vector callback with explicit saving behavior.
    pub fn with_vector_continuous_callback_saving<C, A>(
        mut self,
        event_count: usize,
        save: CallbackSave,
        condition: C,
        affect: A,
    ) -> Self
    where
        C: Fn(&mut [f64], &[f64], &[f64], &P, f64) + 'static,
        A: Fn(&mut [f64], &mut [f64], &P, f64, &[EventCrossing]) -> CallbackAction + 'static,
    {
        self.callbacks.push(PartitionedCallback::VectorContinuous(
            VectorContinuousCallback::new(event_count, save, condition, affect),
        ));
        self
    }

    /// Appends another set, preserving callback order within each set.
    pub fn append(mut self, mut other: Self) -> Self {
        self.callbacks.append(&mut other.callbacks);
        self.initializers.append(&mut other.initializers);
        self.finalizers.append(&mut other.finalizers);
        self.step_guards.append(&mut other.step_guards);
        self
    }
}

impl<P> Default for SecondOrderCallbackSet<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// A second-order initial-value problem `q'' = f(q', q, p, t)`.
///
/// The acceleration function follows SciML's in-place calling convention
/// `f(dv, v, q, p, t)`. Positions and velocities remain separate throughout
/// the public API; callers do not need to flatten the partitioned state.
/// This represents SciML's `SecondOrderODEProblem` specialization `q' = v`,
/// not a general `DynamicalODEProblem` with a separately supplied position
/// rate.
pub struct SecondOrderOdeProblem<F, P> {
    pub(crate) acceleration: F,
    initial_velocity: Vec<f64>,
    initial_position: Vec<f64>,
    time_span: (f64, f64),
    parameters: P,
    callbacks: Vec<PartitionedCallback<P>>,
    initializers: Vec<InitializationHook<P>>,
    finalizers: Vec<Box<LifecycleHook<P>>>,
    step_guards: Vec<StepGuard<P>>,
}

impl<F, P> SecondOrderOdeProblem<F, P> {
    /// Creates a second-order ODE problem.
    pub fn new(
        acceleration: F,
        initial_velocity: impl Into<Vec<f64>>,
        initial_position: impl Into<Vec<f64>>,
        time_span: (f64, f64),
        parameters: P,
    ) -> Self {
        Self {
            acceleration,
            initial_velocity: initial_velocity.into(),
            initial_position: initial_position.into(),
            time_span,
            parameters,
            callbacks: Vec::new(),
            initializers: Vec::new(),
            finalizers: Vec::new(),
            step_guards: Vec::new(),
        }
    }

    /// Appends an ordered callback set to this problem.
    pub fn with_callback_set(mut self, mut callback_set: SecondOrderCallbackSet<P>) -> Self {
        self.callbacks.append(&mut callback_set.callbacks);
        self.initializers.append(&mut callback_set.initializers);
        self.finalizers.append(&mut callback_set.finalizers);
        self.step_guards.append(&mut callback_set.step_guards);
        self
    }

    /// Adds a callback evaluated at the initial state and after accepted steps.
    ///
    /// Conditions and effects receive velocity before position, matching the
    /// `SecondOrderODEProblem` acceleration signature. Effects may modify both
    /// partitions and may terminate integration.
    pub fn with_discrete_callback<C, A>(self, condition: C, affect: A) -> Self
    where
        C: Fn(&[f64], &[f64], &P, f64) -> bool + 'static,
        A: Fn(&mut [f64], &mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        self.with_discrete_callback_saving(CallbackSave::After, condition, affect)
    }

    /// Adds a discrete callback with explicit callback-time saving behavior.
    pub fn with_discrete_callback_saving<C, A>(
        self,
        save: CallbackSave,
        condition: C,
        affect: A,
    ) -> Self
    where
        C: Fn(&[f64], &[f64], &P, f64) -> bool + 'static,
        A: Fn(&mut [f64], &mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        self.with_callback_set(
            SecondOrderCallbackSet::new().with_discrete_callback_saving(save, condition, affect),
        )
    }

    /// Adds a callback that runs at each listed integration time.
    ///
    /// Preset times become mandatory integration stops automatically and are
    /// validated against this problem's time span when solving begins.
    pub fn with_preset_time_callback<A>(
        self,
        times: impl IntoIterator<Item = f64>,
        affect: A,
    ) -> Self
    where
        A: Fn(&mut [f64], &mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        self.with_preset_time_callback_saving(times, CallbackSave::After, affect)
    }

    /// Adds a preset-time callback with explicit callback-time saving behavior.
    pub fn with_preset_time_callback_saving<A>(
        self,
        times: impl IntoIterator<Item = f64>,
        save: CallbackSave,
        affect: A,
    ) -> Self
    where
        A: Fn(&mut [f64], &mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        self.with_callback_set(
            SecondOrderCallbackSet::new().with_preset_time_callback_saving(times, save, affect),
        )
    }

    /// Adds a zero-crossing callback that triggers in either direction.
    pub fn with_continuous_callback<C, A>(self, condition: C, affect: A) -> Self
    where
        C: Fn(&[f64], &[f64], &P, f64) -> f64 + 'static,
        A: Fn(&mut [f64], &mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        self.with_continuous_callback_saving(CallbackSave::Both, condition, affect)
    }

    /// Adds a zero-crossing callback with explicit callback-time saving behavior.
    pub fn with_continuous_callback_saving<C, A>(
        self,
        save: CallbackSave,
        condition: C,
        affect: A,
    ) -> Self
    where
        C: Fn(&[f64], &[f64], &P, f64) -> f64 + 'static,
        A: Fn(&mut [f64], &mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        self.with_continuous_callback_direction_saving(EventDirection::Any, save, condition, affect)
    }

    /// Adds a direction-filtered zero-crossing callback.
    ///
    /// Roots use the method's native extension when available; otherwise they
    /// use a partition-aware segment with cubic-Hermite position and linear
    /// velocity interpolation.
    pub fn with_continuous_callback_direction<C, A>(
        self,
        direction: EventDirection,
        condition: C,
        affect: A,
    ) -> Self
    where
        C: Fn(&[f64], &[f64], &P, f64) -> f64 + 'static,
        A: Fn(&mut [f64], &mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        self.with_continuous_callback_direction_saving(
            direction,
            CallbackSave::Both,
            condition,
            affect,
        )
    }

    /// Adds a direction-filtered callback with explicit saving behavior.
    pub fn with_continuous_callback_direction_saving<C, A>(
        self,
        direction: EventDirection,
        save: CallbackSave,
        condition: C,
        affect: A,
    ) -> Self
    where
        C: Fn(&[f64], &[f64], &P, f64) -> f64 + 'static,
        A: Fn(&mut [f64], &mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        self.with_callback_set(
            SecondOrderCallbackSet::new()
                .with_continuous_callback_direction_saving(direction, save, condition, affect),
        )
    }

    /// Adds a vector-valued partitioned zero-crossing callback.
    pub fn with_vector_continuous_callback<C, A>(
        self,
        event_count: usize,
        condition: C,
        affect: A,
    ) -> Self
    where
        C: Fn(&mut [f64], &[f64], &[f64], &P, f64) + 'static,
        A: Fn(&mut [f64], &mut [f64], &P, f64, &[EventCrossing]) -> CallbackAction + 'static,
    {
        self.with_vector_continuous_callback_saving(
            event_count,
            CallbackSave::Both,
            condition,
            affect,
        )
    }

    /// Adds a partitioned vector callback with explicit saving behavior.
    pub fn with_vector_continuous_callback_saving<C, A>(
        self,
        event_count: usize,
        save: CallbackSave,
        condition: C,
        affect: A,
    ) -> Self
    where
        C: Fn(&mut [f64], &[f64], &[f64], &P, f64) + 'static,
        A: Fn(&mut [f64], &mut [f64], &P, f64, &[EventCrossing]) -> CallbackAction + 'static,
    {
        self.with_callback_set(
            SecondOrderCallbackSet::new().with_vector_continuous_callback_saving(
                event_count,
                save,
                condition,
                affect,
            ),
        )
    }

    /// Initial velocity.
    pub fn initial_velocity(&self) -> &[f64] {
        &self.initial_velocity
    }

    /// Initial position.
    pub fn initial_position(&self) -> &[f64] {
        &self.initial_position
    }

    /// Returns `(start_time, end_time)`.
    pub fn time_span(&self) -> (f64, f64) {
        self.time_span
    }

    /// Problem parameters.
    pub fn parameters(&self) -> &P {
        &self.parameters
    }

    /// Evaluates the acceleration callback for a specialized partitioned
    /// solver without exposing the problem's internal storage.
    pub fn evaluate_acceleration(
        &self,
        output: &mut [f64],
        velocity: &[f64],
        position: &[f64],
        time: f64,
    ) where
        F: Fn(&mut [f64], &[f64], &[f64], &P, f64),
    {
        (self.acceleration)(output, velocity, position, &self.parameters, time);
    }

    pub(crate) fn has_callbacks(&self) -> bool {
        !self.callbacks.is_empty()
            || !self.initializers.is_empty()
            || !self.finalizers.is_empty()
            || !self.step_guards.is_empty()
    }

    pub(crate) fn domain_rejection_factor(
        &self,
        velocity: &[f64],
        position: &[f64],
        time: f64,
    ) -> Option<f64> {
        self.step_guards
            .iter()
            .filter(|guard| (guard.is_out_of_domain)(velocity, position, &self.parameters, time))
            .map(|guard| guard.reduction_factor)
            .reduce(f64::min)
    }

    pub(crate) fn preset_time_sequences(&self) -> impl Iterator<Item = &[f64]> {
        self.callbacks.iter().filter_map(|callback| {
            let PartitionedCallback::Discrete(callback) = callback else {
                return None;
            };
            callback.trigger.preset_times()
        })
    }

    pub(crate) fn next_preset_time(&self, time: f64, direction: f64) -> Option<f64> {
        self.callbacks
            .iter()
            .filter_map(|callback| {
                let PartitionedCallback::Discrete(callback) = callback else {
                    return None;
                };
                callback.trigger.next_preset_time(time, direction)
            })
            .reduce(|earliest, candidate| {
                if direction * (earliest - candidate) <= 0.0 {
                    earliest
                } else {
                    candidate
                }
            })
    }

    pub(crate) fn vector_callback_lengths(&self) -> impl Iterator<Item = usize> + '_ {
        self.callbacks.iter().filter_map(|callback| {
            let PartitionedCallback::VectorContinuous(callback) = callback else {
                return None;
            };
            Some(callback.event_count)
        })
    }
}

/// A saved trajectory for a second-order ODE.
///
/// Callbacks configured with [`CallbackSave::Both`] produce adjacent states
/// at the same time, ordered before-effect then after-effect. Exact
/// interpolation at that time returns the latter state.
#[derive(Clone, Debug, PartialEq)]
pub struct SecondOrderSolution {
    times: Vec<f64>,
    velocities: Vec<f64>,
    positions: Vec<f64>,
    dimension: usize,
    stats: SolverStats,
    dense_segments: Vec<PartitionedDenseSegment>,
}

impl SecondOrderSolution {
    /// Saved times in integration order.
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    /// Number of scalar components in each position or velocity partition.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// All saved velocities in contiguous row-major storage.
    pub fn velocity_values(&self) -> &[f64] {
        &self.velocities
    }

    /// All saved positions in contiguous row-major storage.
    pub fn position_values(&self) -> &[f64] {
        &self.positions
    }

    /// Saved velocity at a time index.
    pub fn velocity(&self, index: usize) -> Option<&[f64]> {
        partition(&self.velocities, self.dimension, index)
    }

    /// Saved position at a time index.
    pub fn position(&self, index: usize) -> Option<&[f64]> {
        partition(&self.positions, self.dimension, index)
    }

    /// Last saved velocity.
    pub fn last_velocity(&self) -> &[f64] {
        let start = self.velocities.len() - self.dimension;
        &self.velocities[start..]
    }

    /// Last saved position.
    pub fn last_position(&self) -> &[f64] {
        let start = self.positions.len() - self.dimension;
        &self.positions[start..]
    }

    /// Solver work counters. Acceleration evaluations contribute to
    /// `rhs_evaluations`; the identity position rate `q' = v` is not evaluated
    /// as a user function.
    pub fn stats(&self) -> SolverStats {
        self.stats
    }

    /// Interpolates `(velocity, position)` at a time covered by the solution.
    ///
    /// When dense output was retained, positions use a cubic Hermite segment
    /// consistent with `q' = v` and velocities use a stable linear segment.
    /// Without retained segments, saved states are linearly interpolated.
    pub fn interpolate(&self, time: f64) -> Option<(Vec<f64>, Vec<f64>)> {
        self.try_interpolate(time).ok()
    }

    /// Interpolates `(velocity, position)` and reports why the query fails.
    pub fn try_interpolate(&self, time: f64) -> Result<(Vec<f64>, Vec<f64>), InterpolationError> {
        if !time.is_finite() {
            return Err(InterpolationError::NonFiniteTime);
        }
        if self.times.is_empty() {
            return Err(InterpolationError::EmptySolution);
        }
        for (index, &saved_time) in self.times.iter().enumerate().rev() {
            if time == saved_time {
                return Ok((
                    self.velocity(index)
                        .ok_or(InterpolationError::InvalidSegmentData {
                            context: "saved second-order velocity",
                        })?
                        .to_vec(),
                    self.position(index)
                        .ok_or(InterpolationError::InvalidSegmentData {
                            context: "saved second-order position",
                        })?
                        .to_vec(),
                ));
            }
        }
        for segment in &self.dense_segments {
            if segment.contains(time) {
                let mut velocity = vec![0.0; self.dimension];
                let mut position = vec![0.0; self.dimension];
                segment
                    .interpolate(time, &mut velocity, &mut position)
                    .ok_or(InterpolationError::InvalidSegmentData {
                        context: "second-order dense segment",
                    })?;
                return Ok((velocity, position));
            }
        }
        for index in 1..self.times.len() {
            let left = self.times[index - 1];
            let right = self.times[index];
            if between(time, left, right) && left != right {
                let fraction = (time - left) / (right - left);
                let mut velocity = vec![0.0; self.dimension];
                let mut position = vec![0.0; self.dimension];
                interpolate(
                    self.velocity(index)
                        .ok_or(InterpolationError::InvalidSegmentData {
                            context: "saved second-order velocity",
                        })?,
                    self.velocity(index - 1)
                        .ok_or(InterpolationError::InvalidSegmentData {
                            context: "saved second-order velocity",
                        })?,
                    fraction,
                    &mut velocity,
                );
                interpolate(
                    self.position(index)
                        .ok_or(InterpolationError::InvalidSegmentData {
                            context: "saved second-order position",
                        })?,
                    self.position(index - 1)
                        .ok_or(InterpolationError::InvalidSegmentData {
                            context: "saved second-order position",
                        })?,
                    fraction,
                    &mut position,
                );
                return Ok((velocity, position));
            }
        }
        Err(InterpolationError::OutsideTimeSpan)
    }
}

fn between(time: f64, left: f64, right: f64) -> bool {
    (left <= time && time <= right) || (right <= time && time <= left)
}

#[derive(Clone, Debug, PartialEq)]
struct PartitionedDenseSegment {
    start_time: f64,
    end_time: f64,
    start_velocity: Vec<f64>,
    end_velocity: Vec<f64>,
    start_position: Vec<f64>,
    end_position: Vec<f64>,
}

impl PartitionedDenseSegment {
    fn new(
        start_time: f64,
        end_time: f64,
        start_velocity: &[f64],
        end_velocity: &[f64],
        start_position: &[f64],
        end_position: &[f64],
    ) -> Self {
        Self {
            start_time,
            end_time,
            start_velocity: start_velocity.to_vec(),
            end_velocity: end_velocity.to_vec(),
            start_position: start_position.to_vec(),
            end_position: end_position.to_vec(),
        }
    }

    fn contains(&self, time: f64) -> bool {
        between(time, self.start_time, self.end_time)
    }

    fn interpolate(&self, time: f64, velocity: &mut [f64], position: &mut [f64]) -> Option<()> {
        if !self.contains(time)
            || velocity.len() != self.start_velocity.len()
            || position.len() != self.start_position.len()
        {
            return None;
        }
        if time == self.start_time {
            velocity.copy_from_slice(&self.start_velocity);
            position.copy_from_slice(&self.start_position);
            return Some(());
        }
        if time == self.end_time {
            velocity.copy_from_slice(&self.end_velocity);
            position.copy_from_slice(&self.end_position);
            return Some(());
        }
        let step = self.end_time - self.start_time;
        let theta = (time - self.start_time) / step;
        let theta2 = theta * theta;
        let theta3 = theta2 * theta;
        let h00 = 2.0 * theta3 - 3.0 * theta2 + 1.0;
        let h10 = theta3 - 2.0 * theta2 + theta;
        let h01 = -2.0 * theta3 + 3.0 * theta2;
        let h11 = theta3 - theta2;
        for index in 0..velocity.len() {
            velocity[index] = self.start_velocity[index]
                + theta * (self.end_velocity[index] - self.start_velocity[index]);
            position[index] = h00 * self.start_position[index]
                + h10 * step * self.start_velocity[index]
                + h01 * self.end_position[index]
                + h11 * step * self.end_velocity[index];
        }
        Some(())
    }
}

fn partition(values: &[f64], dimension: usize, index: usize) -> Option<&[f64]> {
    let start = index.checked_mul(dimension)?;
    values.get(start..start + dimension)
}

/// Configuration or integration failure specific to partitioned ODE states.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
#[non_exhaustive]
pub enum SecondOrderSolveError {
    /// Position and velocity partitions do not have the same dimension.
    #[error("position and velocity dimensions must match")]
    StateDimensionMismatch,
    /// A common ODE validation or integration error.
    #[error("{0}")]
    Solve(
        #[from]
        #[source]
        SolveError,
    ),
}

/// An algorithm for `q' = v` second-order ODE problems.
pub trait SecondOrderOdeAlgorithm {
    /// Solves a problem after validating its partitioned state and options.
    fn solve<F, P>(
        &self,
        problem: &SecondOrderOdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<SecondOrderSolution, SecondOrderSolveError>
    where
        F: Fn(&mut [f64], &[f64], &[f64], &P, f64),
    {
        validate(problem, options)?;
        self.solve_validated(problem, options)
    }

    /// Executes the numerical method after common inputs have been checked.
    ///
    /// Implementors may rely on [`SecondOrderOdeAlgorithm::solve`] having
    /// validated both state partitions, the time span, solver options, and
    /// requested output times. User code should normally call
    /// [`SecondOrderOdeAlgorithm::solve`] or [`solve_second_order`]; direct
    /// callers of this lower-level hook are responsible for those invariants.
    fn solve_validated<F, P>(
        &self,
        problem: &SecondOrderOdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<SecondOrderSolution, SecondOrderSolveError>
    where
        F: Fn(&mut [f64], &[f64], &[f64], &P, f64);
}

/// Solves a second-order ODE without flattening its position and velocity.
pub fn solve_second_order<F, P, A>(
    problem: &SecondOrderOdeProblem<F, P>,
    algorithm: A,
    options: &SolveOptions,
) -> Result<SecondOrderSolution, SecondOrderSolveError>
where
    F: Fn(&mut [f64], &[f64], &[f64], &P, f64),
    A: SecondOrderOdeAlgorithm,
{
    algorithm.solve(problem, options)
}

/// First-order drift-then-kick symplectic Euler method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SymplecticEuler;

/// Second-order velocity Verlet method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VelocityVerlet;

/// Second-order kick-drift-kick leapfrog method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VerletLeapfrog;

/// Second-order drift-kick-drift leapfrog method.
///
/// This variant evaluates acceleration twice and supports acceleration that
/// depends on velocity, matching OrdinaryDiffEq's implementation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LeapfrogDriftKickDrift;

/// Fourth-order Runge--Kutta--Nystrom method for velocity-dependent acceleration.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Nystrom4;

/// Fourth-order Runge--Kutta--Nystrom method for acceleration independent of velocity.
///
/// The acceleration callback must ignore its velocity argument. This restriction
/// matches the pinned upstream algorithm.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Nystrom4VelocityIndependent;

/// Fifth-order Runge--Kutta--Nystrom method for acceleration independent of velocity.
///
/// The acceleration callback must ignore its velocity argument. This restriction
/// matches the pinned upstream algorithm.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Nystrom5VelocityIndependent;

/// Three-stage RKN method for second-order linear inhomogeneous problems.
///
/// The pinned method is fourth order on that problem class and generally only
/// second order outside it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Rkn4;

/// Classical Newmark--beta structural dynamics method.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NewmarkBeta {
    beta: f64,
    gamma: f64,
}

impl NewmarkBeta {
    /// Creates a Newmark method with parameters in the upstream admissible ranges.
    pub fn new(beta: f64, gamma: f64) -> Result<Self, ConfigurationError> {
        if !beta.is_finite() || !(0.0..=0.5).contains(&beta) {
            return Err(ConfigurationError::InvalidParameter {
                parameter: "Newmark beta",
                reason: "must be finite and in [0, 0.5]",
            });
        }
        if !gamma.is_finite() || !(0.0..=1.0).contains(&gamma) {
            return Err(ConfigurationError::InvalidParameter {
                parameter: "Newmark gamma",
                reason: "must be finite and in [0, 1]",
            });
        }
        Ok(Self { beta, gamma })
    }

    /// Position update coefficient.
    pub fn beta(self) -> f64 {
        self.beta
    }

    /// Velocity update coefficient.
    pub fn gamma(self) -> f64 {
        self.gamma
    }
}

impl Default for NewmarkBeta {
    fn default() -> Self {
        Self {
            beta: 0.25,
            gamma: 0.5,
        }
    }
}

/// Generalized-alpha structural dynamics method of Chung and Hulbert.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeneralizedAlpha {
    alpha_m: f64,
    alpha_f: f64,
    beta: f64,
    gamma: f64,
}

impl GeneralizedAlpha {
    /// Creates a method from all four generalized-alpha parameters.
    pub fn new(
        alpha_m: f64,
        alpha_f: f64,
        beta: f64,
        gamma: f64,
    ) -> Result<Self, ConfigurationError> {
        if ![alpha_m, alpha_f, beta, gamma]
            .iter()
            .all(|value| value.is_finite())
        {
            return Err(ConfigurationError::NonFiniteData {
                context: "generalized-alpha parameters",
            });
        }
        if alpha_m > alpha_f || alpha_f > 0.5 {
            return Err(ConfigurationError::InvalidParameter {
                parameter: "generalized-alpha alpha values",
                reason: "must satisfy alpha_m <= alpha_f <= 0.5",
            });
        }
        let minimum_beta = (0.5 + alpha_f - alpha_m).powi(2) / 4.0;
        if beta < minimum_beta || !(0.0..=1.0).contains(&gamma) {
            return Err(ConfigurationError::InvalidParameter {
                parameter: "generalized-alpha beta or gamma",
                reason: "must satisfy the unconditional-stability bounds",
            });
        }
        Ok(Self {
            alpha_m,
            alpha_f,
            beta,
            gamma,
        })
    }

    /// Uses the recommended spectral-radius-at-infinity parameterization.
    pub fn from_spectral_radius(rho_infinity: f64) -> Result<Self, ConfigurationError> {
        if !rho_infinity.is_finite() || !(0.0..=1.0).contains(&rho_infinity) {
            return Err(ConfigurationError::InvalidParameter {
                parameter: "generalized-alpha spectral radius",
                reason: "must be finite and in [0, 1]",
            });
        }
        let alpha_m = (2.0 * rho_infinity - 1.0) / (rho_infinity + 1.0);
        let alpha_f = rho_infinity / (rho_infinity + 1.0);
        let gamma = 0.5 - alpha_m + alpha_f;
        let beta = (0.5 + alpha_f - alpha_m).powi(2) / 4.0;
        Self::new(alpha_m, alpha_f, beta, gamma)
    }

    /// Uses the Hilber--Hughes--Taylor alpha parameterization.
    pub fn from_hht_alpha(alpha: f64) -> Result<Self, ConfigurationError> {
        if !alpha.is_finite() || !(-1.0 / 3.0..=0.0).contains(&alpha) {
            return Err(ConfigurationError::InvalidParameter {
                parameter: "HHT alpha",
                reason: "must be finite and in [-1/3, 0]",
            });
        }
        Self::new(
            0.0,
            -alpha,
            (1.0 - alpha).powi(2) / 4.0,
            (1.0 - 2.0 * alpha) / 2.0,
        )
    }

    /// Returns `(alpha_m, alpha_f, beta, gamma)`.
    pub fn parameters(self) -> (f64, f64, f64, f64) {
        (self.alpha_m, self.alpha_f, self.beta, self.gamma)
    }
}

impl Default for GeneralizedAlpha {
    fn default() -> Self {
        Self {
            alpha_m: 0.5,
            alpha_f: 0.5,
            beta: 0.25,
            gamma: 0.5,
        }
    }
}

/// Dormand--Prince fourth-order adaptive Runge--Kutta--Nystrom method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Dprkn4;

/// Dormand--Prince fifth-order adaptive Runge--Kutta--Nystrom method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Dprkn5;

/// Dormand--Prince sixth-order adaptive RKN method with free sixth-order dense output.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Dprkn6;

/// Fine--Montagnier sixth-order adaptive Runge--Kutta--Nystrom method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Dprkn6Fm;

/// Dormand--Prince eighth-order adaptive Runge--Kutta--Nystrom method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Dprkn8;

/// Dormand--Prince twelfth-order adaptive Runge--Kutta--Nystrom method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Dprkn12;

/// Embedded fourth-order Runge--Kutta--Nystrom method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Erkn4;

/// Embedded fifth-order Runge--Kutta--Nystrom method with position-only error control.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Erkn5;

/// Embedded seventh-order Runge--Kutta--Nystrom method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Erkn7;

/// Fine's fourth-order adaptive RKN method for velocity-dependent acceleration.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FineRkn4;

/// Fine's fifth-order adaptive RKN method for velocity-dependent acceleration.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FineRkn5;

/// Third-order fixed-step improved Runge--Kutta--Nystrom two-step method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Irkn3;

/// Fourth-order fixed-step improved Runge--Kutta--Nystrom two-step method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Irkn4;

/// SciML-compatible constructor spelling for [`Dprkn4`].
pub type DPRKN4 = Dprkn4;
/// SciML-compatible constructor spelling for [`Dprkn5`].
pub type DPRKN5 = Dprkn5;
/// SciML-compatible constructor spelling for [`Dprkn6`].
pub type DPRKN6 = Dprkn6;
/// SciML-compatible constructor spelling for [`Dprkn6Fm`].
pub type DPRKN6FM = Dprkn6Fm;
/// SciML-compatible constructor spelling for [`Dprkn8`].
pub type DPRKN8 = Dprkn8;
/// SciML-compatible constructor spelling for [`Dprkn12`].
pub type DPRKN12 = Dprkn12;
/// SciML-compatible constructor spelling for [`Erkn4`].
pub type ERKN4 = Erkn4;
/// SciML-compatible constructor spelling for [`Erkn5`].
pub type ERKN5 = Erkn5;
/// SciML-compatible constructor spelling for [`Erkn7`].
pub type ERKN7 = Erkn7;
/// SciML-compatible constructor spelling for [`FineRkn4`].
pub type FineRKN4 = FineRkn4;
/// SciML-compatible constructor spelling for [`FineRkn5`].
pub type FineRKN5 = FineRkn5;
/// SciML-compatible constructor spelling for [`Irkn3`].
pub type IRKN3 = Irkn3;
/// SciML-compatible constructor spelling for [`Irkn4`].
pub type IRKN4 = Irkn4;

#[allow(non_upper_case_globals)]
/// Value-form SciML-compatible constructor spelling for [`Dprkn4`].
pub const DPRKN4: Dprkn4 = Dprkn4;
#[allow(non_upper_case_globals)]
/// Value-form SciML-compatible constructor spelling for [`Dprkn5`].
pub const DPRKN5: Dprkn5 = Dprkn5;
#[allow(non_upper_case_globals)]
/// Value-form SciML-compatible constructor spelling for [`Dprkn6`].
pub const DPRKN6: Dprkn6 = Dprkn6;
#[allow(non_upper_case_globals)]
/// Value-form SciML-compatible constructor spelling for [`Dprkn6Fm`].
pub const DPRKN6FM: Dprkn6Fm = Dprkn6Fm;
#[allow(non_upper_case_globals)]
/// Value-form SciML-compatible constructor spelling for [`Dprkn8`].
pub const DPRKN8: Dprkn8 = Dprkn8;
#[allow(non_upper_case_globals)]
/// Value-form SciML-compatible constructor spelling for [`Dprkn12`].
pub const DPRKN12: Dprkn12 = Dprkn12;
#[allow(non_upper_case_globals)]
/// Value-form SciML-compatible constructor spelling for [`Erkn4`].
pub const ERKN4: Erkn4 = Erkn4;
#[allow(non_upper_case_globals)]
/// Value-form SciML-compatible constructor spelling for [`Erkn5`].
pub const ERKN5: Erkn5 = Erkn5;
#[allow(non_upper_case_globals)]
/// Value-form SciML-compatible constructor spelling for [`Erkn7`].
pub const ERKN7: Erkn7 = Erkn7;
#[allow(non_upper_case_globals)]
/// Value-form SciML-compatible constructor spelling for [`FineRkn4`].
pub const FineRKN4: FineRkn4 = FineRkn4;
#[allow(non_upper_case_globals)]
/// Value-form SciML-compatible constructor spelling for [`FineRkn5`].
pub const FineRKN5: FineRkn5 = FineRkn5;
#[allow(non_upper_case_globals)]
/// Value-form SciML-compatible constructor spelling for [`Irkn3`].
pub const IRKN3: Irkn3 = Irkn3;
#[allow(non_upper_case_globals)]
/// Value-form SciML-compatible constructor spelling for [`Irkn4`].
pub const IRKN4: Irkn4 = Irkn4;

struct RknTableau {
    nodes: &'static [f64],
    position_coefficients: &'static [&'static [f64]],
    velocity_coefficients: Option<&'static [&'static [f64]]>,
    position_weights: &'static [f64],
    velocity_weights: &'static [f64],
}

struct AdaptiveRknTableau {
    position_coefficients: &'static [f64],
    velocity_coefficients: Option<&'static [f64]>,
    position_weights: &'static [f64],
    velocity_weights: &'static [f64],
    position_error_weights: &'static [f64],
    velocity_error_weights: &'static [f64],
    nodes: &'static [f64],
    order: usize,
    position_only_error: bool,
    dense_position_coefficients: Option<&'static [f64]>,
    dense_velocity_coefficients: Option<&'static [f64]>,
}

macro_rules! adaptive_rkn_tableau {
    ($name:ident, $a:ident, $b:ident, $bp:ident, $bt:ident, $bpt:ident, $c:ident, $order:ident, $pos_only:ident) => {
        const $name: AdaptiveRknTableau = AdaptiveRknTableau {
            position_coefficients: $a,
            velocity_coefficients: None,
            position_weights: $b,
            velocity_weights: $bp,
            position_error_weights: $bt,
            velocity_error_weights: $bpt,
            nodes: $c,
            order: $order,
            position_only_error: $pos_only,
            dense_position_coefficients: None,
            dense_velocity_coefficients: None,
        };
    };
}

adaptive_rkn_tableau!(
    DPRKN4_ADAPTIVE_TABLEAU,
    DPRKN4_A,
    DPRKN4_B,
    DPRKN4_BP,
    DPRKN4_BTILDE,
    DPRKN4_BPTILDE,
    DPRKN4_C,
    DPRKN4_ORDER,
    DPRKN4_POS_ONLY_ERROR
);

macro_rules! adaptive_velocity_dependent_rkn_tableau {
    ($name:ident, $a:ident, $abar:ident, $b:ident, $bp:ident, $bt:ident, $bpt:ident, $c:ident, $order:ident, $pos_only:ident) => {
        const $name: AdaptiveRknTableau = AdaptiveRknTableau {
            position_coefficients: $a,
            velocity_coefficients: Some($abar),
            position_weights: $b,
            velocity_weights: $bp,
            position_error_weights: $bt,
            velocity_error_weights: $bpt,
            nodes: $c,
            order: $order,
            position_only_error: $pos_only,
            dense_position_coefficients: None,
            dense_velocity_coefficients: None,
        };
    };
}

adaptive_velocity_dependent_rkn_tableau!(
    FINERKN4_ADAPTIVE_TABLEAU,
    FINERKN4_A,
    FINERKN4_ABAR,
    FINERKN4_B,
    FINERKN4_BP,
    FINERKN4_BTILDE,
    FINERKN4_BPTILDE,
    FINERKN4_C,
    FINERKN4_ORDER,
    FINERKN4_POS_ONLY_ERROR
);
adaptive_velocity_dependent_rkn_tableau!(
    FINERKN5_ADAPTIVE_TABLEAU,
    FINERKN5_A,
    FINERKN5_ABAR,
    FINERKN5_B,
    FINERKN5_BP,
    FINERKN5_BTILDE,
    FINERKN5_BPTILDE,
    FINERKN5_C,
    FINERKN5_ORDER,
    FINERKN5_POS_ONLY_ERROR
);

const DPRKN6_ADAPTIVE_TABLEAU: AdaptiveRknTableau = AdaptiveRknTableau {
    position_coefficients: DPRKN6_A,
    velocity_coefficients: None,
    position_weights: DPRKN6_B,
    velocity_weights: DPRKN6_BP,
    position_error_weights: DPRKN6_BTILDE,
    velocity_error_weights: DPRKN6_BPTILDE,
    nodes: DPRKN6_C,
    order: DPRKN6_ORDER,
    position_only_error: DPRKN6_POS_ONLY_ERROR,
    dense_position_coefficients: Some(DPRKN6_R),
    dense_velocity_coefficients: Some(DPRKN6_RP),
};
adaptive_rkn_tableau!(
    DPRKN5_ADAPTIVE_TABLEAU,
    DPRKN5_A,
    DPRKN5_B,
    DPRKN5_BP,
    DPRKN5_BTILDE,
    DPRKN5_BPTILDE,
    DPRKN5_C,
    DPRKN5_ORDER,
    DPRKN5_POS_ONLY_ERROR
);
adaptive_rkn_tableau!(
    DPRKN6FM_ADAPTIVE_TABLEAU,
    DPRKN6FM_A,
    DPRKN6FM_B,
    DPRKN6FM_BP,
    DPRKN6FM_BTILDE,
    DPRKN6FM_BPTILDE,
    DPRKN6FM_C,
    DPRKN6FM_ORDER,
    DPRKN6FM_POS_ONLY_ERROR
);
adaptive_rkn_tableau!(
    DPRKN8_ADAPTIVE_TABLEAU,
    DPRKN8_A,
    DPRKN8_B,
    DPRKN8_BP,
    DPRKN8_BTILDE,
    DPRKN8_BPTILDE,
    DPRKN8_C,
    DPRKN8_ORDER,
    DPRKN8_POS_ONLY_ERROR
);
adaptive_rkn_tableau!(
    DPRKN12_ADAPTIVE_TABLEAU,
    DPRKN12_A,
    DPRKN12_B,
    DPRKN12_BP,
    DPRKN12_BTILDE,
    DPRKN12_BPTILDE,
    DPRKN12_C,
    DPRKN12_ORDER,
    DPRKN12_POS_ONLY_ERROR
);
adaptive_rkn_tableau!(
    ERKN4_ADAPTIVE_TABLEAU,
    ERKN4_A,
    ERKN4_B,
    ERKN4_BP,
    ERKN4_BTILDE,
    ERKN4_BPTILDE,
    ERKN4_C,
    ERKN4_ORDER,
    ERKN4_POS_ONLY_ERROR
);
adaptive_rkn_tableau!(
    ERKN5_ADAPTIVE_TABLEAU,
    ERKN5_A,
    ERKN5_B,
    ERKN5_BP,
    ERKN5_BTILDE,
    ERKN5_BPTILDE,
    ERKN5_C,
    ERKN5_ORDER,
    ERKN5_POS_ONLY_ERROR
);
adaptive_rkn_tableau!(
    ERKN7_ADAPTIVE_TABLEAU,
    ERKN7_A,
    ERKN7_B,
    ERKN7_BP,
    ERKN7_BTILDE,
    ERKN7_BPTILDE,
    ERKN7_C,
    ERKN7_ORDER,
    ERKN7_POS_ONLY_ERROR
);

const NYSTROM4_VI_A: &[&[f64]] = &[EMPTY_ROW, NYSTROM4_VI_A2, NYSTROM4_VI_A3];

const NYSTROM4_VI_TABLEAU: RknTableau = RknTableau {
    nodes: NYSTROM4_VI_NODES,
    position_coefficients: NYSTROM4_VI_A,
    velocity_coefficients: None,
    position_weights: NYSTROM4_VI_B,
    velocity_weights: NYSTROM4_VI_BP,
};

const NYSTROM5_VI_A: &[&[f64]] = &[EMPTY_ROW, NYSTROM5_VI_A2, NYSTROM5_VI_A3, NYSTROM5_VI_A4];

const NYSTROM5_VI_TABLEAU: RknTableau = RknTableau {
    nodes: NYSTROM5_VI_NODES,
    position_coefficients: NYSTROM5_VI_A,
    velocity_coefficients: None,
    position_weights: NYSTROM5_VI_B,
    velocity_weights: NYSTROM5_VI_BP,
};

const NYSTROM4_A: &[&[f64]] = &[EMPTY_ROW, NYSTROM4_A2, NYSTROM4_A3, NYSTROM4_A4];

const NYSTROM4_ABAR: &[&[f64]] = &[EMPTY_ROW, NYSTROM4_ABAR2, NYSTROM4_ABAR3, NYSTROM4_ABAR4];

const NYSTROM4_TABLEAU: RknTableau = RknTableau {
    nodes: NYSTROM4_NODES,
    position_coefficients: NYSTROM4_A,
    velocity_coefficients: Some(NYSTROM4_ABAR),
    position_weights: NYSTROM4_B,
    velocity_weights: NYSTROM4_BP,
};

const RKN4_A: &[&[f64]] = &[EMPTY_ROW, NYSTROM4_A2, NYSTROM4_VI_A3];

const RKN4_ABAR: &[&[f64]] = &[EMPTY_ROW, NYSTROM4_ABAR2, RKN4_ABAR3];
const RKN4_TABLEAU: RknTableau = RknTableau {
    nodes: NYSTROM4_VI_NODES,
    position_coefficients: RKN4_A,
    velocity_coefficients: Some(RKN4_ABAR),
    position_weights: NYSTROM4_VI_B,
    velocity_weights: NYSTROM4_VI_BP,
};

#[derive(Clone, Copy)]
enum Method {
    SymplecticEuler,
    VelocityVerlet,
    VerletLeapfrog,
    LeapfrogDriftKickDrift,
}

macro_rules! impl_algorithm {
    ($algorithm:ty, $method:expr) => {
        impl SecondOrderOdeAlgorithm for $algorithm {
            fn solve_validated<F, P>(
                &self,
                problem: &SecondOrderOdeProblem<F, P>,
                options: &SolveOptions,
            ) -> Result<SecondOrderSolution, SecondOrderSolveError>
            where
                F: Fn(&mut [f64], &[f64], &[f64], &P, f64),
            {
                solve_fixed(problem, options, $method)
            }
        }
    };
}

impl_algorithm!(SymplecticEuler, Method::SymplecticEuler);
impl_algorithm!(VelocityVerlet, Method::VelocityVerlet);
impl_algorithm!(VerletLeapfrog, Method::VerletLeapfrog);
impl_algorithm!(LeapfrogDriftKickDrift, Method::LeapfrogDriftKickDrift);

macro_rules! impl_rkn_algorithm {
    ($algorithm:ty, $tableau:expr) => {
        impl SecondOrderOdeAlgorithm for $algorithm {
            fn solve_validated<F, P>(
                &self,
                problem: &SecondOrderOdeProblem<F, P>,
                options: &SolveOptions,
            ) -> Result<SecondOrderSolution, SecondOrderSolveError>
            where
                F: Fn(&mut [f64], &[f64], &[f64], &P, f64),
            {
                solve_rkn_fixed(problem, options, &$tableau)
            }
        }
    };
}

impl_rkn_algorithm!(Nystrom4, NYSTROM4_TABLEAU);
impl_rkn_algorithm!(Nystrom4VelocityIndependent, NYSTROM4_VI_TABLEAU);
impl_rkn_algorithm!(Nystrom5VelocityIndependent, NYSTROM5_VI_TABLEAU);
impl_rkn_algorithm!(Rkn4, RKN4_TABLEAU);

macro_rules! impl_adaptive_rkn_algorithm {
    ($algorithm:ty, $tableau:ident) => {
        impl SecondOrderOdeAlgorithm for $algorithm {
            fn solve_validated<F, P>(
                &self,
                problem: &SecondOrderOdeProblem<F, P>,
                options: &SolveOptions,
            ) -> Result<SecondOrderSolution, SecondOrderSolveError>
            where
                F: Fn(&mut [f64], &[f64], &[f64], &P, f64),
            {
                solve_rkn_adaptive(problem, options, &$tableau)
            }
        }
    };
}

impl_adaptive_rkn_algorithm!(Dprkn4, DPRKN4_ADAPTIVE_TABLEAU);
impl_adaptive_rkn_algorithm!(Dprkn5, DPRKN5_ADAPTIVE_TABLEAU);
impl_adaptive_rkn_algorithm!(Dprkn6, DPRKN6_ADAPTIVE_TABLEAU);
impl_adaptive_rkn_algorithm!(Dprkn6Fm, DPRKN6FM_ADAPTIVE_TABLEAU);
impl_adaptive_rkn_algorithm!(Dprkn8, DPRKN8_ADAPTIVE_TABLEAU);
impl_adaptive_rkn_algorithm!(Dprkn12, DPRKN12_ADAPTIVE_TABLEAU);
impl_adaptive_rkn_algorithm!(Erkn4, ERKN4_ADAPTIVE_TABLEAU);
impl_adaptive_rkn_algorithm!(Erkn5, ERKN5_ADAPTIVE_TABLEAU);
impl_adaptive_rkn_algorithm!(Erkn7, ERKN7_ADAPTIVE_TABLEAU);
impl_adaptive_rkn_algorithm!(FineRkn4, FINERKN4_ADAPTIVE_TABLEAU);
impl_adaptive_rkn_algorithm!(FineRkn5, FINERKN5_ADAPTIVE_TABLEAU);

impl SecondOrderOdeAlgorithm for NewmarkBeta {
    fn solve_validated<F, P>(
        &self,
        problem: &SecondOrderOdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<SecondOrderSolution, SecondOrderSolveError>
    where
        F: Fn(&mut [f64], &[f64], &[f64], &P, f64),
    {
        solve_newmark(
            problem,
            options,
            StructuralParameters {
                alpha_m: 0.0,
                alpha_f: 0.0,
                beta: self.beta,
                gamma: self.gamma,
            },
        )
    }
}

impl SecondOrderOdeAlgorithm for GeneralizedAlpha {
    fn solve_validated<F, P>(
        &self,
        problem: &SecondOrderOdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<SecondOrderSolution, SecondOrderSolveError>
    where
        F: Fn(&mut [f64], &[f64], &[f64], &P, f64),
    {
        solve_newmark(
            problem,
            options,
            StructuralParameters {
                alpha_m: self.alpha_m,
                alpha_f: self.alpha_f,
                beta: self.beta,
                gamma: self.gamma,
            },
        )
    }
}

impl SecondOrderOdeAlgorithm for Irkn3 {
    fn solve_validated<F, P>(
        &self,
        problem: &SecondOrderOdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<SecondOrderSolution, SecondOrderSolveError>
    where
        F: Fn(&mut [f64], &[f64], &[f64], &P, f64),
    {
        solve_irkn(problem, options, IrknMethod::ThirdOrder)
    }
}

impl SecondOrderOdeAlgorithm for Irkn4 {
    fn solve_validated<F, P>(
        &self,
        problem: &SecondOrderOdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<SecondOrderSolution, SecondOrderSolveError>
    where
        F: Fn(&mut [f64], &[f64], &[f64], &P, f64),
    {
        solve_irkn(problem, options, IrknMethod::FourthOrder)
    }
}

fn validate<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    options: &SolveOptions,
) -> Result<(), SecondOrderSolveError> {
    if problem.initial_position.is_empty() {
        return Err(SolveError::EmptyState.into());
    }
    if problem.initial_position.len() != problem.initial_velocity.len() {
        return Err(SecondOrderSolveError::StateDimensionMismatch);
    }
    validate_state_time_options(&problem.initial_position, problem.time_span, options)?;
    validate_preset_time_sequences(problem.preset_time_sequences(), problem.time_span)?;
    validate_vector_callback_lengths(problem.vector_callback_lengths())?;
    if !problem
        .initial_velocity
        .iter()
        .all(|value| value.is_finite())
    {
        return Err(SolveError::NonFiniteInitialState.into());
    }
    Ok(())
}

struct Workspace {
    candidate_velocity: Vec<f64>,
    candidate_position: Vec<f64>,
    acceleration: Vec<f64>,
    stage_velocity: Vec<f64>,
    stage_position: Vec<f64>,
    previous_effect_velocity: Vec<f64>,
    previous_effect_position: Vec<f64>,
}

impl Workspace {
    fn new(dimension: usize, callbacks: bool) -> Self {
        Self {
            candidate_velocity: vec![0.0; dimension],
            candidate_position: vec![0.0; dimension],
            acceleration: vec![0.0; dimension],
            stage_velocity: vec![0.0; dimension],
            stage_position: vec![0.0; dimension],
            previous_effect_velocity: if callbacks {
                vec![0.0; dimension]
            } else {
                Vec::new()
            },
            previous_effect_position: if callbacks {
                vec![0.0; dimension]
            } else {
                Vec::new()
            },
        }
    }
}

struct RknWorkspace {
    candidate_velocity: Vec<f64>,
    candidate_position: Vec<f64>,
    stage_velocity: Vec<f64>,
    stage_position: Vec<f64>,
    stage_accelerations: Vec<f64>,
    previous_effect_velocity: Vec<f64>,
    previous_effect_position: Vec<f64>,
}

impl RknWorkspace {
    fn new(dimension: usize, stages: usize, callbacks: bool) -> Self {
        Self {
            candidate_velocity: vec![0.0; dimension],
            candidate_position: vec![0.0; dimension],
            stage_velocity: vec![0.0; dimension],
            stage_position: vec![0.0; dimension],
            stage_accelerations: vec![0.0; dimension * stages],
            previous_effect_velocity: if callbacks {
                vec![0.0; dimension]
            } else {
                Vec::new()
            },
            previous_effect_position: if callbacks {
                vec![0.0; dimension]
            } else {
                Vec::new()
            },
        }
    }
}

#[derive(Clone, Copy)]
struct StructuralParameters {
    alpha_m: f64,
    alpha_f: f64,
    beta: f64,
    gamma: f64,
}

struct StructuralWorkspace {
    full_velocity: Vec<f64>,
    full_position: Vec<f64>,
    full_acceleration: Vec<f64>,
    candidate_velocity: Vec<f64>,
    candidate_position: Vec<f64>,
    candidate_acceleration: Vec<f64>,
    half_velocity: Vec<f64>,
    half_position: Vec<f64>,
    half_acceleration: Vec<f64>,
    trial_acceleration: Vec<f64>,
    trial_velocity: Vec<f64>,
    trial_position: Vec<f64>,
    evaluated_acceleration: Vec<f64>,
    perturbed_acceleration: Vec<f64>,
    residual: Vec<f64>,
    perturbed_residual: Vec<f64>,
    correction: Vec<f64>,
    matrix: Vec<f64>,
    pivots: Vec<usize>,
    previous_effect_velocity: Vec<f64>,
    previous_effect_position: Vec<f64>,
}

impl StructuralWorkspace {
    fn new(dimension: usize, callbacks: bool) -> Self {
        Self {
            full_velocity: vec![0.0; dimension],
            full_position: vec![0.0; dimension],
            full_acceleration: vec![0.0; dimension],
            candidate_velocity: vec![0.0; dimension],
            candidate_position: vec![0.0; dimension],
            candidate_acceleration: vec![0.0; dimension],
            half_velocity: vec![0.0; dimension],
            half_position: vec![0.0; dimension],
            half_acceleration: vec![0.0; dimension],
            trial_acceleration: vec![0.0; dimension],
            trial_velocity: vec![0.0; dimension],
            trial_position: vec![0.0; dimension],
            evaluated_acceleration: vec![0.0; dimension],
            perturbed_acceleration: vec![0.0; dimension],
            residual: vec![0.0; dimension],
            perturbed_residual: vec![0.0; dimension],
            correction: vec![0.0; dimension],
            matrix: vec![0.0; dimension * dimension],
            pivots: vec![0; dimension],
            previous_effect_velocity: if callbacks {
                vec![0.0; dimension]
            } else {
                Vec::new()
            },
            previous_effect_position: if callbacks {
                vec![0.0; dimension]
            } else {
                Vec::new()
            },
        }
    }
}

fn finish_successful<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    velocity: &mut [f64],
    position: &mut [f64],
    time: f64,
    mut recorder: PartitionedRecorder<'_>,
    stats: SolverStats,
) -> Result<SecondOrderSolution, SecondOrderSolveError>
where
    F: Fn(&mut [f64], &[f64], &[f64], &P, f64),
{
    if apply_finalize_callbacks(problem, velocity, position, time)? {
        recorder.synchronize_endpoint(time, velocity, position);
    }
    Ok(recorder.finish(stats))
}

fn solve_newmark<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    options: &SolveOptions,
    method: StructuralParameters,
) -> Result<SecondOrderSolution, SecondOrderSolveError>
where
    F: Fn(&mut [f64], &[f64], &[f64], &P, f64),
{
    if !options.adaptive && options.initial_step.is_none() {
        return Err(SolveError::InitialStepRequired.into());
    }
    let dimension = problem.initial_position.len();
    let (start, end) = problem.time_span;
    let direction = (end - start).signum();
    let maximum_step = options.max_step.min((end - start).abs());
    let mut step_magnitude = options
        .initial_step
        .unwrap_or_else(|| ((end - start).abs() / 100.0).min(maximum_step))
        .min(maximum_step);
    if !step_magnitude.is_finite() || step_magnitude <= 0.0 {
        return Err(SolveError::StepSizeUnderflow.into());
    }

    let mut velocity = problem.initial_velocity.clone();
    let mut position = problem.initial_position.clone();
    let mut acceleration = vec![0.0; dimension];
    let mut workspace = StructuralWorkspace::new(dimension, !problem.callbacks.is_empty());
    let mut stats = SolverStats::default();

    let mut recorder = PartitionedRecorder::new(&velocity, &position, start, options);
    let initial = apply_initial_callbacks(problem, &mut velocity, &mut position, start)?;
    stats.callback_invocations += initial.invocations;
    stats.rhs_evaluations += initial.rhs_evaluations;
    if initial.state_modified {
        recorder.record_callback(
            start,
            &problem.initial_velocity,
            &problem.initial_position,
            &velocity,
            &position,
            initial,
            true,
        );
    }
    if problem
        .domain_rejection_factor(&velocity, &position, start)
        .is_some()
    {
        return Err(SolveError::InitialStateOutOfDomain.into());
    }
    if initial.terminate {
        return finish_successful(
            problem,
            &mut velocity,
            &mut position,
            start,
            recorder,
            stats,
        );
    }
    step_magnitude =
        callback_adjusted_step(initial, direction * step_magnitude, direction, maximum_step).abs();
    evaluate_acceleration(
        problem,
        &mut acceleration,
        &velocity,
        &position,
        start,
        &mut stats,
    )?;

    let controller = ControllerConfig::proportional(2, 0.9, 0.2, 5.0, 0.2);
    let mut controller_state = ControllerState::default();
    let mut previous_attempt_rejected = false;
    let mut time = start;
    let mut attempts = 0;
    let mut time_stops = TimeStopSchedule::new(&options.time_stops, start, end);
    while direction * (end - time) > 0.0 {
        if attempts == options.max_steps {
            return Err(SolveError::MaxStepsExceeded.into());
        }
        attempts += 1;
        let step = time_stops.clip_step_with(
            time,
            direction * step_magnitude,
            problem.next_preset_time(time, direction),
        );
        if time + step == time {
            return Err(SolveError::StepSizeUnderflow.into());
        }

        structural_substep(
            problem,
            method,
            &velocity,
            &position,
            &acceleration,
            time,
            step,
            &mut workspace.full_velocity,
            &mut workspace.full_position,
            &mut workspace.full_acceleration,
            &mut workspace.trial_acceleration,
            &mut workspace.trial_velocity,
            &mut workspace.trial_position,
            &mut workspace.evaluated_acceleration,
            &mut workspace.perturbed_acceleration,
            &mut workspace.residual,
            &mut workspace.perturbed_residual,
            &mut workspace.correction,
            &mut workspace.matrix,
            &mut workspace.pivots,
            &mut stats,
        )?;

        let error = if options.adaptive {
            structural_substep(
                problem,
                method,
                &velocity,
                &position,
                &acceleration,
                time,
                0.5 * step,
                &mut workspace.half_velocity,
                &mut workspace.half_position,
                &mut workspace.half_acceleration,
                &mut workspace.trial_acceleration,
                &mut workspace.trial_velocity,
                &mut workspace.trial_position,
                &mut workspace.evaluated_acceleration,
                &mut workspace.perturbed_acceleration,
                &mut workspace.residual,
                &mut workspace.perturbed_residual,
                &mut workspace.correction,
                &mut workspace.matrix,
                &mut workspace.pivots,
                &mut stats,
            )?;
            structural_substep(
                problem,
                method,
                &workspace.half_velocity,
                &workspace.half_position,
                &workspace.half_acceleration,
                time + 0.5 * step,
                0.5 * step,
                &mut workspace.candidate_velocity,
                &mut workspace.candidate_position,
                &mut workspace.candidate_acceleration,
                &mut workspace.trial_acceleration,
                &mut workspace.trial_velocity,
                &mut workspace.trial_position,
                &mut workspace.evaluated_acceleration,
                &mut workspace.perturbed_acceleration,
                &mut workspace.residual,
                &mut workspace.perturbed_residual,
                &mut workspace.correction,
                &mut workspace.matrix,
                &mut workspace.pivots,
                &mut stats,
            )?;
            structural_error_norm(
                &workspace.candidate_velocity,
                &workspace.candidate_position,
                &workspace.full_velocity,
                &workspace.full_position,
                &velocity,
                &position,
                options,
            )
        } else {
            workspace
                .candidate_velocity
                .copy_from_slice(&workspace.full_velocity);
            workspace
                .candidate_position
                .copy_from_slice(&workspace.full_position);
            workspace
                .candidate_acceleration
                .copy_from_slice(&workspace.full_acceleration);
            0.0
        };

        if error > 1.0 {
            stats.rejected_steps += 1;
            controller_state.rejected(error);
            step_magnitude = step.abs() * controller_state.factor(error, controller).min(1.0);
            previous_attempt_rejected = true;
            continue;
        }

        let previous_time = time;
        let attempted_time = time + step;
        let mut next_time = if direction * (end - attempted_time) <= 0.0 {
            end
        } else {
            attempted_time
        };
        if let Some(reduction_factor) = problem.domain_rejection_factor(
            &workspace.candidate_velocity,
            &workspace.candidate_position,
            next_time,
        ) {
            stats.rejected_steps += 1;
            controller_state.reset();
            step_magnitude = step.abs() * reduction_factor;
            previous_attempt_rejected = true;
            continue;
        }
        let callback = apply_step_callbacks(
            problem,
            &velocity,
            &position,
            previous_time,
            &mut workspace.candidate_velocity,
            &mut workspace.candidate_position,
            &mut next_time,
            &mut workspace.previous_effect_velocity,
            &mut workspace.previous_effect_position,
            options.event_tolerance,
            None,
        )?;
        stats.callback_invocations += callback.invocations;
        stats.rhs_evaluations += callback.rhs_evaluations;
        stats.accepted_steps += 1;
        recorder.record_step(
            &velocity,
            &position,
            previous_time,
            if callback.invocations == 0 {
                &workspace.candidate_velocity
            } else {
                &workspace.previous_effect_velocity
            },
            if callback.invocations == 0 {
                &workspace.candidate_position
            } else {
                &workspace.previous_effect_position
            },
            next_time,
            next_time == end,
            None,
        )?;
        if callback.invocations > 0 {
            recorder.record_callback(
                next_time,
                &workspace.previous_effect_velocity,
                &workspace.previous_effect_position,
                &workspace.candidate_velocity,
                &workspace.candidate_position,
                callback,
                next_time == end,
            );
        }
        if callback.terminate {
            return finish_successful(
                problem,
                &mut workspace.candidate_velocity,
                &mut workspace.candidate_position,
                next_time,
                recorder,
                stats,
            );
        }

        time = next_time;
        time_stops.accepted(time);
        std::mem::swap(&mut velocity, &mut workspace.candidate_velocity);
        std::mem::swap(&mut position, &mut workspace.candidate_position);
        if callback.state_modified || next_time != attempted_time {
            evaluate_acceleration(
                problem,
                &mut acceleration,
                &velocity,
                &position,
                time,
                &mut stats,
            )?;
        } else {
            std::mem::swap(&mut acceleration, &mut workspace.candidate_acceleration);
        }

        if callback.state_modified {
            controller_state.reset();
        }
        if callback.requested_step.is_some() {
            step_magnitude = callback_adjusted_step(
                callback,
                direction * step_magnitude,
                direction,
                maximum_step,
            )
            .abs();
        } else if options.adaptive {
            controller_state.accepted(error);
            let mut factor = controller_state.factor(error, controller);
            if previous_attempt_rejected {
                factor = factor.min(1.0);
            }
            step_magnitude = callback_adjusted_step(
                callback,
                direction * step.abs() * factor,
                direction,
                maximum_step,
            )
            .abs();
        } else if callback.step_limit.is_some() {
            step_magnitude = callback_adjusted_step(
                callback,
                direction * step_magnitude,
                direction,
                maximum_step,
            )
            .abs();
        }
        previous_attempt_rejected = false;
    }
    finish_successful(problem, &mut velocity, &mut position, time, recorder, stats)
}

#[allow(clippy::too_many_arguments)]
fn structural_substep<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    method: StructuralParameters,
    velocity: &[f64],
    position: &[f64],
    acceleration: &[f64],
    time: f64,
    step: f64,
    output_velocity: &mut [f64],
    output_position: &mut [f64],
    output_acceleration: &mut [f64],
    trial_acceleration: &mut [f64],
    trial_velocity: &mut [f64],
    trial_position: &mut [f64],
    evaluated_acceleration: &mut [f64],
    perturbed_acceleration: &mut [f64],
    residual: &mut [f64],
    perturbed_residual: &mut [f64],
    correction: &mut [f64],
    matrix: &mut [f64],
    pivots: &mut [usize],
    stats: &mut SolverStats,
) -> Result<(), SecondOrderSolveError>
where
    F: Fn(&mut [f64], &[f64], &[f64], &P, f64),
{
    const MAX_ITERATIONS: usize = 12;
    const TOLERANCE: f64 = 1.0e-12;
    trial_acceleration.copy_from_slice(acceleration);
    let dimension = acceleration.len();
    for _ in 0..MAX_ITERATIONS {
        structural_residual(
            problem,
            method,
            velocity,
            position,
            acceleration,
            time,
            step,
            trial_acceleration,
            trial_velocity,
            trial_position,
            evaluated_acceleration,
            residual,
            stats,
        )?;
        stats.nonlinear_iterations += 1;
        let residual_norm = residual
            .iter()
            .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
        let scale = 1.0
            + trial_acceleration
                .iter()
                .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
        if residual_norm <= TOLERANCE * scale {
            update_structural_state(
                method,
                velocity,
                position,
                acceleration,
                trial_acceleration,
                step,
                output_velocity,
                output_position,
            );
            evaluate_acceleration(
                problem,
                output_acceleration,
                output_velocity,
                output_position,
                time + step,
                stats,
            )?;
            return Ok(());
        }

        stats.jacobian_evaluations += 1;
        for column in 0..dimension {
            let original = trial_acceleration[column];
            let delta = f64::EPSILON.sqrt() * original.abs().max(1.0);
            trial_acceleration[column] = original + delta;
            structural_residual(
                problem,
                method,
                velocity,
                position,
                acceleration,
                time,
                step,
                trial_acceleration,
                trial_velocity,
                trial_position,
                perturbed_acceleration,
                perturbed_residual,
                stats,
            )?;
            for row in 0..dimension {
                matrix[row * dimension + column] =
                    (perturbed_residual[row] - residual[row]) / delta;
            }
            trial_acceleration[column] = original;
        }
        for (correction, residual) in correction.iter_mut().zip(residual.iter()) {
            *correction = -*residual;
        }
        factorize(matrix, pivots, dimension)?;
        stats.linear_factorizations += 1;
        solve_factorized(matrix, pivots, correction, dimension);
        stats.linear_solves += 1;
        for (value, correction) in trial_acceleration.iter_mut().zip(correction.iter()) {
            *value += correction;
        }
    }
    Err(SolveError::NonlinearSolveFailed.into())
}

#[allow(clippy::too_many_arguments)]
fn structural_residual<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    method: StructuralParameters,
    velocity: &[f64],
    position: &[f64],
    acceleration: &[f64],
    time: f64,
    step: f64,
    trial_acceleration: &[f64],
    trial_velocity: &mut [f64],
    trial_position: &mut [f64],
    evaluated_acceleration: &mut [f64],
    residual: &mut [f64],
    stats: &mut SolverStats,
) -> Result<(), SecondOrderSolveError>
where
    F: Fn(&mut [f64], &[f64], &[f64], &P, f64),
{
    update_structural_state(
        method,
        velocity,
        position,
        acceleration,
        trial_acceleration,
        step,
        trial_velocity,
        trial_position,
    );
    for index in 0..trial_acceleration.len() {
        trial_velocity[index] =
            (1.0 - method.alpha_f) * trial_velocity[index] + method.alpha_f * velocity[index];
        trial_position[index] =
            (1.0 - method.alpha_f) * trial_position[index] + method.alpha_f * position[index];
    }
    evaluate_acceleration(
        problem,
        evaluated_acceleration,
        trial_velocity,
        trial_position,
        time + (1.0 - method.alpha_f) * step,
        stats,
    )?;
    for index in 0..residual.len() {
        residual[index] = (1.0 - method.alpha_m) * trial_acceleration[index]
            + method.alpha_m * acceleration[index]
            - evaluated_acceleration[index];
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn update_structural_state(
    method: StructuralParameters,
    velocity: &[f64],
    position: &[f64],
    acceleration: &[f64],
    next_acceleration: &[f64],
    step: f64,
    output_velocity: &mut [f64],
    output_position: &mut [f64],
) {
    for index in 0..velocity.len() {
        output_velocity[index] = velocity[index]
            + step
                * ((1.0 - method.gamma) * acceleration[index]
                    + method.gamma * next_acceleration[index]);
        output_position[index] = position[index]
            + step * velocity[index]
            + 0.5
                * step
                * step
                * ((1.0 - 2.0 * method.beta) * acceleration[index]
                    + 2.0 * method.beta * next_acceleration[index]);
    }
}

#[allow(clippy::too_many_arguments)]
fn structural_error_norm(
    velocity: &[f64],
    position: &[f64],
    coarse_velocity: &[f64],
    coarse_position: &[f64],
    previous_velocity: &[f64],
    previous_position: &[f64],
    options: &SolveOptions,
) -> f64 {
    velocity
        .iter()
        .zip(position)
        .zip(coarse_velocity.iter().zip(coarse_position))
        .zip(previous_velocity.iter().zip(previous_position))
        .fold(
            0.0_f64,
            |maximum,
             (
                ((velocity, position), (coarse_velocity, coarse_position)),
                (previous_velocity, previous_position),
            )| {
                let velocity_scale = options.absolute_tolerance
                    + options.relative_tolerance * velocity.abs().max(previous_velocity.abs());
                let position_scale = options.absolute_tolerance
                    + options.relative_tolerance * position.abs().max(previous_position.abs());
                maximum
                    .max(((velocity - coarse_velocity) / (3.0 * velocity_scale)).abs())
                    .max(((position - coarse_position) / (3.0 * position_scale)).abs())
            },
        )
}

fn solve_rkn_fixed<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    options: &SolveOptions,
    tableau: &RknTableau,
) -> Result<SecondOrderSolution, SecondOrderSolveError>
where
    F: Fn(&mut [f64], &[f64], &[f64], &P, f64),
{
    if options.adaptive {
        return Err(SolveError::AdaptiveStepUnsupported.into());
    }
    let fixed_step = options
        .initial_step
        .ok_or(SolveError::InitialStepRequired)?;
    let (start, end) = problem.time_span;
    let direction = (end - start).signum();
    let maximum_step = options.max_step.min((end - start).abs());
    let mut step_magnitude = fixed_step.min(maximum_step);
    let dimension = problem.initial_position.len();
    let stages = tableau.nodes.len();
    debug_assert_eq!(tableau.position_coefficients.len(), stages);
    debug_assert_eq!(tableau.position_weights.len(), stages);
    debug_assert_eq!(tableau.velocity_weights.len(), stages);
    debug_assert!(
        tableau
            .velocity_coefficients
            .is_none_or(|rows| rows.len() == stages)
    );

    let mut velocity = problem.initial_velocity.clone();
    let mut position = problem.initial_position.clone();
    let mut workspace = RknWorkspace::new(dimension, stages, !problem.callbacks.is_empty());
    let mut stats = SolverStats::default();

    let mut recorder = PartitionedRecorder::new(&velocity, &position, start, options);
    let initial = apply_initial_callbacks(problem, &mut velocity, &mut position, start)?;
    stats.callback_invocations += initial.invocations;
    stats.rhs_evaluations += initial.rhs_evaluations;
    if initial.state_modified {
        recorder.record_callback(
            start,
            &problem.initial_velocity,
            &problem.initial_position,
            &velocity,
            &position,
            initial,
            true,
        );
    }
    if problem
        .domain_rejection_factor(&velocity, &position, start)
        .is_some()
    {
        return Err(SolveError::InitialStateOutOfDomain.into());
    }
    if initial.terminate {
        return finish_successful(
            problem,
            &mut velocity,
            &mut position,
            start,
            recorder,
            stats,
        );
    }
    step_magnitude =
        callback_adjusted_step(initial, direction * step_magnitude, direction, maximum_step).abs();

    let mut time = start;
    let mut steps = 0;
    let mut time_stops = TimeStopSchedule::new(&options.time_stops, start, end);
    while direction * (end - time) > 0.0 {
        if steps == options.max_steps {
            return Err(SolveError::MaxStepsExceeded.into());
        }
        steps += 1;
        let step = time_stops.clip_step_with(
            time,
            direction * step_magnitude,
            problem.next_preset_time(time, direction),
        );
        if time + step == time {
            return Err(SolveError::StepSizeUnderflow.into());
        }

        for stage in 0..stages {
            let node = tableau.nodes[stage];
            workspace.stage_position.copy_from_slice(&position);
            workspace.stage_velocity.copy_from_slice(&velocity);
            for (stage_position, velocity) in workspace.stage_position.iter_mut().zip(&velocity) {
                *stage_position += step * node * velocity;
            }
            for previous_stage in 0..stage {
                let acceleration = workspace.stage_accelerations
                    [previous_stage * dimension..(previous_stage + 1) * dimension]
                    .iter();
                let position_coefficient = tableau.position_coefficients[stage][previous_stage];
                for (value, acceleration) in workspace.stage_position.iter_mut().zip(acceleration) {
                    *value += step * step * position_coefficient * acceleration;
                }
                if let Some(velocity_coefficients) = tableau.velocity_coefficients {
                    let acceleration = &workspace.stage_accelerations
                        [previous_stage * dimension..(previous_stage + 1) * dimension];
                    let coefficient = velocity_coefficients[stage][previous_stage];
                    for (value, acceleration) in
                        workspace.stage_velocity.iter_mut().zip(acceleration)
                    {
                        *value += step * coefficient * acceleration;
                    }
                }
            }
            let stage_velocity = if tableau.velocity_coefficients.is_some() {
                &workspace.stage_velocity
            } else {
                &velocity
            };
            let acceleration =
                &mut workspace.stage_accelerations[stage * dimension..(stage + 1) * dimension];
            evaluate_acceleration(
                problem,
                acceleration,
                stage_velocity,
                &workspace.stage_position,
                time + node * step,
                &mut stats,
            )?;
        }

        workspace.candidate_position.copy_from_slice(&position);
        workspace.candidate_velocity.copy_from_slice(&velocity);
        for (candidate_position, velocity) in workspace.candidate_position.iter_mut().zip(&velocity)
        {
            *candidate_position += step * velocity;
        }
        for stage in 0..stages {
            let acceleration =
                &workspace.stage_accelerations[stage * dimension..(stage + 1) * dimension];
            for ((candidate_position, candidate_velocity), acceleration) in workspace
                .candidate_position
                .iter_mut()
                .zip(&mut workspace.candidate_velocity)
                .zip(acceleration)
            {
                *candidate_position += step * step * tableau.position_weights[stage] * acceleration;
                *candidate_velocity += step * tableau.velocity_weights[stage] * acceleration;
            }
        }
        ensure_finite_state(&workspace.candidate_velocity, &workspace.candidate_position)?;

        let previous_time = time;
        let mut next_time = time + step;
        if direction * (end - next_time) <= 0.0 {
            next_time = end;
        }
        if let Some(reduction_factor) = problem.domain_rejection_factor(
            &workspace.candidate_velocity,
            &workspace.candidate_position,
            next_time,
        ) {
            stats.rejected_steps += 1;
            step_magnitude = step.abs() * reduction_factor;
            continue;
        }
        let callback = apply_step_callbacks(
            problem,
            &velocity,
            &position,
            previous_time,
            &mut workspace.candidate_velocity,
            &mut workspace.candidate_position,
            &mut next_time,
            &mut workspace.previous_effect_velocity,
            &mut workspace.previous_effect_position,
            options.event_tolerance,
            None,
        )?;
        stats.callback_invocations += callback.invocations;
        stats.rhs_evaluations += callback.rhs_evaluations;
        time = next_time;
        time_stops.accepted(time);
        std::mem::swap(&mut velocity, &mut workspace.candidate_velocity);
        std::mem::swap(&mut position, &mut workspace.candidate_position);
        stats.accepted_steps += 1;

        recorder.record_step(
            &workspace.candidate_velocity,
            &workspace.candidate_position,
            previous_time,
            if callback.invocations == 0 {
                &velocity
            } else {
                &workspace.previous_effect_velocity
            },
            if callback.invocations == 0 {
                &position
            } else {
                &workspace.previous_effect_position
            },
            time,
            time == end,
            None,
        )?;
        if callback.invocations > 0 {
            recorder.record_callback(
                time,
                &workspace.previous_effect_velocity,
                &workspace.previous_effect_position,
                &velocity,
                &position,
                callback,
                time == end,
            );
        }
        if callback.terminate {
            return finish_successful(problem, &mut velocity, &mut position, time, recorder, stats);
        }
        step_magnitude = callback_adjusted_step(
            callback,
            direction * step_magnitude,
            direction,
            maximum_step,
        )
        .abs();
    }
    finish_successful(problem, &mut velocity, &mut position, time, recorder, stats)
}

fn solve_rkn_adaptive<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    options: &SolveOptions,
    tableau: &AdaptiveRknTableau,
) -> Result<SecondOrderSolution, SecondOrderSolveError>
where
    F: Fn(&mut [f64], &[f64], &[f64], &P, f64),
{
    if !options.adaptive && options.initial_step.is_none() {
        return Err(SolveError::InitialStepRequired.into());
    }
    let dimension = problem.initial_position.len();
    let stages = tableau.nodes.len();
    debug_assert_eq!(tableau.position_coefficients.len(), stages * stages);
    debug_assert!(
        tableau
            .velocity_coefficients
            .is_none_or(|coefficients| coefficients.len() == stages * stages)
    );
    debug_assert_eq!(tableau.position_weights.len(), stages);
    debug_assert_eq!(tableau.velocity_weights.len(), stages);
    debug_assert_eq!(tableau.position_error_weights.len(), stages);
    debug_assert!(tableau.position_only_error || tableau.velocity_error_weights.len() == stages);

    let (start, end) = problem.time_span;
    let direction = (end - start).signum();
    let span = (end - start).abs();
    let maximum_step = options.max_step.min(span);
    let mut step_magnitude = options
        .initial_step
        .unwrap_or(span / 100.0)
        .min(maximum_step);
    if !step_magnitude.is_finite() || step_magnitude <= 0.0 {
        return Err(SolveError::StepSizeUnderflow.into());
    }

    let mut velocity = problem.initial_velocity.clone();
    let mut position = problem.initial_position.clone();
    let mut workspace = RknWorkspace::new(dimension, stages, !problem.callbacks.is_empty());
    let mut stats = SolverStats::default();
    let controller = ControllerConfig::proportional(tableau.order, 0.9, 0.2, 10.0, 0.2);
    let mut controller_state = ControllerState::default();
    let mut previous_attempt_rejected = false;

    let mut recorder = PartitionedRecorder::new(&velocity, &position, start, options);
    let initial = apply_initial_callbacks(problem, &mut velocity, &mut position, start)?;
    stats.callback_invocations += initial.invocations;
    stats.rhs_evaluations += initial.rhs_evaluations;
    if initial.state_modified {
        recorder.record_callback(
            start,
            &problem.initial_velocity,
            &problem.initial_position,
            &velocity,
            &position,
            initial,
            true,
        );
    }
    if problem
        .domain_rejection_factor(&velocity, &position, start)
        .is_some()
    {
        return Err(SolveError::InitialStateOutOfDomain.into());
    }
    if initial.terminate {
        return finish_successful(
            problem,
            &mut velocity,
            &mut position,
            start,
            recorder,
            stats,
        );
    }
    step_magnitude =
        callback_adjusted_step(initial, direction * step_magnitude, direction, maximum_step).abs();

    let mut time = start;
    let mut attempts = 0;
    let mut time_stops = TimeStopSchedule::new(&options.time_stops, start, end);
    while direction * (end - time) > 0.0 {
        if attempts == options.max_steps {
            return Err(SolveError::MaxStepsExceeded.into());
        }
        attempts += 1;
        let step = time_stops.clip_step_with(
            time,
            direction * step_magnitude,
            problem.next_preset_time(time, direction),
        );
        if time + step == time {
            return Err(SolveError::StepSizeUnderflow.into());
        }

        for stage in 0..stages {
            workspace.stage_position.copy_from_slice(&position);
            workspace.stage_velocity.copy_from_slice(&velocity);
            for (stage_position, velocity) in workspace.stage_position.iter_mut().zip(&velocity) {
                *stage_position += step * tableau.nodes[stage] * velocity;
            }
            for previous_stage in 0..stage {
                let coefficient = tableau.position_coefficients[stage * stages + previous_stage];
                let acceleration = &workspace.stage_accelerations
                    [previous_stage * dimension..(previous_stage + 1) * dimension];
                for (stage_position, acceleration) in
                    workspace.stage_position.iter_mut().zip(acceleration)
                {
                    *stage_position += step * step * coefficient * acceleration;
                }
                if let Some(velocity_coefficients) = tableau.velocity_coefficients {
                    let coefficient = velocity_coefficients[stage * stages + previous_stage];
                    for (stage_velocity, acceleration) in
                        workspace.stage_velocity.iter_mut().zip(acceleration)
                    {
                        *stage_velocity += step * coefficient * acceleration;
                    }
                }
            }
            let acceleration =
                &mut workspace.stage_accelerations[stage * dimension..(stage + 1) * dimension];
            evaluate_acceleration(
                problem,
                acceleration,
                if tableau.velocity_coefficients.is_some() {
                    &workspace.stage_velocity
                } else {
                    &velocity
                },
                &workspace.stage_position,
                time + tableau.nodes[stage] * step,
                &mut stats,
            )?;
        }

        workspace.candidate_position.copy_from_slice(&position);
        workspace.candidate_velocity.copy_from_slice(&velocity);
        for (candidate_position, velocity) in workspace.candidate_position.iter_mut().zip(&velocity)
        {
            *candidate_position += step * velocity;
        }
        for stage in 0..stages {
            let acceleration =
                &workspace.stage_accelerations[stage * dimension..(stage + 1) * dimension];
            for ((candidate_position, candidate_velocity), acceleration) in workspace
                .candidate_position
                .iter_mut()
                .zip(&mut workspace.candidate_velocity)
                .zip(acceleration)
            {
                *candidate_position += step * step * tableau.position_weights[stage] * acceleration;
                *candidate_velocity += step * tableau.velocity_weights[stage] * acceleration;
            }
        }
        ensure_finite_state(&workspace.candidate_velocity, &workspace.candidate_position)?;

        let error = if options.adaptive {
            rkn_error_norm(
                &velocity,
                &position,
                &workspace.candidate_velocity,
                &workspace.candidate_position,
                &workspace.stage_accelerations,
                step,
                tableau,
                options,
            )
        } else {
            0.0
        };

        if error <= 1.0 {
            let previous_time = time;
            let attempted_time = time + step;
            let mut next_time = if direction * (end - attempted_time) <= 0.0 {
                end
            } else {
                attempted_time
            };
            if let Some(reduction_factor) = problem.domain_rejection_factor(
                &workspace.candidate_velocity,
                &workspace.candidate_position,
                next_time,
            ) {
                stats.rejected_steps += 1;
                controller_state.reset();
                step_magnitude = step.abs() * reduction_factor;
                previous_attempt_rejected = true;
                continue;
            }
            let callback = if tableau.dense_position_coefficients.is_some() {
                let stage_accelerations = &workspace.stage_accelerations;
                let mut interpolate =
                    |fraction: f64, output_velocity: &mut [f64], output_position: &mut [f64]| {
                        interpolate_dprkn6(
                            tableau,
                            &velocity,
                            &position,
                            stage_accelerations,
                            step,
                            fraction,
                            output_velocity,
                            output_position,
                        )
                    };
                apply_step_callbacks(
                    problem,
                    &velocity,
                    &position,
                    previous_time,
                    &mut workspace.candidate_velocity,
                    &mut workspace.candidate_position,
                    &mut next_time,
                    &mut workspace.previous_effect_velocity,
                    &mut workspace.previous_effect_position,
                    options.event_tolerance,
                    Some(&mut interpolate),
                )?
            } else {
                apply_step_callbacks(
                    problem,
                    &velocity,
                    &position,
                    previous_time,
                    &mut workspace.candidate_velocity,
                    &mut workspace.candidate_position,
                    &mut next_time,
                    &mut workspace.previous_effect_velocity,
                    &mut workspace.previous_effect_position,
                    options.event_tolerance,
                    None,
                )?
            };
            stats.callback_invocations += callback.invocations;
            stats.rhs_evaluations += callback.rhs_evaluations;
            time = next_time;
            time_stops.accepted(time);
            std::mem::swap(&mut velocity, &mut workspace.candidate_velocity);
            std::mem::swap(&mut position, &mut workspace.candidate_position);
            stats.accepted_steps += 1;

            let recorded_velocity = if callback.invocations == 0 {
                &velocity
            } else {
                &workspace.previous_effect_velocity
            };
            let recorded_position = if callback.invocations == 0 {
                &position
            } else {
                &workspace.previous_effect_position
            };
            if tableau.dense_position_coefficients.is_some() {
                let previous_velocity = &workspace.candidate_velocity;
                let previous_position = &workspace.candidate_position;
                let stage_accelerations = &workspace.stage_accelerations;
                let mut interpolate =
                    |target: f64, output_velocity: &mut [f64], output_position: &mut [f64]| {
                        let fraction = (target - previous_time) / step;
                        interpolate_dprkn6(
                            tableau,
                            previous_velocity,
                            previous_position,
                            stage_accelerations,
                            step,
                            fraction,
                            output_velocity,
                            output_position,
                        )
                    };
                recorder.record_step(
                    previous_velocity,
                    previous_position,
                    previous_time,
                    recorded_velocity,
                    recorded_position,
                    time,
                    time == end,
                    Some(&mut interpolate),
                )?;
            } else {
                recorder.record_step(
                    &workspace.candidate_velocity,
                    &workspace.candidate_position,
                    previous_time,
                    recorded_velocity,
                    recorded_position,
                    time,
                    time == end,
                    None,
                )?;
            }
            if callback.invocations > 0 {
                recorder.record_callback(
                    time,
                    &workspace.previous_effect_velocity,
                    &workspace.previous_effect_position,
                    &velocity,
                    &position,
                    callback,
                    time == end,
                );
            }
            if callback.state_modified {
                controller_state.reset();
            }
            if callback.terminate {
                return finish_successful(
                    problem,
                    &mut velocity,
                    &mut position,
                    time,
                    recorder,
                    stats,
                );
            }

            if callback.requested_step.is_some() {
                step_magnitude = callback_adjusted_step(
                    callback,
                    direction * step_magnitude,
                    direction,
                    maximum_step,
                )
                .abs();
                previous_attempt_rejected = false;
            } else if options.adaptive {
                let factor = controller_state.factor(error, controller);
                controller_state.accepted(error);
                let factor = if previous_attempt_rejected {
                    factor.min(1.0)
                } else {
                    factor
                };
                step_magnitude = callback_adjusted_step(
                    callback,
                    direction * step.abs() * factor,
                    direction,
                    maximum_step,
                )
                .abs();
                previous_attempt_rejected = false;
            } else if callback.step_limit.is_some() {
                step_magnitude = callback_adjusted_step(
                    callback,
                    direction * step_magnitude,
                    direction,
                    maximum_step,
                )
                .abs();
            }
        } else {
            stats.rejected_steps += 1;
            controller_state.rejected(error);
            let factor = controller_state.factor(error, controller).min(1.0);
            step_magnitude = (step.abs() * factor).min(maximum_step);
            previous_attempt_rejected = true;
            if time + direction * step_magnitude == time {
                return Err(SolveError::StepSizeUnderflow.into());
            }
        }
    }
    finish_successful(problem, &mut velocity, &mut position, time, recorder, stats)
}

#[allow(clippy::too_many_arguments)]
fn rkn_error_norm(
    velocity: &[f64],
    position: &[f64],
    candidate_velocity: &[f64],
    candidate_position: &[f64],
    stage_accelerations: &[f64],
    step: f64,
    tableau: &AdaptiveRknTableau,
    options: &SolveOptions,
) -> f64 {
    let dimension = position.len();
    let stages = tableau.nodes.len();
    let mut sum = 0.0;
    for component in 0..dimension {
        let mut position_error = 0.0;
        let mut velocity_error = 0.0;
        for stage in 0..stages {
            let acceleration = stage_accelerations[stage * dimension + component];
            position_error += tableau.position_error_weights[stage] * acceleration;
            if !tableau.position_only_error {
                velocity_error += tableau.velocity_error_weights[stage] * acceleration;
            }
        }
        position_error *= step * step;
        let position_scale = options.absolute_tolerance
            + options.relative_tolerance
                * position[component]
                    .abs()
                    .max(candidate_position[component].abs());
        sum += (position_error / position_scale).powi(2);
        if !tableau.position_only_error {
            velocity_error *= step;
            let velocity_scale = options.absolute_tolerance
                + options.relative_tolerance
                    * velocity[component]
                        .abs()
                        .max(candidate_velocity[component].abs());
            sum += (velocity_error / velocity_scale).powi(2);
        }
    }
    let components = if tableau.position_only_error {
        dimension
    } else {
        2 * dimension
    };
    (sum / components as f64).sqrt()
}

#[allow(clippy::too_many_arguments)]
fn interpolate_dprkn6(
    tableau: &AdaptiveRknTableau,
    previous_velocity: &[f64],
    previous_position: &[f64],
    stage_accelerations: &[f64],
    step: f64,
    fraction: f64,
    output_velocity: &mut [f64],
    output_position: &mut [f64],
) -> Result<(), SolveError> {
    let position_coefficients = tableau
        .dense_position_coefficients
        .ok_or(SolveError::InvalidTableau)?;
    let velocity_coefficients = tableau
        .dense_velocity_coefficients
        .ok_or(SolveError::InvalidTableau)?;
    let dimension = previous_position.len();
    let stages = tableau.nodes.len();
    debug_assert_eq!(position_coefficients.len(), stages * 5);
    debug_assert_eq!(velocity_coefficients.len(), stages * 5);
    for component in 0..dimension {
        let mut position_sum = 0.0;
        let mut velocity_sum = 0.0;
        for stage in 0..stages {
            let position_row = &position_coefficients[stage * 5..stage * 5 + 5];
            let velocity_row = &velocity_coefficients[stage * 5..stage * 5 + 5];
            let position_weight = position_row
                .iter()
                .rev()
                .fold(0.0, |value, coefficient| value * fraction + coefficient);
            let velocity_weight = velocity_row
                .iter()
                .rev()
                .fold(0.0, |value, coefficient| value * fraction + coefficient);
            let acceleration = stage_accelerations[stage * dimension + component];
            position_sum += position_weight * acceleration;
            velocity_sum += velocity_weight * acceleration;
        }
        output_velocity[component] = previous_velocity[component] + step * fraction * velocity_sum;
        output_position[component] = previous_position[component]
            + step * fraction * (previous_velocity[component] + step * fraction * position_sum);
    }
    ensure_finite_state(output_velocity, output_position)
}

#[derive(Clone, Copy)]
enum IrknMethod {
    ThirdOrder,
    FourthOrder,
}

struct IrknWorkspace {
    candidate_velocity: Vec<f64>,
    candidate_position: Vec<f64>,
    k2: Vec<f64>,
    k3: Vec<f64>,
    next_acceleration: Vec<f64>,
    old_acceleration: Vec<f64>,
    old_internal_first: Vec<f64>,
    old_internal_second: Vec<f64>,
    internal_first: Vec<f64>,
    internal_second: Vec<f64>,
    previous_velocity: Vec<f64>,
    previous_effect_velocity: Vec<f64>,
    previous_effect_position: Vec<f64>,
}

impl IrknWorkspace {
    fn new(dimension: usize, callbacks: bool) -> Self {
        Self {
            candidate_velocity: vec![0.0; dimension],
            candidate_position: vec![0.0; dimension],
            k2: vec![0.0; dimension],
            k3: vec![0.0; dimension],
            next_acceleration: vec![0.0; dimension],
            old_acceleration: vec![0.0; dimension],
            old_internal_first: vec![0.0; dimension],
            old_internal_second: vec![0.0; dimension],
            internal_first: vec![0.0; dimension],
            internal_second: vec![0.0; dimension],
            previous_velocity: vec![0.0; dimension],
            previous_effect_velocity: if callbacks {
                vec![0.0; dimension]
            } else {
                Vec::new()
            },
            previous_effect_position: if callbacks {
                vec![0.0; dimension]
            } else {
                Vec::new()
            },
        }
    }
}

fn solve_irkn<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    options: &SolveOptions,
    method: IrknMethod,
) -> Result<SecondOrderSolution, SecondOrderSolveError>
where
    F: Fn(&mut [f64], &[f64], &[f64], &P, f64),
{
    match method {
        IrknMethod::ThirdOrder => {
            debug_assert_eq!(IRKN3_ORDER, 3);
            debug_assert_eq!(IRKN3_BOOTSTRAP_ORDER, 4);
            debug_assert_eq!(IRKN3_INTERNAL_STAGES, 1);
            debug_assert_eq!(IRKN3_RETAINED_ENDPOINT_ACCELERATIONS, 2);
            debug_assert_eq!(IRKN3_RETAINED_INTERNAL_STAGES, 1);
        }
        IrknMethod::FourthOrder => {
            debug_assert_eq!(IRKN4_ORDER, 4);
            debug_assert_eq!(IRKN4_BOOTSTRAP_ORDER, 4);
            debug_assert_eq!(IRKN4_INTERNAL_STAGES, 2);
            debug_assert_eq!(IRKN4_RETAINED_ENDPOINT_ACCELERATIONS, 2);
            debug_assert_eq!(IRKN4_RETAINED_INTERNAL_STAGES, 2);
        }
    }
    if options.adaptive {
        return Err(SolveError::AdaptiveStepUnsupported.into());
    }
    let fixed_step = options
        .initial_step
        .ok_or(SolveError::InitialStepRequired)?;
    let (start, end) = problem.time_span;
    let direction = (end - start).signum();
    let maximum_step = options.max_step.min((end - start).abs());
    let mut step_magnitude = fixed_step.min(maximum_step);
    let dimension = problem.initial_position.len();
    let mut velocity = problem.initial_velocity.clone();
    let mut position = problem.initial_position.clone();
    let mut acceleration = vec![0.0; dimension];
    let mut workspace = IrknWorkspace::new(dimension, !problem.callbacks.is_empty());
    let mut stats = SolverStats::default();

    let mut recorder = PartitionedRecorder::new(&velocity, &position, start, options);
    let initial = apply_initial_callbacks(problem, &mut velocity, &mut position, start)?;
    stats.callback_invocations += initial.invocations;
    stats.rhs_evaluations += initial.rhs_evaluations;
    if initial.state_modified {
        recorder.record_callback(
            start,
            &problem.initial_velocity,
            &problem.initial_position,
            &velocity,
            &position,
            initial,
            true,
        );
    }
    if problem
        .domain_rejection_factor(&velocity, &position, start)
        .is_some()
    {
        return Err(SolveError::InitialStateOutOfDomain.into());
    }
    if initial.terminate {
        return finish_successful(
            problem,
            &mut velocity,
            &mut position,
            start,
            recorder,
            stats,
        );
    }
    step_magnitude =
        callback_adjusted_step(initial, direction * step_magnitude, direction, maximum_step).abs();
    evaluate_acceleration(
        problem,
        &mut acceleration,
        &velocity,
        &position,
        start,
        &mut stats,
    )?;

    let mut time = start;
    let mut attempts = 0;
    let mut time_stops = TimeStopSchedule::new(&options.time_stops, start, end);
    let mut history_valid = false;
    while direction * (end - time) > 0.0 {
        if attempts == options.max_steps {
            return Err(SolveError::MaxStepsExceeded.into());
        }
        attempts += 1;
        let step = time_stops.clip_step_with(
            time,
            direction * step_magnitude,
            problem.next_preset_time(time, direction),
        );
        if time + step == time {
            return Err(SolveError::StepSizeUnderflow.into());
        }
        let constant_step = step.abs() == step_magnitude;
        let bootstrap = !history_valid || !constant_step;

        if bootstrap {
            // Exact pinned Nyström4VelocityIndependent startup.
            for component in 0..dimension {
                workspace.candidate_position[component] = position[component]
                    + 0.5 * step * velocity[component]
                    + step * step * acceleration[component] / 8.0;
            }
            evaluate_acceleration(
                problem,
                &mut workspace.k2,
                &velocity,
                &workspace.candidate_position,
                time + 0.5 * step,
                &mut stats,
            )?;
            for component in 0..dimension {
                workspace.candidate_position[component] = position[component]
                    + step * velocity[component]
                    + 0.5 * step * step * workspace.k2[component];
            }
            evaluate_acceleration(
                problem,
                &mut workspace.k3,
                &velocity,
                &workspace.candidate_position,
                time + step,
                &mut stats,
            )?;
            for component in 0..dimension {
                workspace.candidate_position[component] = position[component]
                    + step * velocity[component]
                    + step * step * (acceleration[component] + 2.0 * workspace.k2[component]) / 6.0;
                workspace.candidate_velocity[component] = velocity[component]
                    + step
                        * (acceleration[component]
                            + 4.0 * workspace.k2[component]
                            + workspace.k3[component])
                        / 6.0;
            }
            evaluate_acceleration(
                problem,
                &mut workspace.next_acceleration,
                &workspace.candidate_velocity,
                &workspace.candidate_position,
                time + step,
                &mut stats,
            )?;

            let (c1, a21) = match method {
                IrknMethod::ThirdOrder => (IRKN3_C[0], IRKN3_A[0]),
                IrknMethod::FourthOrder => (IRKN4_C[0], IRKN4_A[0]),
            };
            // Preserve the pinned in-place cache seeds, including their time arguments.
            evaluate_acceleration(
                problem,
                &mut workspace.old_acceleration,
                &velocity,
                &position,
                time + c1 * step,
                &mut stats,
            )?;
            for component in 0..dimension {
                let seed_acceleration = match method {
                    // The pinned in-place IRKN3 cache seeds G0 from H0, while
                    // IRKN4 seeds G0 from the newly bootstrapped endpoint A1.
                    IrknMethod::ThirdOrder => workspace.old_acceleration[component],
                    IrknMethod::FourthOrder => workspace.next_acceleration[component],
                };
                workspace.k2[component] = position[component]
                    + step * (c1 * velocity[component] + step * a21 * seed_acceleration);
            }
            evaluate_acceleration(
                problem,
                &mut workspace.old_internal_first,
                &velocity,
                &workspace.k2,
                time + c1 * step,
                &mut stats,
            )?;
            if matches!(method, IrknMethod::FourthOrder) {
                for component in 0..dimension {
                    workspace.k2[component] = position[component]
                        + step
                            * (IRKN4_C[1] * velocity[component]
                                + step * IRKN4_A[1] * workspace.old_acceleration[component]);
                }
                evaluate_acceleration(
                    problem,
                    &mut workspace.old_internal_second,
                    &velocity,
                    &workspace.k2,
                    time + IRKN4_C[0] * step,
                    &mut stats,
                )?;
            }
        } else {
            match method {
                IrknMethod::ThirdOrder => {
                    for component in 0..dimension {
                        workspace.k2[component] = position[component]
                            + step
                                * (IRKN3_C[0] * velocity[component]
                                    + step * IRKN3_A[0] * workspace.old_acceleration[component]);
                    }
                    evaluate_acceleration(
                        problem,
                        &mut workspace.internal_first,
                        &velocity,
                        &workspace.k2,
                        time + IRKN3_C[0] * step,
                        &mut stats,
                    )?;
                    for component in 0..dimension {
                        let difference = workspace.internal_first[component]
                            - workspace.old_internal_first[component];
                        workspace.candidate_velocity[component] = velocity[component]
                            + step
                                * (IRKN3_VELOCITY_WEIGHTS[0] * acceleration[component]
                                    + IRKN3_HISTORY_WEIGHTS[0]
                                        * workspace.old_acceleration[component]
                                    + IRKN3_VELOCITY_WEIGHTS[1] * difference);
                        workspace.candidate_position[component] = position[component]
                            + step
                                * (IRKN3_VELOCITY_HISTORY[0] * velocity[component]
                                    + IRKN3_VELOCITY_HISTORY[1]
                                        * workspace.previous_velocity[component])
                            + step * step * IRKN3_HISTORY_WEIGHTS[1] * difference;
                    }
                }
                IrknMethod::FourthOrder => {
                    for component in 0..dimension {
                        workspace.k2[component] = position[component]
                            + step
                                * (IRKN4_C[0] * velocity[component]
                                    + step * IRKN4_A[0] * acceleration[component]);
                    }
                    evaluate_acceleration(
                        problem,
                        &mut workspace.internal_first,
                        &velocity,
                        &workspace.k2,
                        time + IRKN4_C[0] * step,
                        &mut stats,
                    )?;
                    for component in 0..dimension {
                        workspace.k2[component] = position[component]
                            + step
                                * (IRKN4_C[1] * velocity[component]
                                    + step * IRKN4_A[1] * workspace.internal_first[component]);
                    }
                    evaluate_acceleration(
                        problem,
                        &mut workspace.internal_second,
                        &velocity,
                        &workspace.k2,
                        time + IRKN4_C[1] * step,
                        &mut stats,
                    )?;
                    for component in 0..dimension {
                        let first_difference = workspace.internal_first[component]
                            - workspace.old_internal_first[component];
                        let second_difference = workspace.internal_second[component]
                            - workspace.old_internal_second[component];
                        workspace.candidate_velocity[component] = velocity[component]
                            + step
                                * (IRKN4_VELOCITY_WEIGHTS[0] * acceleration[component]
                                    + IRKN4_HISTORY_WEIGHTS[0]
                                        * workspace.old_acceleration[component]
                                    + IRKN4_VELOCITY_WEIGHTS[1] * first_difference
                                    + IRKN4_VELOCITY_WEIGHTS[2] * second_difference);
                        workspace.candidate_position[component] = position[component]
                            + step
                                * (IRKN4_VELOCITY_HISTORY[0] * velocity[component]
                                    + IRKN4_VELOCITY_HISTORY[1]
                                        * workspace.previous_velocity[component])
                            + step
                                * step
                                * (IRKN4_HISTORY_WEIGHTS[1] * first_difference
                                    + IRKN4_HISTORY_WEIGHTS[2] * second_difference);
                    }
                }
            }
            evaluate_acceleration(
                problem,
                &mut workspace.next_acceleration,
                &workspace.candidate_velocity,
                &workspace.candidate_position,
                time + step,
                &mut stats,
            )?;
        }
        ensure_finite_state(&workspace.candidate_velocity, &workspace.candidate_position)?;

        let previous_time = time;
        let mut next_time = if direction * (end - (time + step)) <= 0.0 {
            end
        } else {
            time + step
        };
        if let Some(reduction_factor) = problem.domain_rejection_factor(
            &workspace.candidate_velocity,
            &workspace.candidate_position,
            next_time,
        ) {
            stats.rejected_steps += 1;
            step_magnitude = step.abs() * reduction_factor;
            history_valid = false;
            continue;
        }
        let callback = apply_step_callbacks(
            problem,
            &velocity,
            &position,
            previous_time,
            &mut workspace.candidate_velocity,
            &mut workspace.candidate_position,
            &mut next_time,
            &mut workspace.previous_effect_velocity,
            &mut workspace.previous_effect_position,
            options.event_tolerance,
            None,
        )?;
        stats.callback_invocations += callback.invocations;
        stats.rhs_evaluations += callback.rhs_evaluations;
        time = next_time;
        time_stops.accepted(time);
        std::mem::swap(&mut velocity, &mut workspace.candidate_velocity);
        std::mem::swap(&mut position, &mut workspace.candidate_position);
        stats.accepted_steps += 1;
        recorder.record_step(
            &workspace.candidate_velocity,
            &workspace.candidate_position,
            previous_time,
            if callback.invocations == 0 {
                &velocity
            } else {
                &workspace.previous_effect_velocity
            },
            if callback.invocations == 0 {
                &position
            } else {
                &workspace.previous_effect_position
            },
            time,
            time == end,
            None,
        )?;
        if callback.invocations > 0 {
            recorder.record_callback(
                time,
                &workspace.previous_effect_velocity,
                &workspace.previous_effect_position,
                &velocity,
                &position,
                callback,
                time == end,
            );
        }
        if callback.terminate {
            return finish_successful(problem, &mut velocity, &mut position, time, recorder, stats);
        }

        if callback.state_modified {
            evaluate_acceleration(
                problem,
                &mut acceleration,
                &velocity,
                &position,
                time,
                &mut stats,
            )?;
            history_valid = false;
        } else {
            workspace
                .previous_velocity
                .copy_from_slice(&workspace.candidate_velocity);
            if !bootstrap {
                workspace.old_acceleration.copy_from_slice(&acceleration);
                workspace
                    .old_internal_first
                    .copy_from_slice(&workspace.internal_first);
                if matches!(method, IrknMethod::FourthOrder) {
                    workspace
                        .old_internal_second
                        .copy_from_slice(&workspace.internal_second);
                }
            }
            acceleration.copy_from_slice(&workspace.next_acceleration);
            history_valid = constant_step;
        }
        if callback.requested_step.is_some() || callback.step_limit.is_some() {
            let adjusted = callback_adjusted_step(
                callback,
                direction * step_magnitude,
                direction,
                maximum_step,
            )
            .abs();
            history_valid &= adjusted == step_magnitude;
            step_magnitude = adjusted;
        }
    }
    finish_successful(problem, &mut velocity, &mut position, time, recorder, stats)
}

fn solve_fixed<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    options: &SolveOptions,
    method: Method,
) -> Result<SecondOrderSolution, SecondOrderSolveError>
where
    F: Fn(&mut [f64], &[f64], &[f64], &P, f64),
{
    if options.adaptive {
        return Err(SolveError::AdaptiveStepUnsupported.into());
    }
    let fixed_step = options
        .initial_step
        .ok_or(SolveError::InitialStepRequired)?;
    let (start, end) = problem.time_span;
    let direction = (end - start).signum();
    let maximum_step = options.max_step.min((end - start).abs());
    let mut step_magnitude = fixed_step.min(maximum_step);
    let dimension = problem.initial_position.len();
    let mut velocity = problem.initial_velocity.clone();
    let mut position = problem.initial_position.clone();
    let mut workspace = Workspace::new(dimension, !problem.callbacks.is_empty());
    let mut stats = SolverStats::default();

    let mut recorder = PartitionedRecorder::new(&velocity, &position, start, options);
    let initial = apply_initial_callbacks(problem, &mut velocity, &mut position, start)?;
    stats.callback_invocations += initial.invocations;
    stats.rhs_evaluations += initial.rhs_evaluations;
    if initial.state_modified {
        recorder.record_callback(
            start,
            &problem.initial_velocity,
            &problem.initial_position,
            &velocity,
            &position,
            initial,
            true,
        );
    }
    if problem
        .domain_rejection_factor(&velocity, &position, start)
        .is_some()
    {
        return Err(SolveError::InitialStateOutOfDomain.into());
    }
    if initial.terminate {
        return finish_successful(
            problem,
            &mut velocity,
            &mut position,
            start,
            recorder,
            stats,
        );
    }
    step_magnitude =
        callback_adjusted_step(initial, direction * step_magnitude, direction, maximum_step).abs();

    let caches_acceleration = matches!(method, Method::VelocityVerlet | Method::VerletLeapfrog);
    if caches_acceleration {
        evaluate_acceleration(
            problem,
            &mut workspace.acceleration,
            &velocity,
            &position,
            start,
            &mut stats,
        )?;
    }

    let mut time = start;
    let mut steps = 0;
    let mut time_stops = TimeStopSchedule::new(&options.time_stops, start, end);
    while direction * (end - time) > 0.0 {
        if steps == options.max_steps {
            return Err(SolveError::MaxStepsExceeded.into());
        }
        steps += 1;
        let step = time_stops.clip_step_with(
            time,
            direction * step_magnitude,
            problem.next_preset_time(time, direction),
        );
        if time + step == time {
            return Err(SolveError::StepSizeUnderflow.into());
        }
        perform_step(
            problem,
            method,
            &velocity,
            &position,
            time,
            step,
            &mut workspace,
            &mut stats,
        )?;

        let previous_time = time;
        let mut next_time = time + step;
        if direction * (end - next_time) <= 0.0 {
            next_time = end;
        }
        if let Some(reduction_factor) = problem.domain_rejection_factor(
            &workspace.candidate_velocity,
            &workspace.candidate_position,
            next_time,
        ) {
            stats.rejected_steps += 1;
            step_magnitude = step.abs() * reduction_factor;
            if caches_acceleration {
                evaluate_acceleration(
                    problem,
                    &mut workspace.acceleration,
                    &velocity,
                    &position,
                    time,
                    &mut stats,
                )?;
            }
            continue;
        }
        let callback = apply_step_callbacks(
            problem,
            &velocity,
            &position,
            previous_time,
            &mut workspace.candidate_velocity,
            &mut workspace.candidate_position,
            &mut next_time,
            &mut workspace.previous_effect_velocity,
            &mut workspace.previous_effect_position,
            options.event_tolerance,
            None,
        )?;
        stats.callback_invocations += callback.invocations;
        stats.rhs_evaluations += callback.rhs_evaluations;
        time = next_time;
        time_stops.accepted(time);
        std::mem::swap(&mut velocity, &mut workspace.candidate_velocity);
        std::mem::swap(&mut position, &mut workspace.candidate_position);
        stats.accepted_steps += 1;

        recorder.record_step(
            &workspace.candidate_velocity,
            &workspace.candidate_position,
            previous_time,
            if callback.invocations == 0 {
                &velocity
            } else {
                &workspace.previous_effect_velocity
            },
            if callback.invocations == 0 {
                &position
            } else {
                &workspace.previous_effect_position
            },
            time,
            time == end,
            None,
        )?;
        if callback.invocations > 0 {
            recorder.record_callback(
                time,
                &workspace.previous_effect_velocity,
                &workspace.previous_effect_position,
                &velocity,
                &position,
                callback,
                time == end,
            );
        }
        if callback.terminate {
            return finish_successful(problem, &mut velocity, &mut position, time, recorder, stats);
        }
        step_magnitude = callback_adjusted_step(
            callback,
            direction * step_magnitude,
            direction,
            maximum_step,
        )
        .abs();
        if callback.state_modified && caches_acceleration {
            evaluate_acceleration(
                problem,
                &mut workspace.acceleration,
                &velocity,
                &position,
                time,
                &mut stats,
            )?;
        }
    }
    finish_successful(problem, &mut velocity, &mut position, time, recorder, stats)
}

#[allow(clippy::too_many_arguments)]
fn perform_step<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    method: Method,
    velocity: &[f64],
    position: &[f64],
    time: f64,
    step: f64,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: Fn(&mut [f64], &[f64], &[f64], &P, f64),
{
    match method {
        Method::SymplecticEuler => {
            for ((next_position, position), velocity) in workspace
                .candidate_position
                .iter_mut()
                .zip(position)
                .zip(velocity)
            {
                *next_position = position + step * velocity;
            }
            evaluate_acceleration(
                problem,
                &mut workspace.acceleration,
                velocity,
                &workspace.candidate_position,
                time,
                stats,
            )?;
            for ((next_velocity, velocity), acceleration) in workspace
                .candidate_velocity
                .iter_mut()
                .zip(velocity)
                .zip(&workspace.acceleration)
            {
                *next_velocity = velocity + step * acceleration;
            }
        }
        Method::VelocityVerlet => {
            for (((next_position, position), velocity), acceleration) in workspace
                .candidate_position
                .iter_mut()
                .zip(position)
                .zip(velocity)
                .zip(&workspace.acceleration)
            {
                *next_position = position + step * velocity + 0.5 * step * step * acceleration;
            }
            evaluate_acceleration(
                problem,
                &mut workspace.stage_position,
                velocity,
                &workspace.candidate_position,
                time + step,
                stats,
            )?;
            for (((next_velocity, velocity), old), new) in workspace
                .candidate_velocity
                .iter_mut()
                .zip(velocity)
                .zip(&workspace.acceleration)
                .zip(&workspace.stage_position)
            {
                *next_velocity = velocity + 0.5 * step * (old + new);
            }
            workspace
                .acceleration
                .copy_from_slice(&workspace.stage_position);
        }
        Method::VerletLeapfrog => {
            for (((stage_velocity, velocity), acceleration), next_position) in workspace
                .stage_velocity
                .iter_mut()
                .zip(velocity)
                .zip(&workspace.acceleration)
                .zip(&mut workspace.candidate_position)
            {
                *stage_velocity = velocity + 0.5 * step * acceleration;
                *next_position = 0.0;
            }
            for ((next_position, position), stage_velocity) in workspace
                .candidate_position
                .iter_mut()
                .zip(position)
                .zip(&workspace.stage_velocity)
            {
                *next_position = position + step * stage_velocity;
            }
            evaluate_acceleration(
                problem,
                &mut workspace.stage_position,
                &workspace.stage_velocity,
                &workspace.candidate_position,
                time + step,
                stats,
            )?;
            for ((next_velocity, stage_velocity), acceleration) in workspace
                .candidate_velocity
                .iter_mut()
                .zip(&workspace.stage_velocity)
                .zip(&workspace.stage_position)
            {
                *next_velocity = stage_velocity + 0.5 * step * acceleration;
            }
            workspace
                .acceleration
                .copy_from_slice(&workspace.stage_position);
        }
        Method::LeapfrogDriftKickDrift => {
            for ((stage_position, position), velocity) in workspace
                .stage_position
                .iter_mut()
                .zip(position)
                .zip(velocity)
            {
                *stage_position = position + 0.5 * step * velocity;
            }
            evaluate_acceleration(
                problem,
                &mut workspace.acceleration,
                velocity,
                position,
                time,
                stats,
            )?;
            for ((stage_velocity, velocity), acceleration) in workspace
                .stage_velocity
                .iter_mut()
                .zip(velocity)
                .zip(&workspace.acceleration)
            {
                *stage_velocity = velocity + 0.5 * step * acceleration;
            }
            evaluate_acceleration(
                problem,
                &mut workspace.acceleration,
                &workspace.stage_velocity,
                &workspace.stage_position,
                time + 0.5 * step,
                stats,
            )?;
            for ((next_velocity, velocity), acceleration) in workspace
                .candidate_velocity
                .iter_mut()
                .zip(velocity)
                .zip(&workspace.acceleration)
            {
                *next_velocity = velocity + step * acceleration;
            }
            for ((next_position, stage_position), next_velocity) in workspace
                .candidate_position
                .iter_mut()
                .zip(&workspace.stage_position)
                .zip(&workspace.candidate_velocity)
            {
                *next_position = stage_position + 0.5 * step * next_velocity;
            }
        }
    }
    Ok(())
}

fn evaluate_acceleration<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    output: &mut [f64],
    velocity: &[f64],
    position: &[f64],
    time: f64,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: Fn(&mut [f64], &[f64], &[f64], &P, f64),
{
    (problem.acceleration)(output, velocity, position, &problem.parameters, time);
    stats.rhs_evaluations += 1;
    output
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteDerivative)
}

pub(super) fn apply_initial_callbacks<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    velocity: &mut [f64],
    position: &mut [f64],
    time: f64,
) -> Result<CallbackOutcome, SolveError>
where
    F: Fn(&mut [f64], &[f64], &[f64], &P, f64),
{
    let mut outcome = CallbackOutcome::default();
    for initialization in &problem.initializers {
        (initialization.hook)(velocity, position, &problem.parameters, time);
        ensure_finite_state(velocity, position)?;
        outcome.register_initialization(initialization.save);
    }
    for callback in &problem.callbacks {
        let PartitionedCallback::Discrete(callback) = callback else {
            continue;
        };
        callback
            .trigger
            .initialize(velocity, position, &problem.parameters, time)?;
        if callback.trigger.is_triggered(
            velocity,
            position,
            &problem.parameters,
            time,
            |du, _| {
                let (acceleration, rate) = du.split_at_mut(velocity.len());
                (problem.acceleration)(acceleration, velocity, position, &problem.parameters, time);
                rate.copy_from_slice(velocity);
                outcome.rhs_evaluations += 1;
                Ok(())
            },
        )? {
            outcome.register(callback.save);
            outcome.apply_action((callback.affect)(
                velocity,
                position,
                &problem.parameters,
                time,
            )?)?;
            ensure_finite_state(velocity, position)?;
            if outcome.terminate {
                break;
            }
        }
    }
    Ok(outcome)
}

pub(super) fn apply_finalize_callbacks<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    velocity: &mut [f64],
    position: &mut [f64],
    time: f64,
) -> Result<bool, SolveError> {
    for finalize in &problem.finalizers {
        finalize(velocity, position, &problem.parameters, time);
        ensure_finite_state(velocity, position)?;
    }
    Ok(!problem.finalizers.is_empty())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn apply_step_callbacks<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    previous_velocity: &[f64],
    previous_position: &[f64],
    previous_time: f64,
    velocity: &mut [f64],
    position: &mut [f64],
    time: &mut f64,
    state_before_velocity: &mut [f64],
    state_before_position: &mut [f64],
    event_tolerance: f64,
    mut interpolator: Option<&mut PartitionedInterpolator<'_>>,
) -> Result<CallbackOutcome, SolveError>
where
    F: Fn(&mut [f64], &[f64], &[f64], &P, f64),
{
    if problem.callbacks.is_empty() {
        return Ok(CallbackOutcome::default());
    }
    let mut outcome = CallbackOutcome::default();
    let mut root = None;
    for (index, callback) in problem.callbacks.iter().enumerate() {
        match callback {
            PartitionedCallback::Continuous(callback) => {
                let before = (callback.condition)(
                    previous_velocity,
                    previous_position,
                    &problem.parameters,
                    previous_time,
                );
                let after = (callback.condition)(velocity, position, &problem.parameters, *time);
                if !before.is_finite() || !after.is_finite() {
                    return Err(SolveError::NonFiniteCallbackCondition);
                }
                if callback.direction.accepts(before, after) {
                    let fraction = locate_root(
                        callback,
                        previous_velocity,
                        previous_position,
                        previous_time,
                        velocity,
                        position,
                        *time,
                        before,
                        state_before_velocity,
                        state_before_position,
                        &problem.parameters,
                        event_tolerance,
                        interpolator.as_deref_mut(),
                    )?;
                    if root.is_none_or(|(_, earliest)| fraction < earliest) {
                        root = Some((index, fraction));
                    }
                }
            }
            PartitionedCallback::VectorContinuous(callback) => {
                let mut scratch = callback.scratch.borrow_mut();
                evaluate_partitioned_vector_condition(
                    callback,
                    &mut scratch.before,
                    previous_velocity,
                    previous_position,
                    &problem.parameters,
                    previous_time,
                )?;
                evaluate_partitioned_vector_condition(
                    callback,
                    &mut scratch.after,
                    velocity,
                    position,
                    &problem.parameters,
                    *time,
                )?;
                scratch.root_fractions.fill(f64::INFINITY);
                scratch.crossings.fill(EventCrossing::None);
                for event_index in 0..callback.event_count {
                    let before = scratch.before[event_index];
                    let crossing = EventDirection::Any.crossing(before, scratch.after[event_index]);
                    if crossing == EventCrossing::None {
                        continue;
                    }
                    let fraction = locate_partitioned_vector_root(
                        callback,
                        event_index,
                        previous_velocity,
                        previous_position,
                        previous_time,
                        velocity,
                        position,
                        *time,
                        before,
                        state_before_velocity,
                        state_before_position,
                        &problem.parameters,
                        event_tolerance,
                        interpolator.as_deref_mut(),
                        &mut scratch.middle,
                    )?;
                    scratch.root_fractions[event_index] = fraction;
                    scratch.crossings[event_index] = crossing;
                    if root.is_none_or(|(_, earliest)| fraction < earliest) {
                        root = Some((index, fraction));
                    }
                }
            }
            PartitionedCallback::Discrete(_) => {}
        }
    }
    if let Some((index, fraction)) = root {
        let end_time = *time;
        if let Some(interpolator) = interpolator {
            interpolator(fraction, state_before_velocity, state_before_position)?;
        } else {
            interpolate_partitioned(
                previous_velocity,
                previous_position,
                velocity,
                position,
                end_time - previous_time,
                fraction,
                state_before_velocity,
                state_before_position,
            );
        }
        velocity.copy_from_slice(state_before_velocity);
        position.copy_from_slice(state_before_position);
        *time = previous_time + fraction * (end_time - previous_time);
        match &problem.callbacks[index] {
            PartitionedCallback::Continuous(callback) => {
                outcome.register(callback.save);
                outcome.apply_action((callback.affect)(
                    velocity,
                    position,
                    &problem.parameters,
                    *time,
                ))?;
            }
            PartitionedCallback::VectorContinuous(callback) => {
                let root_time = *time;
                let mut scratch = callback.scratch.borrow_mut();
                for event_index in 0..callback.event_count {
                    let event_time = previous_time
                        + scratch.root_fractions[event_index] * (end_time - previous_time);
                    let tolerance =
                        effective_event_tolerance(event_tolerance, root_time, event_time);
                    let crossing = scratch.crossings[event_index];
                    scratch.simultaneous_events[event_index] =
                        if (event_time - root_time).abs() <= tolerance {
                            crossing
                        } else {
                            EventCrossing::None
                        };
                }
                outcome.register(callback.save);
                outcome.apply_action((callback.affect)(
                    velocity,
                    position,
                    &problem.parameters,
                    *time,
                    &scratch.simultaneous_events,
                ))?;
            }
            PartitionedCallback::Discrete(_) => return Err(SolveError::InvalidCallbackState),
        }
        // The localized root truncates the attempted step even when its
        // effect is observation-only, so endpoint-dependent caches cannot be
        // reused for the next step.
        outcome.state_modified = true;
        ensure_finite_state(velocity, position)?;
    }
    if !outcome.terminate {
        for callback in &problem.callbacks {
            let PartitionedCallback::Discrete(callback) = callback else {
                continue;
            };
            if callback.trigger.is_triggered(
                velocity,
                position,
                &problem.parameters,
                *time,
                |du, _| {
                    let (acceleration, rate) = du.split_at_mut(velocity.len());
                    (problem.acceleration)(
                        acceleration,
                        velocity,
                        position,
                        &problem.parameters,
                        *time,
                    );
                    rate.copy_from_slice(velocity);
                    outcome.rhs_evaluations += 1;
                    Ok(())
                },
            )? {
                if outcome.invocations == 0 {
                    state_before_velocity.copy_from_slice(velocity);
                    state_before_position.copy_from_slice(position);
                }
                outcome.register(callback.save);
                outcome.apply_action((callback.affect)(
                    velocity,
                    position,
                    &problem.parameters,
                    *time,
                )?)?;
                ensure_finite_state(velocity, position)?;
                if outcome.terminate {
                    break;
                }
            }
        }
    }
    Ok(outcome)
}

#[allow(clippy::too_many_arguments)]
fn locate_root<P>(
    callback: &ContinuousCallback<P>,
    previous_velocity: &[f64],
    previous_position: &[f64],
    previous_time: f64,
    velocity: &[f64],
    position: &[f64],
    time: f64,
    before: f64,
    interpolation_velocity: &mut [f64],
    interpolation_position: &mut [f64],
    parameters: &P,
    event_tolerance: f64,
    mut interpolator: Option<&mut PartitionedInterpolator<'_>>,
) -> Result<f64, SolveError> {
    let mut left = 0.0;
    let mut right = 1.0;
    let mut left_value = before;
    for _ in 0..MAX_EVENT_ROOT_ITERATIONS {
        let middle = 0.5 * (left + right);
        if middle == left || middle == right {
            break;
        }
        if let Some(interpolator) = interpolator.as_deref_mut() {
            interpolator(middle, interpolation_velocity, interpolation_position)?;
        } else {
            interpolate_partitioned(
                previous_velocity,
                previous_position,
                velocity,
                position,
                time - previous_time,
                middle,
                interpolation_velocity,
                interpolation_position,
            );
        }
        let middle_time = previous_time + middle * (time - previous_time);
        let value = (callback.condition)(
            interpolation_velocity,
            interpolation_position,
            parameters,
            middle_time,
        );
        if !value.is_finite() {
            return Err(SolveError::NonFiniteCallbackCondition);
        }
        if value == 0.0 {
            return Ok(middle);
        }
        if left_value.signum() == value.signum() {
            left = middle;
            left_value = value;
        } else {
            right = middle;
        }
        if event_interval_converged(event_tolerance, previous_time, time, left, right) {
            break;
        }
    }
    // Keep the accepted state on the post-crossing side so a continuing
    // callback cannot immediately retrigger the same root.
    Ok(right)
}

fn evaluate_partitioned_vector_condition<P>(
    callback: &VectorContinuousCallback<P>,
    output: &mut [f64],
    velocity: &[f64],
    position: &[f64],
    parameters: &P,
    time: f64,
) -> Result<(), SolveError> {
    output.fill(f64::NAN);
    (callback.condition)(output, velocity, position, parameters, time);
    output
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteCallbackCondition)
}

#[allow(clippy::too_many_arguments)]
fn locate_partitioned_vector_root<P>(
    callback: &VectorContinuousCallback<P>,
    event_index: usize,
    previous_velocity: &[f64],
    previous_position: &[f64],
    previous_time: f64,
    velocity: &[f64],
    position: &[f64],
    time: f64,
    before: f64,
    interpolation_velocity: &mut [f64],
    interpolation_position: &mut [f64],
    parameters: &P,
    event_tolerance: f64,
    mut interpolator: Option<&mut PartitionedInterpolator<'_>>,
    condition_values: &mut [f64],
) -> Result<f64, SolveError> {
    let mut left = 0.0;
    let mut right = 1.0;
    let mut left_value = before;
    for _ in 0..MAX_EVENT_ROOT_ITERATIONS {
        let middle = 0.5 * (left + right);
        if middle == left || middle == right {
            break;
        }
        if let Some(interpolator) = interpolator.as_deref_mut() {
            interpolator(middle, interpolation_velocity, interpolation_position)?;
        } else {
            interpolate_partitioned(
                previous_velocity,
                previous_position,
                velocity,
                position,
                time - previous_time,
                middle,
                interpolation_velocity,
                interpolation_position,
            );
        }
        let middle_time = previous_time + middle * (time - previous_time);
        evaluate_partitioned_vector_condition(
            callback,
            condition_values,
            interpolation_velocity,
            interpolation_position,
            parameters,
            middle_time,
        )?;
        let value = condition_values[event_index];
        if value == 0.0 {
            return Ok(middle);
        }
        if left_value.signum() == value.signum() {
            left = middle;
            left_value = value;
        } else {
            right = middle;
        }
        if event_interval_converged(event_tolerance, previous_time, time, left, right) {
            break;
        }
    }
    Ok(right)
}

fn interpolate(current: &[f64], previous: &[f64], fraction: f64, output: &mut [f64]) {
    for ((output, previous), current) in output.iter_mut().zip(previous).zip(current) {
        *output = previous + fraction * (current - previous);
    }
}

#[allow(clippy::too_many_arguments)]
fn interpolate_partitioned(
    start_velocity: &[f64],
    start_position: &[f64],
    end_velocity: &[f64],
    end_position: &[f64],
    step: f64,
    theta: f64,
    velocity: &mut [f64],
    position: &mut [f64],
) {
    let theta2 = theta * theta;
    let theta3 = theta2 * theta;
    let h00 = 2.0 * theta3 - 3.0 * theta2 + 1.0;
    let h10 = theta3 - 2.0 * theta2 + theta;
    let h01 = -2.0 * theta3 + 3.0 * theta2;
    let h11 = theta3 - theta2;
    for index in 0..velocity.len() {
        velocity[index] =
            start_velocity[index] + theta * (end_velocity[index] - start_velocity[index]);
        position[index] = h00 * start_position[index]
            + h10 * step * start_velocity[index]
            + h01 * end_position[index]
            + h11 * step * end_velocity[index];
    }
}

fn ensure_finite_state(velocity: &[f64], position: &[f64]) -> Result<(), SolveError> {
    velocity
        .iter()
        .chain(position)
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteCallbackState)
}

struct PartitionedRecorder<'a> {
    times: Vec<f64>,
    velocities: Vec<f64>,
    positions: Vec<f64>,
    dimension: usize,
    save_at: &'a [f64],
    next_save: usize,
    save_mode: SaveMode,
    interpolation_velocity: Vec<f64>,
    interpolation_position: Vec<f64>,
    dense_segments: Vec<PartitionedDenseSegment>,
    retain_dense_output: bool,
}

impl<'a> PartitionedRecorder<'a> {
    fn new(velocity: &[f64], position: &[f64], time: f64, options: &'a SolveOptions) -> Self {
        let save_initial = options.save_at.is_empty() || options.save_at.first() == Some(&time);
        let capacity = if options.save_at.is_empty() {
            2
        } else {
            options.save_at.len()
        };
        let mut recorder = Self {
            times: Vec::with_capacity(capacity),
            velocities: Vec::with_capacity(capacity * velocity.len()),
            positions: Vec::with_capacity(capacity * position.len()),
            dimension: position.len(),
            save_at: &options.save_at,
            next_save: usize::from(!options.save_at.is_empty() && save_initial),
            save_mode: options.save,
            interpolation_velocity: if options.save_at.is_empty() {
                Vec::new()
            } else {
                vec![0.0; velocity.len()]
            },
            interpolation_position: if options.save_at.is_empty() {
                Vec::new()
            } else {
                vec![0.0; position.len()]
            },
            dense_segments: Vec::new(),
            retain_dense_output: options.retain_dense_output,
        };
        if save_initial {
            recorder.push_unique(time, velocity, position);
        }
        recorder
    }

    #[allow(clippy::too_many_arguments)]
    fn record_step(
        &mut self,
        previous_velocity: &[f64],
        previous_position: &[f64],
        previous_time: f64,
        velocity: &[f64],
        position: &[f64],
        time: f64,
        final_time: bool,
        mut interpolator: Option<&mut PartitionedInterpolator<'_>>,
    ) -> Result<(), SolveError> {
        let generic_segment = PartitionedDenseSegment::new(
            previous_time,
            time,
            previous_velocity,
            velocity,
            previous_position,
            position,
        );
        if self.retain_dense_output {
            self.dense_segments.push(generic_segment.clone());
        }
        if self.save_at.is_empty() {
            if self.save_mode == SaveMode::EveryStep || final_time {
                self.push_unique(time, velocity, position);
            }
            return Ok(());
        }
        let direction = (time - previous_time).signum();
        while let Some(&target) = self.save_at.get(self.next_save) {
            if direction * (target - previous_time) <= 0.0 {
                self.next_save += 1;
                continue;
            }
            if direction * (time - target) < 0.0 {
                break;
            }
            if let Some(interpolator) = interpolator.as_deref_mut() {
                interpolator(
                    target,
                    &mut self.interpolation_velocity,
                    &mut self.interpolation_position,
                )
                .map_err(|_| SolveError::DenseOutputFailed)?;
            } else {
                generic_segment
                    .interpolate(
                        target,
                        &mut self.interpolation_velocity,
                        &mut self.interpolation_position,
                    )
                    .ok_or(SolveError::DenseOutputFailed)?;
            }
            self.times.push(target);
            self.velocities
                .extend_from_slice(&self.interpolation_velocity);
            self.positions
                .extend_from_slice(&self.interpolation_position);
            self.next_save += 1;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn record_callback(
        &mut self,
        time: f64,
        before_velocity: &[f64],
        before_position: &[f64],
        after_velocity: &[f64],
        after_position: &[f64],
        outcome: CallbackOutcome,
        boundary: bool,
    ) {
        let canonical_time = self
            .save_at
            .iter()
            .copied()
            .find(|target| times_are_numerically_equal(*target, time))
            .unwrap_or(time);
        let requested_at = self
            .save_at
            .iter()
            .any(|target| times_are_numerically_equal(*target, time));
        let globally_saved_after = (self.save_at.is_empty()
            && (self.save_mode == SaveMode::EveryStep || boundary))
            || outcome.terminate;
        let save_before = outcome.save_before || requested_at;
        let save_after = outcome.save_after || globally_saved_after;

        if save_before {
            self.push_unique(canonical_time, before_velocity, before_position);
        }
        if save_after {
            if save_before {
                self.push(canonical_time, after_velocity, after_position);
            } else {
                self.push_unique(canonical_time, after_velocity, after_position);
            }
        }
    }

    fn synchronize_endpoint(&mut self, time: f64, velocity: &[f64], position: &[f64]) {
        if self
            .times
            .last()
            .is_some_and(|saved| times_are_representably_equal(*saved, time))
        {
            let start = self.velocities.len() - self.dimension;
            self.velocities[start..].copy_from_slice(velocity);
            self.positions[start..].copy_from_slice(position);
        }
    }

    fn push_unique(&mut self, time: f64, velocity: &[f64], position: &[f64]) {
        if self
            .times
            .last()
            .is_some_and(|saved| times_are_representably_equal(*saved, time))
        {
            let start = self.velocities.len() - self.dimension;
            self.velocities[start..].copy_from_slice(velocity);
            self.positions[start..].copy_from_slice(position);
        } else {
            self.push(time, velocity, position);
        }
    }

    fn push(&mut self, time: f64, velocity: &[f64], position: &[f64]) {
        debug_assert_eq!(velocity.len(), self.dimension);
        debug_assert_eq!(position.len(), self.dimension);
        self.times.push(time);
        self.velocities.extend_from_slice(velocity);
        self.positions.extend_from_slice(position);
    }

    fn finish(self, stats: SolverStats) -> SecondOrderSolution {
        SecondOrderSolution {
            times: self.times,
            velocities: self.velocities,
            positions: self.positions,
            dimension: self.dimension,
            stats,
            dense_segments: self.dense_segments,
        }
    }
}
