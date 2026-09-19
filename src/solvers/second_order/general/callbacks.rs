use crate::callback::{
    CallbackSave, IterativeTimes, PeriodicTimes, PresetTimes, VectorCallbackScratch,
};
use crate::callbacks::SteadyStateCondition;
use crate::{CallbackAction, EventCrossing, EventDirection, SolveError};
use std::cell::RefCell;
use std::rc::Rc;

pub(super) type DiscreteCondition<P> = dyn Fn(&[f64], &[f64], &P, f64) -> bool;
pub(super) type ContinuousCondition<P> = dyn Fn(&[f64], &[f64], &P, f64) -> f64;
pub(super) type Affect<P> = dyn Fn(&mut [f64], &mut [f64], &P, f64) -> CallbackAction;
pub(super) type FallibleAffect<P> =
    dyn Fn(&mut [f64], &mut [f64], &P, f64) -> Result<CallbackAction, SolveError>;
pub(super) type IterativeInitialization<P> =
    dyn Fn(&[f64], &[f64], &P, f64) -> Result<(), SolveError>;
pub(super) type VectorContinuousCondition<P> = dyn Fn(&mut [f64], &[f64], &[f64], &P, f64);
pub(super) type VectorAffect<P> =
    dyn Fn(&mut [f64], &mut [f64], &P, f64, &[EventCrossing]) -> CallbackAction;
pub(super) type LifecycleHook<P> = dyn Fn(&mut [f64], &mut [f64], &P, f64);
pub(super) type DomainCondition<P> = dyn Fn(&[f64], &[f64], &P, f64) -> bool;
pub(super) type PartitionedInterpolator<'a> =
    dyn FnMut(f64, &mut [f64], &mut [f64]) -> Result<(), SolveError> + 'a;

pub(super) enum DiscreteTrigger<P> {
    Condition(Box<DiscreteCondition<P>>),
    SteadyState(SteadyStateCondition),
    PresetTimes(PresetTimes),
    Periodic(PeriodicTimes),
    Iterative {
        times: Rc<IterativeTimes>,
        initialize: Box<IterativeInitialization<P>>,
    },
}

pub(super) struct DiscreteCallback<P> {
    pub(super) trigger: DiscreteTrigger<P>,
    pub(super) affect: Box<FallibleAffect<P>>,
    pub(super) save: CallbackSave,
}

impl<P> DiscreteTrigger<P> {
    pub(super) fn initialize(
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

    pub(super) fn is_triggered(
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

    pub(super) fn preset_times(&self) -> Option<&[f64]> {
        match self {
            Self::Condition(_) | Self::SteadyState(_) => None,
            Self::PresetTimes(times) => Some(times.as_slice()),
            Self::Periodic(_) => None,
            Self::Iterative { .. } => None,
        }
    }

    pub(super) fn next_preset_time(&self, time: f64, direction: f64) -> Option<f64> {
        match self {
            Self::Condition(_) | Self::SteadyState(_) => None,
            Self::PresetTimes(times) => times.next(time, direction),
            Self::Periodic(times) => times.next(time, direction),
            Self::Iterative { times, .. } => times.next(time, direction),
        }
    }
}

pub(super) struct ContinuousCallback<P> {
    pub(super) condition: Box<ContinuousCondition<P>>,
    pub(super) affect: Box<Affect<P>>,
    pub(super) direction: EventDirection,
    pub(super) save: CallbackSave,
}

pub(super) struct VectorContinuousCallback<P> {
    pub(super) condition: Box<VectorContinuousCondition<P>>,
    pub(super) affect: Box<VectorAffect<P>>,
    pub(super) event_count: usize,
    pub(super) save: CallbackSave,
    pub(super) scratch: RefCell<VectorCallbackScratch>,
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

pub(super) enum PartitionedCallback<P> {
    Discrete(DiscreteCallback<P>),
    Continuous(ContinuousCallback<P>),
    VectorContinuous(VectorContinuousCallback<P>),
}

pub(super) struct InitializationHook<P> {
    pub(super) hook: Box<LifecycleHook<P>>,
    pub(super) save: CallbackSave,
}

pub(super) struct StepGuard<P> {
    pub(super) is_out_of_domain: Box<DomainCondition<P>>,
    pub(super) reduction_factor: f64,
}

/// An ordered collection of callbacks for a second-order ODE problem.
///
/// Conditions receive velocity before position, and effects receive mutable
/// velocity and position partitions in the same order.
#[must_use]
pub struct SecondOrderCallbackSet<P> {
    pub(super) callbacks: Vec<PartitionedCallback<P>>,
    pub(super) initializers: Vec<InitializationHook<P>>,
    pub(super) finalizers: Vec<Box<LifecycleHook<P>>>,
    pub(super) step_guards: Vec<StepGuard<P>>,
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
