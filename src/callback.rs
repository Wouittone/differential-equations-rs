use std::cell::RefCell;

/// The action requested after an ODE callback changes the state.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub enum CallbackAction {
    /// Resume integration from the callback time and state.
    #[default]
    Continue,
    /// Resume integration and use this positive step-size magnitude next.
    ///
    /// The request overrides the adaptive controller or fixed-step size for
    /// the next attempt, but remains bounded by [`crate::SolveOptions::max_step`]
    /// and any pending exact time stop. When several callbacks request a step
    /// at the same time, the last request wins.
    ContinueWithStepSize(f64),
    /// Stop integration and return the callback time and state as the endpoint.
    Terminate,
}

/// Selects which states a callback forces into the saved trajectory.
///
/// These saves are in addition to the accepted-step and requested-time output
/// configured through [`crate::SolveOptions`]. When both positions are saved,
/// the solution contains two adjacent entries at the callback time: the
/// left-limit state followed by the affected state.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum CallbackSave {
    /// Do not force an additional callback-time save.
    None,
    /// Save only the state immediately before the callback effect.
    Before,
    /// Save only the affected state.
    #[default]
    After,
    /// Save the state before and after the callback effect.
    Both,
}

impl CallbackSave {
    /// Returns whether the left-limit state is retained.
    pub const fn saves_before(self) -> bool {
        matches!(self, Self::Before | Self::Both)
    }

    /// Returns whether the affected state is retained.
    pub const fn saves_after(self) -> bool {
        matches!(self, Self::After | Self::Both)
    }
}

/// Selects which zero crossings trigger a continuous callback.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum EventDirection {
    /// Trigger on either a negative-to-positive or positive-to-negative crossing.
    #[default]
    Any,
    /// Trigger only when the condition increases through zero along the trajectory.
    Rising,
    /// Trigger only when the condition decreases through zero along the trajectory.
    Falling,
}

/// Describes whether and how one vector callback condition crossed zero.
///
/// A vector continuous callback receives one entry per condition. Several
/// entries may be non-[`None`](Self::None) when roots are simultaneous.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(i8)]
#[non_exhaustive]
pub enum EventCrossing {
    /// This condition did not trigger at the localized event time.
    #[default]
    None = 0,
    /// The condition crossed from negative to non-negative.
    Rising = 1,
    /// The condition crossed from positive to non-positive.
    Falling = -1,
}

pub(crate) type Condition<P> = dyn Fn(&[f64], &P, f64) -> bool;
pub(crate) type EventCondition<P> = dyn Fn(&[f64], &P, f64) -> f64;
pub(crate) type Affect<P> = dyn Fn(&mut [f64], &P, f64) -> CallbackAction;
pub(crate) type VectorEventCondition<P> = dyn Fn(&mut [f64], &[f64], &P, f64);
pub(crate) type VectorAffect<P> = dyn Fn(&mut [f64], &P, f64, &[EventCrossing]) -> CallbackAction;
pub(crate) type LifecycleHook<P> = dyn Fn(&mut [f64], &P, f64);

pub(crate) struct InitializationHook<P> {
    pub hook: Box<LifecycleHook<P>>,
    pub save: CallbackSave,
}

pub(crate) struct PresetTimes(Vec<f64>);

impl PresetTimes {
    pub(crate) fn new(times: impl IntoIterator<Item = f64>) -> Self {
        Self(times.into_iter().collect())
    }

    pub(crate) fn as_slice(&self) -> &[f64] {
        &self.0
    }

    pub(crate) fn contains(&self, time: f64) -> bool {
        self.0.contains(&time)
    }

    pub(crate) fn next(&self, time: f64, direction: f64) -> Option<f64> {
        self.0
            .iter()
            .copied()
            .find(|candidate| direction * (*candidate - time) > 0.0)
    }
}

pub(crate) enum DiscreteTrigger<P> {
    Condition(Box<Condition<P>>),
    PresetTimes(PresetTimes),
}

pub(crate) struct DiscreteCallback<P> {
    pub trigger: DiscreteTrigger<P>,
    pub affect: Box<Affect<P>>,
    pub save: CallbackSave,
}

pub(crate) struct ContinuousCallback<P> {
    pub condition: Box<EventCondition<P>>,
    pub affect: Box<Affect<P>>,
    pub direction: EventDirection,
    pub save: CallbackSave,
}

pub(crate) struct VectorCallbackScratch {
    pub before: Vec<f64>,
    pub after: Vec<f64>,
    pub middle: Vec<f64>,
    pub root_fractions: Vec<f64>,
    pub crossings: Vec<EventCrossing>,
    pub simultaneous_events: Vec<EventCrossing>,
}

impl VectorCallbackScratch {
    pub(crate) fn new(event_count: usize) -> Self {
        Self {
            before: vec![f64::NAN; event_count],
            after: vec![f64::NAN; event_count],
            middle: vec![f64::NAN; event_count],
            root_fractions: vec![f64::INFINITY; event_count],
            crossings: vec![EventCrossing::None; event_count],
            simultaneous_events: vec![EventCrossing::None; event_count],
        }
    }
}

pub(crate) struct VectorContinuousCallback<P> {
    pub condition: Box<VectorEventCondition<P>>,
    pub affect: Box<VectorAffect<P>>,
    pub event_count: usize,
    pub save: CallbackSave,
    pub scratch: RefCell<VectorCallbackScratch>,
}

impl<P> VectorContinuousCallback<P> {
    pub(crate) fn new<C, A>(event_count: usize, save: CallbackSave, condition: C, affect: A) -> Self
    where
        C: Fn(&mut [f64], &[f64], &P, f64) + 'static,
        A: Fn(&mut [f64], &P, f64, &[EventCrossing]) -> CallbackAction + 'static,
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

pub(crate) enum Callback<P> {
    Discrete(DiscreteCallback<P>),
    Continuous(ContinuousCallback<P>),
    VectorContinuous(VectorContinuousCallback<P>),
}

/// An ordered collection of callbacks that can be attached to an ODE problem.
///
/// A set is useful when callback configuration is built separately from a
/// problem or assembled from several reusable pieces. Callbacks retain their
/// insertion order within each callback kind; continuous events are localized
/// before discrete conditions are evaluated at the resulting time.
#[must_use]
pub struct CallbackSet<P> {
    pub(crate) callbacks: Vec<Callback<P>>,
    pub(crate) initializers: Vec<InitializationHook<P>>,
    pub(crate) finalizers: Vec<Box<LifecycleHook<P>>>,
}

impl<P> CallbackSet<P> {
    /// Creates an empty callback set.
    pub const fn new() -> Self {
        Self {
            callbacks: Vec::new(),
            initializers: Vec::new(),
            finalizers: Vec::new(),
        }
    }

    /// Returns the number of callbacks in the set.
    pub fn len(&self) -> usize {
        self.callbacks.len()
    }

    /// Returns whether the set contains no callbacks.
    pub fn is_empty(&self) -> bool {
        self.callbacks.is_empty() && self.initializers.is_empty() && self.finalizers.is_empty()
    }

    /// Adds a state initialization hook that saves the initialized state.
    ///
    /// Initialization hooks run in insertion order before initial discrete
    /// callback conditions are evaluated. A hook is treated as a state
    /// mutation so solver caches are initialized from its resulting state.
    pub fn with_initialize<I>(self, initialize: I) -> Self
    where
        I: Fn(&mut [f64], &P, f64) + 'static,
    {
        self.with_initialize_saving(CallbackSave::After, initialize)
    }

    /// Adds an initialization hook with explicit initial-state saving behavior.
    pub fn with_initialize_saving<I>(mut self, save: CallbackSave, initialize: I) -> Self
    where
        I: Fn(&mut [f64], &P, f64) + 'static,
    {
        self.initializers.push(InitializationHook {
            hook: Box::new(initialize),
            save,
        });
        self
    }

    /// Adds an end-of-solve state finalization hook.
    ///
    /// Finalizers run in insertion order after normal completion or callback
    /// termination, but not after a solve error. If the endpoint is part of
    /// the saved trajectory, it is synchronized with the finalized state.
    pub fn with_finalize<F>(mut self, finalize: F) -> Self
    where
        F: Fn(&mut [f64], &P, f64) + 'static,
    {
        self.finalizers.push(Box::new(finalize));
        self
    }

    /// Adds a callback evaluated at initialization and after accepted steps.
    pub fn with_discrete_callback<C, A>(self, condition: C, affect: A) -> Self
    where
        C: Fn(&[f64], &P, f64) -> bool + 'static,
        A: Fn(&mut [f64], &P, f64) -> CallbackAction + 'static,
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
        C: Fn(&[f64], &P, f64) -> bool + 'static,
        A: Fn(&mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        self.callbacks.push(Callback::Discrete(DiscreteCallback {
            trigger: DiscreteTrigger::Condition(Box::new(condition)),
            affect: Box::new(affect),
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
        A: Fn(&mut [f64], &P, f64) -> CallbackAction + 'static,
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
        A: Fn(&mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        self.callbacks.push(Callback::Discrete(DiscreteCallback {
            trigger: DiscreteTrigger::PresetTimes(PresetTimes::new(times)),
            affect: Box::new(affect),
            save,
        }));
        self
    }

    /// Adds a zero-crossing callback that triggers in either direction.
    pub fn with_continuous_callback<C, A>(self, condition: C, affect: A) -> Self
    where
        C: Fn(&[f64], &P, f64) -> f64 + 'static,
        A: Fn(&mut [f64], &P, f64) -> CallbackAction + 'static,
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
        C: Fn(&[f64], &P, f64) -> f64 + 'static,
        A: Fn(&mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        self.with_continuous_callback_direction_saving(EventDirection::Any, save, condition, affect)
    }

    /// Adds a direction-filtered continuous callback.
    pub fn with_continuous_callback_direction<C, A>(
        self,
        direction: EventDirection,
        condition: C,
        affect: A,
    ) -> Self
    where
        C: Fn(&[f64], &P, f64) -> f64 + 'static,
        A: Fn(&mut [f64], &P, f64) -> CallbackAction + 'static,
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
        C: Fn(&[f64], &P, f64) -> f64 + 'static,
        A: Fn(&mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        self.callbacks
            .push(Callback::Continuous(ContinuousCallback {
                condition: Box::new(condition),
                affect: Box::new(affect),
                direction,
                save,
            }));
        self
    }

    /// Adds a vector-valued zero-crossing callback.
    ///
    /// `condition` must overwrite all `event_count` output entries. The effect
    /// runs once at the earliest localized root and receives a crossing mask;
    /// simultaneous conditions are reported together in index order.
    pub fn with_vector_continuous_callback<C, A>(
        self,
        event_count: usize,
        condition: C,
        affect: A,
    ) -> Self
    where
        C: Fn(&mut [f64], &[f64], &P, f64) + 'static,
        A: Fn(&mut [f64], &P, f64, &[EventCrossing]) -> CallbackAction + 'static,
    {
        self.with_vector_continuous_callback_saving(
            event_count,
            CallbackSave::Both,
            condition,
            affect,
        )
    }

    /// Adds a vector continuous callback with explicit saving behavior.
    pub fn with_vector_continuous_callback_saving<C, A>(
        mut self,
        event_count: usize,
        save: CallbackSave,
        condition: C,
        affect: A,
    ) -> Self
    where
        C: Fn(&mut [f64], &[f64], &P, f64) + 'static,
        A: Fn(&mut [f64], &P, f64, &[EventCrossing]) -> CallbackAction + 'static,
    {
        self.callbacks
            .push(Callback::VectorContinuous(VectorContinuousCallback::new(
                event_count,
                save,
                condition,
                affect,
            )));
        self
    }

    /// Appends another set, preserving callback order within each set.
    pub fn append(mut self, mut other: Self) -> Self {
        self.callbacks.append(&mut other.callbacks);
        self.initializers.append(&mut other.initializers);
        self.finalizers.append(&mut other.finalizers);
        self
    }
}

impl<P> Default for CallbackSet<P> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct CallbackOutcome {
    pub invocations: usize,
    pub terminate: bool,
    pub state_modified: bool,
    pub save_before: bool,
    pub save_after: bool,
    pub requested_step: Option<f64>,
}

impl CallbackOutcome {
    pub(crate) fn register(&mut self, save: CallbackSave) {
        self.invocations += 1;
        self.state_modified = true;
        self.register_save(save);
    }

    pub(crate) fn register_initialization(&mut self, save: CallbackSave) {
        self.state_modified = true;
        self.register_save(save);
    }

    pub(crate) fn apply_action(&mut self, action: CallbackAction) -> Result<(), crate::SolveError> {
        match action {
            CallbackAction::Continue => {}
            CallbackAction::ContinueWithStepSize(step) if step.is_finite() && step > 0.0 => {
                self.requested_step = Some(step);
            }
            CallbackAction::ContinueWithStepSize(_) => {
                return Err(crate::SolveError::InvalidCallbackStepSize);
            }
            CallbackAction::Terminate => self.terminate = true,
        }
        Ok(())
    }

    fn register_save(&mut self, save: CallbackSave) {
        self.save_before |= save.saves_before();
        self.save_after |= save.saves_after();
    }
}

impl EventDirection {
    pub(crate) fn crossing(self, before: f64, after: f64) -> EventCrossing {
        if before < 0.0 && after >= 0.0 && !matches!(self, Self::Falling) {
            EventCrossing::Rising
        } else if before > 0.0 && after <= 0.0 && !matches!(self, Self::Rising) {
            EventCrossing::Falling
        } else {
            EventCrossing::None
        }
    }

    pub(crate) fn accepts(self, before: f64, after: f64) -> bool {
        self.crossing(before, after) != EventCrossing::None
    }
}

impl<P> DiscreteTrigger<P> {
    pub(crate) fn is_triggered(&self, state: &[f64], parameters: &P, time: f64) -> bool {
        match self {
            Self::Condition(condition) => condition(state, parameters, time),
            Self::PresetTimes(times) => times.contains(time),
        }
    }

    pub(crate) fn preset_times(&self) -> Option<&[f64]> {
        match self {
            Self::Condition(_) => None,
            Self::PresetTimes(times) => Some(times.as_slice()),
        }
    }

    pub(crate) fn next_preset_time(&self, time: f64, direction: f64) -> Option<f64> {
        match self {
            Self::Condition(_) => None,
            Self::PresetTimes(times) => times.next(time, direction),
        }
    }
}
