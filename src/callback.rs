/// The action requested after an ODE callback changes the state.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum CallbackAction {
    /// Resume integration from the callback time and state.
    #[default]
    Continue,
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

pub(crate) type Condition<P> = dyn Fn(&[f64], &P, f64) -> bool;
pub(crate) type EventCondition<P> = dyn Fn(&[f64], &P, f64) -> f64;
pub(crate) type Affect<P> = dyn Fn(&mut [f64], &P, f64) -> CallbackAction;

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

pub(crate) enum Callback<P> {
    Discrete(DiscreteCallback<P>),
    Continuous(ContinuousCallback<P>),
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
}

impl<P> CallbackSet<P> {
    /// Creates an empty callback set.
    pub const fn new() -> Self {
        Self {
            callbacks: Vec::new(),
        }
    }

    /// Returns the number of callbacks in the set.
    pub fn len(&self) -> usize {
        self.callbacks.len()
    }

    /// Returns whether the set contains no callbacks.
    pub fn is_empty(&self) -> bool {
        self.callbacks.is_empty()
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

    /// Appends another set, preserving callback order within each set.
    pub fn append(mut self, mut other: Self) -> Self {
        self.callbacks.append(&mut other.callbacks);
        self
    }
}

impl<P> Default for CallbackSet<P> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct CallbackOutcome {
    pub invocations: usize,
    pub terminate: bool,
    pub save_before: bool,
    pub save_after: bool,
}

impl CallbackOutcome {
    pub(crate) fn register(&mut self, save: CallbackSave) {
        self.invocations += 1;
        self.save_before |= save.saves_before();
        self.save_after |= save.saves_after();
    }
}

impl EventDirection {
    pub(crate) fn accepts(self, before: f64, after: f64) -> bool {
        match self {
            Self::Any => (before < 0.0 && after >= 0.0) || (before > 0.0 && after <= 0.0),
            Self::Rising => before < 0.0 && after >= 0.0,
            Self::Falling => before > 0.0 && after <= 0.0,
        }
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
