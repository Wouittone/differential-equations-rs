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
