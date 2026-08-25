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

pub(crate) struct DiscreteCallback<P> {
    pub condition: Box<Condition<P>>,
    pub affect: Box<Affect<P>>,
}

pub(crate) struct ContinuousCallback<P> {
    pub condition: Box<EventCondition<P>>,
    pub affect: Box<Affect<P>>,
    pub direction: EventDirection,
}

pub(crate) enum Callback<P> {
    Discrete(DiscreteCallback<P>),
    Continuous(ContinuousCallback<P>),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct CallbackOutcome {
    pub invocations: usize,
    pub terminate: bool,
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
