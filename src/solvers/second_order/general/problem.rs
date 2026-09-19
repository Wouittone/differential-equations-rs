use super::super::function::SecondOrderFunction;
use super::callbacks::{
    InitializationHook, LifecycleHook, PartitionedCallback, SecondOrderCallbackSet, StepGuard,
};
use crate::callback::CallbackSave;
use crate::{CallbackAction, EventCrossing, EventDirection, SolveError};
use ndarray::IxDyn;

/// A second-order initial-value problem `q'' = f(q', q, p, t)`.
///
/// The acceleration function follows SciML's in-place calling convention
/// `f(dv, v, q, p, t)`. Positions and velocities remain separate throughout
/// the public API; callers do not need to flatten the partitioned state.
/// [`Self::from_array`] accepts ndarray states, and
/// [`Self::from_array_out_of_place`] accepts functions returning accelerations.
/// This represents SciML's `SecondOrderODEProblem` specialization `q' = v`,
/// not a general `DynamicalODEProblem` with a separately supplied position
/// rate.
pub struct SecondOrderOdeProblem<F, P> {
    pub(crate) acceleration: F,
    pub(super) initial_velocity: Vec<f64>,
    pub(super) initial_position: Vec<f64>,
    pub(super) state_shape: IxDyn,
    pub(super) time_span: (f64, f64),
    pub(super) parameters: P,
    pub(super) callbacks: Vec<PartitionedCallback<P>>,
    pub(super) initializers: Vec<InitializationHook<P>>,
    pub(super) finalizers: Vec<Box<LifecycleHook<P>>>,
    pub(super) step_guards: Vec<StepGuard<P>>,
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
        let initial_position = initial_position.into();
        Self {
            acceleration,
            initial_velocity: initial_velocity.into(),
            state_shape: IxDyn(&[initial_position.len()]),
            initial_position,
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

    /// Returns the number of scalar components in each state partition.
    ///
    /// Mismatched initial partitions are rejected by the checked solver entry
    /// point; this reports the position partition's configured dimension.
    pub fn dimension(&self) -> usize {
        self.initial_position.len()
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
    ///
    /// All buffers must match the problem's partition dimension, otherwise
    /// [`SolveError::EvaluationDimensionMismatch`] is returned. Function
    /// errors are propagated unchanged, and a successful evaluation containing
    /// NaN or infinity returns [`SolveError::NonFiniteDerivative`].
    pub fn evaluate_acceleration(
        &self,
        output: &mut [f64],
        velocity: &[f64],
        position: &[f64],
        time: f64,
    ) -> Result<(), SolveError>
    where
        F: SecondOrderFunction<P>,
    {
        let dimension = self.initial_position.len();
        if output.len() != dimension || velocity.len() != dimension || position.len() != dimension {
            return Err(SolveError::EvaluationDimensionMismatch);
        }
        self.acceleration
            .evaluate(output, velocity, position, &self.parameters, time)?;
        output
            .iter()
            .all(|value| value.is_finite())
            .then_some(())
            .ok_or(SolveError::NonFiniteDerivative)
    }

    /// Returns whether callback policies, lifecycle hooks, or guards were supplied.
    ///
    /// Downstream [`crate::solvers::second_order::SecondOrderOdeAlgorithm`]
    /// implementations that do not
    /// provide the callback lifecycle must use this query to reject the
    /// problem with [`SolveError::CallbacksUnsupported`]. Silently ignoring a
    /// callback-bearing problem violates the algorithm contract.
    pub fn has_callbacks(&self) -> bool {
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
