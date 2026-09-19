use super::callbacks::{
    RootSegment, ensure_finite_callback_state, evaluate_vector_condition, interpolate, locate_root,
    locate_vector_root, predictive_domain_adjusted_step,
};
use super::{SplitOdeProblem, StepInterpolator};
use crate::SolveError;
use crate::callback::{
    Callback, CallbackAction, CallbackOutcome, CallbackSave, CallbackSet, DiscreteTrigger,
    EventCrossing, EventDirection,
};
use crate::event::effective_event_tolerance;
use ndarray::{ArrayViewD, Dimension, IxDyn};

#[allow(dead_code)]
impl<FE, FI, P> SplitOdeProblem<FE, FI, P> {
    /// Constructs a split problem from explicit and implicit right-hand sides.
    pub fn new(
        explicit: FE,
        implicit: FI,
        initial_state: impl Into<Vec<f64>>,
        time_span: (f64, f64),
        parameters: P,
    ) -> Self {
        let initial_state = initial_state.into();
        let state_shape = IxDyn(&[initial_state.len()]);
        Self {
            explicit,
            implicit,
            initial_state,
            state_shape,
            time_span,
            parameters,
            implicit_jacobian: None,
            callbacks: Vec::new(),
            initializers: Vec::new(),
            finalizers: Vec::new(),
            step_guards: Vec::new(),
            predictive_domains: Vec::new(),
        }
    }

    /// Appends an ordered callback set to this problem.
    pub fn with_callback_set(mut self, mut callback_set: CallbackSet<P>) -> Self {
        self.callbacks.append(&mut callback_set.callbacks);
        self.initializers.append(&mut callback_set.initializers);
        self.finalizers.append(&mut callback_set.finalizers);
        self.step_guards.append(&mut callback_set.step_guards);
        self.predictive_domains
            .append(&mut callback_set.predictive_domains);
        self
    }

    /// Adds a callback evaluated after every accepted step and at the initial state.
    pub fn with_discrete_callback<C, A>(self, condition: C, affect: A) -> Self
    where
        C: Fn(&[f64], &P, f64) -> bool + 'static,
        A: Fn(&mut [f64], &P, f64) -> CallbackAction + 'static,
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
        C: Fn(&[f64], &P, f64) -> bool + 'static,
        A: Fn(&mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        self.with_callback_set(
            CallbackSet::new().with_discrete_callback_saving(save, condition, affect),
        )
    }

    /// Adds a callback that runs at each listed integration time.
    ///
    /// The times are also treated as mandatory integration stops, so callers
    /// do not need to duplicate them in [`crate::SolveOptions::time_stops`].
    /// They are validated when the problem is solved.
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
        self,
        times: impl IntoIterator<Item = f64>,
        save: CallbackSave,
        affect: A,
    ) -> Self
    where
        A: Fn(&mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        self.with_callback_set(
            CallbackSet::new().with_preset_time_callback_saving(times, save, affect),
        )
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
        self,
        direction: EventDirection,
        save: CallbackSave,
        condition: C,
        affect: A,
    ) -> Self
    where
        C: Fn(&[f64], &P, f64) -> f64 + 'static,
        A: Fn(&mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        self.with_callback_set(
            CallbackSet::new()
                .with_continuous_callback_direction_saving(direction, save, condition, affect),
        )
    }

    /// Adds a vector-valued zero-crossing callback.
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
        self,
        event_count: usize,
        save: CallbackSave,
        condition: C,
        affect: A,
    ) -> Self
    where
        C: Fn(&mut [f64], &[f64], &P, f64) + 'static,
        A: Fn(&mut [f64], &P, f64, &[EventCrossing]) -> CallbackAction + 'static,
    {
        self.with_callback_set(CallbackSet::new().with_vector_continuous_callback_saving(
            event_count,
            save,
            condition,
            affect,
        ))
    }

    /// Supplies the analytic state Jacobian of the implicit component.
    ///
    /// The callback receives a row-major `dimension x dimension` output matrix
    /// and must overwrite every entry with the Jacobian of the implicit right-
    /// hand side. IMEX methods use finite differences when this is absent.
    pub fn with_implicit_jacobian<J>(mut self, jacobian: J) -> Self
    where
        J: Fn(&mut [f64], &[f64], &P, f64) + 'static,
    {
        self.implicit_jacobian = Some(Box::new(jacobian));
        self
    }

    /// Returns the initial state.
    pub fn initial_state(&self) -> &[f64] {
        &self.initial_state
    }

    /// Returns the initial state with its ndarray dimensionality.
    pub fn initial_state_array(&self) -> ArrayViewD<'_, f64> {
        ArrayViewD::from_shape(self.state_shape.clone(), &self.initial_state)
            .expect("problem state shape must match its contiguous storage")
    }

    /// Returns the logical ndarray shape of the state.
    pub fn state_shape(&self) -> &[usize] {
        self.state_shape.slice()
    }

    /// Returns the integration time span.
    pub fn time_span(&self) -> (f64, f64) {
        self.time_span
    }

    /// Returns the shared user parameters.
    pub fn parameters(&self) -> &P {
        &self.parameters
    }

    /// Returns the state dimension.
    pub fn dimension(&self) -> usize {
        self.initial_state.len()
    }

    /// Returns whether callback policies, lifecycle hooks, or guards were supplied.
    pub fn has_callbacks(&self) -> bool {
        !self.callbacks.is_empty()
            || !self.initializers.is_empty()
            || !self.finalizers.is_empty()
            || !self.step_guards.is_empty()
            || !self.predictive_domains.is_empty()
    }

    pub(crate) fn domain_rejection_factor(&self, state: &[f64], time: f64) -> Option<f64> {
        self.step_guards
            .iter()
            .filter(|guard| (guard.is_out_of_domain)(state, &self.parameters, time))
            .map(|guard| guard.reduction_factor)
            .reduce(f64::min)
    }

    pub(crate) fn has_predictive_domain(&self) -> bool {
        !self.predictive_domains.is_empty()
    }

    pub(crate) fn predictive_domain_adjusted_step(
        &self,
        state: &[f64],
        derivative: &[f64],
        time: f64,
        proposed_step: f64,
        default_tolerance: f64,
        prediction: &mut [f64],
    ) -> Result<f64, SolveError> {
        predictive_domain_adjusted_step(
            &self.predictive_domains,
            &self.parameters,
            state,
            derivative,
            time,
            proposed_step,
            default_tolerance,
            prediction,
        )
    }

    /// Evaluates the explicit right-hand side.
    ///
    /// Propagates errors from the function, including a returned array with an
    /// incompatible shape.
    pub fn evaluate_explicit(
        &self,
        derivative: &mut [f64],
        state: &[f64],
        time: f64,
    ) -> Result<(), SolveError>
    where
        FE: crate::OdeFunction<P>,
    {
        if derivative.len() != self.dimension() || state.len() != self.dimension() {
            return Err(SolveError::EvaluationDimensionMismatch);
        }
        self.explicit
            .evaluate(derivative, state, &self.parameters, time)?;
        derivative
            .iter()
            .all(|value| value.is_finite())
            .then_some(())
            .ok_or(SolveError::NonFiniteDerivative)
    }

    /// Evaluates the implicit right-hand side.
    ///
    /// Propagates errors from the function, including a returned array with an
    /// incompatible shape.
    pub fn evaluate_implicit(
        &self,
        derivative: &mut [f64],
        state: &[f64],
        time: f64,
    ) -> Result<(), SolveError>
    where
        FI: crate::OdeFunction<P>,
    {
        if derivative.len() != self.dimension() || state.len() != self.dimension() {
            return Err(SolveError::EvaluationDimensionMismatch);
        }
        self.implicit
            .evaluate(derivative, state, &self.parameters, time)?;
        derivative
            .iter()
            .all(|value| value.is_finite())
            .then_some(())
            .ok_or(SolveError::NonFiniteDerivative)
    }

    fn discrete_is_triggered(
        &self,
        trigger: &DiscreteTrigger<P>,
        state: &[f64],
        time: f64,
        evaluations: &mut usize,
    ) -> Result<bool, SolveError>
    where
        FE: crate::OdeFunction<P>,
        FI: crate::OdeFunction<P>,
    {
        trigger.is_triggered(state, &self.parameters, time, |du, work| {
            self.evaluate_explicit(du, state, time)?;
            self.evaluate_implicit(work, state, time)?;
            *evaluations += 2;
            for (du, implicit) in du.iter_mut().zip(work) {
                *du += *implicit;
            }
            Ok(())
        })
    }

    pub(crate) fn evaluate_implicit_jacobian(
        &self,
        jacobian: &mut [f64],
        state: &[f64],
        time: f64,
    ) -> bool {
        let Some(function) = &self.implicit_jacobian else {
            return false;
        };
        function(jacobian, state, &self.parameters, time);
        true
    }

    pub(crate) fn apply_initial_callbacks(
        &self,
        state: &mut [f64],
        time: f64,
    ) -> Result<CallbackOutcome, SolveError>
    where
        FE: crate::OdeFunction<P>,
        FI: crate::OdeFunction<P>,
    {
        let mut outcome = CallbackOutcome::default();
        for initialization in &self.initializers {
            (initialization.hook)(state, &self.parameters, time);
            ensure_finite_callback_state(state)?;
            outcome.register_initialization(initialization.save);
        }
        for callback in &self.callbacks {
            let Callback::Discrete(callback) = callback else {
                continue;
            };
            callback.trigger.initialize(state, &self.parameters, time)?;
            if self.discrete_is_triggered(
                &callback.trigger,
                state,
                time,
                &mut outcome.rhs_evaluations,
            )? {
                outcome.register(callback.save);
                outcome.apply_action((callback.affect)(state, &self.parameters, time)?)?;
                ensure_finite_callback_state(state)?;
                if outcome.terminate {
                    break;
                }
            }
        }
        Ok(outcome)
    }

    pub(crate) fn apply_finalize_callbacks(
        &self,
        state: &mut [f64],
        time: f64,
    ) -> Result<bool, SolveError> {
        for finalize in &self.finalizers {
            finalize(state, &self.parameters, time);
            ensure_finite_callback_state(state)?;
        }
        Ok(!self.finalizers.is_empty())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_step_callbacks(
        &self,
        previous_state: &[f64],
        previous_time: f64,
        state: &mut [f64],
        time: &mut f64,
        state_before_effect: &mut [f64],
        event_tolerance: f64,
        mut interpolator: Option<&mut StepInterpolator<'_>>,
    ) -> Result<CallbackOutcome, SolveError>
    where
        FE: crate::OdeFunction<P>,
        FI: crate::OdeFunction<P>,
    {
        if self.callbacks.is_empty() {
            return Ok(CallbackOutcome::default());
        }
        let mut outcome = CallbackOutcome::default();
        let mut root = None;
        for (index, callback) in self.callbacks.iter().enumerate() {
            match callback {
                Callback::Continuous(callback) => {
                    let before =
                        (callback.condition)(previous_state, &self.parameters, previous_time);
                    let after = (callback.condition)(state, &self.parameters, *time);
                    if !before.is_finite() || !after.is_finite() {
                        return Err(SolveError::NonFiniteCallbackCondition);
                    }
                    if callback.direction.accepts(before, after) {
                        let fraction = locate_root(
                            callback,
                            RootSegment {
                                previous_state,
                                previous_time,
                                state,
                                time: *time,
                            },
                            before,
                            state_before_effect,
                            &self.parameters,
                            event_tolerance,
                            &mut interpolator,
                        )?;
                        if root.is_none_or(|(_, earliest)| fraction < earliest) {
                            root = Some((index, fraction));
                        }
                    }
                }
                Callback::VectorContinuous(callback) => {
                    let mut scratch = callback.scratch.borrow_mut();
                    evaluate_vector_condition(
                        callback,
                        &mut scratch.before,
                        previous_state,
                        &self.parameters,
                        previous_time,
                    )?;
                    evaluate_vector_condition(
                        callback,
                        &mut scratch.after,
                        state,
                        &self.parameters,
                        *time,
                    )?;
                    scratch.root_fractions.fill(f64::INFINITY);
                    scratch.crossings.fill(EventCrossing::None);
                    for event_index in 0..callback.event_count {
                        let before = scratch.before[event_index];
                        let crossing =
                            EventDirection::Any.crossing(before, scratch.after[event_index]);
                        if crossing == EventCrossing::None {
                            continue;
                        }
                        let fraction = locate_vector_root(
                            callback,
                            event_index,
                            RootSegment {
                                previous_state,
                                previous_time,
                                state,
                                time: *time,
                            },
                            before,
                            state_before_effect,
                            &self.parameters,
                            event_tolerance,
                            &mut interpolator,
                            &mut scratch.middle,
                        )?;
                        scratch.root_fractions[event_index] = fraction;
                        scratch.crossings[event_index] = crossing;
                        if root.is_none_or(|(_, earliest)| fraction < earliest) {
                            root = Some((index, fraction));
                        }
                    }
                }
                Callback::Discrete(_) => {}
            }
        }
        if let Some((index, fraction)) = root {
            let end_time = *time;
            let root_time = previous_time + fraction * (end_time - previous_time);
            if let Some(interpolator) = interpolator.as_mut() {
                interpolator(root_time, state_before_effect)?;
            } else {
                interpolate(state, previous_state, fraction, state_before_effect);
            }
            state.copy_from_slice(state_before_effect);
            *time = root_time;
            match &self.callbacks[index] {
                Callback::Continuous(callback) => {
                    outcome.register(callback.save);
                    outcome.apply_action((callback.affect)(state, &self.parameters, *time))?;
                }
                Callback::VectorContinuous(callback) => {
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
                        state,
                        &self.parameters,
                        *time,
                        &scratch.simultaneous_events,
                    ))?;
                }
                Callback::Discrete(_) => return Err(SolveError::InvalidCallbackState),
            }
            // The localized root truncates the attempted step even when its
            // effect is observation-only, so endpoint-dependent caches cannot
            // be reused for the next step.
            outcome.state_modified = true;
            ensure_finite_callback_state(state)?;
        }
        if !outcome.terminate {
            for callback in &self.callbacks {
                let Callback::Discrete(callback) = callback else {
                    continue;
                };
                if self.discrete_is_triggered(
                    &callback.trigger,
                    state,
                    *time,
                    &mut outcome.rhs_evaluations,
                )? {
                    if outcome.invocations == 0 {
                        state_before_effect.copy_from_slice(state);
                    }
                    outcome.register(callback.save);
                    outcome.apply_action((callback.affect)(state, &self.parameters, *time)?)?;
                    ensure_finite_callback_state(state)?;
                    if outcome.terminate {
                        break;
                    }
                }
            }
        }
        Ok(outcome)
    }

    pub(crate) fn preset_time_sequences(&self) -> impl Iterator<Item = &[f64]> {
        self.callbacks.iter().filter_map(|callback| {
            let Callback::Discrete(callback) = callback else {
                return None;
            };
            callback.trigger.preset_times()
        })
    }

    pub(crate) fn vector_callback_lengths(&self) -> impl Iterator<Item = usize> + '_ {
        self.callbacks.iter().filter_map(|callback| {
            let Callback::VectorContinuous(callback) = callback else {
                return None;
            };
            Some(callback.event_count)
        })
    }

    pub(crate) fn next_preset_time(&self, time: f64, direction: f64) -> Option<f64> {
        self.callbacks
            .iter()
            .filter_map(|callback| {
                let Callback::Discrete(callback) = callback else {
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
}
