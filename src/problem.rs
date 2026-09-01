use crate::SolveError;
use crate::callback::{
    Callback, CallbackAction, CallbackOutcome, CallbackSave, CallbackSet, ContinuousCallback,
    DiscreteCallback, DiscreteTrigger, EventCrossing, EventDirection, InitializationHook,
    LifecycleHook, PresetTimes, StepGuard, VectorContinuousCallback,
};
use crate::event::{
    MAX_EVENT_ROOT_ITERATIONS, effective_event_tolerance, event_interval_converged,
};
use ndarray::{
    Array, ArrayView, ArrayViewD, ArrayViewMut, ArrayViewMut1, ArrayViewMut2, ArrayViewMutD,
    Dimension, IxDyn,
};

/// An initial-value ordinary differential equation problem.
///
/// The right-hand side uses the in-place SciML calling convention
/// `f(du, u, p, t)`: it must overwrite every element of `du` with the
/// derivative at `(u, p, t)`.
pub struct OdeProblem<F, P> {
    pub(crate) rhs: F,
    initial_state: Vec<f64>,
    state_shape: IxDyn,
    time_span: (f64, f64),
    parameters: P,
    jacobian: Option<Box<JacobianFunction<P>>>,
    callbacks: Vec<Callback<P>>,
    initializers: Vec<InitializationHook<P>>,
    finalizers: Vec<Box<LifecycleHook<P>>>,
    step_guards: Vec<StepGuard<P>>,
}

type JacobianFunction<P> = dyn Fn(&mut [f64], &[f64], &P, f64);
type StepInterpolator<'a> = dyn FnMut(f64, &mut [f64]) -> Result<(), SolveError> + 'a;

/// A split/IMEX ODE representation retaining explicit and implicit components.
///
/// The representation is solver-neutral: kernels choose how to combine the
/// two components, while dimensions, parameters, and time semantics remain
/// shared and checked in one place.
#[allow(dead_code)]
pub struct SplitOdeProblem<FE, FI, P> {
    explicit: FE,
    implicit: FI,
    initial_state: Vec<f64>,
    state_shape: IxDyn,
    time_span: (f64, f64),
    parameters: P,
    implicit_jacobian: Option<Box<JacobianFunction<P>>>,
    callbacks: Vec<Callback<P>>,
    initializers: Vec<InitializationHook<P>>,
    finalizers: Vec<Box<LifecycleHook<P>>>,
    step_guards: Vec<StepGuard<P>>,
}

#[allow(dead_code)]
impl SplitOdeProblem<(), (), ()> {
    /// Constructs a split problem from ndarray-shaped functions and state.
    ///
    /// Both functions receive mutable/read-only dynamic ndarray views with the
    /// same dimensionality as `initial_state`. The generated adapters are
    /// monomorphized and expose contiguous slices only to numerical kernels.
    #[allow(clippy::type_complexity)] // Preserve monomorphized RHS adapters instead of boxing.
    pub fn from_array<FE, FI, P, D>(
        explicit: FE,
        implicit: FI,
        initial_state: Array<f64, D>,
        time_span: (f64, f64),
        parameters: P,
    ) -> SplitOdeProblem<
        impl Fn(&mut [f64], &[f64], &P, f64),
        impl Fn(&mut [f64], &[f64], &P, f64),
        P,
    >
    where
        D: Dimension,
        FE: for<'a, 'b> Fn(ArrayViewMut<'a, f64, D>, ArrayView<'b, f64, D>, &P, f64),
        FI: for<'a, 'b> Fn(ArrayViewMut<'a, f64, D>, ArrayView<'b, f64, D>, &P, f64),
    {
        let rhs_shape = initial_state.raw_dim();
        let state_shape = rhs_shape.clone().into_dyn();
        let initial_state = initial_state.iter().copied().collect();
        let explicit_shape = rhs_shape.clone();
        let implicit_shape = rhs_shape;
        let explicit = move |derivative: &mut [f64], state: &[f64], parameters: &P, time| {
            let derivative = ArrayViewMut::from_shape(explicit_shape.clone(), derivative)
                .expect("split derivative shape must match its contiguous storage");
            let state = ArrayView::from_shape(explicit_shape.clone(), state)
                .expect("split state shape must match its contiguous storage");
            explicit(derivative, state, parameters, time);
        };
        let implicit = move |derivative: &mut [f64], state: &[f64], parameters: &P, time| {
            let derivative = ArrayViewMut::from_shape(implicit_shape.clone(), derivative)
                .expect("split derivative shape must match its contiguous storage");
            let state = ArrayView::from_shape(implicit_shape.clone(), state)
                .expect("split state shape must match its contiguous storage");
            implicit(derivative, state, parameters, time);
        };
        SplitOdeProblem {
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
        }
    }
}

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
        }
    }

    /// Appends an ordered callback set to this problem.
    pub fn with_callback_set(mut self, mut callback_set: CallbackSet<P>) -> Self {
        self.callbacks.append(&mut callback_set.callbacks);
        self.initializers.append(&mut callback_set.initializers);
        self.finalizers.append(&mut callback_set.finalizers);
        self.step_guards.append(&mut callback_set.step_guards);
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
    }

    pub(crate) fn domain_rejection_factor(&self, state: &[f64], time: f64) -> Option<f64> {
        self.step_guards
            .iter()
            .filter(|guard| (guard.is_out_of_domain)(state, &self.parameters, time))
            .map(|guard| guard.reduction_factor)
            .reduce(f64::min)
    }

    /// Evaluates the explicit right-hand side.
    pub fn evaluate_explicit(&self, derivative: &mut [f64], state: &[f64], time: f64)
    where
        FE: Fn(&mut [f64], &[f64], &P, f64),
    {
        (self.explicit)(derivative, state, &self.parameters, time);
    }

    /// Evaluates the implicit right-hand side.
    pub fn evaluate_implicit(&self, derivative: &mut [f64], state: &[f64], time: f64)
    where
        FI: Fn(&mut [f64], &[f64], &P, f64),
    {
        (self.implicit)(derivative, state, &self.parameters, time);
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
    ) -> Result<CallbackOutcome, SolveError> {
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
            if callback.trigger.is_triggered(state, &self.parameters, time) {
                outcome.register(callback.save);
                outcome.apply_action((callback.affect)(state, &self.parameters, time))?;
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
    ) -> Result<CallbackOutcome, SolveError> {
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
                if callback
                    .trigger
                    .is_triggered(state, &self.parameters, *time)
                {
                    if outcome.invocations == 0 {
                        state_before_effect.copy_from_slice(state);
                    }
                    outcome.register(callback.save);
                    outcome.apply_action((callback.affect)(state, &self.parameters, *time))?;
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

#[allow(dead_code)]
pub(crate) struct JacobianProvider<'a, F, P> {
    problem: &'a OdeProblem<F, P>,
}

#[allow(dead_code)]
impl<'a, F, P> JacobianProvider<'a, F, P> {
    pub(crate) fn new(problem: &'a OdeProblem<F, P>) -> Self {
        Self { problem }
    }

    pub(crate) fn evaluate(&self, jacobian: &mut [f64], state: &[f64], time: f64) -> bool {
        self.problem.evaluate_jacobian(jacobian, state, time)
    }

    pub(crate) fn is_analytic(&self) -> bool {
        self.problem.has_jacobian()
    }
}

impl OdeProblem<(), ()> {
    /// Creates an ODE problem from an ndarray-shaped function and state.
    ///
    /// The function receives mutable/read-only dynamic ndarray views with the
    /// same dimensionality as `initial_state`. The generated adapter is
    /// monomorphized and exposes contiguous slices only to numerical kernels.
    #[allow(clippy::type_complexity)] // Preserve a monomorphized RHS adapter instead of boxing.
    pub fn from_array<F, P, D>(
        rhs: F,
        initial_state: Array<f64, D>,
        time_span: (f64, f64),
        parameters: P,
    ) -> OdeProblem<impl Fn(&mut [f64], &[f64], &P, f64), P>
    where
        D: Dimension,
        F: for<'a, 'b> Fn(ArrayViewMut<'a, f64, D>, ArrayView<'b, f64, D>, &P, f64),
    {
        let rhs_shape = initial_state.raw_dim();
        let state_shape = rhs_shape.clone().into_dyn();
        let initial_state = initial_state.iter().copied().collect();
        let rhs = move |derivative: &mut [f64], state: &[f64], parameters: &P, time| {
            let derivative = ArrayViewMut::from_shape(rhs_shape.clone(), derivative)
                .expect("derivative shape must match its contiguous storage");
            let state = ArrayView::from_shape(rhs_shape.clone(), state)
                .expect("state shape must match its contiguous storage");
            rhs(derivative, state, parameters, time);
        };
        OdeProblem {
            rhs,
            initial_state,
            state_shape,
            time_span,
            parameters,
            jacobian: None,
            callbacks: Vec::new(),
            initializers: Vec::new(),
            finalizers: Vec::new(),
            step_guards: Vec::new(),
        }
    }
}

impl<F, P> OdeProblem<F, P> {
    /// Creates an ODE problem `du/dt = f(u, p, t)`.
    pub fn new(
        rhs: F,
        initial_state: impl Into<Vec<f64>>,
        time_span: (f64, f64),
        parameters: P,
    ) -> Self {
        let initial_state = initial_state.into();
        let state_shape = IxDyn(&[initial_state.len()]);
        Self {
            rhs,
            initial_state,
            state_shape,
            time_span,
            parameters,
            jacobian: None,
            callbacks: Vec::new(),
            initializers: Vec::new(),
            finalizers: Vec::new(),
            step_guards: Vec::new(),
        }
    }

    /// Appends an ordered callback set to this problem.
    pub fn with_callback_set(mut self, mut callback_set: CallbackSet<P>) -> Self {
        self.callbacks.append(&mut callback_set.callbacks);
        self.initializers.append(&mut callback_set.initializers);
        self.finalizers.append(&mut callback_set.finalizers);
        self.step_guards.append(&mut callback_set.step_guards);
        self
    }

    /// Supplies an analytic state Jacobian for implicit and Rosenbrock methods.
    ///
    /// The callback receives a row-major `dimension × dimension` output matrix
    /// and must overwrite every entry with `∂fᵢ/∂uⱼ` at `(state, parameters,
    /// time)`. Solvers that do not use a Jacobian ignore this callback.
    pub fn with_jacobian<J>(mut self, jacobian: J) -> Self
    where
        J: Fn(&mut [f64], &[f64], &P, f64) + 'static,
    {
        self.jacobian = Some(Box::new(jacobian));
        self
    }

    /// Supplies an analytic Jacobian using ndarray views.
    ///
    /// The Jacobian output is a two-dimensional `dimension × dimension`
    /// matrix. The state view retains the scalar, vector, or matrix shape used
    /// to construct this problem with [`OdeProblem::from_array`].
    pub fn with_array_jacobian<J>(mut self, jacobian: J) -> Self
    where
        J: for<'a, 'b> Fn(ArrayViewMut2<'a, f64>, ArrayViewD<'b, f64>, &P, f64) + 'static,
    {
        let state_shape = self.state_shape.clone();
        let dimension = self.initial_state.len();
        self.jacobian = Some(Box::new(move |output, state, parameters, time| {
            let output = ArrayViewMut2::from_shape((dimension, dimension), output)
                .expect("Jacobian shape must match its contiguous storage");
            let state = ArrayViewD::from_shape(state_shape.clone(), state)
                .expect("state shape must match its contiguous storage");
            jacobian(output, state, parameters, time);
        }));
        self
    }

    /// Adds a callback evaluated after every accepted step (and at the initial state).
    ///
    /// When `condition(state, parameters, time)` is true, `affect` may modify the
    /// state and may request termination. Multiple callbacks run in insertion order.
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
    /// Preset times become mandatory integration stops automatically and are
    /// validated against this problem's time span when solving begins.
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

    /// Adds a discrete callback using shape-aware ndarray state views.
    pub fn with_array_discrete_callback<C, A>(self, condition: C, affect: A) -> Self
    where
        C: for<'a> Fn(ArrayViewD<'a, f64>, &P, f64) -> bool + 'static,
        A: for<'a> Fn(ArrayViewMutD<'a, f64>, &P, f64) -> CallbackAction + 'static,
    {
        self.with_array_discrete_callback_saving(CallbackSave::After, condition, affect)
    }

    /// Adds an ndarray discrete callback with explicit saving behavior.
    pub fn with_array_discrete_callback_saving<C, A>(
        mut self,
        save: CallbackSave,
        condition: C,
        affect: A,
    ) -> Self
    where
        C: for<'a> Fn(ArrayViewD<'a, f64>, &P, f64) -> bool + 'static,
        A: for<'a> Fn(ArrayViewMutD<'a, f64>, &P, f64) -> CallbackAction + 'static,
    {
        let condition_shape = self.state_shape.clone();
        let affect_shape = self.state_shape.clone();
        self.callbacks.push(Callback::Discrete(DiscreteCallback {
            trigger: DiscreteTrigger::Condition(Box::new(move |state, parameters, time| {
                let state = ArrayViewD::from_shape(condition_shape.clone(), state)
                    .expect("callback state shape must match its contiguous storage");
                condition(state, parameters, time)
            })),
            affect: Box::new(move |state, parameters, time| {
                let state = ArrayViewMutD::from_shape(affect_shape.clone(), state)
                    .expect("callback state shape must match its contiguous storage");
                affect(state, parameters, time)
            }),
            save,
        }));
        self
    }

    /// Adds a preset-time callback using a shape-aware ndarray state view.
    pub fn with_array_preset_time_callback<A>(
        self,
        times: impl IntoIterator<Item = f64>,
        affect: A,
    ) -> Self
    where
        A: for<'a> Fn(ArrayViewMutD<'a, f64>, &P, f64) -> CallbackAction + 'static,
    {
        self.with_array_preset_time_callback_saving(times, CallbackSave::After, affect)
    }

    /// Adds an ndarray preset-time callback with explicit saving behavior.
    pub fn with_array_preset_time_callback_saving<A>(
        mut self,
        times: impl IntoIterator<Item = f64>,
        save: CallbackSave,
        affect: A,
    ) -> Self
    where
        A: for<'a> Fn(ArrayViewMutD<'a, f64>, &P, f64) -> CallbackAction + 'static,
    {
        let affect_shape = self.state_shape.clone();
        self.callbacks.push(Callback::Discrete(DiscreteCallback {
            trigger: DiscreteTrigger::PresetTimes(PresetTimes::new(times)),
            affect: Box::new(move |state, parameters, time| {
                let state = ArrayViewMutD::from_shape(affect_shape.clone(), state)
                    .expect("callback state shape must match its contiguous storage");
                affect(state, parameters, time)
            }),
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

    /// Adds a zero-crossing callback using shape-aware ndarray state views.
    pub fn with_array_continuous_callback<C, A>(self, condition: C, affect: A) -> Self
    where
        C: for<'a> Fn(ArrayViewD<'a, f64>, &P, f64) -> f64 + 'static,
        A: for<'a> Fn(ArrayViewMutD<'a, f64>, &P, f64) -> CallbackAction + 'static,
    {
        self.with_array_continuous_callback_saving(CallbackSave::Both, condition, affect)
    }

    /// Adds an ndarray zero-crossing callback with explicit saving behavior.
    pub fn with_array_continuous_callback_saving<C, A>(
        self,
        save: CallbackSave,
        condition: C,
        affect: A,
    ) -> Self
    where
        C: for<'a> Fn(ArrayViewD<'a, f64>, &P, f64) -> f64 + 'static,
        A: for<'a> Fn(ArrayViewMutD<'a, f64>, &P, f64) -> CallbackAction + 'static,
    {
        self.with_array_continuous_callback_direction_saving(
            EventDirection::Any,
            save,
            condition,
            affect,
        )
    }

    /// Adds a direction-filtered continuous callback using ndarray views.
    pub fn with_array_continuous_callback_direction<C, A>(
        self,
        direction: EventDirection,
        condition: C,
        affect: A,
    ) -> Self
    where
        C: for<'a> Fn(ArrayViewD<'a, f64>, &P, f64) -> f64 + 'static,
        A: for<'a> Fn(ArrayViewMutD<'a, f64>, &P, f64) -> CallbackAction + 'static,
    {
        self.with_array_continuous_callback_direction_saving(
            direction,
            CallbackSave::Both,
            condition,
            affect,
        )
    }

    /// Adds a direction-filtered ndarray callback with explicit saving behavior.
    pub fn with_array_continuous_callback_direction_saving<C, A>(
        mut self,
        direction: EventDirection,
        save: CallbackSave,
        condition: C,
        affect: A,
    ) -> Self
    where
        C: for<'a> Fn(ArrayViewD<'a, f64>, &P, f64) -> f64 + 'static,
        A: for<'a> Fn(ArrayViewMutD<'a, f64>, &P, f64) -> CallbackAction + 'static,
    {
        let condition_shape = self.state_shape.clone();
        let affect_shape = self.state_shape.clone();
        self.callbacks
            .push(Callback::Continuous(ContinuousCallback {
                condition: Box::new(move |state, parameters, time| {
                    let state = ArrayViewD::from_shape(condition_shape.clone(), state)
                        .expect("callback state shape must match its contiguous storage");
                    condition(state, parameters, time)
                }),
                affect: Box::new(move |state, parameters, time| {
                    let state = ArrayViewMutD::from_shape(affect_shape.clone(), state)
                        .expect("callback state shape must match its contiguous storage");
                    affect(state, parameters, time)
                }),
                direction,
                save,
            }));
        self
    }

    /// Adds a vector continuous callback using shape-aware ndarray views.
    pub fn with_array_vector_continuous_callback<C, A>(
        self,
        event_count: usize,
        condition: C,
        affect: A,
    ) -> Self
    where
        C: for<'a, 'b> Fn(ArrayViewMut1<'a, f64>, ArrayViewD<'b, f64>, &P, f64) + 'static,
        A: for<'a> Fn(ArrayViewMutD<'a, f64>, &P, f64, &[EventCrossing]) -> CallbackAction
            + 'static,
    {
        self.with_array_vector_continuous_callback_saving(
            event_count,
            CallbackSave::Both,
            condition,
            affect,
        )
    }

    /// Adds an ndarray vector callback with explicit saving behavior.
    pub fn with_array_vector_continuous_callback_saving<C, A>(
        mut self,
        event_count: usize,
        save: CallbackSave,
        condition: C,
        affect: A,
    ) -> Self
    where
        C: for<'a, 'b> Fn(ArrayViewMut1<'a, f64>, ArrayViewD<'b, f64>, &P, f64) + 'static,
        A: for<'a> Fn(ArrayViewMutD<'a, f64>, &P, f64, &[EventCrossing]) -> CallbackAction
            + 'static,
    {
        let condition_shape = self.state_shape.clone();
        let affect_shape = self.state_shape.clone();
        self.callbacks
            .push(Callback::VectorContinuous(VectorContinuousCallback::new(
                event_count,
                save,
                move |output, state, parameters, time| {
                    let output = ArrayViewMut1::from(output);
                    let state = ArrayViewD::from_shape(condition_shape.clone(), state)
                        .expect("callback state shape must match its contiguous storage");
                    condition(output, state, parameters, time);
                },
                move |state, parameters, time, events| {
                    let state = ArrayViewMutD::from_shape(affect_shape.clone(), state)
                        .expect("callback state shape must match its contiguous storage");
                    affect(state, parameters, time, events)
                },
            )));
        self
    }

    /// Adds a direction-filtered zero-crossing callback.
    ///
    /// A root is localized by bisection over the accepted step's state segment.
    /// The callback receives the localized time and interpolated state.
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

    /// Returns `(start_time, end_time)`.
    pub fn time_span(&self) -> (f64, f64) {
        self.time_span
    }

    /// Returns whether an analytic state Jacobian was supplied.
    pub fn has_jacobian(&self) -> bool {
        self.jacobian.is_some()
    }

    /// Returns whether callback policies, lifecycle hooks, or guards were supplied.
    pub fn has_callbacks(&self) -> bool {
        !self.callbacks.is_empty()
            || !self.initializers.is_empty()
            || !self.finalizers.is_empty()
            || !self.step_guards.is_empty()
    }

    pub(crate) fn domain_rejection_factor(&self, state: &[f64], time: f64) -> Option<f64> {
        self.step_guards
            .iter()
            .filter(|guard| (guard.is_out_of_domain)(state, &self.parameters, time))
            .map(|guard| guard.reduction_factor)
            .reduce(f64::min)
    }

    pub(crate) fn has_continuous_callbacks(&self) -> bool {
        self.callbacks.iter().any(|callback| {
            matches!(
                callback,
                Callback::Continuous(_) | Callback::VectorContinuous(_)
            )
        })
    }

    /// Returns the problem parameters.
    pub fn parameters(&self) -> &P {
        &self.parameters
    }

    pub(crate) fn evaluate_jacobian(&self, jacobian: &mut [f64], state: &[f64], time: f64) -> bool {
        let Some(function) = &self.jacobian else {
            return false;
        };
        function(jacobian, state, &self.parameters, time);
        true
    }

    pub(crate) fn apply_initial_callbacks(
        &self,
        state: &mut [f64],
        time: f64,
    ) -> Result<CallbackOutcome, SolveError> {
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
            if callback.trigger.is_triggered(state, &self.parameters, time) {
                outcome.register(callback.save);
                outcome.apply_action((callback.affect)(state, &self.parameters, time))?;
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
    ) -> Result<CallbackOutcome, SolveError> {
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
                if callback
                    .trigger
                    .is_triggered(state, &self.parameters, *time)
                {
                    if outcome.invocations == 0 {
                        state_before_effect.copy_from_slice(state);
                    }
                    outcome.register(callback.save);
                    outcome.apply_action((callback.affect)(state, &self.parameters, *time))?;
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

    pub(crate) fn vector_callback_lengths(&self) -> impl Iterator<Item = usize> + '_ {
        self.callbacks.iter().filter_map(|callback| {
            let Callback::VectorContinuous(callback) = callback else {
                return None;
            };
            Some(callback.event_count)
        })
    }
}

struct RootSegment<'a> {
    previous_state: &'a [f64],
    previous_time: f64,
    state: &'a [f64],
    time: f64,
}

fn locate_root<P>(
    callback: &ContinuousCallback<P>,
    segment: RootSegment<'_>,
    before: f64,
    interpolation: &mut [f64],
    parameters: &P,
    event_tolerance: f64,
    interpolator: &mut Option<&mut StepInterpolator<'_>>,
) -> Result<f64, SolveError> {
    let mut left = 0.0;
    let mut right = 1.0;
    let mut left_value = before;
    for _ in 0..MAX_EVENT_ROOT_ITERATIONS {
        let middle = 0.5 * (left + right);
        if middle == left || middle == right {
            break;
        }
        let middle_time = segment.previous_time + middle * (segment.time - segment.previous_time);
        if let Some(interpolator) = interpolator.as_mut() {
            interpolator(middle_time, interpolation)?;
        } else {
            interpolate(segment.state, segment.previous_state, middle, interpolation);
        }
        let value = (callback.condition)(interpolation, parameters, middle_time);
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
        if event_interval_converged(
            event_tolerance,
            segment.previous_time,
            segment.time,
            left,
            right,
        ) {
            break;
        }
    }
    // Return the post-crossing side of the final bracket. This prevents a
    // continuing callback from immediately detecting the same root again on
    // the next step when the midpoint lies microscopically before the root.
    Ok(right)
}

fn evaluate_vector_condition<P>(
    callback: &VectorContinuousCallback<P>,
    output: &mut [f64],
    state: &[f64],
    parameters: &P,
    time: f64,
) -> Result<(), SolveError> {
    output.fill(f64::NAN);
    (callback.condition)(output, state, parameters, time);
    output
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteCallbackCondition)
}

#[allow(clippy::too_many_arguments)]
fn locate_vector_root<P>(
    callback: &VectorContinuousCallback<P>,
    event_index: usize,
    segment: RootSegment<'_>,
    before: f64,
    interpolation: &mut [f64],
    parameters: &P,
    event_tolerance: f64,
    interpolator: &mut Option<&mut StepInterpolator<'_>>,
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
        let middle_time = segment.previous_time + middle * (segment.time - segment.previous_time);
        if let Some(interpolator) = interpolator.as_mut() {
            interpolator(middle_time, interpolation)?;
        } else {
            interpolate(segment.state, segment.previous_state, middle, interpolation);
        }
        evaluate_vector_condition(
            callback,
            condition_values,
            interpolation,
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
        if event_interval_converged(
            event_tolerance,
            segment.previous_time,
            segment.time,
            left,
            right,
        ) {
            break;
        }
    }
    Ok(right)
}

fn interpolate(state: &[f64], previous_state: &[f64], fraction: f64, output: &mut [f64]) {
    for ((output, previous), current) in output.iter_mut().zip(previous_state).zip(state) {
        *output = previous + fraction * (current - previous);
    }
}

fn ensure_finite_callback_state(state: &[f64]) -> Result<(), SolveError> {
    state
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteCallbackState)
}

#[cfg(test)]
mod tests {
    use super::{JacobianProvider, OdeProblem, SplitOdeProblem};

    #[test]
    fn jacobian_provider_reports_analytic_callbacks() {
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
            vec![1.0],
            (0.0, 1.0),
            (),
        )
        .with_jacobian(|jacobian: &mut [f64], _: &[f64], _: &(), _: f64| jacobian[0] = 2.0);
        let provider = JacobianProvider::new(&problem);
        let mut jacobian = [0.0];
        assert!(provider.is_analytic());
        assert!(provider.evaluate(&mut jacobian, &[1.0], 0.0));
        assert_eq!(jacobian, [2.0]);
    }

    #[test]
    fn split_representation_preserves_components() {
        let split = SplitOdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -u[0],
            vec![1.0],
            (0.0, 1.0),
            (),
        );
        let mut explicit = [0.0];
        let mut implicit = [0.0];
        split.evaluate_explicit(&mut explicit, &[2.0], 0.0);
        split.evaluate_implicit(&mut implicit, &[2.0], 0.0);
        assert_eq!(explicit, [2.0]);
        assert_eq!(implicit, [-2.0]);
    }
}
