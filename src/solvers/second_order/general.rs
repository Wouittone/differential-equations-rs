mod arrays;
mod driver;

use driver::{
    StructuralParameters, interpolate, solve_fixed, solve_irkn, solve_newmark, solve_rkn_adaptive,
    solve_rkn_fixed, validate,
};
pub(super) use driver::{apply_finalize_callbacks, apply_initial_callbacks, apply_step_callbacks};

use super::function::SecondOrderFunction;
use crate::callbacks::SteadyStateCondition;
use ndarray::IxDyn;
use std::cell::RefCell;
use std::rc::Rc;

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
use crate::solution::{
    finite_partitioned_interpolation, interpolate_value, interpolation_fraction,
};
use crate::solver::{
    validate_preset_time_sequences, validate_state_time_options, validate_vector_callback_lengths,
};
use crate::tableau::{
    IrknBootstrapSeed, IrknTableau, LazyRungeKuttaNystromTableau, RungeKuttaNystromKind,
    RungeKuttaNystromTableau, TableauError, define_irkn_tableau_from_file,
    define_rkn_tableau_from_file, load_tableau,
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
/// [`Self::from_array`] accepts ndarray states, and
/// [`Self::from_array_out_of_place`] accepts functions returning accelerations.
/// This represents SciML's `SecondOrderODEProblem` specialization `q' = v`,
/// not a general `DynamicalODEProblem` with a separately supplied position
/// rate.
pub struct SecondOrderOdeProblem<F, P> {
    pub(crate) acceleration: F,
    initial_velocity: Vec<f64>,
    initial_position: Vec<f64>,
    state_shape: IxDyn,
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
    state_shape: IxDyn,
    stats: SolverStats,
    dense_segments: Vec<PartitionedDenseSegment>,
}

impl SecondOrderSolution {
    /// Constructs a solution from already-saved velocity and position states.
    ///
    /// Each partition contains one flattened row-major state for every entry
    /// in `times`. `state_shape` is the shared logical ndarray shape; an empty
    /// shape denotes a scalar. The constructor checks shape arithmetic, buffer
    /// lengths, finiteness, and monotonic time order. The resulting solution
    /// has no retained method-specific dense segments, so interpolation
    /// between saved states uses the saved-state fallback.
    ///
    /// This is the construction seam for downstream
    /// [`SecondOrderOdeAlgorithm`] implementations.
    pub fn from_saved(
        times: Vec<f64>,
        velocities: Vec<f64>,
        positions: Vec<f64>,
        state_shape: &[usize],
        stats: SolverStats,
    ) -> Result<Self, crate::SolutionConstructionError> {
        let dimension = crate::solution::validate_saved_solution(
            &times,
            state_shape,
            &[&velocities, &positions],
        )?;
        Ok(Self {
            times,
            velocities,
            positions,
            dimension,
            state_shape: IxDyn(state_shape),
            stats,
            dense_segments: Vec::new(),
        })
    }

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

    fn set_state_shape_checked(
        &mut self,
        state_shape: &[usize],
    ) -> Result<(), crate::SolutionConstructionError> {
        if crate::solution::checked_state_dimension(state_shape)? != self.dimension {
            return Err(crate::SolutionConstructionError::DimensionMismatch);
        }
        self.state_shape = IxDyn(state_shape);
        Ok(())
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
                let velocity = self
                    .velocity(index)
                    .ok_or(InterpolationError::InvalidSegmentData {
                        context: "saved second-order velocity",
                    })?
                    .to_vec();
                let position = self
                    .position(index)
                    .ok_or(InterpolationError::InvalidSegmentData {
                        context: "saved second-order position",
                    })?
                    .to_vec();
                return finite_partitioned_interpolation(
                    velocity,
                    position,
                    "saved second-order state",
                );
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
                return finite_partitioned_interpolation(
                    velocity,
                    position,
                    "second-order dense segment",
                );
            }
        }
        for index in 1..self.times.len() {
            let left = self.times[index - 1];
            let right = self.times[index];
            if between(time, left, right) && left != right {
                let fraction = interpolation_fraction(time, left, right).clamp(0.0, 1.0);
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
                return finite_partitioned_interpolation(
                    velocity,
                    position,
                    "saved second-order interpolation",
                );
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
            velocity[index] =
                interpolate_value(self.start_velocity[index], self.end_velocity[index], theta);
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
    let end = start.checked_add(dimension)?;
    values.get(start..end)
}

/// Configuration or integration failure specific to partitioned ODE states.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
#[non_exhaustive]
pub enum SecondOrderSolveError {
    /// Position and velocity partitions do not have the same dimension.
    #[error("position and velocity dimensions must match")]
    StateDimensionMismatch,
    /// An algorithm produced malformed saved trajectory data.
    #[error("invalid saved solution: {0}")]
    InvalidSolution(#[from] crate::SolutionConstructionError),
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
        F: SecondOrderFunction<P>,
    {
        validate(problem, options)?;
        let mut solution = self.solve_validated(problem, options)?;
        solution.set_state_shape_checked(problem.state_shape())?;
        Ok(solution)
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
        F: SecondOrderFunction<P>;
}

/// Solves a second-order ODE without flattening its position and velocity.
pub fn solve_second_order<F, P, A>(
    problem: &SecondOrderOdeProblem<F, P>,
    algorithm: A,
    options: &SolveOptions,
) -> Result<SecondOrderSolution, SecondOrderSolveError>
where
    F: SecondOrderFunction<P>,
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

/// A generic second-order algorithm backed by a validated RKN resource.
///
/// Most users create a named zero-sized algorithm with
/// [`crate::tableau::define_rkn_from_file!`]. This wrapper is public so other
/// compile-time tooling can reuse the built-in fixed and adaptive RKN drivers.
#[derive(Clone, Copy)]
pub struct ResourceRungeKuttaNystrom {
    resource: &'static LazyRungeKuttaNystromTableau,
}

impl ResourceRungeKuttaNystrom {
    /// Wraps one compile-time-validated lazy RKN tableau.
    pub const fn new(resource: &'static LazyRungeKuttaNystromTableau) -> Self {
        Self { resource }
    }

    /// Returns the lazily initialized tableau.
    pub fn tableau(self) -> Result<&'static RungeKuttaNystromTableau, TableauError> {
        load_tableau(self.resource)
    }
}

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

define_rkn_tableau_from_file!(
    pub(super) NYSTROM4_TABLEAU,
    "Nystrom4",
    "src/tableau/resources/second_order/nystrom4.json",
    crate = crate
);
define_rkn_tableau_from_file!(
    pub(super) NYSTROM4_VI_TABLEAU,
    "Nystrom4VelocityIndependent",
    "src/tableau/resources/second_order/nystrom4-velocity-independent.json",
    crate = crate
);
define_rkn_tableau_from_file!(
    pub(super) NYSTROM5_VI_TABLEAU,
    "Nystrom5VelocityIndependent",
    "src/tableau/resources/second_order/nystrom5-velocity-independent.json",
    crate = crate
);
define_rkn_tableau_from_file!(
    pub(super) RKN4_TABLEAU,
    "Rkn4",
    "src/tableau/resources/second_order/rkn4.json",
    crate = crate
);
define_rkn_tableau_from_file!(
    pub(super) DPRKN4_ADAPTIVE_TABLEAU,
    "Dprkn4",
    "src/tableau/resources/second_order/dprkn4.json",
    crate = crate
);
define_rkn_tableau_from_file!(
    pub(super) DPRKN5_ADAPTIVE_TABLEAU,
    "Dprkn5",
    "src/tableau/resources/second_order/dprkn5.json",
    crate = crate
);
define_rkn_tableau_from_file!(
    pub(super) DPRKN6_ADAPTIVE_TABLEAU,
    "Dprkn6",
    "src/tableau/resources/second_order/dprkn6.json",
    crate = crate
);
define_rkn_tableau_from_file!(
    pub(super) DPRKN6FM_ADAPTIVE_TABLEAU,
    "Dprkn6Fm",
    "src/tableau/resources/second_order/dprkn6fm.json",
    crate = crate
);
define_rkn_tableau_from_file!(
    pub(super) DPRKN8_ADAPTIVE_TABLEAU,
    "Dprkn8",
    "src/tableau/resources/second_order/dprkn8.json",
    crate = crate
);
define_rkn_tableau_from_file!(
    pub(super) DPRKN12_ADAPTIVE_TABLEAU,
    "Dprkn12",
    "src/tableau/resources/second_order/dprkn12.json",
    crate = crate
);
define_rkn_tableau_from_file!(
    pub(super) ERKN4_ADAPTIVE_TABLEAU,
    "Erkn4",
    "src/tableau/resources/second_order/erkn4.json",
    crate = crate
);
define_rkn_tableau_from_file!(
    pub(super) ERKN5_ADAPTIVE_TABLEAU,
    "Erkn5",
    "src/tableau/resources/second_order/erkn5.json",
    crate = crate
);
define_rkn_tableau_from_file!(
    pub(super) ERKN7_ADAPTIVE_TABLEAU,
    "Erkn7",
    "src/tableau/resources/second_order/erkn7.json",
    crate = crate
);
define_rkn_tableau_from_file!(
    pub(super) FINERKN4_ADAPTIVE_TABLEAU,
    "FineRkn4",
    "src/tableau/resources/second_order/fine-rkn4.json",
    crate = crate
);
define_rkn_tableau_from_file!(
    pub(super) FINERKN5_ADAPTIVE_TABLEAU,
    "FineRkn5",
    "src/tableau/resources/second_order/fine-rkn5.json",
    crate = crate
);
define_irkn_tableau_from_file!(
    pub(super) IRKN3_TABLEAU,
    "Irkn3",
    "src/tableau/resources/second_order/irkn3.json",
    crate = crate
);
define_irkn_tableau_from_file!(
    pub(super) IRKN4_TABLEAU,
    "Irkn4",
    "src/tableau/resources/second_order/irkn4.json",
    crate = crate
);

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
                F: SecondOrderFunction<P>,
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
    ($algorithm:ty, $tableau:ident) => {
        impl $algorithm {
            /// Returns this method's lazily initialized, validated tableau.
            pub fn tableau(self) -> Result<&'static RungeKuttaNystromTableau, TableauError> {
                load_tableau(&$tableau)
            }
        }

        impl SecondOrderOdeAlgorithm for $algorithm {
            fn solve_validated<F, P>(
                &self,
                problem: &SecondOrderOdeProblem<F, P>,
                options: &SolveOptions,
            ) -> Result<SecondOrderSolution, SecondOrderSolveError>
            where
                F: SecondOrderFunction<P>,
            {
                let tableau = self.tableau().map_err(|_| SolveError::InvalidTableau)?;
                if tableau.kind() != RungeKuttaNystromKind::Fixed {
                    return Err(SolveError::InvalidTableau.into());
                }
                solve_rkn_fixed(problem, options, tableau)
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
        impl $algorithm {
            /// Returns this method's lazily initialized, validated tableau.
            pub fn tableau(self) -> Result<&'static RungeKuttaNystromTableau, TableauError> {
                load_tableau(&$tableau)
            }
        }

        impl SecondOrderOdeAlgorithm for $algorithm {
            fn solve_validated<F, P>(
                &self,
                problem: &SecondOrderOdeProblem<F, P>,
                options: &SolveOptions,
            ) -> Result<SecondOrderSolution, SecondOrderSolveError>
            where
                F: SecondOrderFunction<P>,
            {
                let tableau = self.tableau().map_err(|_| SolveError::InvalidTableau)?;
                if tableau.kind() != RungeKuttaNystromKind::Adaptive {
                    return Err(SolveError::InvalidTableau.into());
                }
                solve_rkn_adaptive(problem, options, tableau)
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

impl SecondOrderOdeAlgorithm for ResourceRungeKuttaNystrom {
    fn solve_validated<F, P>(
        &self,
        problem: &SecondOrderOdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<SecondOrderSolution, SecondOrderSolveError>
    where
        F: SecondOrderFunction<P>,
    {
        let tableau = self.tableau().map_err(|_| SolveError::InvalidTableau)?;
        match tableau.kind() {
            RungeKuttaNystromKind::Fixed => solve_rkn_fixed(problem, options, tableau),
            RungeKuttaNystromKind::Adaptive => solve_rkn_adaptive(problem, options, tableau),
        }
    }
}

impl SecondOrderOdeAlgorithm for NewmarkBeta {
    fn solve_validated<F, P>(
        &self,
        problem: &SecondOrderOdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<SecondOrderSolution, SecondOrderSolveError>
    where
        F: SecondOrderFunction<P>,
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
        F: SecondOrderFunction<P>,
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

macro_rules! impl_irkn_algorithm {
    ($algorithm:ty, $tableau:ident) => {
        impl $algorithm {
            /// Returns this method's lazily initialized, validated tableau.
            pub fn tableau(self) -> Result<&'static IrknTableau, TableauError> {
                load_tableau(&$tableau)
            }
        }

        impl SecondOrderOdeAlgorithm for $algorithm {
            fn solve_validated<F, P>(
                &self,
                problem: &SecondOrderOdeProblem<F, P>,
                options: &SolveOptions,
            ) -> Result<SecondOrderSolution, SecondOrderSolveError>
            where
                F: SecondOrderFunction<P>,
            {
                let tableau = self.tableau().map_err(|_| SolveError::InvalidTableau)?;
                let bootstrap = Nystrom4VelocityIndependent
                    .tableau()
                    .map_err(|_| SolveError::InvalidTableau)?;
                solve_irkn(problem, options, tableau, bootstrap)
            }
        }
    };
}

impl_irkn_algorithm!(Irkn3, IRKN3_TABLEAU);
impl_irkn_algorithm!(Irkn4, IRKN4_TABLEAU);
