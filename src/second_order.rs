use crate::{CallbackAction, EventDirection, SaveMode, SolveError, SolveOptions, SolverStats};
use thiserror::Error;

type DiscreteCondition<P> = dyn Fn(&[f64], &[f64], &P, f64) -> bool;
type ContinuousCondition<P> = dyn Fn(&[f64], &[f64], &P, f64) -> f64;
type Affect<P> = dyn Fn(&mut [f64], &mut [f64], &P, f64) -> CallbackAction;

struct DiscreteCallback<P> {
    condition: Box<DiscreteCondition<P>>,
    affect: Box<Affect<P>>,
}

struct ContinuousCallback<P> {
    condition: Box<ContinuousCondition<P>>,
    affect: Box<Affect<P>>,
    direction: EventDirection,
}

enum PartitionedCallback<P> {
    Discrete(DiscreteCallback<P>),
    Continuous(ContinuousCallback<P>),
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
        }
    }

    /// Adds a callback evaluated at the initial state and after accepted steps.
    ///
    /// Conditions and effects receive velocity before position, matching the
    /// `SecondOrderODEProblem` acceleration signature. Effects may modify both
    /// partitions and may terminate integration.
    pub fn with_discrete_callback<C, A>(mut self, condition: C, affect: A) -> Self
    where
        C: Fn(&[f64], &[f64], &P, f64) -> bool + 'static,
        A: Fn(&mut [f64], &mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        self.callbacks
            .push(PartitionedCallback::Discrete(DiscreteCallback {
                condition: Box::new(condition),
                affect: Box::new(affect),
            }));
        self
    }

    /// Adds a zero-crossing callback that triggers in either direction.
    pub fn with_continuous_callback<C, A>(self, condition: C, affect: A) -> Self
    where
        C: Fn(&[f64], &[f64], &P, f64) -> f64 + 'static,
        A: Fn(&mut [f64], &mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        self.with_continuous_callback_direction(EventDirection::Any, condition, affect)
    }

    /// Adds a direction-filtered zero-crossing callback.
    ///
    /// Roots are localized on the line segment between accepted partitioned
    /// states. This is the interpolation used by OrdinaryDiffEq's leapfrog
    /// variants; higher-order second-order interpolants are not implied.
    pub fn with_continuous_callback_direction<C, A>(
        mut self,
        direction: EventDirection,
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
            }));
        self
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
}

/// A saved trajectory for a second-order ODE.
#[derive(Clone, Debug, PartialEq)]
pub struct SecondOrderSolution {
    times: Vec<f64>,
    velocities: Vec<f64>,
    positions: Vec<f64>,
    dimension: usize,
    stats: SolverStats,
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
}

fn partition(values: &[f64], dimension: usize, index: usize) -> Option<&[f64]> {
    let start = index.checked_mul(dimension)?;
    values.get(start..start + dimension)
}

/// Configuration or integration failure specific to partitioned ODE states.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
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

/// A fixed-step algorithm for `q' = v` second-order ODE problems.
pub trait SecondOrderOdeAlgorithm {
    fn solve<F, P>(
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
    validate(problem, options)?;
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
            fn solve<F, P>(
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
    if !problem
        .initial_position
        .iter()
        .chain(&problem.initial_velocity)
        .all(|value| value.is_finite())
    {
        return Err(SolveError::NonFiniteInitialState.into());
    }
    let (start, end) = problem.time_span;
    if !start.is_finite() || !end.is_finite() || start == end {
        return Err(SolveError::InvalidTimeSpan.into());
    }
    if !options.absolute_tolerance.is_finite()
        || options.absolute_tolerance <= 0.0
        || !options.relative_tolerance.is_finite()
        || options.relative_tolerance <= 0.0
    {
        return Err(SolveError::InvalidTolerance.into());
    }
    if options
        .initial_step
        .is_some_and(|step| !step.is_finite() || step <= 0.0)
    {
        return Err(SolveError::InvalidInitialStep.into());
    }
    if options.max_step.is_nan() || options.max_step <= 0.0 {
        return Err(SolveError::InvalidMaxStep.into());
    }
    if options.max_steps == 0 {
        return Err(SolveError::InvalidMaxSteps.into());
    }
    if !options.event_tolerance.is_finite() || options.event_tolerance <= 0.0 {
        return Err(SolveError::InvalidEventTolerance.into());
    }
    let direction = (end - start).signum();
    if !options.save_at.iter().all(|time| {
        time.is_finite() && direction * (*time - start) >= 0.0 && direction * (end - *time) >= 0.0
    }) || options
        .save_at
        .windows(2)
        .any(|pair| direction * (pair[1] - pair[0]) <= 0.0)
    {
        return Err(SolveError::InvalidSaveAt.into());
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
    let maximum_step = fixed_step.min(options.max_step);
    let dimension = problem.initial_position.len();
    let (start, end) = problem.time_span;
    let direction = (end - start).signum();
    let mut velocity = problem.initial_velocity.clone();
    let mut position = problem.initial_position.clone();
    let mut workspace = Workspace::new(dimension, !problem.callbacks.is_empty());
    let mut stats = SolverStats::default();

    let initial = apply_initial_callbacks(problem, &mut velocity, &mut position, start)?;
    stats.callback_invocations += initial.invocations;
    let mut recorder = PartitionedRecorder::new(&velocity, &position, start, options);
    if initial.terminate {
        recorder.force_state(start, &velocity, &position);
        return Ok(recorder.finish(stats));
    }

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
    while direction * (end - time) > 0.0 {
        if steps == options.max_steps {
            return Err(SolveError::MaxStepsExceeded.into());
        }
        steps += 1;
        let step = direction * maximum_step.min((end - time).abs());
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
        )?;
        stats.callback_invocations += callback.invocations;
        time = next_time;
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
        );
        if callback.invocations > 0 {
            recorder.force_state(time, &velocity, &position);
        }
        if callback.terminate {
            return Ok(recorder.finish(stats));
        }
        if callback.invocations > 0 && caches_acceleration {
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
    Ok(recorder.finish(stats))
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

#[derive(Default)]
struct CallbackOutcome {
    invocations: usize,
    terminate: bool,
}

fn apply_initial_callbacks<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    velocity: &mut [f64],
    position: &mut [f64],
    time: f64,
) -> Result<CallbackOutcome, SolveError> {
    let mut outcome = CallbackOutcome::default();
    for callback in &problem.callbacks {
        let PartitionedCallback::Discrete(callback) = callback else {
            continue;
        };
        if (callback.condition)(velocity, position, &problem.parameters, time) {
            outcome.invocations += 1;
            outcome.terminate = (callback.affect)(velocity, position, &problem.parameters, time)
                == CallbackAction::Terminate;
            ensure_finite_state(velocity, position)?;
            if outcome.terminate {
                break;
            }
        }
    }
    Ok(outcome)
}

#[allow(clippy::too_many_arguments)]
fn apply_step_callbacks<F, P>(
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
) -> Result<CallbackOutcome, SolveError> {
    if problem.callbacks.is_empty() {
        return Ok(CallbackOutcome::default());
    }
    let mut outcome = CallbackOutcome::default();
    let mut root = None;
    for (index, callback) in problem.callbacks.iter().enumerate() {
        let PartitionedCallback::Continuous(callback) = callback else {
            continue;
        };
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
            )?;
            if root.is_none_or(|(_, earliest)| fraction < earliest) {
                root = Some((index, fraction));
            }
        }
    }
    if let Some((index, fraction)) = root {
        let end_time = *time;
        interpolate(velocity, previous_velocity, fraction, state_before_velocity);
        interpolate(position, previous_position, fraction, state_before_position);
        velocity.copy_from_slice(state_before_velocity);
        position.copy_from_slice(state_before_position);
        *time = previous_time + fraction * (end_time - previous_time);
        let PartitionedCallback::Continuous(callback) = &problem.callbacks[index] else {
            unreachable!();
        };
        outcome.invocations += 1;
        outcome.terminate = (callback.affect)(velocity, position, &problem.parameters, *time)
            == CallbackAction::Terminate;
        ensure_finite_state(velocity, position)?;
    }
    if !outcome.terminate {
        for callback in &problem.callbacks {
            let PartitionedCallback::Discrete(callback) = callback else {
                continue;
            };
            if (callback.condition)(velocity, position, &problem.parameters, *time) {
                if outcome.invocations == 0 {
                    state_before_velocity.copy_from_slice(velocity);
                    state_before_position.copy_from_slice(position);
                }
                outcome.invocations += 1;
                outcome.terminate =
                    (callback.affect)(velocity, position, &problem.parameters, *time)
                        == CallbackAction::Terminate;
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
) -> Result<f64, SolveError> {
    let mut left = 0.0;
    let mut right = 1.0;
    let mut left_value = before;
    for _ in 0..64 {
        let middle = 0.5 * (left + right);
        interpolate(velocity, previous_velocity, middle, interpolation_velocity);
        interpolate(position, previous_position, middle, interpolation_position);
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
        if (right - left) * (time - previous_time).abs() <= event_tolerance {
            break;
        }
    }
    Ok(0.5 * (left + right))
}

fn interpolate(current: &[f64], previous: &[f64], fraction: f64, output: &mut [f64]) {
    for ((output, previous), current) in output.iter_mut().zip(previous).zip(current) {
        *output = previous + fraction * (current - previous);
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
    ) {
        if self.save_at.is_empty() {
            if self.save_mode == SaveMode::EveryStep || final_time {
                self.push_unique(time, velocity, position);
            }
            return;
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
            let fraction = (target - previous_time) / (time - previous_time);
            interpolate(
                velocity,
                previous_velocity,
                fraction,
                &mut self.interpolation_velocity,
            );
            interpolate(
                position,
                previous_position,
                fraction,
                &mut self.interpolation_position,
            );
            self.times.push(target);
            self.velocities
                .extend_from_slice(&self.interpolation_velocity);
            self.positions
                .extend_from_slice(&self.interpolation_position);
            self.next_save += 1;
        }
    }

    fn force_state(&mut self, time: f64, velocity: &[f64], position: &[f64]) {
        self.push_unique(time, velocity, position);
    }

    fn push_unique(&mut self, time: f64, velocity: &[f64], position: &[f64]) {
        if self.times.last() == Some(&time) {
            let start = self.velocities.len() - self.dimension;
            self.velocities[start..].copy_from_slice(velocity);
            self.positions[start..].copy_from_slice(position);
        } else {
            self.times.push(time);
            self.velocities.extend_from_slice(velocity);
            self.positions.extend_from_slice(position);
        }
    }

    fn finish(self, stats: SolverStats) -> SecondOrderSolution {
        SecondOrderSolution {
            times: self.times,
            velocities: self.velocities,
            positions: self.positions,
            dimension: self.dimension,
            stats,
        }
    }
}
