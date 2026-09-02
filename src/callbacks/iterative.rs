use std::rc::Rc;

use super::validate_time_span;
use crate::callback::{CallbackOutcome, IterativeTimes};
use crate::solvers::second_order::SecondOrderCallbackSet;
use crate::{CallbackAction, CallbackSave, CallbackSet, ConfigurationError, SolveError};

/// Schedules each next effect from the current state and integration time.
///
/// The time-choice function is called during initialization and after each
/// effect, returning an absolute time or `None` to end the schedule. Its
/// decision becomes an exact integration stop and is not recomputed during
/// intermediate steps. After an effect it sees the updated state and
/// parameters. Returning a time beyond the configured endpoint ends the
/// schedule; returning a non-finite, repeated, or backward-moving time is a
/// [`SolveError::InvalidIterativeCallbackTime`]. "Forward" follows the solve's
/// integration direction, including for backward solves.
///
/// The schedule stores only one pending time and is reset for every solve.
/// Scheduling is observation-only: the time-choice function must not mutate
/// right-hand-side inputs through parameter interior mutability. Effects may
/// mutate state and parameters using the usual [`CallbackAction`] contract.
/// No further time is requested after an effect terminates the solve or runs
/// at the configured endpoint.
///
/// # Example
///
/// ```
/// use differential_equations::callbacks::IterativeCallback;
/// use differential_equations::solvers::explicit::Euler;
/// use differential_equations::{CallbackAction, OdeProblem, SolveOptions, solve};
///
/// let span = (0.0, 1.0);
/// let events = IterativeCallback::new(|_: &[f64], _: &(), time: f64| Some(time + 0.25))
///     .into_callback_set(span, |state, _, _| {
///         state[0] += 1.0;
///         CallbackAction::Continue
///     })?;
/// let problem = OdeProblem::new(
///     |du: &mut [f64], _: &[f64], _: &(), _| du.fill(0.0),
///     [0.0], span, (),
/// ).with_callback_set(events);
/// let solution = solve(&problem, Euler, &SolveOptions::new()
///     .with_adaptive(false).with_initial_step(0.7))?;
/// assert_eq!(solution.last_state(), &[4.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct IterativeCallback<S> {
    time_choice: S,
    initial_affect: bool,
    save: CallbackSave,
}

impl<S> IterativeCallback<S> {
    /// Creates a scheduler that chooses its first time without firing an effect.
    pub const fn new(time_choice: S) -> Self {
        Self {
            time_choice,
            initial_affect: false,
            save: CallbackSave::After,
        }
    }

    /// Selects whether an initial effect precedes the first time-choice call.
    pub const fn with_initial_affect(mut self, enabled: bool) -> Self {
        self.initial_affect = enabled;
        self
    }

    /// Selects which states are saved around each effect, not scheduling calls.
    pub const fn with_save(mut self, save: CallbackSave) -> Self {
        self.save = save;
        self
    }

    /// Builds an ordinary or split first-order callback for the given time span.
    ///
    /// The span must match the eventual problem. State arguments use contiguous
    /// flat slices, including for ndarray-shaped problems.
    pub fn into_callback_set<P, A>(
        self,
        time_span: (f64, f64),
        affect: A,
    ) -> Result<CallbackSet<P>, ConfigurationError>
    where
        S: Fn(&[f64], &P, f64) -> Option<f64> + 'static,
        A: Fn(&mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        validate_time_span(time_span)?;
        let times = Rc::new(IterativeTimes::new(time_span));
        let initial_times = Rc::clone(&times);
        let effect_times = Rc::clone(&times);
        let choice = Rc::new(self.time_choice);
        let initial_choice = Rc::clone(&choice);
        let initial_affect = self.initial_affect;
        Ok(CallbackSet::new().with_iterative_callback(
            times,
            self.save,
            move |state, parameters, time| {
                initial_times.initialize(time, initial_affect)?;
                if !initial_affect {
                    initial_times.schedule(time, initial_choice(state, parameters, time))?;
                }
                Ok(())
            },
            move |state, parameters, time| {
                let action = affect(state, parameters, time);
                finish_effect(
                    &effect_times,
                    time,
                    action,
                    state.iter().all(|value| value.is_finite()),
                    || choice(state, parameters, time),
                )
            },
        ))
    }

    /// Builds a partitioned callback receiving velocity before position.
    ///
    /// The span must match the eventual second-order problem.
    pub fn into_second_order_callback_set<P, A>(
        self,
        time_span: (f64, f64),
        affect: A,
    ) -> Result<SecondOrderCallbackSet<P>, ConfigurationError>
    where
        S: Fn(&[f64], &[f64], &P, f64) -> Option<f64> + 'static,
        A: Fn(&mut [f64], &mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        validate_time_span(time_span)?;
        let times = Rc::new(IterativeTimes::new(time_span));
        let initial_times = Rc::clone(&times);
        let effect_times = Rc::clone(&times);
        let choice = Rc::new(self.time_choice);
        let initial_choice = Rc::clone(&choice);
        let initial_affect = self.initial_affect;
        Ok(SecondOrderCallbackSet::new().with_iterative_callback(
            times,
            self.save,
            move |velocity, position, parameters, time| {
                initial_times.initialize(time, initial_affect)?;
                if !initial_affect {
                    initial_times
                        .schedule(time, initial_choice(velocity, position, parameters, time))?;
                }
                Ok(())
            },
            move |velocity, position, parameters, time| {
                let action = affect(velocity, position, parameters, time);
                finish_effect(
                    &effect_times,
                    time,
                    action,
                    velocity
                        .iter()
                        .chain(position.iter())
                        .all(|value| value.is_finite()),
                    || choice(velocity, position, parameters, time),
                )
            },
        ))
    }
}

fn finish_effect(
    times: &IterativeTimes,
    time: f64,
    action: CallbackAction,
    finite_state: bool,
    choose: impl FnOnce() -> Option<f64>,
) -> Result<CallbackAction, SolveError> {
    if !finite_state {
        return Err(SolveError::NonFiniteCallbackState);
    }
    let mut outcome = CallbackOutcome::default();
    outcome.apply_action(action)?;
    let next = if outcome.terminate || times.finished(time) {
        None
    } else {
        choose()
    };
    times.schedule(time, next)?;
    Ok(action)
}
