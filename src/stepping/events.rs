//! Terminal scalar root finding for output-free RK propagation.
use super::{
    AdaptiveController, ExplicitRungeKuttaStepper, IntegrationError, IntegrationOutcome,
    Observation, ObserverAction, StepError, StepFailure, endpoint_step,
};

/// Direction of a sign change in integration order (also for backward solves).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootDirection {
    /// Either sign change.
    Any,
    /// Negative to nonnegative.
    Rising,
    /// Positive to nonpositive.
    Falling,
}

/// Scalar terminal-event localization policy.
#[derive(Clone, Copy, Debug)]
pub struct RootOptions {
    /// Absolute tolerance in independent-variable units, strictly positive.
    pub time_tolerance: f64,
    /// Absolute condition tolerance; zero requires a representable exact zero.
    pub value_tolerance: f64,
    /// Maximum bisection attempts within a sign-changing accepted interval.
    pub maximum_iterations: usize,
    /// Sign direction in integration order.
    pub direction: RootDirection,
}
impl Default for RootOptions {
    fn default() -> Self {
        Self {
            time_tolerance: 1e-10,
            value_tolerance: 0.0,
            maximum_iterations: 64,
            direction: RootDirection::Any,
        }
    }
}

/// Terminal-event driver result; the state remains in its original workspace.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootOutcome {
    /// Ordinary integration statistics/outcome.
    pub integration: IntegrationOutcome,
    /// Condition at the localized terminal event, or none when no root occurred.
    pub event_value: Option<f64>,
}

/// Event configuration, condition, or numerical integration failure.
#[derive(Debug, thiserror::Error)]
pub enum RootError<E> {
    /// Underlying solver/controller/application error with original payload.
    #[error(transparent)]
    Integration(#[from] IntegrationError<E>),
    /// Tolerances or localization iteration count are invalid.
    #[error("invalid root localization options")]
    Configuration,
    /// The event condition returned a nonfinite value.
    #[error("nonfinite event condition")]
    NonFiniteCondition,
    /// The bracket could not be localized within the requested iteration budget.
    #[error("root localization iteration limit exceeded")]
    IterationLimit,
}

fn crosses(a: f64, b: f64, direction: RootDirection) -> bool {
    let rising = a < 0.0 && b >= 0.0;
    let falling = a > 0.0 && b <= 0.0;
    match direction {
        RootDirection::Any => rising || falling,
        RootDirection::Rising => rising,
        RootDirection::Falling => falling,
    }
}
fn failure<E>(error: StepFailure) -> RootError<E> {
    RootError::Integration(IntegrationError::Step(StepError::Solver(error)))
}
fn application<E>(error: E) -> RootError<E> {
    RootError::Integration(IntegrationError::Step(StepError::User(error)))
}

/// Integrate to the first scalar sign-change root without retaining a trajectory.
///
/// Accepted-step and requested-output observations follow [`super::integrate_rk`].
/// A root stops before any later requested times; the observer receives its
/// accepted state once. An initial condition within `value_tolerance` stops
/// immediately. Tangential roots without a sign change are not guaranteed.
///
/// Localization recomputes shorter RK attempts from the bracket's initial
/// state, so it does not silently substitute linear dense output. These attempts
/// reuse workspace, are counted as attempts/rejections, and do not advance
/// controller history. A root candidate must also satisfy the error policy.
/// Error-control history records only the finally accepted root interval.
/// maximum_attempts includes every ordinary and localization attempt;
/// maximum_iterations independently bounds a single root search.
/// The caller may reset state/caches and controller after applying event effects.
/// All closures may borrow, mutate, and return the same application error type.
#[allow(clippy::too_many_arguments)]
pub fn integrate_rk_until_event<F, N, O, C, E>(
    stepper: &mut ExplicitRungeKuttaStepper<'_>,
    controller: &mut AdaptiveController,
    endpoint: f64,
    requested_times: &[f64],
    maximum_attempts: usize,
    options: RootOptions,
    rhs: &mut F,
    norm: &mut N,
    condition: &mut C,
    observer: &mut O,
) -> Result<RootOutcome, RootError<E>>
where
    F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), E>,
    N: FnMut(&[f64], &[f64], &[f64]) -> Result<f64, E>,
    O: FnMut(Observation<'_>) -> Result<ObserverAction, E>,
    C: FnMut(f64, &[f64]) -> Result<f64, E>,
{
    if !options.time_tolerance.is_finite()
        || options.time_tolerance <= 0.0
        || !options.value_tolerance.is_finite()
        || options.value_tolerance < 0.0
        || options.maximum_iterations == 0
    {
        return Err(RootError::Configuration);
    }
    if !endpoint.is_finite() {
        return Err(failure(StepFailure::NonFinite));
    }
    let start = stepper.time();
    let direction = (endpoint - start).signum();
    let mut previous = start;
    for &time in requested_times {
        if !time.is_finite()
            || (time - previous) * direction < 0.0
            || (time - start) * direction < 0.0
            || (endpoint - time) * direction < 0.0
            || (endpoint == start && time != start)
        {
            return Err(IntegrationError::RequestedTimes.into());
        }
        previous = time;
    }
    let mut value = condition(start, stepper.state()).map_err(application)?;
    if !value.is_finite() {
        return Err(RootError::NonFiniteCondition);
    }
    let mut index = 0;
    while requested_times.get(index) == Some(&start) {
        index += 1;
    }
    let initial_root = value.abs() <= options.value_tolerance;
    if index > 0 || initial_root || start == endpoint {
        let stop = observer(Observation {
            time: start,
            state: stepper.state(),
            requested: index > 0,
            endpoint: start == endpoint,
        })
        .map_err(application)?
            == ObserverAction::Stop;
        if stop || initial_root || start == endpoint {
            return Ok(RootOutcome {
                integration: IntegrationOutcome {
                    time: start,
                    interrupted: stop || initial_root,
                    forced_acceptances: 0,
                },
                event_value: initial_root.then_some(value),
            });
        }
    }
    let mut forced = 0;
    let mut remaining_attempts = maximum_attempts;
    while remaining_attempts > 0 {
        let target = requested_times.get(index).copied().unwrap_or(endpoint);
        let start = stepper.time();
        let h = endpoint_step(start, target, controller.next_step()).map_err(failure)?;
        let mut next_value = 0.0;
        remaining_attempts -= 1;
        let error = stepper.attempt_with_norm(target, h, rhs, &mut |old, new, e| {
            let error = norm(old, new, e)?;
            next_value = condition(start + h, new)?;
            Ok(error)
        })?;
        if !next_value.is_finite() {
            stepper.reject().map_err(failure)?;
            return Err(RootError::NonFiniteCondition);
        }
        let mut trial_controller = controller.clone();
        let decision = match trial_controller.assess(h, error) {
            Ok(d) => d,
            Err(e) => {
                stepper.reject().map_err(failure)?;
                return Err(IntegrationError::Controller(e).into());
            }
        };
        if !decision.accepted {
            *controller = trial_controller;
            stepper.reject().map_err(failure)?;
            continue;
        }
        let has_root = crosses(value, next_value, options.direction);
        if has_root && next_value.abs() > options.value_tolerance {
            stepper.reject().map_err(failure)?;
            let (mut low, mut high, mut low_value) = (0.0, h, value);
            let mut located = false;
            let mut retry_adaptive = false;
            for _ in 0..options.maximum_iterations {
                // Localize at a representable absolute time and use its actual
                // distance from the accepted state in the numerical formula.
                let mut middle = (start + (low + (high - low) * 0.5)) - start;
                if start + middle == start {
                    middle = high;
                }
                if remaining_attempts == 0 {
                    return Err(IntegrationError::AttemptLimit.into());
                }
                remaining_attempts -= 1;
                let error = stepper.attempt_with_norm(
                    start + middle,
                    middle,
                    rhs,
                    &mut |old, new, e| {
                        let error = norm(old, new, e)?;
                        next_value = condition(start + middle, new)?;
                        Ok(error)
                    },
                )?;
                if !next_value.is_finite() {
                    stepper.reject().map_err(failure)?;
                    return Err(RootError::NonFiniteCondition);
                }
                // Endpoint arithmetic can round the proposed interval at large epochs.
                // Decisions and bracket updates must use the interval actually attempted.
                middle = stepper.pending_step().expect("successful root attempt");
                let mut root_controller = controller.clone();
                let root_decision = match root_controller.assess(middle, error) {
                    Ok(d) => d,
                    Err(e) => {
                        stepper.reject().map_err(failure)?;
                        return Err(IntegrationError::Controller(e).into());
                    }
                };
                if !root_decision.accepted {
                    // Local error need not decrease monotonically. Restart
                    // ordinary adaptive stepping using this smaller proposal.
                    *controller = root_controller;
                    stepper.reject().map_err(failure)?;
                    retry_adaptive = true;
                    break;
                }
                if next_value.abs() <= options.value_tolerance
                    || (high - low).abs() <= options.time_tolerance
                    || start + middle == start + low
                    || start + middle == start + high
                {
                    *controller = root_controller;
                    forced += usize::from(root_decision.forced);
                    located = true;
                    break;
                }
                if crosses(low_value, next_value, RootDirection::Any) {
                    high = middle;
                } else {
                    low = middle;
                    low_value = next_value;
                }
                stepper.reject().map_err(failure)?;
            }
            if !located {
                if stepper.pending_step().is_some() {
                    stepper.reject().map_err(failure)?;
                }
                // A controller rejection above supplies a valid retry; an
                // exhausted localization budget must be reported explicitly.
                if retry_adaptive {
                    continue;
                }
                return Err(RootError::IterationLimit);
            }
        } else {
            *controller = trial_controller;
            forced += usize::from(decision.forced);
        }
        stepper.accept().map_err(failure)?;
        value = next_value;
        let requested = requested_times.get(index) == Some(&stepper.time());
        while requested_times.get(index) == Some(&stepper.time()) {
            index += 1;
        }
        let done = stepper.time() == endpoint;
        let stopped = observer(Observation {
            time: stepper.time(),
            state: stepper.state(),
            requested,
            endpoint: done,
        })
        .map_err(application)?
            == ObserverAction::Stop;
        if has_root || stopped || done {
            return Ok(RootOutcome {
                integration: IntegrationOutcome {
                    time: stepper.time(),
                    interrupted: has_root || stopped,
                    forced_acceptances: forced,
                },
                event_value: has_root.then_some(value),
            });
        }
    }
    Err(IntegrationError::AttemptLimit.into())
}
