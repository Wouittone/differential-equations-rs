use super::{
    AdaptiveController, ControllerError, ExplicitRungeKuttaStepper, StepError, StepFailure,
};

/// Accepted-step observer decision. Captures may borrow and mutate caller data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObserverAction {
    /// Continue towards the endpoint.
    Continue,
    /// Stop successfully at this accepted state.
    Stop,
}
/// A requested-output or accepted-step observation.
#[derive(Clone, Copy, Debug)]
pub struct Observation<'a> {
    /// Accepted time.
    pub time: f64,
    /// Accepted state, borrowed without allocating.
    pub state: &'a [f64],
    /// True when this time was explicitly requested.
    pub requested: bool,
    /// True at the integration endpoint.
    pub endpoint: bool,
}
/// Failure from the optional reusable adaptive driver.
#[derive(Debug, thiserror::Error)]
pub enum IntegrationError<E> {
    /// A numerical attempt, observer or norm failed.
    #[error(transparent)]
    Step(#[from] StepError<E>),
    /// Controller policy could not provide an admissible next step.
    #[error(transparent)]
    Controller(#[from] ControllerError),
    /// Attempt budget was exhausted.
    #[error("integration attempt limit exceeded")]
    AttemptLimit,
    /// Requested times are not finite, ordered and within this interval.
    #[error("invalid requested output times")]
    RequestedTimes,
    /// Adaptive control requires an embedded component-error formula.
    #[error("method has no embedded error formula")]
    MissingErrorEstimate,
}
/// Reusable driver output; accepted state remains in the stepper.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntegrationOutcome {
    /// Reached time, possibly shortened by an observer interruption.
    pub time: f64,
    /// Whether the observer requested a successful early stop.
    pub interrupted: bool,
    /// Number of explicit minimum-step forced acceptances.
    pub forced_acceptances: usize,
}
/// Integrate using reusable state, caller-controlled norm and an output observer.
///
/// The norm is applied to the primary error vector and, if present, once to
/// the secondary vector. Their maximum reproduces this crate's native generic
/// RK estimator convention (including BS5/DP8); it is not a claim of upstream
/// DOP853 E5/E3 compound-estimator equivalence. Hosts needing a different
/// compound formula must drive the low-level step view containing both vectors.
/// No trajectory is retained. `requested_times` are exact step boundaries and
/// are never suppressed; each accepted step is observed once, with a flag for
/// requested outputs. Caller-owned output buffers can be filled by the observer;
/// their resizing policy remains the caller's. Empty requested times plus a
/// no-op observer provides final-state-only operation without allocations.
/// Observers run after acceptance and errors preserve their original payload.
/// A stopped stepper and its controller can be moved into `Continuation` and
/// resumed. Parameter/control changes between calls require cache invalidation.
/// This lightweight driver intentionally leaves event root finding to hosts;
/// an observer can stop at scheduled boundaries and restart after event effects.
pub fn integrate_rk<F, N, O, E>(
    stepper: &mut ExplicitRungeKuttaStepper<'_>,
    controller: &mut AdaptiveController,
    endpoint: f64,
    requested_times: &[f64],
    maximum_attempts: usize,
    rhs: &mut F,
    norm: &mut N,
    observer: &mut O,
) -> Result<IntegrationOutcome, IntegrationError<E>>
where
    F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), E>,
    N: FnMut(&[f64], &[f64], &[f64]) -> Result<f64, E>,
    O: FnMut(Observation<'_>) -> Result<ObserverAction, E>,
{
    let start = stepper.time();
    let direction = (endpoint - start).signum();
    if !endpoint.is_finite() {
        return Err(StepError::Solver(StepFailure::NonFinite).into());
    }
    let mut previous = start;
    for &time in requested_times {
        if !time.is_finite()
            || (time - previous) * direction < 0.
            || (time - start) * direction < 0.
            || (endpoint - time) * direction < 0.
            || (direction == 0. && time != start)
        {
            return Err(IntegrationError::RequestedTimes);
        }
        previous = time;
    }
    let mut index = 0;
    let mut forced = 0;
    if requested_times.first() == Some(&start) || start == endpoint {
        while requested_times.get(index) == Some(&start) {
            index += 1;
        }
        let stop = observer(Observation {
            time: start,
            state: stepper.state(),
            requested: index > 0,
            endpoint: start == endpoint,
        })
        .map_err(|e| StepError::User(e))?
            == ObserverAction::Stop;
        if stop || start == endpoint {
            return Ok(IntegrationOutcome {
                time: start,
                interrupted: stop,
                forced_acceptances: 0,
            });
        }
    }
    for _ in 0..maximum_attempts {
        let target = requested_times.get(index).copied().unwrap_or(endpoint);
        // The immutable accepted state and mutable scratch view cannot be borrowed
        // simultaneously through the public API; the method supplies both internally.
        let error = stepper.attempt_with_norm(target, controller.next_step(), rhs, norm)?;

        let step = stepper.pending_step().expect("successful attempt");
        let decision = match controller.assess(step, error) {
            Ok(d) => d,
            Err(e) => {
                stepper.reject().map_err(|e| StepError::Solver(e))?;
                return Err(e.into());
            }
        };
        if !decision.accepted {
            stepper.reject().map_err(|e| StepError::Solver(e))?;
            continue;
        }
        stepper.accept().map_err(|e| StepError::Solver(e))?;
        if decision.forced {
            forced += 1;
        }
        let requested = requested_times.get(index) == Some(&stepper.time());
        while requested_times.get(index) == Some(&stepper.time()) {
            index += 1;
        }
        let done = stepper.time() == endpoint;
        let stop = observer(Observation {
            time: stepper.time(),
            state: stepper.state(),
            requested,
            endpoint: done,
        })
        .map_err(|e| StepError::User(e))?
            == ObserverAction::Stop;
        if stop || done {
            return Ok(IntegrationOutcome {
                time: stepper.time(),
                interrupted: stop,
                forced_acceptances: forced,
            });
        }
    }
    Err(IntegrationError::AttemptLimit)
}
