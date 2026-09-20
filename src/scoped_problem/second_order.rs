use crate::stepping::{
    AdaptiveController, IntegrationError, IntegrationOutcome, ObserverAction, RknStepView,
    RknStepper, StepError, StepFailure,
};

/// Accepted second-order state borrowed by an observer.
#[derive(Clone, Copy, Debug)]
pub struct RknObservation<'a> {
    /// Accepted integration time.
    pub time: f64,
    /// Accepted position.
    pub position: &'a [f64],
    /// Accepted velocity.
    pub velocity: &'a [f64],
    /// Whether this is an explicitly requested time.
    pub requested: bool,
    /// Whether the integration endpoint was reached.
    pub endpoint: bool,
}
/// Default fallible second-order observer.
pub type ScopedRknObserver<E> = for<'a> fn(RknObservation<'a>) -> Result<ObserverAction, E>;
/// Default fallible second-order finalizer.
pub type ScopedRknFinalizer<E> = fn(IntegrationOutcome, &[f64], &[f64]) -> Result<(), E>;

/// Concrete borrowed/thread-eligible second-order acceleration and lifecycle hooks.
///
/// The acceleration convention is `(time, position, velocity, output)`, matching
/// the reusable RKN kernel. Each hook retains its original application error.
pub struct ScopedSecondOrderProblem<F, E, O = ScopedRknObserver<E>, L = ScopedRknFinalizer<E>> {
    acceleration: F,
    position: Vec<f64>,
    velocity: Vec<f64>,
    span: (f64, f64),
    observer: O,
    finalizer: L,
    error: std::marker::PhantomData<fn() -> E>,
}
fn observe<E>(_: RknObservation<'_>) -> Result<ObserverAction, E> {
    Ok(ObserverAction::Continue)
}
fn finalize<E>(_: IntegrationOutcome, _: &[f64], _: &[f64]) -> Result<(), E> {
    Ok(())
}
impl<F, E> ScopedSecondOrderProblem<F, E>
where
    F: FnMut(f64, &[f64], &[f64], &mut [f64]) -> Result<(), E>,
{
    /// Stores an acceleration closure and separate initial position/velocity.
    pub fn new(
        acceleration: F,
        position: impl Into<Vec<f64>>,
        velocity: impl Into<Vec<f64>>,
        span: (f64, f64),
    ) -> Self {
        Self {
            acceleration,
            position: position.into(),
            velocity: velocity.into(),
            span,
            observer: observe::<E>,
            finalizer: finalize::<E>,
            error: std::marker::PhantomData,
        }
    }
}
impl<F, E, O, L> ScopedSecondOrderProblem<F, E, O, L> {
    /// Initial position.
    pub fn initial_position(&self) -> &[f64] {
        &self.position
    }
    /// Initial velocity.
    pub fn initial_velocity(&self) -> &[f64] {
        &self.velocity
    }
    /// Start and endpoint.
    pub fn time_span(&self) -> (f64, f64) {
        self.span
    }
    /// Adds an accepted-step observer that may borrow local data.
    pub fn with_observer<N>(self, observer: N) -> ScopedSecondOrderProblem<F, E, N, L>
    where
        N: FnMut(RknObservation<'_>) -> Result<ObserverAction, E>,
    {
        ScopedSecondOrderProblem {
            acceleration: self.acceleration,
            position: self.position,
            velocity: self.velocity,
            span: self.span,
            observer,
            finalizer: self.finalizer,
            error: std::marker::PhantomData,
        }
    }
    /// Adds a finalizer for successful completion or observer interruption.
    pub fn with_finalizer<N>(self, finalizer: N) -> ScopedSecondOrderProblem<F, E, O, N>
    where
        N: FnMut(IntegrationOutcome, &[f64], &[f64]) -> Result<(), E>,
    {
        ScopedSecondOrderProblem {
            acceleration: self.acceleration,
            position: self.position,
            velocity: self.velocity,
            span: self.span,
            observer: self.observer,
            finalizer,
            error: std::marker::PhantomData,
        }
    }
    /// Borrows the acceleration closure for host-managed attempts.
    pub fn acceleration_mut(&mut self) -> &mut F {
        &mut self.acceleration
    }
}
impl<F, E, O, L> ScopedSecondOrderProblem<F, E, O, L>
where
    F: FnMut(f64, &[f64], &[f64], &mut [f64]) -> Result<(), E>,
    O: FnMut(RknObservation<'_>) -> Result<ObserverAction, E>,
    L: FnMut(IntegrationOutcome, &[f64], &[f64]) -> Result<(), E>,
{
    /// Starts a fresh trajectory in a reusable RKN workspace without saving output.
    /// The norm sees both partitions, their previous values, and embedded errors.
    pub fn integrate<N>(
        &mut self,
        stepper: &mut RknStepper<'_>,
        controller: &mut AdaptiveController,
        requested_times: &[f64],
        maximum_attempts: usize,
        norm: &mut N,
    ) -> Result<IntegrationOutcome, IntegrationError<E>>
    where
        N: FnMut(&RknStepView<'_>) -> Result<f64, E>,
    {
        stepper
            .reset(self.span.0, &self.position, &self.velocity)
            .map_err(StepError::Solver)?;
        controller.reset(controller.next_step())?;
        let outcome = integrate_rkn(
            stepper,
            controller,
            self.span.1,
            requested_times,
            maximum_attempts,
            &mut self.acceleration,
            norm,
            &mut self.observer,
        )?;
        (self.finalizer)(outcome, stepper.position(), stepper.velocity())
            .map_err(StepError::User)?;
        Ok(outcome)
    }
}

/// Output-free RKN integration with typed borrowed acceleration/norm/observer hooks.
///
/// Every requested time is an exact step boundary. Accepted steps are observed
/// once, duplicate requested times coalesce, and callback failures preserve `E`.
/// On a norm or controller error the pending attempt is rejected, permitting reuse.
/// Root finding and state-changing callbacks are handled by host-controlled
/// attempts; this observer is read-only and may request an early stop.
pub fn integrate_rkn<F, N, O, E>(
    stepper: &mut RknStepper<'_>,
    controller: &mut AdaptiveController,
    endpoint: f64,
    requested_times: &[f64],
    maximum_attempts: usize,
    acceleration: &mut F,
    norm: &mut N,
    observer: &mut O,
) -> Result<IntegrationOutcome, IntegrationError<E>>
where
    F: FnMut(f64, &[f64], &[f64], &mut [f64]) -> Result<(), E>,
    N: FnMut(&RknStepView<'_>) -> Result<f64, E>,
    O: FnMut(RknObservation<'_>) -> Result<ObserverAction, E>,
{
    let start = stepper.time();
    if !endpoint.is_finite() {
        return Err(StepError::Solver(StepFailure::NonFinite).into());
    }
    let direction = (endpoint - start).signum();
    let mut previous = start;
    for &time in requested_times {
        if !time.is_finite()
            || direction * (time - previous) < 0.0
            || direction * (time - start) < 0.0
            || direction * (endpoint - time) < 0.0
            || (start == endpoint && time != start)
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
        let stop = observer(RknObservation {
            time: start,
            position: stepper.position(),
            velocity: stepper.velocity(),
            requested: index > 0,
            endpoint: start == endpoint,
        })
        .map_err(StepError::User)?
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
        let view = stepper.attempt_to(target, controller.next_step(), acceleration)?;
        let step = view.end_time - view.start_time;
        if view.position_error.is_none() || view.velocity_error.is_none() {
            stepper.reject().map_err(StepError::Solver)?;
            return Err(IntegrationError::MissingErrorEstimate);
        }
        let error = match norm(&view) {
            Ok(value) => value,
            Err(error) => {
                stepper.reject().map_err(StepError::Solver)?;
                return Err(StepError::User(error).into());
            }
        };
        let decision = match controller.assess(step, error) {
            Ok(value) => value,
            Err(error) => {
                stepper.reject().map_err(StepError::Solver)?;
                return Err(error.into());
            }
        };
        if !decision.accepted {
            stepper.reject().map_err(StepError::Solver)?;
            continue;
        }
        stepper.accept().map_err(StepError::Solver)?;
        if decision.forced {
            forced += 1;
        }
        let requested = requested_times.get(index) == Some(&stepper.time());
        while requested_times.get(index) == Some(&stepper.time()) {
            index += 1;
        }
        let done = stepper.time() == endpoint;
        let stop = observer(RknObservation {
            time: stepper.time(),
            position: stepper.position(),
            velocity: stepper.velocity(),
            requested,
            endpoint: done,
        })
        .map_err(StepError::User)?
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
