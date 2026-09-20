use super::ScopedOdeProblem;
use crate::stepping::{
    AdaptiveController, DerivativeHook, IntegrationError, IntegrationOutcome, Observation,
    ObserverAction, RosenbrockStepView, RosenbrockStepper, StepError, StepFailure, endpoint_step,
};

impl<F, E, O, J, L> ScopedOdeProblem<F, E, O, J, L>
where
    F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), E>,
    O: FnMut(Observation<'_>) -> Result<ObserverAction, E>,
    J: FnMut(f64, &[f64], &mut [f64]) -> Result<(), E>,
    L: FnMut(IntegrationOutcome, &[f64]) -> Result<(), E>,
{
    /// Integrates with a reusable Rosenbrock workspace and this problem's analytic Jacobian.
    ///
    /// `time_partial` is an optional borrowed fixed-state partial derivative,
    /// not the total derivative along the solution. Its application error is
    /// preserved like RHS, Jacobian, observer and finalizer errors. A missing
    /// Jacobian or time partial uses the workspace's numerical policy.
    pub fn integrate_rosenbrock<N>(
        &mut self,
        stepper: &mut RosenbrockStepper<'_>,
        controller: &mut AdaptiveController,
        requested_times: &[f64],
        maximum_attempts: usize,
        time_partial: Option<&mut DerivativeHook<'_, E>>,
        norm: &mut N,
    ) -> Result<IntegrationOutcome, IntegrationError<E>>
    where
        N: FnMut(&RosenbrockStepView<'_>) -> Result<f64, E>,
    {
        stepper
            .reset(self.time_span.0, &self.initial_state)
            .map_err(StepError::Solver)?;
        controller.reset(controller.next_step())?;
        let jacobian = self
            .jacobian
            .as_mut()
            .map(|j| j as &mut DerivativeHook<'_, E>);
        let outcome = integrate_rosenbrock(
            stepper,
            controller,
            self.time_span.1,
            requested_times,
            maximum_attempts,
            &mut self.rhs,
            jacobian,
            time_partial,
            norm,
            &mut self.observer,
        )?;
        (self.finalizer)(outcome, stepper.state()).map_err(StepError::User)?;
        Ok(outcome)
    }
}

/// Output-free Rosenbrock integration with borrowed typed derivative hooks.
///
/// RHS, Jacobian, time-partial, norm and observer errors retain their original
/// payload. Requested outputs are exact boundaries; no trajectory is retained.
/// Rejected attempts keep the accepted state unchanged. Norm/controller failures
/// reject pending candidates before returning, so the workspace remains reusable.
/// Methods requiring Richardson estimation are rejected before evaluating forces
/// or derivative hooks. They remain usable through fixed low-level attempts.
pub fn integrate_rosenbrock<F, N, O, E>(
    stepper: &mut RosenbrockStepper<'_>,
    controller: &mut AdaptiveController,
    endpoint: f64,
    requested_times: &[f64],
    maximum_attempts: usize,
    rhs: &mut F,
    mut jacobian: Option<&mut DerivativeHook<'_, E>>,
    mut time_partial: Option<&mut DerivativeHook<'_, E>>,
    norm: &mut N,
    observer: &mut O,
) -> Result<IntegrationOutcome, IntegrationError<E>>
where
    F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), E>,
    N: FnMut(&RosenbrockStepView<'_>) -> Result<f64, E>,
    O: FnMut(Observation<'_>) -> Result<ObserverAction, E>,
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
        let stop = observer(Observation {
            time: start,
            state: stepper.state(),
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
    if stepper.tableau().error_estimator() != crate::tableau::RosenbrockErrorEstimator::Embedded
        || stepper.tableau().btilde().is_none()
    {
        return Err(IntegrationError::MissingErrorEstimate);
    }
    for _ in 0..maximum_attempts {
        let target = requested_times.get(index).copied().unwrap_or(endpoint);
        let step = endpoint_step(stepper.time(), target, controller.next_step())
            .map_err(StepError::Solver)?;
        let view = stepper.attempt(
            step,
            rhs,
            jacobian
                .as_mut()
                .map(|j| &mut **j as &mut DerivativeHook<'_, E>),
            time_partial
                .as_mut()
                .map(|f| &mut **f as &mut DerivativeHook<'_, E>),
        )?;
        let step = view.end_time - view.start_time;
        if view.component_error.is_none() {
            stepper.reject().map_err(StepError::Solver)?;
            return Err(IntegrationError::MissingErrorEstimate);
        }
        let error = match norm(&view) {
            Ok(e) => e,
            Err(e) => {
                stepper.reject().map_err(StepError::Solver)?;
                return Err(StepError::User(e).into());
            }
        };
        let decision = match controller.assess(step, error) {
            Ok(d) => d,
            Err(e) => {
                stepper.reject().map_err(StepError::Solver)?;
                return Err(e.into());
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
        let stop = observer(Observation {
            time: stepper.time(),
            state: stepper.state(),
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
