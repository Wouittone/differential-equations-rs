use super::{StepFailure, finite};

/// Behavior when the error criterion cannot be met at the minimum step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MinimumStepPolicy {
    /// Return a failure; never silently accept an inaccurate step.
    Error,
    /// Explicitly accept the candidate and mark the decision as forced.
    ForceAccept,
}
/// Public controller coefficients, expressed as error exponents.
///
/// The raw ratio is `safety * error^(-beta[0]) * previous_error^(beta[1])
/// * older_error^(-beta[2])`. Rejected errors never advance accepted history.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ControllerConfig {
    /// P, PI or PID exponents including the method's error-order scaling.
    pub beta: [f64; 3],
    /// Safety multiplier.
    pub safety: f64,
    /// Smallest proposal/current-step ratio.
    pub minimum_factor: f64,
    /// Largest proposal/current-step ratio.
    pub maximum_factor: f64,
    /// Maximum ratio on a rejected attempt.
    pub rejection_maximum: f64,
    /// Maximum ratio after an acceptance following any rejection.
    pub rejected_acceptance_maximum: f64,
    /// Minimum absolute step (endpoint clipping may make a smaller final step).
    pub minimum_step: f64,
    /// Maximum absolute step.
    pub maximum_step: f64,
    /// Explicit minimum-step acceptance policy.
    pub minimum_step_policy: MinimumStepPolicy,
    /// Seed errors used before sufficient accepted history exists.
    pub initial_error_history: [f64; 2],
}
impl ControllerConfig {
    /// Conventional proportional controller; existing whole-solve defaults are unchanged.
    pub fn proportional(error_order: usize) -> Result<Self, ControllerError> {
        if error_order == 0 {
            return Err(ControllerError::Configuration);
        }
        Ok(Self {
            beta: [1. / error_order as f64, 0., 0.],
            safety: 0.9,
            minimum_factor: 0.2,
            maximum_factor: 10.,
            rejection_maximum: 1.,
            rejected_acceptance_maximum: 1.,
            minimum_step: 0.,
            maximum_step: f64::MAX,
            minimum_step_policy: MinimumStepPolicy::Error,
            initial_error_history: [1., 1.],
        })
    }
    /// PI coefficients with the supplied already-scaled error exponents.
    pub fn pi(beta1: f64, beta2: f64) -> Self {
        Self {
            beta: [beta1, beta2, 0.],
            ..Self::proportional(1).expect("positive order")
        }
    }
    /// PID coefficients with the supplied already-scaled error exponents.
    pub fn pid(beta: [f64; 3]) -> Self {
        Self {
            beta,
            ..Self::proportional(1).expect("positive order")
        }
    }
    fn validate(self) -> Result<(), ControllerError> {
        let values = [
            self.beta[0],
            self.beta[1],
            self.beta[2],
            self.safety,
            self.minimum_factor,
            self.maximum_factor,
            self.rejection_maximum,
            self.rejected_acceptance_maximum,
            self.minimum_step,
            self.maximum_step,
            self.initial_error_history[0],
            self.initial_error_history[1],
        ];
        if values.iter().any(|x| !x.is_finite())
            || self.beta[0] <= 0.
            || self.safety <= 0.
            || self.minimum_factor <= 0.
            || self.maximum_factor < self.minimum_factor
            || self.rejection_maximum <= 0.
            || self.rejection_maximum > 1.
            || self.rejected_acceptance_maximum <= 0.
            || self.minimum_step < 0.
            || self.maximum_step <= 0.
            || self.maximum_step < self.minimum_step
            || self.initial_error_history.iter().any(|x| *x <= 0.)
        {
            return Err(ControllerError::Configuration);
        }
        Ok(())
    }
}
/// Invalid controller state or unattainable requested accuracy.
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ControllerError {
    /// Coefficients, limits, state or proposal are invalid.
    #[error("invalid controller configuration or state")]
    Configuration,
    /// Error cannot be reduced at the configured minimum step.
    #[error("accuracy cannot be met at minimum step (error {error})")]
    MinimumStep {
        /// Last normalized error.
        error: f64,
    },
}
/// Portable exact controller history; `next_step` is the actual next proposal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ControllerState {
    /// Last two accepted errors, most recent first.
    pub accepted_errors: [f64; 2],
    /// Whether an attempt has been rejected since the last acceptance.
    pub rejected_since_acceptance: bool,
    /// Signed next proposal, independent of any clipped last interval.
    pub next_step: f64,
}
/// Explicit acceptance decision and following signed proposal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepDecision {
    /// Commit when true, otherwise reject and retry.
    pub accepted: bool,
    /// True only for the explicitly requested minimum-step override.
    pub forced: bool,
    /// Actual next proposed step, before endpoint clipping.
    pub next_step: f64,
}
/// Standalone P/PI/PID controller suitable for external host drivers.
#[derive(Clone, Debug)]
pub struct AdaptiveController {
    config: ControllerConfig,
    state: ControllerState,
}
impl AdaptiveController {
    /// Construct with an explicit initial signed step.
    pub fn new(config: ControllerConfig, initial_step: f64) -> Result<Self, ControllerError> {
        config.validate()?;
        if !initial_step.is_finite() || initial_step == 0. {
            return Err(ControllerError::Configuration);
        }
        let next_step = initial_step.signum()
            * initial_step
                .abs()
                .clamp(config.minimum_step, config.maximum_step);
        Ok(Self {
            config,
            state: ControllerState {
                accepted_errors: config.initial_error_history,
                rejected_since_acceptance: false,
                next_step,
            },
        })
    }
    /// Configuration used for every decision.
    pub fn config(&self) -> ControllerConfig {
        self.config
    }
    /// Export exact replay/continuation state.
    pub fn state(&self) -> ControllerState {
        self.state
    }
    /// Restore history and the actual next proposal after validation.
    pub fn restore(&mut self, state: ControllerState) -> Result<(), ControllerError> {
        if !state.next_step.is_finite()
            || state.next_step == 0.
            || state
                .accepted_errors
                .iter()
                .any(|e| !e.is_finite() || *e <= 0.)
            || state.next_step.abs() < self.config.minimum_step
            || state.next_step.abs() > self.config.maximum_step
        {
            return Err(ControllerError::Configuration);
        }
        self.state = state;
        Ok(())
    }
    /// Reset history after a discontinuity, retaining a caller-selected proposal.
    pub fn reset(&mut self, next_step: f64) -> Result<(), ControllerError> {
        *self = Self::new(self.config, next_step)?;
        Ok(())
    }
    /// Current proposal; endpoint clipping must not overwrite this value.
    pub fn next_step(&self) -> f64 {
        self.state.next_step
    }
    /// Decide using the actual attempted interval and a nonnegative normalized error.
    pub fn assess(&mut self, step: f64, error: f64) -> Result<StepDecision, ControllerError> {
        if !step.is_finite() || step == 0. || error.is_nan() || error < 0. {
            return Err(ControllerError::Configuration);
        }
        let forced = error > 1.
            && step.abs() <= self.config.minimum_step
            && self.config.minimum_step_policy == MinimumStepPolicy::ForceAccept;
        let accepted = error <= 1. || forced;
        if !accepted && step.abs() <= self.config.minimum_step {
            return Err(ControllerError::MinimumStep { error });
        }
        let mut factor = if error == 0. {
            self.config.maximum_factor
        } else if error.is_infinite() {
            self.config.minimum_factor
        } else {
            self.config.safety
                * error.powf(-self.config.beta[0])
                * self.state.accepted_errors[0].powf(self.config.beta[1])
                * self.state.accepted_errors[1].powf(-self.config.beta[2])
        }
        .clamp(self.config.minimum_factor, self.config.maximum_factor);
        if !accepted {
            factor = factor.min(self.config.rejection_maximum);
        } else if self.state.rejected_since_acceptance {
            factor = factor.min(self.config.rejected_acceptance_maximum);
        }
        let next_step = step.signum()
            * (step.abs() * factor).clamp(self.config.minimum_step, self.config.maximum_step);
        if !next_step.is_finite() || next_step == 0. {
            return Err(ControllerError::Configuration);
        }
        if !accepted && next_step.abs() >= step.abs() {
            return Err(ControllerError::MinimumStep { error });
        }
        if accepted {
            self.state.accepted_errors = [
                error.max(f64::MIN_POSITIVE).min(f64::MAX),
                self.state.accepted_errors[0],
            ];
            self.state.rejected_since_acceptance = false;
        } else {
            self.state.rejected_since_acceptance = true;
        }
        self.state.next_step = next_step;
        Ok(StepDecision {
            accepted,
            forced,
            next_step,
        })
    }
}
/// Scale-aware initial proposal from state and initial derivative, without RHS calls.
///
/// Uses RMS norms over caller-provided positive component scales. The caller can
/// refine this inexpensive estimate with additional evaluations if its host
/// policy requires that. Direction is exactly +1 or -1.
pub fn initial_step(
    state: &[f64],
    derivative: &[f64],
    scales: &[f64],
    direction: f64,
    maximum: f64,
) -> Result<f64, StepFailure> {
    if state.is_empty() || state.len() != derivative.len() || state.len() != scales.len() {
        return Err(StepFailure::Dimension);
    }
    finite(state)?;
    finite(derivative)?;
    finite(scales)?;
    finite(&[direction, maximum])?;
    if scales.iter().any(|s| *s <= 0.) || direction.abs() != 1. || maximum <= 0. {
        return Err(StepFailure::Direction);
    }
    let mut d0 = 0.;
    let mut d1 = 0.;
    for i in 0..state.len() {
        d0 += (state[i] / scales[i]).powi(2);
        d1 += (derivative[i] / scales[i]).powi(2);
    }
    d0 = (d0 / state.len() as f64).sqrt();
    d1 = (d1 / state.len() as f64).sqrt();
    let h = if d0 < 1e-5 || d1 < 1e-5 {
        1e-6
    } else {
        0.01 * d0 / d1
    };
    if !h.is_finite() {
        return Err(StepFailure::NonFinite);
    }
    Ok(direction * h.min(maximum))
}
/// Ownership-preserving continuation containing a workspace and exact controller.
///
/// No method/state deserialization or unchecked cache transfer is performed.
/// After changing state, parameters or controls reset the workspace cache; after
/// discontinuities also reset controller history.
#[derive(Debug)]
pub struct Continuation<S> {
    /// Reusable numerical workspace, including accepted state and caches.
    pub stepper: S,
    /// Error-control history and actual next proposal.
    pub controller: AdaptiveController,
}
