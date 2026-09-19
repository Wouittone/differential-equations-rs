const DEFAULT_SAFETY: f64 = 0.9;
const DEFAULT_MIN_FACTOR: f64 = 0.2;
const DEFAULT_MAX_FACTOR: f64 = 10.0;

/// Per-family metadata for the proportional step-size controller.
///
/// Keeping the complete policy on the kernel capability lets solver families
/// preserve their existing constants while sharing the integration lifecycle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ControllerConfig {
    pub(super) error_order: usize,
    pub(super) safety: f64,
    pub(super) minimum_factor: f64,
    pub(super) maximum_factor: f64,
    pub(super) rejected_acceptance_maximum: f64,
    pub(super) rejection_maximum: f64,
    pub(super) failed_attempt_factor: f64,
    strategy: ControllerStrategy,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum ControllerStrategy {
    Proportional {
        integral_exponent: f64,
    },
    Pi {
        beta1: f64,
        beta2: f64,
        initial_previous_error: f64,
    },
    Pid {
        beta: [f64; 3],
        acceptance_safety: f64,
    },
}

impl ControllerConfig {
    pub(crate) const fn proportional(
        error_order: usize,
        safety: f64,
        minimum_factor: f64,
        maximum_factor: f64,
        failed_attempt_factor: f64,
    ) -> Self {
        Self {
            error_order,
            safety,
            minimum_factor,
            maximum_factor,
            rejected_acceptance_maximum: 1.0,
            rejection_maximum: 1.0,
            failed_attempt_factor,
            strategy: ControllerStrategy::Proportional {
                integral_exponent: 0.0,
            },
        }
    }

    /// Adds an optional integral-history exponent for PI controller metadata.
    /// Zero preserves the existing proportional controller exactly.
    #[allow(dead_code)]
    pub(crate) const fn with_integral_exponent(mut self, integral_exponent: f64) -> Self {
        self.strategy = ControllerStrategy::Proportional { integral_exponent };
        self
    }

    /// SciML's standard PI policy for an adaptive method of fixed order.
    pub(crate) fn standard_pi(error_order: usize) -> Self {
        let order = error_order as f64;
        Self {
            error_order,
            safety: 0.9,
            minimum_factor: 0.2,
            maximum_factor: 10.0,
            rejected_acceptance_maximum: 1.0,
            rejection_maximum: 1.0,
            failed_attempt_factor: 0.2,
            strategy: ControllerStrategy::Pi {
                beta1: 0.7 / order,
                beta2: 0.4 / order,
                initial_previous_error: 1.0e-4,
            },
        }
    }

    /// Smooth PID policy used by optimized low-storage embedded pairs.
    pub(crate) const fn pid(error_order: usize, beta: [f64; 3], acceptance_safety: f64) -> Self {
        Self {
            error_order,
            safety: 1.0,
            minimum_factor: 0.0,
            maximum_factor: f64::INFINITY,
            rejected_acceptance_maximum: f64::INFINITY,
            rejection_maximum: f64::INFINITY,
            failed_attempt_factor: 0.2,
            strategy: ControllerStrategy::Pid {
                beta,
                acceptance_safety,
            },
        }
    }

    pub(super) const fn default_for_order(error_order: usize) -> Self {
        Self::proportional(
            error_order,
            DEFAULT_SAFETY,
            DEFAULT_MIN_FACTOR,
            DEFAULT_MAX_FACTOR,
            DEFAULT_MIN_FACTOR,
        )
    }
}

pub(super) fn step_factor(error: f64, controller: ControllerConfig) -> f64 {
    if error == 0.0 {
        controller.maximum_factor
    } else if error.is_finite() {
        (controller.safety * error.powf(-1.0 / controller.error_order as f64))
            .clamp(controller.minimum_factor, controller.maximum_factor)
    } else {
        controller.minimum_factor
    }
}

#[allow(dead_code)]
pub(super) fn step_factor_with_history(
    error: f64,
    previous_error: Option<f64>,
    controller: ControllerConfig,
) -> f64 {
    let proportional = step_factor(error, controller);
    let ControllerStrategy::Proportional { integral_exponent } = controller.strategy else {
        return proportional;
    };
    if integral_exponent == 0.0 {
        return proportional;
    }
    let Some(previous_error) = previous_error.filter(|value| value.is_finite() && *value > 0.0)
    else {
        return proportional;
    };
    if !error.is_finite() || error <= 0.0 {
        return proportional;
    }
    (proportional * previous_error.powf(integral_exponent))
        .clamp(controller.minimum_factor, controller.maximum_factor)
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct ControllerState {
    pub(super) previous_error: Option<f64>,
    pub(super) older_error: Option<f64>,
}

#[allow(dead_code)]
impl ControllerState {
    pub(crate) fn factor(&self, error: f64, controller: ControllerConfig) -> f64 {
        match controller.strategy {
            ControllerStrategy::Proportional { .. } => {
                step_factor_with_history(error, self.previous_error, controller)
            }
            ControllerStrategy::Pi {
                beta1,
                beta2,
                initial_previous_error,
            } => {
                if error == 0.0 {
                    return controller.maximum_factor;
                }
                if !error.is_finite() || error < 0.0 {
                    return controller.minimum_factor;
                }
                let previous = self.previous_error.unwrap_or(initial_previous_error);
                (controller.safety * error.powf(-beta1) * previous.powf(beta2))
                    .clamp(controller.minimum_factor, controller.maximum_factor)
            }
            ControllerStrategy::Pid { beta, .. } => {
                if !error.is_finite() || error < 0.0 {
                    return 1.0 + (-1.0_f64).atan();
                }
                let error = error.max(f64::EPSILON);
                let previous = self.previous_error.unwrap_or(1.0);
                let older = self.older_error.unwrap_or(1.0);
                let order = controller.error_order as f64;
                let unlimited = error.powf(-beta[0] / order)
                    * previous.powf(-beta[1] / order)
                    * older.powf(-beta[2] / order);
                1.0 + (unlimited - 1.0).atan()
            }
        }
    }

    pub(crate) fn rejection_factor(&self, error: f64, controller: ControllerConfig) -> f64 {
        match controller.strategy {
            ControllerStrategy::Pi { beta1, .. } => {
                if error == 0.0 {
                    return controller.maximum_factor;
                }
                if !error.is_finite() || error < 0.0 {
                    return controller.minimum_factor;
                }
                (controller.safety * error.powf(-beta1))
                    .clamp(controller.minimum_factor, controller.maximum_factor)
            }
            ControllerStrategy::Proportional { .. } | ControllerStrategy::Pid { .. } => {
                self.factor(error, controller)
            }
        }
    }

    pub(crate) fn accepts(&self, error: f64, controller: ControllerConfig) -> bool {
        if !error.is_finite() || error < 0.0 {
            return false;
        }
        match controller.strategy {
            ControllerStrategy::Pid {
                acceptance_safety, ..
            } => self.factor(error, controller) >= acceptance_safety,
            ControllerStrategy::Proportional { .. } | ControllerStrategy::Pi { .. } => error <= 1.0,
        }
    }

    pub(crate) fn accepted(&mut self, error: f64, controller: ControllerConfig) {
        match controller.strategy {
            ControllerStrategy::Pid { .. } => {
                self.older_error = self.previous_error;
                self.previous_error = error.is_finite().then_some(error.max(f64::EPSILON));
            }
            ControllerStrategy::Pi {
                initial_previous_error,
                ..
            } => {
                self.previous_error = error
                    .is_finite()
                    .then_some(error.max(initial_previous_error));
            }
            ControllerStrategy::Proportional { .. } => {
                self.previous_error = error.is_finite().then_some(error.max(f64::MIN_POSITIVE));
            }
        }
    }

    pub(crate) fn rejected(&mut self, error: f64, controller: ControllerConfig) {
        if matches!(controller.strategy, ControllerStrategy::Proportional { .. }) {
            self.previous_error = error.is_finite().then_some(error.max(f64::MIN_POSITIVE));
        }
    }

    pub(crate) fn reset(&mut self) {
        self.previous_error = None;
        self.older_error = None;
    }
}
