//! Configuration for deterministic automatic solver switching.

use thiserror::Error;

/// Identifies the active side of an automatic solver pair.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AutomaticBranch {
    /// The branch intended for non-stiff portions of a problem.
    NonStiff,
    /// The branch intended for stiff portions of a problem.
    Stiff,
}

/// An invalid automatic-switching configuration.
#[derive(Clone, Copy, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum AutoSwitchConfigError {
    /// The stiffness thresholds are non-finite, non-positive, or unordered.
    #[error(
        "invalid stiffness thresholds: require finite values with 0 < exit_stiffness < enter_stiffness, got exit_stiffness={exit_stiffness} and enter_stiffness={enter_stiffness}"
    )]
    InvalidStiffnessThresholds {
        /// The configured threshold for returning to the non-stiff branch.
        exit_stiffness: f64,
        /// The configured threshold for entering the stiff branch.
        enter_stiffness: f64,
    },
    /// A counter used by the switching policy was configured as zero.
    #[error("{parameter} must be greater than zero")]
    ZeroCount {
        /// The name of the invalid counter.
        parameter: &'static str,
    },
    /// The step-size factor is non-finite or is not positive.
    #[error("switch_step_factor must be finite and greater than zero, got {value}")]
    InvalidSwitchStepFactor {
        /// The invalid factor.
        value: f64,
    },
}

/// Controls the deterministic policy used by automatic solver pairs.
///
/// The enter and exit thresholds form a hysteresis band. Evidence must be
/// consecutive, and a branch change based on stiffness diagnostics is allowed
/// only after the configured minimum residence. Consecutive rejected attempts
/// can force the safer stiff branch independently of that residence limit.
///
/// For an accepted step of size `h`, the diagnostic is
/// `abs(h) * spectral_radius_estimate / explicit_real_stability_radius`.
/// Explicit Runge--Kutta branches estimate the spectral radius from their last
/// two endpoint-stage state/derivative pairs, following the Hairer-style
/// estimator used by SciML. Indeterminate `0 / 0` component pairs contribute
/// no evidence, which prevents equilibria and conserved components from
/// causing false switches. Rosenbrock branches reuse the infinity norm of the
/// current Jacobian estimate. Other non-finite estimates are conservatively
/// treated as stiff evidence. Domain-policy rejections clear detector evidence,
/// while numerical error or step-attempt rejections count toward
/// [`Self::rejection_confirmations`].
///
/// Confirmation values name the exact number of consecutive observations: a
/// value of `n` switches on observation `n`. The default `0.75`/`0.90`
/// hysteresis follows SciML's documented thresholds. These choices are
/// deliberate where the pinned upstream implementation uses a strict
/// counter comparison and currently constructs two equal `0.90` thresholds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AutoSwitchConfig {
    enter_stiffness: f64,
    exit_stiffness: f64,
    stiff_confirmations: usize,
    nonstiff_confirmations: usize,
    rejection_confirmations: usize,
    minimum_residence_steps: usize,
    switch_step_factor: f64,
    initial_branch: AutomaticBranch,
    allow_switch_back: bool,
}

impl AutoSwitchConfig {
    /// Creates a configuration with the default automatic-switching policy.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            enter_stiffness: 0.90,
            exit_stiffness: 0.75,
            stiff_confirmations: 10,
            nonstiff_confirmations: 3,
            rejection_confirmations: 2,
            minimum_residence_steps: 3,
            switch_step_factor: 2.0,
            initial_branch: AutomaticBranch::NonStiff,
            allow_switch_back: true,
        }
    }

    /// Returns the diagnostic threshold for entering the stiff branch.
    #[must_use]
    pub const fn enter_stiffness(&self) -> f64 {
        self.enter_stiffness
    }

    /// Returns the diagnostic threshold for returning to the non-stiff branch.
    #[must_use]
    pub const fn exit_stiffness(&self) -> f64 {
        self.exit_stiffness
    }

    /// Returns the consecutive stiff diagnostics required to switch branches.
    #[must_use]
    pub const fn stiff_confirmations(&self) -> usize {
        self.stiff_confirmations
    }

    /// Returns the consecutive non-stiff diagnostics required to switch back.
    #[must_use]
    pub const fn nonstiff_confirmations(&self) -> usize {
        self.nonstiff_confirmations
    }

    /// Returns the consecutive rejected attempts that force the stiff branch.
    #[must_use]
    pub const fn rejection_confirmations(&self) -> usize {
        self.rejection_confirmations
    }

    /// Returns the accepted-step residence required before a diagnostic switch.
    #[must_use]
    pub const fn minimum_residence_steps(&self) -> usize {
        self.minimum_residence_steps
    }

    /// Returns the factor used to adjust the proposed step after a switch.
    ///
    /// The driver multiplies the proposal when entering the stiff branch and
    /// divides by the same factor when returning to the non-stiff branch. The
    /// result is still bounded by `maximum_step`; callback step requests take
    /// precedence. Fixed-step solves resume their configured step after the
    /// one transition proposal.
    #[must_use]
    pub const fn switch_step_factor(&self) -> f64 {
        self.switch_step_factor
    }

    /// Returns the branch selected when detector state is initialized or reset.
    #[must_use]
    pub const fn initial_branch(&self) -> AutomaticBranch {
        self.initial_branch
    }

    /// Returns whether the stiff branch may switch back to the non-stiff branch.
    #[must_use]
    pub const fn allow_switch_back(&self) -> bool {
        self.allow_switch_back
    }

    /// Sets the diagnostic threshold for entering the stiff branch.
    ///
    /// Use [`Self::with_stiffness_thresholds`] when replacing both thresholds
    /// would temporarily violate their required ordering.
    pub fn with_enter_stiffness(
        mut self,
        enter_stiffness: f64,
    ) -> Result<Self, AutoSwitchConfigError> {
        self.enter_stiffness = enter_stiffness;
        self.validate()?;
        Ok(self)
    }

    /// Sets the diagnostic threshold for returning to the non-stiff branch.
    ///
    /// Use [`Self::with_stiffness_thresholds`] when replacing both thresholds
    /// would temporarily violate their required ordering.
    pub fn with_exit_stiffness(
        mut self,
        exit_stiffness: f64,
    ) -> Result<Self, AutoSwitchConfigError> {
        self.exit_stiffness = exit_stiffness;
        self.validate()?;
        Ok(self)
    }

    /// Sets both stiffness thresholds and validates their relationship together.
    pub fn with_stiffness_thresholds(
        mut self,
        exit_stiffness: f64,
        enter_stiffness: f64,
    ) -> Result<Self, AutoSwitchConfigError> {
        self.exit_stiffness = exit_stiffness;
        self.enter_stiffness = enter_stiffness;
        self.validate()?;
        Ok(self)
    }

    /// Sets the consecutive stiff diagnostics required to switch branches.
    pub fn with_stiff_confirmations(
        mut self,
        stiff_confirmations: usize,
    ) -> Result<Self, AutoSwitchConfigError> {
        self.stiff_confirmations = stiff_confirmations;
        self.validate()?;
        Ok(self)
    }

    /// Sets the consecutive non-stiff diagnostics required to switch back.
    pub fn with_nonstiff_confirmations(
        mut self,
        nonstiff_confirmations: usize,
    ) -> Result<Self, AutoSwitchConfigError> {
        self.nonstiff_confirmations = nonstiff_confirmations;
        self.validate()?;
        Ok(self)
    }

    /// Sets the consecutive rejected attempts that force the stiff branch.
    pub fn with_rejection_confirmations(
        mut self,
        rejection_confirmations: usize,
    ) -> Result<Self, AutoSwitchConfigError> {
        self.rejection_confirmations = rejection_confirmations;
        self.validate()?;
        Ok(self)
    }

    /// Sets the accepted-step residence required before a diagnostic switch.
    pub fn with_minimum_residence_steps(
        mut self,
        minimum_residence_steps: usize,
    ) -> Result<Self, AutoSwitchConfigError> {
        self.minimum_residence_steps = minimum_residence_steps;
        self.validate()?;
        Ok(self)
    }

    /// Sets the factor used to adjust the proposed step after a switch.
    ///
    /// Entering the stiff branch multiplies the next proposal by this value;
    /// returning to the non-stiff branch divides by it. Normal step bounds and
    /// callback overrides are applied by the shared driver.
    pub fn with_switch_step_factor(
        mut self,
        switch_step_factor: f64,
    ) -> Result<Self, AutoSwitchConfigError> {
        self.switch_step_factor = switch_step_factor;
        self.validate()?;
        Ok(self)
    }

    /// Selects the branch used when detector state is initialized or reset.
    #[must_use]
    pub const fn with_initial_branch(mut self, initial_branch: AutomaticBranch) -> Self {
        self.initial_branch = initial_branch;
        self
    }

    /// Enables or disables switching back from the stiff branch.
    #[must_use]
    pub const fn with_allow_switch_back(mut self, allow_switch_back: bool) -> Self {
        self.allow_switch_back = allow_switch_back;
        self
    }

    /// Validates all automatic-switching invariants.
    ///
    /// This is useful at API boundaries that receive a complete configuration.
    pub fn validate(&self) -> Result<(), AutoSwitchConfigError> {
        if !self.enter_stiffness.is_finite()
            || !self.exit_stiffness.is_finite()
            || self.exit_stiffness <= 0.0
            || self.exit_stiffness >= self.enter_stiffness
        {
            return Err(AutoSwitchConfigError::InvalidStiffnessThresholds {
                exit_stiffness: self.exit_stiffness,
                enter_stiffness: self.enter_stiffness,
            });
        }

        for (parameter, value) in [
            ("stiff_confirmations", self.stiff_confirmations),
            ("nonstiff_confirmations", self.nonstiff_confirmations),
            ("rejection_confirmations", self.rejection_confirmations),
            ("minimum_residence_steps", self.minimum_residence_steps),
        ] {
            if value == 0 {
                return Err(AutoSwitchConfigError::ZeroCount { parameter });
            }
        }

        if !self.switch_step_factor.is_finite() || self.switch_step_factor <= 0.0 {
            return Err(AutoSwitchConfigError::InvalidSwitchStepFactor {
                value: self.switch_step_factor,
            });
        }

        Ok(())
    }
}

impl Default for AutoSwitchConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{AutoSwitchConfig, AutoSwitchConfigError, AutomaticBranch};

    #[test]
    fn defaults_are_stable_and_valid() {
        let config = AutoSwitchConfig::default();

        assert_eq!(config.enter_stiffness(), 0.90);
        assert_eq!(config.exit_stiffness(), 0.75);
        assert_eq!(config.stiff_confirmations(), 10);
        assert_eq!(config.nonstiff_confirmations(), 3);
        assert_eq!(config.rejection_confirmations(), 2);
        assert_eq!(config.minimum_residence_steps(), 3);
        assert_eq!(config.switch_step_factor(), 2.0);
        assert_eq!(config.initial_branch(), AutomaticBranch::NonStiff);
        assert!(config.allow_switch_back());
        assert_eq!(config.validate(), Ok(()));
    }

    #[test]
    fn checked_builders_reject_invalid_values() {
        assert!(matches!(
            AutoSwitchConfig::new().with_stiffness_thresholds(0.9, 0.9),
            Err(AutoSwitchConfigError::InvalidStiffnessThresholds { .. })
        ));
        assert!(matches!(
            AutoSwitchConfig::new().with_enter_stiffness(f64::INFINITY),
            Err(AutoSwitchConfigError::InvalidStiffnessThresholds { .. })
        ));
        assert!(matches!(
            AutoSwitchConfig::new().with_stiff_confirmations(0),
            Err(AutoSwitchConfigError::ZeroCount {
                parameter: "stiff_confirmations"
            })
        ));
        assert!(matches!(
            AutoSwitchConfig::new().with_minimum_residence_steps(0),
            Err(AutoSwitchConfigError::ZeroCount {
                parameter: "minimum_residence_steps"
            })
        ));
        assert!(matches!(
            AutoSwitchConfig::new().with_switch_step_factor(f64::NAN),
            Err(AutoSwitchConfigError::InvalidSwitchStepFactor { .. })
        ));
    }

    #[test]
    fn infallible_policy_builders_set_their_values() {
        let config = AutoSwitchConfig::new()
            .with_initial_branch(AutomaticBranch::Stiff)
            .with_allow_switch_back(false);

        assert_eq!(config.initial_branch(), AutomaticBranch::Stiff);
        assert!(!config.allow_switch_back());
    }
}
