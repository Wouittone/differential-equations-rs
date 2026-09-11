//! Stateful policy for automatic solver switching.

use super::{AutoSwitchConfig, AutomaticBranch};

/// Deterministic detector state shared by automatic solver pairs.
#[derive(Clone, Debug)]
pub(crate) struct AutoSwitchDetector {
    config: AutoSwitchConfig,
    branch: AutomaticBranch,
    residence_steps: usize,
    stiff_evidence: usize,
    nonstiff_evidence: usize,
    rejection_evidence: usize,
}

impl AutoSwitchDetector {
    /// Initializes detector state from an already validated configuration.
    pub(crate) fn new(config: AutoSwitchConfig) -> Self {
        debug_assert!(config.validate().is_ok());
        Self {
            branch: config.initial_branch(),
            config,
            residence_steps: 0,
            stiff_evidence: 0,
            nonstiff_evidence: 0,
            rejection_evidence: 0,
        }
    }

    /// Returns the active branch.
    pub(crate) const fn branch(&self) -> AutomaticBranch {
        self.branch
    }

    /// Returns the detector configuration.
    pub(crate) const fn config(&self) -> &AutoSwitchConfig {
        &self.config
    }

    /// Records an accepted step and its stiffness diagnostic.
    ///
    /// A non-finite diagnostic is conservatively classified as stiff. The
    /// returned branch is present only when this observation caused a switch.
    pub(crate) fn observe_accepted(
        &mut self,
        stiffness_diagnostic: f64,
    ) -> Option<AutomaticBranch> {
        self.residence_steps = self.residence_steps.saturating_add(1);
        self.rejection_evidence = 0;

        match self.branch {
            AutomaticBranch::NonStiff => {
                self.nonstiff_evidence = 0;
                if !stiffness_diagnostic.is_finite()
                    || stiffness_diagnostic >= self.config.enter_stiffness()
                {
                    self.stiff_evidence = self.stiff_evidence.saturating_add(1);
                } else {
                    self.stiff_evidence = 0;
                }

                if self.residence_is_satisfied()
                    && self.stiff_evidence >= self.config.stiff_confirmations()
                {
                    self.switch_to(AutomaticBranch::Stiff)
                } else {
                    None
                }
            }
            AutomaticBranch::Stiff => {
                self.stiff_evidence = 0;
                if self.config.allow_switch_back()
                    && stiffness_diagnostic.is_finite()
                    && stiffness_diagnostic <= self.config.exit_stiffness()
                {
                    self.nonstiff_evidence = self.nonstiff_evidence.saturating_add(1);
                } else {
                    self.nonstiff_evidence = 0;
                }

                if self.config.allow_switch_back()
                    && self.residence_is_satisfied()
                    && self.nonstiff_evidence >= self.config.nonstiff_confirmations()
                {
                    self.switch_to(AutomaticBranch::NonStiff)
                } else {
                    None
                }
            }
        }
    }

    /// Records a rejected step attempt.
    ///
    /// Consecutive rejections force the non-stiff branch to switch to the stiff
    /// branch even before the minimum accepted-step residence is met.
    pub(crate) fn observe_rejected(&mut self) -> Option<AutomaticBranch> {
        self.stiff_evidence = 0;
        self.nonstiff_evidence = 0;

        if self.branch == AutomaticBranch::NonStiff {
            self.rejection_evidence = self.rejection_evidence.saturating_add(1);
            if self.rejection_evidence >= self.config.rejection_confirmations() {
                return self.switch_to(AutomaticBranch::Stiff);
            }
        } else {
            self.rejection_evidence = 0;
        }

        None
    }

    /// Clears consecutive diagnostic and rejection evidence after a discontinuity.
    ///
    /// Callback effects can use this without losing the current branch or its
    /// residence history.
    pub(crate) fn reset_evidence(&mut self) {
        self.stiff_evidence = 0;
        self.nonstiff_evidence = 0;
        self.rejection_evidence = 0;
    }

    fn residence_is_satisfied(&self) -> bool {
        self.residence_steps >= self.config.minimum_residence_steps()
    }

    fn switch_to(&mut self, branch: AutomaticBranch) -> Option<AutomaticBranch> {
        if branch == self.branch {
            return None;
        }
        self.branch = branch;
        self.residence_steps = 0;
        self.reset_evidence();
        Some(branch)
    }
}

#[cfg(test)]
mod tests {
    use super::AutoSwitchDetector;
    use crate::solvers::automatic::{AutoSwitchConfig, AutomaticBranch};

    fn config() -> AutoSwitchConfig {
        AutoSwitchConfig::new()
            .with_stiff_confirmations(2)
            .unwrap()
            .with_nonstiff_confirmations(2)
            .unwrap()
            .with_rejection_confirmations(2)
            .unwrap()
            .with_minimum_residence_steps(1)
            .unwrap()
    }

    #[test]
    fn thresholds_are_inclusive_and_dead_band_breaks_consecutive_evidence() {
        let mut detector = AutoSwitchDetector::new(config());

        assert_eq!(detector.observe_accepted(0.90), None);
        assert_eq!(
            detector.observe_accepted(0.90),
            Some(AutomaticBranch::Stiff)
        );
        assert_eq!(detector.observe_accepted(0.75), None);
        assert_eq!(detector.observe_accepted(0.80), None);
        assert_eq!(detector.observe_accepted(0.75), None);
        assert_eq!(
            detector.observe_accepted(0.75),
            Some(AutomaticBranch::NonStiff)
        );
    }

    #[test]
    fn switches_on_exact_confirmation_count() {
        let config = config().with_stiff_confirmations(3).unwrap();
        let mut detector = AutoSwitchDetector::new(config);

        assert_eq!(detector.observe_accepted(1.0), None);
        assert_eq!(detector.observe_accepted(1.0), None);
        assert_eq!(detector.branch(), AutomaticBranch::NonStiff);
        assert_eq!(detector.observe_accepted(1.0), Some(AutomaticBranch::Stiff));
    }

    #[test]
    fn minimum_residence_delays_diagnostic_switches() {
        let config = config().with_minimum_residence_steps(4).unwrap();
        let mut detector = AutoSwitchDetector::new(config);

        for _ in 0..3 {
            assert_eq!(detector.observe_accepted(1.0), None);
        }
        assert_eq!(detector.branch(), AutomaticBranch::NonStiff);
        assert_eq!(detector.observe_accepted(1.0), Some(AutomaticBranch::Stiff));
    }

    #[test]
    fn accepted_steps_reset_rejections_and_rejections_bypass_residence() {
        let config = config().with_minimum_residence_steps(100).unwrap();
        let mut detector = AutoSwitchDetector::new(config);

        assert_eq!(detector.observe_rejected(), None);
        assert_eq!(detector.observe_accepted(0.0), None);
        assert_eq!(detector.observe_rejected(), None);
        assert_eq!(detector.observe_rejected(), Some(AutomaticBranch::Stiff));
    }

    #[test]
    fn switch_back_can_be_disabled() {
        let config = config()
            .with_initial_branch(AutomaticBranch::Stiff)
            .with_allow_switch_back(false);
        let mut detector = AutoSwitchDetector::new(config);

        for _ in 0..10 {
            assert_eq!(detector.observe_accepted(0.0), None);
        }
        assert_eq!(detector.branch(), AutomaticBranch::Stiff);
    }

    #[test]
    fn evidence_can_be_reset_without_changing_the_branch() {
        let mut detector = AutoSwitchDetector::new(config());

        assert_eq!(detector.observe_accepted(1.0), None);
        detector.reset_evidence();
        assert_eq!(detector.observe_accepted(1.0), None);
        assert_eq!(detector.observe_accepted(1.0), Some(AutomaticBranch::Stiff));
        detector.reset_evidence();
        assert_eq!(detector.branch(), AutomaticBranch::Stiff);
    }

    #[test]
    fn nonfinite_diagnostics_are_conservative_stiff_evidence() {
        let mut detector = AutoSwitchDetector::new(config());

        assert_eq!(detector.observe_accepted(f64::NAN), None);
        assert_eq!(
            detector.observe_accepted(f64::INFINITY),
            Some(AutomaticBranch::Stiff)
        );

        for diagnostic in [f64::NAN, f64::NEG_INFINITY, f64::INFINITY] {
            assert_eq!(detector.observe_accepted(diagnostic), None);
            assert_eq!(detector.branch(), AutomaticBranch::Stiff);
        }
    }
}
