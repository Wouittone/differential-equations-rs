use super::switching::solve_automatic;
use super::{AutoSwitchConfig, AutoSwitchConfigError, AutomaticStiffAlgorithm};
use crate::solvers::explicit::Dp5;
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions};

/// Automatic low-order Dormand--Prince composite.
///
/// `AutoDp5` begins with DP5 by default and changes branches at the current
/// accepted state when the shared stiffness policy accumulates enough
/// evidence. The integration driver, callback lifecycle, saved trajectory,
/// dense output, step budget, and statistics remain continuous across every
/// transition; the problem is never restarted or replayed.
#[derive(Clone, Debug, PartialEq)]
pub struct AutoDp5<A> {
    stiff_algorithm: A,
    switch_config: AutoSwitchConfig,
}

impl<A> AutoDp5<A> {
    /// Constructs an automatic DP5 pair with the default switching policy.
    pub const fn new(stiff_algorithm: A) -> Self {
        Self {
            stiff_algorithm,
            switch_config: AutoSwitchConfig::new(),
        }
    }

    /// Returns the configured stiff component.
    pub const fn stiff_algorithm(&self) -> &A {
        &self.stiff_algorithm
    }

    /// Returns the inspectable switching policy.
    pub const fn switch_config(&self) -> &AutoSwitchConfig {
        &self.switch_config
    }

    /// Replaces the switching policy after validating all invariants.
    pub fn with_switch_config(
        mut self,
        switch_config: AutoSwitchConfig,
    ) -> Result<Self, AutoSwitchConfigError> {
        switch_config.validate()?;
        self.switch_config = switch_config;
        Ok(self)
    }
}

impl<A: AutomaticStiffAlgorithm> OdeAlgorithm for AutoDp5<A> {
    fn solve_validated<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        let tableau = Dp5.tableau().map_err(SolveError::from)?;
        solve_automatic(
            problem,
            options,
            tableau,
            self.stiff_algorithm,
            self.switch_config,
        )
    }
}

/// Uppercase acronym spelling used by the pinned Julia algorithm name.
#[allow(non_camel_case_types)]
pub type AutoDP5<A> = AutoDp5<A>;
