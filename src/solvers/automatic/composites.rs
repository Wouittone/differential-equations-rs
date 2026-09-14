//! Public automatic/default algorithms.
//!
//! Automatic algorithms share one driver and change numerical kernels only at
//! accepted-state boundaries. Callback, saving, dense-output, controller, and
//! statistics state is therefore preserved without replaying the problem.

use super::switching::solve_automatic;
use super::{AutoSwitchConfig, AutoSwitchConfigError, AutomaticStiffAlgorithm};
use crate::solvers::explicit::{Tsit5, Vern6, Vern7, Vern8, Vern9};
use crate::solvers::rosenbrock::Rodas5P;
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions};

/// Defines an automatic non-stiff-first algorithm with in-flight switching.
macro_rules! automatic_facade {
    ($name:ident, $component:ident, $documentation:literal) => {
        #[doc = $documentation]
        #[derive(Clone, Debug, PartialEq)]
        pub struct $name<A> {
            stiff_algorithm: A,
            switch_config: AutoSwitchConfig,
        }

        impl<A> $name<A> {
            /// Constructs the automatic facade with its requested stiff branch.
            pub const fn new(stiff_algorithm: A) -> Self {
                Self {
                    stiff_algorithm,
                    switch_config: AutoSwitchConfig::new(),
                }
            }

            /// Returns the configured stiff branch.
            pub const fn stiff_algorithm(&self) -> &A {
                &self.stiff_algorithm
            }

            /// Returns the inspectable switching policy.
            pub const fn switch_config(&self) -> &AutoSwitchConfig {
                &self.switch_config
            }

            /// Replaces the switching policy after validating its invariants.
            pub fn with_switch_config(
                mut self,
                switch_config: AutoSwitchConfig,
            ) -> Result<Self, AutoSwitchConfigError> {
                switch_config.validate()?;
                self.switch_config = switch_config;
                Ok(self)
            }
        }

        impl<A: AutomaticStiffAlgorithm> OdeAlgorithm for $name<A> {
            fn solve_validated<F, P>(
                &self,
                problem: &OdeProblem<F, P>,
                options: &SolveOptions,
            ) -> Result<Solution, SolveError>
            where
                F: crate::OdeFunction<P>,
            {
                let tableau = $component
                    .tableau()
                    .map_err(|_| SolveError::InvalidTableau)?;
                solve_automatic(
                    problem,
                    options,
                    tableau,
                    self.stiff_algorithm,
                    self.switch_config,
                )
            }
        }
    };
}

automatic_facade!(
    AutoTsit5,
    Tsit5,
    "Automatically switches between `Tsit5` and a compatible stiff branch at the current accepted state without restarting the solve."
);
automatic_facade!(
    AutoVern6,
    Vern6,
    "Automatically switches between `Vern6` and a compatible stiff branch at the current accepted state without restarting the solve."
);
automatic_facade!(
    AutoVern7,
    Vern7,
    "Automatically switches between `Vern7` and a compatible stiff branch at the current accepted state without restarting the solve."
);
automatic_facade!(
    AutoVern8,
    Vern8,
    "Automatically switches between `Vern8` and a compatible stiff branch at the current accepted state without restarting the solve."
);
automatic_facade!(
    AutoVern9,
    Vern9,
    "Automatically switches between `Vern9` and a compatible stiff branch at the current accepted state without restarting the solve."
);

/// Default nonstiff algorithm facade over `Tsit5`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DefaultOdeAlgorithm;

impl OdeAlgorithm for DefaultOdeAlgorithm {
    fn solve_validated<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        Tsit5.solve(problem, options)
    }
}

/// Spelling-compatible alias for OrdinaryDiffEq's default nonstiff facade.
#[allow(non_camel_case_types)]
pub type DefaultODEAlgorithm = DefaultOdeAlgorithm;

/// Default stiff algorithm facade over `Rodas5P`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DefaultImplicitOdeAlgorithm;

impl OdeAlgorithm for DefaultImplicitOdeAlgorithm {
    fn solve_validated<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        Rodas5P.solve(problem, options)
    }
}

/// Spelling-compatible alias for OrdinaryDiffEq's default stiff facade.
#[allow(non_camel_case_types)]
pub type DefaultImplicitODEAlgorithm = DefaultImplicitOdeAlgorithm;

#[cfg(test)]
mod tests {
    use super::{AutoTsit5, AutoVern6, AutoVern7, AutoVern8, AutoVern9};
    use crate::solvers::explicit::{Tsit5, Vern6, Vern7, Vern8, Vern9};
    use crate::solvers::rosenbrock::Rodas5P;
    use crate::{OdeProblem, SaveMode, SolveOptions, solve};

    type ScalarRhs = fn(&mut [f64], &[f64], &(), f64);

    fn problem() -> OdeProblem<ScalarRhs, ()> {
        fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
            du[0] = u[0];
        }
        OdeProblem::new(rhs as ScalarRhs, vec![1.0], (0.0, 1.0), ())
    }

    fn options() -> SolveOptions {
        SolveOptions {
            absolute_tolerance: 1.0e-10,
            relative_tolerance: 1.0e-10,
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        }
    }

    #[test]
    fn automatic_facades_preserve_native_component_results() {
        assert_eq!(
            solve(&problem(), AutoTsit5::new(Rodas5P), &options())
                .unwrap()
                .last_state(),
            solve(&problem(), Tsit5, &options()).unwrap().last_state()
        );
        assert_eq!(
            solve(&problem(), AutoVern6::new(Rodas5P), &options())
                .unwrap()
                .last_state(),
            solve(&problem(), Vern6, &options()).unwrap().last_state()
        );
        assert_eq!(
            solve(&problem(), AutoVern7::new(Rodas5P), &options())
                .unwrap()
                .last_state(),
            solve(&problem(), Vern7, &options()).unwrap().last_state()
        );
        assert_eq!(
            solve(&problem(), AutoVern8::new(Rodas5P), &options())
                .unwrap()
                .last_state(),
            solve(&problem(), Vern8, &options()).unwrap().last_state()
        );
        assert_eq!(
            solve(&problem(), AutoVern9::new(Rodas5P), &options())
                .unwrap()
                .last_state(),
            solve(&problem(), Vern9, &options()).unwrap().last_state()
        );
    }
}
