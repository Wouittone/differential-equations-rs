//! Public automatic/default algorithms.
//!
//! The automatic algorithms run their native non-stiff component first. The
//! Rust driver does not yet expose the state needed for an in-flight algorithm
//! switch, so a numerical failure restarts the problem from its initial state
//! with the configured stiff component. This deterministic fallback ensures
//! the stiff branch is functional rather than retained as ignored metadata.

use std::marker::PhantomData;

use crate::solvers::explicit::tsit5::Tsit5;
use crate::solvers::explicit::verner::{Vern6, Vern7, Vern8, Vern9};
use crate::solvers::rosenbrock::rosenbrock_extended::Rodas5P;
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions};

fn should_retry_with_stiff(error: SolveError) -> bool {
    matches!(
        error,
        SolveError::NonFiniteDerivative
            | SolveError::StepSizeUnderflow
            | SolveError::MaxStepsExceeded
    )
}

/// Defines an automatic non-stiff-first algorithm with a stiff fallback.
macro_rules! automatic_facade {
    ($name:ident, $component:ident, $documentation:literal) => {
        #[doc = $documentation]
        #[derive(Clone, Debug, PartialEq)]
        pub struct $name<A> {
            /// Stiff component used after a recoverable numerical failure.
            pub stiff_algorithm: A,
            marker: PhantomData<fn() -> $component>,
        }

        impl<A> $name<A> {
            /// Constructs the automatic facade with its requested stiff branch.
            pub const fn new(stiff_algorithm: A) -> Self {
                Self {
                    stiff_algorithm,
                    marker: PhantomData,
                }
            }

            /// Returns the configured stiff branch.
            pub const fn stiff_algorithm(&self) -> &A {
                &self.stiff_algorithm
            }
        }

        impl<A: OdeAlgorithm> OdeAlgorithm for $name<A> {
            fn solve_validated<F, P>(
                &self,
                problem: &OdeProblem<F, P>,
                options: &SolveOptions,
            ) -> Result<Solution, SolveError>
            where
                F: crate::OdeFunction<P>,
            {
                match $component.solve(problem, options) {
                    Err(error) if should_retry_with_stiff(error) => {
                        self.stiff_algorithm.solve(problem, options)
                    }
                    result => result,
                }
            }
        }
    };
}

automatic_facade!(
    AutoTsit5,
    Tsit5,
    "Runs `Tsit5` first and restarts with the configured stiff algorithm after a recoverable numerical failure. This is a full-solve fallback, not an in-flight switch; right-hand-side functions and callbacks with external side effects can be evaluated again."
);
automatic_facade!(
    AutoVern6,
    Vern6,
    "Runs `Vern6` first and restarts with the configured stiff algorithm after a recoverable numerical failure. This is a full-solve fallback, not an in-flight switch; right-hand-side functions and callbacks with external side effects can be evaluated again."
);
automatic_facade!(
    AutoVern7,
    Vern7,
    "Runs `Vern7` first and restarts with the configured stiff algorithm after a recoverable numerical failure. This is a full-solve fallback, not an in-flight switch; right-hand-side functions and callbacks with external side effects can be evaluated again."
);
automatic_facade!(
    AutoVern8,
    Vern8,
    "Runs `Vern8` first and restarts with the configured stiff algorithm after a recoverable numerical failure. This is a full-solve fallback, not an in-flight switch; right-hand-side functions and callbacks with external side effects can be evaluated again."
);
automatic_facade!(
    AutoVern9,
    Vern9,
    "Runs `Vern9` first and restarts with the configured stiff algorithm after a recoverable numerical failure. This is a full-solve fallback, not an in-flight switch; right-hand-side functions and callbacks with external side effects can be evaluated again."
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
    use crate::solvers::explicit::tsit5::Tsit5;
    use crate::solvers::explicit::verner::{Vern6, Vern7, Vern8, Vern9};
    use crate::solvers::rosenbrock::rosenbrock_extended::Rodas5P;
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
