//! Public automatic/default algorithm facades.
//!
//! The pinned OrdinaryDiffEq constructors in this module are composites whose
//! runtime switching policy is not yet represented by the Rust driver. The
//! facades retain the public algorithm names and delegate regular ODE solves
//! to the corresponding native component while that capability is completed.

use std::marker::PhantomData;

use crate::{OdeAlgorithm, OdeProblem, Rodas5P, Solution, SolveError, SolveOptions, Tsit5};
use crate::{Vern6, Vern7, Vern8, Vern9};

/// Automatic Tsit5 facade over the native `Tsit5` component.
macro_rules! automatic_facade {
    ($name:ident, $component:ident, $documentation:literal) => {
        #[doc = $documentation]
        #[derive(Clone, Debug, PartialEq)]
        pub struct $name<A> {
            /// Stiff component retained for future runtime switching.
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

        impl<A> OdeAlgorithm for $name<A> {
            fn solve<F, P>(
                &self,
                problem: &OdeProblem<F, P>,
                options: &SolveOptions,
            ) -> Result<Solution, SolveError>
            where
                F: Fn(&mut [f64], &[f64], &P, f64),
            {
                let _ = &self.stiff_algorithm;
                $component.solve(problem, options)
            }
        }
    };
}

automatic_facade!(
    AutoTsit5,
    Tsit5,
    "Automatic Tsit5 facade over the native `Tsit5` component."
);
automatic_facade!(
    AutoVern6,
    Vern6,
    "Automatic Verner-6 facade over the native `Vern6` component."
);
automatic_facade!(
    AutoVern7,
    Vern7,
    "Automatic Verner-7 facade over the native `Vern7` component."
);
automatic_facade!(
    AutoVern8,
    Vern8,
    "Automatic Verner-8 facade over the native `Vern8` component."
);
automatic_facade!(
    AutoVern9,
    Vern9,
    "Automatic Verner-9 facade over the native `Vern9` component."
);

/// Default nonstiff algorithm facade over `Tsit5`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DefaultOdeAlgorithm;

impl OdeAlgorithm for DefaultOdeAlgorithm {
    fn solve<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
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
    fn solve<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
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
    use crate::{
        OdeProblem, Rodas5P, SaveMode, SolveOptions, Tsit5, Vern6, Vern7, Vern8, Vern9, solve,
    };

    fn problem() -> OdeProblem<impl Fn(&mut [f64], &[f64], &(), f64), ()> {
        OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
            vec![1.0],
            (0.0, 1.0),
            (),
        )
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
