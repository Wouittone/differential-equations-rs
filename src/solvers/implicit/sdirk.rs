//! Native regular-ODE SDIRK methods.
//!
//! This module currently contains the two-stage adaptive SDIRK2/ESDIRK
//! method from the pinned OrdinaryDiffEqSDIRK tableau.

// The pinned SDIRK/ESDIRK catalogue intentionally preserves upstream decimal
// literals, including values with more written digits than f64 can represent.
#![allow(clippy::excessive_precision)]

use crate::integrator::integrate as drive_integration;
use crate::tableau::{RungeKuttaTableau, load_tableau};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions};

mod catalog;
mod kernel;

use catalog::{ExtendedKind, extended_resource, sdirk2_resource};
use kernel::{ExtendedKernel, Sdirk2Kernel};

/// The adaptive second-order two-stage SDIRK/ESDIRK method.
///
/// The pinned tableau has stage times (1, 0), unit diagonal, coupling
/// a21 = -1, primary weights (1/2, 1/2), and embedded weights
/// (1/2, -1/2). This is a regular identity-mass ODE method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Sdirk2;

impl Sdirk2 {
    /// Returns this method's lazily materialized, validated tableau.
    pub fn tableau(&self) -> Result<&'static RungeKuttaTableau, crate::tableau::TableauError> {
        load_tableau(sdirk2_resource())
    }
}

impl OdeAlgorithm for Sdirk2 {
    fn solve_validated<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        let tableau = self.tableau().map_err(SolveError::from)?;
        drive_integration(
            problem,
            options,
            Sdirk2Kernel::new(problem.initial_state().len(), tableau),
        )
    }
}

macro_rules! extended_algorithm {
    ($name:ident, $kind:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        pub struct $name;

        impl $name {
            /// Returns this method's lazily materialized, validated tableau.
            pub fn tableau(
                &self,
            ) -> Result<&'static RungeKuttaTableau, crate::tableau::TableauError> {
                load_tableau(extended_resource(ExtendedKind::$kind))
            }
        }

        impl OdeAlgorithm for $name {
            fn solve_validated<F, P>(
                &self,
                problem: &OdeProblem<F, P>,
                options: &SolveOptions,
            ) -> Result<Solution, SolveError>
            where
                F: crate::OdeFunction<P>,
            {
                let tableau = self.tableau().map_err(SolveError::from)?;
                drive_integration(
                    problem,
                    options,
                    ExtendedKernel::new(tableau, problem.initial_state().len()),
                )
            }
        }
    };
}

extended_algorithm!(Ars222, Ars222, "Pinned ARS(2,2,2) regular-ODE projection.");
extended_algorithm!(Ars232, Ars232, "Pinned ARS(2,3,2) regular-ODE projection.");
extended_algorithm!(Ars343, Ars343, "Pinned ARS(3,4,3) regular-ODE projection.");
extended_algorithm!(Ars443, Ars443, "Pinned ARS(4,4,3) regular-ODE projection.");
extended_algorithm!(Bhr553, Bhr553, "Pinned BHR(5,5,3) regular-ODE projection.");
extended_algorithm!(Cfnlirk3, Cfnlirk3, "Pinned CFNLIRK3 regular-ODE method.");
extended_algorithm!(
    Esdirk325L2Sa,
    Esdirk325,
    "Pinned ESDIRK325L2SA regular-ODE method."
);
extended_algorithm!(
    Esdirk436L2Sa2,
    Esdirk436,
    "Pinned ESDIRK436L2SA2 regular-ODE method."
);
extended_algorithm!(
    Esdirk437L2Sa,
    Esdirk437,
    "Pinned ESDIRK437L2SA regular-ODE method."
);
extended_algorithm!(
    Esdirk547L2Sa2,
    Esdirk547,
    "Pinned ESDIRK547L2SA2 regular-ODE method."
);
extended_algorithm!(
    Esdirk54I8L2Sa,
    Esdirk54,
    "Pinned ESDIRK54I8L2SA regular-ODE method."
);
extended_algorithm!(
    Esdirk659L2Sa,
    Esdirk659,
    "Pinned ESDIRK659L2SA regular-ODE method."
);
extended_algorithm!(Hairer4, Hairer4, "Pinned Hairer4 regular-ODE method.");
extended_algorithm!(Hairer42, Hairer42, "Pinned Hairer42 regular-ODE method.");
extended_algorithm!(
    ImexSsp222,
    ImexSsp222,
    "Pinned IMEXSSP222 implicit regular-ODE projection."
);
extended_algorithm!(
    ImexSsp2322,
    ImexSsp2322,
    "Pinned IMEXSSP2322 implicit regular-ODE projection."
);
extended_algorithm!(
    ImexSsp3332,
    ImexSsp3332,
    "Pinned IMEXSSP3332 implicit regular-ODE projection."
);
extended_algorithm!(
    ImexSsp3433,
    ImexSsp3433,
    "Pinned IMEXSSP3433 implicit regular-ODE projection."
);
extended_algorithm!(
    KenCarp3,
    KenCarp3,
    "Pinned KenCarp3 regular-ODE projection."
);
extended_algorithm!(
    KenCarp4,
    KenCarp4,
    "Pinned KenCarp4 regular-ODE projection."
);
extended_algorithm!(
    KenCarp47,
    KenCarp47,
    "Pinned KenCarp47 regular-ODE projection."
);
extended_algorithm!(
    KenCarp5,
    KenCarp5,
    "Pinned KenCarp5 regular-ODE projection."
);
extended_algorithm!(
    KenCarp58,
    KenCarp58,
    "Pinned KenCarp58 regular-ODE projection."
);
extended_algorithm!(Kvaerno3, Kvaerno3, "Pinned Kvaerno3 regular-ODE method.");
extended_algorithm!(Kvaerno4, Kvaerno4, "Pinned Kvaerno4 regular-ODE method.");
extended_algorithm!(Kvaerno5, Kvaerno5, "Pinned Kvaerno5 regular-ODE method.");
extended_algorithm!(Sdirk22, Sdirk22, "Pinned SDIRK22 regular-ODE method.");
extended_algorithm!(Sfsdirk4, Sfsdirk4, "Pinned SFSDIRK4 regular-ODE method.");
extended_algorithm!(Sfsdirk5, Sfsdirk5, "Pinned SFSDIRK5 regular-ODE method.");
extended_algorithm!(Sfsdirk6, Sfsdirk6, "Pinned SFSDIRK6 regular-ODE method.");
extended_algorithm!(Sfsdirk7, Sfsdirk7, "Pinned SFSDIRK7 regular-ODE method.");
extended_algorithm!(Sfsdirk8, Sfsdirk8, "Pinned SFSDIRK8 regular-ODE method.");
extended_algorithm!(SspSdirk2, SspSdirk2, "Pinned SSPSDIRK2 regular-ODE method.");

#[cfg(test)]
mod tests;
