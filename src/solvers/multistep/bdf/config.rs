use super::super::tableaux::backward_differentiation;
use super::kernel::{FbdfKernel, QndfKernel};
use crate::integrator::integrate as drive_integration;
use crate::tableau::{LinearMultistepTableau, TableauAccessError};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions};

/// Adaptive-order quasi-constant-step NDF, orders one through five.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qndf;

/// Adaptive-order quasi-constant-step BDF (`QNDF` with all kappa values zero).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qbdf;

/// Adaptive-order fixed-leading-coefficient BDF, orders one through five.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Fbdf;

/// Exact Julia-compatible spelling alias for [`Qndf`].
pub type QNDF = Qndf;
/// Exact Julia-compatible spelling alias for [`Qbdf`].
pub type QBDF = Qbdf;
/// Exact Julia-compatible spelling alias for [`Fbdf`].
pub type FBDF = Fbdf;

#[allow(non_upper_case_globals)]
/// Exact OrdinaryDiffEq-compatible value spelling for [`Qndf`].
pub const QNDF: Qndf = Qndf;
#[allow(non_upper_case_globals)]
/// Exact OrdinaryDiffEq-compatible value spelling for [`Qbdf`].
pub const QBDF: Qbdf = Qbdf;
#[allow(non_upper_case_globals)]
/// Exact OrdinaryDiffEq-compatible value spelling for [`Fbdf`].
pub const FBDF: Fbdf = Fbdf;

macro_rules! tableau_access {
    ($($method:ty),+ $(,)?) => {$(
        impl $method {
            /// Returns an order's shared BDF base formula and optional NDF modifier.
            ///
            /// QNDF applies `ndf_kappa()`; QBDF and FBDF use the unmodified
            /// `alpha()`/`beta()` formula.
            ///
            /// # Errors
            ///
            /// Returns [`TableauAccessError::UnsupportedOrder`] for orders
            /// outside `1..=5`, and preserves resource validation or
            /// family-invariant failures.
            pub fn tableau(
                &self,
                order: usize,
            ) -> Result<&'static LinearMultistepTableau, TableauAccessError> {
                backward_differentiation(order)
            }
        }
    )+};
}
tableau_access!(Qndf, Qbdf, Fbdf);

impl OdeAlgorithm for Qndf {
    fn solve_validated<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        drive_integration(
            problem,
            options,
            QndfKernel::new(problem.initial_state().len(), true),
        )
    }
}

impl OdeAlgorithm for Qbdf {
    fn solve_validated<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        drive_integration(
            problem,
            options,
            QndfKernel::new(problem.initial_state().len(), false),
        )
    }
}

impl OdeAlgorithm for Fbdf {
    fn solve_validated<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        drive_integration(
            problem,
            options,
            FbdfKernel::new(problem.initial_state().len()),
        )
    }
}
