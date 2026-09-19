//! Public variable-Adams algorithms and generic-driver dispatch.

use super::fixed::VariableAdamsKernel;
use super::method::{
    VCAB3_METHOD, VCAB4_METHOD, VCAB5_METHOD, VCABM3_METHOD, VCABM4_METHOD, VCABM5_METHOD,
};
use super::variable_order::VariableOrderAdamsKernel;
use crate::integrator::integrate;
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions};

macro_rules! algorithm {
    ($name:ident, $documentation:literal, $method:ident) => {
        #[doc = $documentation]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        pub struct $name;

        impl OdeAlgorithm for $name {
            fn solve_validated<F, P>(
                &self,
                problem: &OdeProblem<F, P>,
                options: &SolveOptions,
            ) -> Result<Solution, SolveError>
            where
                F: crate::OdeFunction<P>,
            {
                integrate(problem, options, VariableAdamsKernel::new(&$method))
            }
        }
    };
}

algorithm!(
    Vcab3,
    "The adaptive third-order variable-coefficient Adams--Bashforth method.",
    VCAB3_METHOD
);
algorithm!(
    Vcab4,
    "The adaptive fourth-order variable-coefficient Adams--Bashforth method.",
    VCAB4_METHOD
);
algorithm!(
    Vcab5,
    "The adaptive fifth-order variable-coefficient Adams--Bashforth method.",
    VCAB5_METHOD
);
algorithm!(
    Vcabm3,
    "The adaptive third-order variable-coefficient Adams--Moulton method.",
    VCABM3_METHOD
);
algorithm!(
    Vcabm4,
    "The adaptive fourth-order variable-coefficient Adams--Moulton method.",
    VCABM4_METHOD
);
algorithm!(
    Vcabm5,
    "The adaptive fifth-order variable-coefficient Adams--Moulton method.",
    VCABM5_METHOD
);

/// Adaptive-order, adaptive-step Adams--Moulton predictor/corrector method.
///
/// The order starts at one and rises through order twelve as accepted divided-
/// difference history becomes available. Rejections lower the active order.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Vcabm;

/// Exact Julia-compatible spelling alias for [`Vcabm`].
pub type VCABM = Vcabm;

#[allow(non_upper_case_globals)]
/// Exact OrdinaryDiffEq-compatible value spelling for [`Vcabm`].
pub const VCABM: Vcabm = Vcabm;

impl OdeAlgorithm for Vcabm {
    fn solve_validated<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        integrate(problem, options, VariableOrderAdamsKernel::new())
    }
}
