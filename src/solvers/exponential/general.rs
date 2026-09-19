use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions};

mod engine;
mod evaluation;
mod matrix;
mod methods;

use engine::solve_scheme;
pub(crate) use matrix::{identity, mat_mul, mat_vec, matrix_exp};

#[cfg(test)]
use methods::phi_action;

const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 6.0;

/// Common metadata for exponential Runge--Kutta algorithms.
pub trait ExponentialAlgorithm: OdeAlgorithm {
    /// Classical convergence order of the method.
    fn order(&self) -> usize;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Scheme {
    LawsonEuler,
    NorsettEuler,
    Etdrk2,
    Etdrk3,
    Etdrk4,
    HochOst4,
    Exp4,
    Epirk4s3A,
    Epirk4s3B,
    Epirk5s3,
    Exprb53s3,
    Epirk5P1,
    Epirk5P2,
    Etd2,
    Exprb32,
    Exprb43,
}

impl Scheme {
    const fn order(self) -> usize {
        match self {
            Self::LawsonEuler | Self::NorsettEuler => 1,
            Self::Etdrk2 | Self::Etd2 => 2,
            Self::Etdrk3 | Self::Exprb32 => 3,
            Self::Etdrk4
            | Self::HochOst4
            | Self::Exp4
            | Self::Epirk4s3A
            | Self::Epirk4s3B
            | Self::Exprb43 => 4,
            Self::Epirk5s3 | Self::Exprb53s3 | Self::Epirk5P1 | Self::Epirk5P2 => 5,
        }
    }

    const fn adaptive(self) -> bool {
        matches!(self, Self::Exprb32 | Self::Exprb43)
    }
}

macro_rules! exponential_algorithm {
    ($name:ident, $scheme:ident, $order:expr, $documentation:literal) => {
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
                solve_scheme(problem, options, Scheme::$scheme)
            }
        }

        impl ExponentialAlgorithm for $name {
            fn order(&self) -> usize {
                $order
            }
        }
    };
}

exponential_algorithm!(
    LawsonEuler,
    LawsonEuler,
    1,
    "First-order Lawson exponential Euler method."
);
exponential_algorithm!(
    NorsettEuler,
    NorsettEuler,
    1,
    "First-order Nørsett exponential Euler method."
);
/// Exact spelling alias for [`NorsettEuler`].
pub type ETD1 = NorsettEuler;
/// Value constructor for the exact [`NorsettEuler`] spelling alias.
#[allow(non_upper_case_globals)]
pub const ETD1: NorsettEuler = NorsettEuler;
exponential_algorithm!(
    ETDRK2,
    Etdrk2,
    2,
    "Second-order exponential Runge--Kutta method."
);
exponential_algorithm!(
    ETDRK3,
    Etdrk3,
    3,
    "Third-order exponential Runge--Kutta method."
);
exponential_algorithm!(
    ETDRK4,
    Etdrk4,
    4,
    "Fourth-order exponential Runge--Kutta method."
);
exponential_algorithm!(
    HochOst4,
    HochOst4,
    4,
    "Hochbruck--Ostermann stiff-order-four method."
);
exponential_algorithm!(Exp4, Exp4, 4, "Hochbruck--Lubich--Selhofer Exp4 method.");
exponential_algorithm!(
    EPIRK4s3A,
    Epirk4s3A,
    4,
    "Three-stage stiff-order-four EPIRK method A."
);
exponential_algorithm!(
    EPIRK4s3B,
    Epirk4s3B,
    4,
    "Three-stage stiff-order-four EPIRK method B."
);
exponential_algorithm!(
    EPIRK5s3,
    Epirk5s3,
    5,
    "Three-stage fifth-order horizontal EPIRK method."
);
exponential_algorithm!(
    EXPRB53s3,
    Exprb53s3,
    5,
    "Three-stage fifth-order exponential Rosenbrock method."
);
exponential_algorithm!(EPIRK5P1, Epirk5P1, 5, "Fifth-order EPIRK method P1.");
exponential_algorithm!(EPIRK5P2, Epirk5P2, 5, "Fifth-order EPIRK method P2.");
exponential_algorithm!(
    ETD2,
    Etd2,
    2,
    "Second-order multistep exponential time-differencing method."
);
exponential_algorithm!(
    Exprb32,
    Exprb32,
    3,
    "Adaptive embedded exponential Rosenbrock 3(2) method."
);
exponential_algorithm!(
    Exprb43,
    Exprb43,
    4,
    "Adaptive embedded exponential Rosenbrock 4(3) method."
);

#[cfg(test)]
mod tests;
