use crate::operator_problem::{LieGroupProblem, LieRepresentation, LinearOperatorProblem};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions};

mod engine;
mod schemes;

use engine::{solve_ode, solve_typed_group, solve_typed_operator, validate_inputs};

const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 6.0;

type OperatorResult = Result<(), SolveError>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Scheme {
    LieEuler,
    LinearExponential,
    MagnusMidpoint,
    MagnusLeapfrog,
    Rkmk2,
    Rkmk4,
    LieRk4,
    Cg2,
    Cg3,
    Cg4a,
    MagnusAdapt4,
    MagnusGauss4,
    MagnusGl4,
    MagnusGl6,
    MagnusNc6,
    MagnusGl8,
    MagnusNc8,
}

impl Scheme {
    const fn order(self) -> usize {
        match self {
            Self::LieEuler | Self::LinearExponential => 1,
            Self::MagnusMidpoint | Self::MagnusLeapfrog | Self::Rkmk2 | Self::Cg2 => 2,
            Self::Cg3 => 3,
            Self::Rkmk4
            | Self::LieRk4
            | Self::Cg4a
            | Self::MagnusAdapt4
            | Self::MagnusGauss4
            | Self::MagnusGl4 => 4,
            Self::MagnusGl6 | Self::MagnusNc6 => 6,
            Self::MagnusGl8 | Self::MagnusNc8 => 8,
        }
    }

    const fn adaptive(self) -> bool {
        matches!(self, Self::MagnusAdapt4)
    }
}

/// Algorithms acting on `u' = A(u,p,t)u` through dense exponential actions.
///
/// This trait is a downstream extension point. Implementors can evaluate the
/// checked operator through [`LinearOperatorProblem::evaluate_operator`] and
/// construct checked output with [`Solution::from_saved`].
pub trait LinearOperatorAlgorithm {
    /// Classical order reported by OrdinaryDiffEqLinear.
    fn order(&self) -> usize;

    /// Solves a linear-operator problem after validating its common inputs.
    fn solve_operator<O, P>(
        &self,
        problem: &LinearOperatorProblem<O, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        O: Fn(&mut [f64], &[f64], &P, f64),
    {
        validate_inputs(problem.initial_state(), problem.time_span(), options)?;
        let mut solution = self.solve_operator_validated(problem, options)?;
        solution.set_state_shape_checked(&[problem.dimension()])?;
        Ok(solution)
    }

    /// Executes the numerical method after common inputs have been checked.
    ///
    /// Implementors may rely on [`LinearOperatorAlgorithm::solve_operator`]
    /// having validated the state, time span, tolerances, step bounds, callback
    /// tolerance, and requested output times. Direct callers of this lower-level
    /// hook are responsible for preserving those invariants. Implementors
    /// remain responsible for honoring adaptive stepping, step bounds, saving,
    /// time stops, and dense-output retention, or for returning the
    /// corresponding typed error when a capability is not supported.
    fn solve_operator_validated<O, P>(
        &self,
        problem: &LinearOperatorProblem<O, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        O: Fn(&mut [f64], &[f64], &P, f64);
}

/// Algorithms acting on vector homogeneous spaces or matrix Lie groups.
///
/// This trait is a downstream extension point. Implementors can evaluate the
/// checked generator through [`LieGroupProblem::evaluate_operator`] and
/// construct checked output with [`Solution::from_saved`]. The checked entry
/// point preserves matrix state shape in the returned solution.
pub trait LieGroupAlgorithm {
    /// Classical order reported by OrdinaryDiffEqLinear.
    fn order(&self) -> usize;

    /// Solves a Lie-group problem after validating its common inputs.
    fn solve_group<O, P>(
        &self,
        problem: &LieGroupProblem<O, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        O: Fn(&mut [f64], &[f64], &P, f64),
    {
        validate_inputs(problem.initial_state(), problem.time_span(), options)?;
        let mut solution = self.solve_group_validated(problem, options)?;
        if problem.is_matrix_state() {
            solution
                .set_state_shape_checked(&[problem.group_dimension(), problem.group_dimension()])?;
        } else {
            solution.set_state_shape_checked(&[problem.group_dimension()])?;
        }
        Ok(solution)
    }

    /// Executes the numerical method after common inputs have been checked.
    ///
    /// Implementors may rely on [`LieGroupAlgorithm::solve_group`] having
    /// validated the state, time span, tolerances, step bounds, callback
    /// tolerance, and requested output times. Direct callers of this lower-level
    /// hook are responsible for preserving those invariants. Implementors
    /// remain responsible for honoring adaptive stepping, step bounds, saving,
    /// time stops, and dense-output retention, or for returning the
    /// corresponding typed error when a capability is not supported.
    fn solve_group_validated<O, P>(
        &self,
        problem: &LieGroupProblem<O, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        O: Fn(&mut [f64], &[f64], &P, f64);
}

/// Solves a typed dense linear-operator problem.
pub fn solve_linear_operator<O, P, A>(
    problem: &LinearOperatorProblem<O, P>,
    algorithm: A,
    options: &SolveOptions,
) -> Result<Solution, SolveError>
where
    O: Fn(&mut [f64], &[f64], &P, f64),
    A: LinearOperatorAlgorithm,
{
    algorithm.solve_operator(problem, options)
}

/// Solves a typed vector or matrix Lie-group problem.
pub fn solve_lie_group<O, P, A>(
    problem: &LieGroupProblem<O, P>,
    algorithm: A,
    options: &SolveOptions,
) -> Result<Solution, SolveError>
where
    O: Fn(&mut [f64], &[f64], &P, f64),
    A: LieGroupAlgorithm,
{
    algorithm.solve_group(problem, options)
}

macro_rules! linear_algorithm {
    ($name:ident, $scheme:ident, $documentation:literal) => {
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
                solve_ode(problem, options, Scheme::$scheme)
            }
        }

        impl LinearOperatorAlgorithm for $name {
            fn order(&self) -> usize {
                Scheme::$scheme.order()
            }

            fn solve_operator_validated<O, P>(
                &self,
                problem: &LinearOperatorProblem<O, P>,
                options: &SolveOptions,
            ) -> Result<Solution, SolveError>
            where
                O: Fn(&mut [f64], &[f64], &P, f64),
            {
                solve_typed_operator(problem, options, Scheme::$scheme)
            }
        }

        impl LieGroupAlgorithm for $name {
            fn order(&self) -> usize {
                Scheme::$scheme.order()
            }

            fn solve_group_validated<O, P>(
                &self,
                problem: &LieGroupProblem<O, P>,
                options: &SolveOptions,
            ) -> Result<Solution, SolveError>
            where
                O: Fn(&mut [f64], &[f64], &P, f64),
            {
                if problem.representation != LieRepresentation::Vector {
                    return Err(SolveError::UnsupportedProblemRepresentation);
                }
                solve_typed_group(problem, options, Scheme::$scheme)
            }
        }
    };
}

linear_algorithm!(
    LieEuler,
    LieEuler,
    "First-order Lie--Euler exponential action."
);
linear_algorithm!(
    LinearExponential,
    LinearExponential,
    "Exact exponential stepping for a constant dense linear operator."
);
linear_algorithm!(
    MagnusMidpoint,
    MagnusMidpoint,
    "Second-order exponential midpoint Magnus method."
);
linear_algorithm!(
    MagnusLeapfrog,
    MagnusLeapfrog,
    "Two-step exponential Magnus leapfrog method."
);
linear_algorithm!(
    RKMK2,
    Rkmk2,
    "Second-order Runge--Kutta--Munthe-Kaas method."
);
linear_algorithm!(
    RKMK4,
    Rkmk4,
    "Fourth-order Runge--Kutta--Munthe-Kaas method."
);
linear_algorithm!(
    LieRK4,
    LieRk4,
    "Fourth-order Lie-group Runge--Kutta method."
);
linear_algorithm!(
    CG2,
    Cg2,
    "Second-order Crouch--Grossman composition method."
);
linear_algorithm!(CG3, Cg3, "Third-order Crouch--Grossman composition method.");
linear_algorithm!(
    CG4a,
    Cg4a,
    "Fourth-order Crouch--Grossman composition method A."
);
linear_algorithm!(
    MagnusAdapt4,
    MagnusAdapt4,
    "Adaptive embedded fourth-order commutator-free Magnus method."
);
linear_algorithm!(
    MagnusGauss4,
    MagnusGauss4,
    "Fourth-order two-node Gauss Magnus method."
);
linear_algorithm!(
    MagnusGL4,
    MagnusGl4,
    "Fourth-order Gauss--Legendre Magnus method."
);
linear_algorithm!(
    MagnusGL6,
    MagnusGl6,
    "Sixth-order Gauss--Legendre Magnus method."
);
linear_algorithm!(
    MagnusNC6,
    MagnusNc6,
    "Sixth-order Newton--Cotes Magnus method."
);
linear_algorithm!(
    MagnusGL8,
    MagnusGl8,
    "Eighth-order Gauss--Legendre Magnus method."
);
linear_algorithm!(
    MagnusNC8,
    MagnusNc8,
    "Eighth-order Newton--Cotes Magnus method."
);

/// Cayley transform method for matrix states updated by group conjugation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CayleyEuler;
