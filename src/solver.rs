use std::error::Error;
use std::fmt::{self, Display, Formatter};

use crate::{OdeProblem, Solution};

/// Controls which accepted states are retained in a [`Solution`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SaveMode {
    /// Save the initial state and every accepted step.
    #[default]
    EveryStep,
    /// Save only the initial and final states.
    Endpoints,
}

/// Common options for adaptive ODE solvers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolveOptions {
    /// Absolute local-error tolerance.
    pub absolute_tolerance: f64,
    /// Relative local-error tolerance.
    pub relative_tolerance: f64,
    /// Initial step-size magnitude. The solver estimates it when absent.
    pub initial_step: Option<f64>,
    /// Maximum allowed step-size magnitude.
    pub max_step: f64,
    /// Maximum number of attempted steps.
    pub max_steps: usize,
    /// Accepted states retained in the solution.
    pub save: SaveMode,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            absolute_tolerance: 1.0e-6,
            relative_tolerance: 1.0e-3,
            initial_step: None,
            max_step: f64::INFINITY,
            max_steps: 100_000,
            save: SaveMode::EveryStep,
        }
    }
}

/// A failure to configure or complete an ODE solve.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SolveError {
    EmptyState,
    NonFiniteInitialState,
    InvalidTimeSpan,
    InvalidTolerance,
    InvalidInitialStep,
    InvalidMaxStep,
    InvalidMaxSteps,
    NonFiniteDerivative,
    StepSizeUnderflow,
    MaxStepsExceeded,
}

impl Display for SolveError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::EmptyState => "the initial state is empty",
            Self::NonFiniteInitialState => "the initial state contains a non-finite value",
            Self::InvalidTimeSpan => "the time span must contain distinct finite values",
            Self::InvalidTolerance => {
                "absolute and relative tolerances must be finite and positive"
            }
            Self::InvalidInitialStep => "the initial step must be finite and positive",
            Self::InvalidMaxStep => "the maximum step must be positive and not NaN",
            Self::InvalidMaxSteps => "the maximum step count must be positive",
            Self::NonFiniteDerivative => "the right-hand side produced a non-finite derivative",
            Self::StepSizeUnderflow => "the adaptive step size underflowed",
            Self::MaxStepsExceeded => "the solver exceeded its maximum attempted step count",
        })
    }
}

impl Error for SolveError {}

/// An ODE integration algorithm.
pub trait OdeAlgorithm {
    fn solve<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64);
}

/// Solves an ODE problem with a selected algorithm.
pub fn solve<F, P, A>(
    problem: &OdeProblem<F, P>,
    algorithm: A,
    options: &SolveOptions,
) -> Result<Solution, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
    A: OdeAlgorithm,
{
    validate(problem, options)?;
    algorithm.solve(problem, options)
}

fn validate<F, P>(problem: &OdeProblem<F, P>, options: &SolveOptions) -> Result<(), SolveError> {
    if problem.initial_state().is_empty() {
        return Err(SolveError::EmptyState);
    }
    if !problem
        .initial_state()
        .iter()
        .all(|value| value.is_finite())
    {
        return Err(SolveError::NonFiniteInitialState);
    }

    let (start, end) = problem.time_span();
    if !start.is_finite() || !end.is_finite() || start == end {
        return Err(SolveError::InvalidTimeSpan);
    }

    if !options.absolute_tolerance.is_finite()
        || options.absolute_tolerance <= 0.0
        || !options.relative_tolerance.is_finite()
        || options.relative_tolerance <= 0.0
    {
        return Err(SolveError::InvalidTolerance);
    }
    if options
        .initial_step
        .is_some_and(|step| !step.is_finite() || step <= 0.0)
    {
        return Err(SolveError::InvalidInitialStep);
    }
    if options.max_step.is_nan() || options.max_step <= 0.0 {
        return Err(SolveError::InvalidMaxStep);
    }
    if options.max_steps == 0 {
        return Err(SolveError::InvalidMaxSteps);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{OdeAlgorithm, OdeProblem, Solution, SolverStats};

    use super::{SaveMode, SolveError, SolveOptions, solve};

    struct Noop;
    type TestRhs = fn(&mut [f64], &[f64], &(), f64);

    impl OdeAlgorithm for Noop {
        fn solve<F, P>(
            &self,
            problem: &OdeProblem<F, P>,
            _: &SolveOptions,
        ) -> Result<Solution, SolveError>
        where
            F: Fn(&mut [f64], &[f64], &P, f64),
        {
            let state = problem.initial_state().to_vec();
            Ok(Solution::new(
                vec![problem.time_span().0],
                state,
                problem.initial_state().len(),
                SolverStats::default(),
            ))
        }
    }

    fn problem(
        initial_state: Vec<f64>,
        time_span: (f64, f64),
    ) -> OdeProblem<TestRhs, ()> {
        fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
            du.copy_from_slice(u);
        }

        OdeProblem::new(rhs, initial_state, time_span, ())
    }

    #[test]
    fn defaults_match_sciml_tolerances() {
        let options = SolveOptions::default();

        assert_eq!(options.absolute_tolerance, 1.0e-6);
        assert_eq!(options.relative_tolerance, 1.0e-3);
        assert_eq!(options.save, SaveMode::EveryStep);
    }

    #[test]
    fn validates_problem_before_dispatch() {
        assert_eq!(
            solve(
                &problem(Vec::new(), (0.0, 1.0)),
                Noop,
                &SolveOptions::default()
            ),
            Err(SolveError::EmptyState)
        );
        assert_eq!(
            solve(
                &problem(vec![f64::NAN], (0.0, 1.0)),
                Noop,
                &SolveOptions::default()
            ),
            Err(SolveError::NonFiniteInitialState)
        );
        assert_eq!(
            solve(
                &problem(vec![1.0], (0.0, 0.0)),
                Noop,
                &SolveOptions::default()
            ),
            Err(SolveError::InvalidTimeSpan)
        );
    }

    #[test]
    fn validates_solver_options_before_dispatch() {
        let mut options = SolveOptions {
            absolute_tolerance: 0.0,
            ..SolveOptions::default()
        };
        assert_eq!(
            solve(&problem(vec![1.0], (0.0, 1.0)), Noop, &options),
            Err(SolveError::InvalidTolerance)
        );

        options = SolveOptions {
            initial_step: Some(f64::INFINITY),
            ..SolveOptions::default()
        };
        assert_eq!(
            solve(&problem(vec![1.0], (0.0, 1.0)), Noop, &options),
            Err(SolveError::InvalidInitialStep)
        );
    }
}
