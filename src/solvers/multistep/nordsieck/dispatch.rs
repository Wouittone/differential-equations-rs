use super::config::{AN5, JVODE, JvodeAdams, JvodeBdf};
use super::kernel::NordsieckKernel;
use crate::integrator::integrate as drive_integration;
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions};

impl OdeAlgorithm for AN5 {
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
            NordsieckKernel::an5(problem.initial_state().len()),
        )
    }
}

impl OdeAlgorithm for JVODE {
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
            NordsieckKernel::jvode(problem.initial_state().len(), *self),
        )
    }
}

impl OdeAlgorithm for JvodeAdams {
    fn solve_validated<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        JVODE::adams().solve(problem, options)
    }
}

impl OdeAlgorithm for JvodeBdf {
    fn solve_validated<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        JVODE::bdf().solve(problem, options)
    }
}
