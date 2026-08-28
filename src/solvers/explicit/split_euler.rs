//! First-order Euler integration for typed split ODE problems.

use crate::callback::CallbackOutcome;
use crate::integrator::{KernelCapabilities, StepEstimate, StepKernel, integrate};
use crate::solver::validate_state_time_options;
use crate::{OdeProblem, Solution, SolveError, SolveOptions, SolverStats, SplitOdeProblem};

/// Explicit Euler applied to the sum of a split problem's two right-hand sides.
///
/// Solver statistics count each paired explicit/implicit evaluation as one
/// logical right-hand-side evaluation, matching the historical combined-
/// problem implementation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SplitEuler;

/// Algorithm contract for typed split ODE problems.
pub trait SplitOdeAlgorithm {
    /// Solve a typed split problem with this algorithm.
    fn solve<FE, FI, P>(
        &self,
        problem: &SplitOdeProblem<FE, FI, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        FE: Fn(&mut [f64], &[f64], &P, f64),
        FI: Fn(&mut [f64], &[f64], &P, f64),
    {
        validate_state_time_options(problem.initial_state(), problem.time_span(), options)?;
        self.solve_validated(problem, options)
    }

    /// Executes the numerical method after common inputs have been checked.
    ///
    /// Implementors may rely on [`SplitOdeAlgorithm::solve`] having validated
    /// the shared state, time span, tolerances, step bounds, callback tolerance,
    /// and requested output times. User code should normally call
    /// [`SplitOdeAlgorithm::solve`] or [`solve_split`]; direct callers of this
    /// lower-level hook are responsible for preserving those invariants.
    fn solve_validated<FE, FI, P>(
        &self,
        problem: &SplitOdeProblem<FE, FI, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        FE: Fn(&mut [f64], &[f64], &P, f64),
        FI: Fn(&mut [f64], &[f64], &P, f64);
}

/// Solves a typed split problem with a selected split/IMEX algorithm.
pub fn solve_split<FE, FI, P, A>(
    problem: &SplitOdeProblem<FE, FI, P>,
    algorithm: A,
    options: &SolveOptions,
) -> Result<Solution, SolveError>
where
    FE: Fn(&mut [f64], &[f64], &P, f64),
    FI: Fn(&mut [f64], &[f64], &P, f64),
    A: SplitOdeAlgorithm,
{
    algorithm.solve(problem, options)
}

impl SplitOdeAlgorithm for SplitEuler {
    fn solve_validated<FE, FI, P>(
        &self,
        problem: &SplitOdeProblem<FE, FI, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        FE: Fn(&mut [f64], &[f64], &P, f64),
        FI: Fn(&mut [f64], &[f64], &P, f64),
    {
        let placeholder = OdeProblem::new(
            noop as fn(&mut [f64], &[f64], &(), f64),
            problem.initial_state().to_vec(),
            problem.time_span(),
            (),
        );
        integrate(&placeholder, options, SplitEulerKernel::new(problem))
    }
}

fn noop(_: &mut [f64], _: &[f64], _: &(), _: f64) {}

struct SplitEulerKernel<'a, FE, FI, P> {
    problem: &'a SplitOdeProblem<FE, FI, P>,
    explicit: Vec<f64>,
    implicit: Vec<f64>,
}

impl<'a, FE, FI, P> SplitEulerKernel<'a, FE, FI, P> {
    fn new(problem: &'a SplitOdeProblem<FE, FI, P>) -> Self {
        Self {
            problem,
            explicit: vec![0.0; problem.dimension()],
            implicit: vec![0.0; problem.dimension()],
        }
    }

    fn evaluate(
        &mut self,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError>
    where
        FE: Fn(&mut [f64], &[f64], &P, f64),
        FI: Fn(&mut [f64], &[f64], &P, f64),
    {
        self.problem
            .evaluate_explicit(&mut self.explicit, state, time);
        self.problem
            .evaluate_implicit(&mut self.implicit, state, time);
        // A split pair is one logical right-hand-side evaluation, matching the
        // accounting used by the previous combined-problem implementation.
        stats.rhs_evaluations += 1;
        self.explicit
            .iter()
            .chain(&self.implicit)
            .all(|value| value.is_finite())
            .then_some(())
            .ok_or(SolveError::NonFiniteDerivative)
    }
}

impl<FE, FI, P> StepKernel<fn(&mut [f64], &[f64], &(), f64), ()> for SplitEulerKernel<'_, FE, FI, P>
where
    FE: Fn(&mut [f64], &[f64], &P, f64),
    FI: Fn(&mut [f64], &[f64], &P, f64),
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::new(false, 1)
    }

    fn has_callbacks(&self, _: &OdeProblem<fn(&mut [f64], &[f64], &(), f64), ()>) -> bool {
        self.problem.has_callbacks()
    }

    fn apply_initial_callbacks(
        &mut self,
        _: &OdeProblem<fn(&mut [f64], &[f64], &(), f64), ()>,
        state: &mut [f64],
        time: f64,
    ) -> Result<CallbackOutcome, SolveError> {
        self.problem.apply_initial_callbacks(state, time)
    }

    fn has_custom_callback_handling(&self) -> bool {
        // Route callback dispatch through the typed split problem while still
        // allowing the shared Hermite lifecycle to serve save-at requests.
        true
    }

    fn evaluate_dense_derivative(
        &mut self,
        _: &OdeProblem<fn(&mut [f64], &[f64], &(), f64), ()>,
        output: &mut [f64],
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        self.evaluate(state, time, stats)?;
        for ((output, explicit), implicit) in
            output.iter_mut().zip(&self.explicit).zip(&self.implicit)
        {
            *output = explicit + implicit;
        }
        Ok(())
    }

    fn initialize(
        &mut self,
        _: &OdeProblem<fn(&mut [f64], &[f64], &(), f64), ()>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        let _ = (state, time, stats);
        Ok(())
    }

    fn estimate_initial_step(
        &mut self,
        _: &OdeProblem<fn(&mut [f64], &[f64], &(), f64), ()>,
        _: &[f64],
        _: f64,
        _: f64,
        maximum_step: f64,
        _: &mut [f64],
        _: &SolveOptions,
        _: &mut SolverStats,
    ) -> Result<f64, SolveError> {
        Ok(maximum_step)
    }

    fn attempt_step(
        &mut self,
        _: &OdeProblem<fn(&mut [f64], &[f64], &(), f64), ()>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        _: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError> {
        self.evaluate(state, time, stats)?;
        for (((candidate, state), explicit), implicit) in candidate
            .iter_mut()
            .zip(state)
            .zip(&self.explicit)
            .zip(&self.implicit)
        {
            *candidate = state + step * (explicit + implicit);
        }
        Ok(StepEstimate::new(0.0))
    }

    fn apply_step_callbacks(
        &mut self,
        _: &OdeProblem<fn(&mut [f64], &[f64], &(), f64), ()>,
        previous_state: &[f64],
        previous_time: f64,
        state: &mut [f64],
        time: &mut f64,
        state_before_effect: &mut [f64],
        event_tolerance: f64,
        _: &mut SolverStats,
    ) -> Result<CallbackOutcome, SolveError> {
        self.problem.apply_step_callbacks(
            previous_state,
            previous_time,
            state,
            time,
            state_before_effect,
            event_tolerance,
            None,
        )
    }

    fn accept_step(
        &mut self,
        _: &OdeProblem<fn(&mut [f64], &[f64], &(), f64), ()>,
        _: &[f64],
        _: &[f64],
        _: f64,
        _: f64,
        _: bool,
        _: &mut SolverStats,
    ) -> Result<(), SolveError> {
        Ok(())
    }

    fn reject_step(&mut self) {}
}

/// Solve a typed [`SplitOdeProblem`] with [`SplitEuler`].
///
/// Each derivative evaluation invokes both split components at the same state
/// and time, matching OrdinaryDiffEq's `SplitEuler` update.
pub fn solve_split_euler<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    algorithm: SplitEuler,
    options: &SolveOptions,
) -> Result<Solution, SolveError>
where
    FE: Fn(&mut [f64], &[f64], &P, f64),
    FI: Fn(&mut [f64], &[f64], &P, f64),
{
    algorithm.solve(problem, options)
}
