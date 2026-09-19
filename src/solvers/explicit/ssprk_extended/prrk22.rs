use super::Prrk22;
use super::dense::{apply_hermite_callbacks, record_hermite_step};
use crate::SolverStats;
use crate::callback::CallbackOutcome;
use crate::integrator::{
    KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::solution::TrajectoryRecorder;
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions};

struct Prrk22Kernel {
    kappa: f64,
    first_derivative: Vec<f64>,
    second_derivative: Vec<f64>,
    stage_state: Vec<f64>,
    dense_endpoint_state: Vec<f64>,
    dense_endpoint_derivative: Vec<f64>,
    dense_endpoint_prepared: bool,
}

impl Prrk22Kernel {
    fn new(kappa: f64, dimension: usize) -> Self {
        Self {
            kappa,
            first_derivative: vec![0.0; dimension],
            second_derivative: vec![0.0; dimension],
            stage_state: vec![0.0; dimension],
            dense_endpoint_state: vec![0.0; dimension],
            dense_endpoint_derivative: vec![0.0; dimension],
            dense_endpoint_prepared: false,
        }
    }

    fn evaluate<F, P>(
        problem: &OdeProblem<F, P>,
        derivative: &mut [f64],
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        problem
            .rhs
            .evaluate(derivative, state, problem.parameters(), time)?;
        stats.rhs_evaluations += 1;
        Ok(())
    }

    fn ensure_finite(values: &[f64]) -> Result<(), SolveError> {
        values
            .iter()
            .all(|value| value.is_finite())
            .then_some(())
            .ok_or(SolveError::NonFiniteDerivative)
    }
}

impl<F, P> StepKernel<F, P> for Prrk22Kernel
where
    F: crate::OdeFunction<P>,
{
    fn has_custom_dense_output(&self) -> bool {
        true
    }

    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::new(false, 2)
    }

    fn initialize(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        Self::evaluate(problem, &mut self.first_derivative, state, time, stats)?;
        Self::ensure_finite(&self.first_derivative)
    }

    fn estimate_initial_step(
        &mut self,
        _: &OdeProblem<F, P>,
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
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        _: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError> {
        Self::evaluate(problem, &mut self.first_derivative, state, time, stats)?;
        Self::ensure_finite(&self.first_derivative)?;

        let z = self.kappa * step;
        let psi1 = 1.0 + z;
        let psi2 = 0.5 + psi1 * (0.5 + 0.5 * z);
        let alpha_hat10 = 1.0;
        let beta_hat10 = 1.0 / psi1;
        let alpha_hat20 = 0.5 / psi2;
        let alpha_hat21 = psi1 * (0.5 + 0.5 * z) / psi2;
        let beta_hat21 = psi1 * 0.5 / psi2;
        let c_hat1 = beta_hat10;
        let c_hat2 = alpha_hat21 * c_hat1 + beta_hat21;
        let step_hat = c_hat2 * step;

        for ((output, value), derivative) in self
            .stage_state
            .iter_mut()
            .zip(state)
            .zip(&self.first_derivative)
        {
            *output = alpha_hat10 * value + beta_hat10 * step_hat * derivative;
        }
        Self::evaluate(
            problem,
            &mut self.second_derivative,
            &self.stage_state,
            time + c_hat1 * step_hat,
            stats,
        )?;
        Self::ensure_finite(&self.second_derivative)?;
        for (((output, value), stage), derivative) in candidate
            .iter_mut()
            .zip(state)
            .zip(&self.stage_state)
            .zip(&self.second_derivative)
        {
            *output =
                alpha_hat20 * value + alpha_hat21 * stage + beta_hat21 * step_hat * derivative;
        }
        Self::ensure_finite(candidate)?;
        Ok(StepEstimate::new(0.0))
    }

    fn apply_step_callbacks(
        &mut self,
        problem: &OdeProblem<F, P>,
        previous_state: &[f64],
        previous_time: f64,
        state: &mut [f64],
        time: &mut f64,
        state_before_effect: &mut [f64],
        event_tolerance: f64,
        stats: &mut SolverStats,
    ) -> Result<CallbackOutcome, SolveError> {
        apply_hermite_callbacks(
            problem,
            previous_state,
            previous_time,
            state,
            time,
            state_before_effect,
            event_tolerance,
            &self.first_derivative,
            &mut self.dense_endpoint_state,
            &mut self.dense_endpoint_derivative,
            &mut self.dense_endpoint_prepared,
            stats,
        )
    }

    fn record_dense_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        previous_state: &[f64],
        state: &[f64],
        previous_time: f64,
        attempted_time: f64,
        time: f64,
        final_time: bool,
        recorder: &mut TrajectoryRecorder<'_>,
        stats: &mut SolverStats,
    ) -> Result<bool, SolveError> {
        record_hermite_step(
            problem,
            previous_state,
            state,
            &self.first_derivative,
            &mut self.dense_endpoint_state,
            &mut self.dense_endpoint_derivative,
            &mut self.dense_endpoint_prepared,
            previous_time,
            attempted_time,
            time,
            final_time,
            recorder,
            stats,
        )
    }

    fn accept_step(
        &mut self,
        _: &OdeProblem<F, P>,
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

impl OdeAlgorithm for Prrk22 {
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
            Prrk22Kernel::new(self.kappa, problem.initial_state().len()),
        )
    }
}
