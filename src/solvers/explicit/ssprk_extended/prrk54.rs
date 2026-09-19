use super::Prrk54;
use super::dense::{apply_hermite_callbacks, record_hermite_step};
use crate::SolverStats;
use crate::callback::CallbackOutcome;
use crate::integrator::{
    KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::solution::TrajectoryRecorder;
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions};

struct Prrk54Kernel {
    kappa: f64,
    start_derivative: Vec<f64>,
    first_derivative: Vec<f64>,
    second_derivative: Vec<f64>,
    third_derivative: Vec<f64>,
    fourth_derivative: Vec<f64>,
    stage_one: Vec<f64>,
    stage_two: Vec<f64>,
    stage_three: Vec<f64>,
    stage_four: Vec<f64>,
    dense_endpoint_state: Vec<f64>,
    dense_endpoint_derivative: Vec<f64>,
    dense_endpoint_prepared: bool,
}

impl Prrk54Kernel {
    fn new(kappa: f64, dimension: usize) -> Self {
        Self {
            kappa,
            start_derivative: vec![0.0; dimension],
            first_derivative: vec![0.0; dimension],
            second_derivative: vec![0.0; dimension],
            third_derivative: vec![0.0; dimension],
            fourth_derivative: vec![0.0; dimension],
            stage_one: vec![0.0; dimension],
            stage_two: vec![0.0; dimension],
            stage_three: vec![0.0; dimension],
            stage_four: vec![0.0; dimension],
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

impl<F, P> StepKernel<F, P> for Prrk54Kernel
where
    F: crate::OdeFunction<P>,
{
    fn has_custom_dense_output(&self) -> bool {
        true
    }

    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::new(false, 4)
    }

    fn initialize(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        Self::evaluate(problem, &mut self.start_derivative, state, time, stats)?;
        Self::ensure_finite(&self.start_derivative)
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

    #[allow(clippy::too_many_lines)]
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
        // Base Shu--Osher coefficients from the pinned
        // OrdinaryDiffEqSSPRK pRRK54ConstantCache.  Keep these decimal
        // literals rather than reconstructing them from a tableau: the
        // relaxation transform is defined in this representation upstream.
        let beta10 = 0.391_752_226_571_89;
        let alpha20 = 0.444_370_493_651_235;
        let alpha21 = 0.555_629_506_348_765;
        let beta21 = 0.368_410_593_050_371;
        let alpha30 = 0.620_101_851_488_403;
        let alpha32 = 0.379_898_148_511_597;
        let beta32 = 0.251_891_774_271_694;
        let alpha40 = 0.178_079_954_393_132;
        let alpha43 = 0.821_920_045_606_868;
        let beta43 = 0.544_974_750_228_521;
        let alpha52 = 0.517_231_671_970_585;
        let alpha53 = 0.096_059_710_526_147;
        let beta53 = 0.063_692_468_666_29;
        let alpha54 = 0.386_708_617_503_269;
        let beta54 = 0.226_007_483_236_906;

        Self::evaluate(problem, &mut self.start_derivative, state, time, stats)?;
        Self::ensure_finite(&self.start_derivative)?;

        let z = self.kappa * step;
        let psi1 = 1.0 + z * beta10;
        let psi2 = alpha20 + psi1 * (alpha21 + z * beta21);
        let psi3 = alpha30 + psi2 * (alpha32 + z * beta32);
        let psi4 = alpha40 + psi3 * (alpha43 + z * beta43);
        let psi5 = psi2 * alpha52 + psi3 * (alpha53 + z * beta53) + psi4 * (alpha54 + z * beta54);

        let alpha_hat10 = (1.0 + z * beta10) / psi1;
        let beta_hat10 = beta10 / psi1;
        let alpha_hat20 = alpha20 / psi2;
        let alpha_hat21 = psi1 * (alpha21 + z * beta21) / psi2;
        let beta_hat21 = psi1 * beta21 / psi2;
        let alpha_hat30 = alpha30 / psi3;
        let alpha_hat32 = psi2 * (alpha32 + z * beta32) / psi3;
        let beta_hat32 = psi2 * beta32 / psi3;
        let alpha_hat40 = alpha40 / psi4;
        let alpha_hat43 = psi3 * (alpha43 + z * beta43) / psi4;
        let beta_hat43 = psi3 * beta43 / psi4;
        let alpha_hat52 = psi2 * alpha52 / psi5;
        let alpha_hat53 = psi3 * (alpha53 + z * beta53) / psi5;
        let beta_hat53 = psi3 * beta53 / psi5;
        let alpha_hat54 = psi4 * (alpha54 + z * beta54) / psi5;
        let beta_hat54 = psi4 * beta54 / psi5;

        let c_hat1 = beta_hat10;
        let c_hat2 = alpha_hat21 * c_hat1 + beta_hat21;
        let c_hat3 = alpha_hat32 * c_hat2 + beta_hat32;
        let c_hat4 = alpha_hat43 * c_hat3 + beta_hat43;
        let c_hat5 = alpha_hat52 * c_hat2
            + alpha_hat53 * c_hat3
            + beta_hat53
            + alpha_hat54 * c_hat4
            + beta_hat54;
        let step_hat = c_hat5 * step;

        for ((output, value), derivative) in self
            .stage_one
            .iter_mut()
            .zip(state)
            .zip(&self.start_derivative)
        {
            *output = alpha_hat10 * value + beta_hat10 * step_hat * derivative;
        }
        Self::evaluate(
            problem,
            &mut self.second_derivative,
            &self.stage_one,
            time + c_hat1 * step_hat,
            stats,
        )?;
        Self::ensure_finite(&self.second_derivative)?;

        for (((output, value), stage), derivative) in self
            .stage_two
            .iter_mut()
            .zip(state)
            .zip(&self.stage_one)
            .zip(&self.second_derivative)
        {
            *output =
                alpha_hat20 * value + alpha_hat21 * stage + beta_hat21 * step_hat * derivative;
        }
        Self::evaluate(
            problem,
            &mut self.third_derivative,
            &self.stage_two,
            time + c_hat2 * step_hat,
            stats,
        )?;
        Self::ensure_finite(&self.third_derivative)?;

        for (((output, value), stage), derivative) in self
            .stage_three
            .iter_mut()
            .zip(state)
            .zip(&self.stage_two)
            .zip(&self.third_derivative)
        {
            *output =
                alpha_hat30 * value + alpha_hat32 * stage + beta_hat32 * step_hat * derivative;
        }
        Self::evaluate(
            problem,
            &mut self.fourth_derivative,
            &self.stage_three,
            time + c_hat3 * step_hat,
            stats,
        )?;
        Self::ensure_finite(&self.fourth_derivative)?;

        for (((output, value), stage), derivative) in self
            .stage_four
            .iter_mut()
            .zip(state)
            .zip(&self.stage_three)
            .zip(&self.fourth_derivative)
        {
            *output =
                alpha_hat40 * value + alpha_hat43 * stage + beta_hat43 * step_hat * derivative;
        }
        Self::evaluate(
            problem,
            &mut self.first_derivative,
            &self.stage_four,
            time + c_hat4 * step_hat,
            stats,
        )?;
        Self::ensure_finite(&self.first_derivative)?;

        for (
            ((((output, stage_two), stage_three), derivative_three), stage_four),
            derivative_four,
        ) in candidate
            .iter_mut()
            .zip(&self.stage_two)
            .zip(&self.stage_three)
            .zip(&self.fourth_derivative)
            .zip(&self.stage_four)
            .zip(&self.first_derivative)
        {
            *output = alpha_hat52 * stage_two
                + alpha_hat53 * stage_three
                + beta_hat53 * step_hat * derivative_three
                + alpha_hat54 * stage_four
                + beta_hat54 * step_hat * derivative_four;
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
            &self.start_derivative,
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
            &self.start_derivative,
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

impl OdeAlgorithm for Prrk54 {
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
            Prrk54Kernel::new(self.kappa, problem.initial_state().len()),
        )
    }
}
