use crate::integrator::{
    KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::solution::{BorrowedHermiteSegment, TrajectoryRecorder};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

/// Anastassi–Simos optimized fifth-order Runge–Kutta method for periodic
/// problems. `w` is the periodicity estimate used by the upstream method;
/// `Anas5::default()` uses the pinned default `w = 1`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Anas5 {
    pub w: f64,
}

impl Anas5 {
    pub const fn new(w: f64) -> Self {
        Self { w }
    }
}

impl Default for Anas5 {
    fn default() -> Self {
        Self { w: 1.0 }
    }
}

impl OdeAlgorithm for Anas5 {
    fn solve<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        drive_integration(
            problem,
            options,
            Anas5Kernel::new(self.w, problem.initial_state().len()),
        )
    }
}

struct Anas5Kernel {
    w: f64,
    first_derivative: Vec<f64>,
    second_derivative: Vec<f64>,
    third_derivative: Vec<f64>,
    fourth_derivative: Vec<f64>,
    fifth_derivative: Vec<f64>,
    sixth_derivative: Vec<f64>,
    last_derivative: Vec<f64>,
    stage_state: Vec<f64>,
    first_is_current: bool,
}

impl Anas5Kernel {
    fn new(w: f64, dimension: usize) -> Self {
        Self {
            w,
            first_derivative: vec![0.0; dimension],
            second_derivative: vec![0.0; dimension],
            third_derivative: vec![0.0; dimension],
            fourth_derivative: vec![0.0; dimension],
            fifth_derivative: vec![0.0; dimension],
            sixth_derivative: vec![0.0; dimension],
            last_derivative: vec![0.0; dimension],
            stage_state: vec![0.0; dimension],
            first_is_current: false,
        }
    }

    fn evaluate<F, P>(
        problem: &OdeProblem<F, P>,
        derivative: &mut [f64],
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        (problem.rhs)(derivative, state, problem.parameters(), time);
        stats.rhs_evaluations += 1;
    }

    fn ensure_finite(values: &[f64]) -> Result<(), SolveError> {
        values
            .iter()
            .all(|value| value.is_finite())
            .then_some(())
            .ok_or(SolveError::NonFiniteDerivative)
    }

    fn dynamic_coefficients(&self, step: f64) -> Result<(f64, f64, f64, f64), SolveError> {
        let v = self.w * step;
        let v2 = v * v;
        let v3 = v2 * v;
        let v4 = v3 * v;
        let v5 = v4 * v;
        let tan_v = v.tan();
        let numerator = -2.0 * v5 + 6.0 * tan_v * v4 + 24.0 * v3 - 72.0 * tan_v * v2 - 144.0 * v
            + 144.0 * tan_v;
        let denominator = v5 * (2.0 * tan_v * v - 8.0);
        let a65 = (-8000.0 / 1071.0) * numerator / denominator;
        let a61 = -4.0 - (119.0 / 200.0) * a65;
        let a63 = (189.0 / 100.0) * a65;
        let a64 = -(459.0 / 200.0) * a65;
        [a61, a63, a64, a65]
            .iter()
            .all(|value| value.is_finite())
            .then_some((a61, a63, a64, a65))
            .ok_or(SolveError::NonFiniteDerivative)
    }
}

impl<F, P> StepKernel<F, P> for Anas5Kernel
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::new(false, 5)
    }

    fn initialize(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        if !self.w.is_finite() {
            return Err(SolveError::InvalidTableau);
        }
        Self::evaluate(problem, &mut self.first_derivative, state, time, stats);
        Self::ensure_finite(&self.first_derivative)?;
        self.first_is_current = true;
        Ok(())
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
        if !self.first_is_current {
            Self::evaluate(problem, &mut self.first_derivative, state, time, stats);
            Self::ensure_finite(&self.first_derivative)?;
            self.first_is_current = true;
        }
        let (a61, a63, a64, a65) = self.dynamic_coefficients(step)?;
        let dt = step;

        for ((output, value), derivative) in self
            .stage_state
            .iter_mut()
            .zip(state)
            .zip(&self.first_derivative)
        {
            *output = value + dt * 0.1 * derivative;
        }
        Self::evaluate(
            problem,
            &mut self.second_derivative,
            &self.stage_state,
            time + 0.1 * dt,
            stats,
        );

        for (((output, value), first), second) in self
            .stage_state
            .iter_mut()
            .zip(state)
            .zip(&self.first_derivative)
            .zip(&self.second_derivative)
        {
            *output = value + dt * (-2.0 / 9.0 * first + 5.0 / 9.0 * second);
        }
        Self::evaluate(
            problem,
            &mut self.third_derivative,
            &self.stage_state,
            time + dt / 3.0,
            stats,
        );

        for ((((output, value), first), second), third) in self
            .stage_state
            .iter_mut()
            .zip(state)
            .zip(&self.first_derivative)
            .zip(&self.second_derivative)
            .zip(&self.third_derivative)
        {
            *output = value + dt * (28.0 / 9.0 * first - 40.0 / 9.0 * second + 2.0 * third);
        }
        Self::evaluate(
            problem,
            &mut self.fourth_derivative,
            &self.stage_state,
            time + 2.0 * dt / 3.0,
            stats,
        );

        for (((((output, value), first), second), third), fourth) in self
            .stage_state
            .iter_mut()
            .zip(state)
            .zip(&self.first_derivative)
            .zip(&self.second_derivative)
            .zip(&self.third_derivative)
            .zip(&self.fourth_derivative)
        {
            *output = value
                + dt * (-11277.0 / 8000.0 * first + 171.0 / 80.0 * second - 459.0 / 2000.0 * third
                    + 3213.0 / 8000.0 * fourth);
        }
        Self::evaluate(
            problem,
            &mut self.fifth_derivative,
            &self.stage_state,
            time + 0.9 * dt,
            stats,
        );

        for ((((((output, value), first), second), third), fourth), fifth) in self
            .stage_state
            .iter_mut()
            .zip(state)
            .zip(&self.first_derivative)
            .zip(&self.second_derivative)
            .zip(&self.third_derivative)
            .zip(&self.fourth_derivative)
            .zip(&self.fifth_derivative)
        {
            *output = value
                + dt * (a61 * first + 5.0 * second + a63 * third + a64 * fourth + a65 * fifth);
        }
        Self::evaluate(
            problem,
            &mut self.sixth_derivative,
            &self.stage_state,
            time + dt,
            stats,
        );

        for ((((((output, value), first), third), fourth), fifth), sixth) in candidate
            .iter_mut()
            .zip(state)
            .zip(&self.first_derivative)
            .zip(&self.third_derivative)
            .zip(&self.fourth_derivative)
            .zip(&self.fifth_derivative)
            .zip(&self.sixth_derivative)
        {
            *output = value
                + dt * (23.0 / 216.0 * first
                    + 63.0 / 136.0 * third
                    + 9.0 / 56.0 * fourth
                    + 1000.0 / 3213.0 * fifth
                    - 1.0 / 24.0 * sixth);
        }
        Self::evaluate(
            problem,
            &mut self.last_derivative,
            candidate,
            time + dt,
            stats,
        );
        Self::ensure_finite(&self.last_derivative)?;
        Self::ensure_finite(candidate)?;
        Ok(StepEstimate::new(0.0))
    }

    fn record_dense_step(
        &mut self,
        _: &OdeProblem<F, P>,
        previous_state: &[f64],
        state: &[f64],
        previous_time: f64,
        time: f64,
        final_time: bool,
        recorder: &mut TrajectoryRecorder<'_>,
        _: &mut SolverStats,
    ) -> Result<bool, SolveError> {
        let segment = BorrowedHermiteSegment::new(
            previous_time,
            time,
            previous_state,
            state,
            &self.first_derivative,
            &self.last_derivative,
        )
        .map_err(|_| SolveError::NonFiniteDerivative)?;
        recorder
            .record_step_dense(
                previous_state,
                previous_time,
                state,
                time,
                final_time,
                &segment,
            )
            .map_err(|_| SolveError::NonFiniteDerivative)?;
        Ok(true)
    }

    fn accept_step(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: &[f64],
        _: f64,
        _: f64,
        callback_applied: bool,
        _: &mut SolverStats,
    ) -> Result<(), SolveError> {
        if callback_applied {
            self.first_is_current = false;
        } else {
            std::mem::swap(&mut self.first_derivative, &mut self.last_derivative);
            self.first_is_current = true;
        }
        Ok(())
    }

    fn reject_step(&mut self) {}
}
