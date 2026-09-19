use crate::integrator::{
    ControllerConfig, KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::{OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

use super::evaluation::evaluate;
use super::methods::perform_step;
use super::{MAX_FACTOR, MIN_FACTOR, SAFETY, Scheme};

pub(super) fn solve_scheme<F, P>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
    scheme: Scheme,
) -> Result<Solution, SolveError>
where
    F: crate::OdeFunction<P>,
{
    drive_integration(
        problem,
        options,
        ExponentialKernel::new(problem.initial_state().len(), scheme),
    )
}

struct ExponentialKernel {
    scheme: Scheme,
    derivative: Vec<f64>,
    jacobian: Vec<f64>,
    previous_nonlinear: Option<Vec<f64>>,
    pending_nonlinear: Vec<f64>,
}

impl ExponentialKernel {
    fn new(dimension: usize, scheme: Scheme) -> Self {
        Self {
            scheme,
            derivative: vec![0.0; dimension],
            jacobian: vec![0.0; dimension * dimension],
            previous_nonlinear: None,
            pending_nonlinear: vec![0.0; dimension],
        }
    }
}

impl<F, P> StepKernel<F, P> for ExponentialKernel
where
    F: crate::OdeFunction<P>,
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::with_controller(
            self.scheme.adaptive(),
            ControllerConfig::proportional(
                self.scheme.order(),
                SAFETY,
                MIN_FACTOR,
                MAX_FACTOR,
                MIN_FACTOR,
            ),
        )
    }

    fn initialize(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        evaluate(problem, &mut self.derivative, state, time, stats)
    }

    fn estimate_initial_step(
        &mut self,
        _: &OdeProblem<F, P>,
        state: &[f64],
        _: f64,
        _: f64,
        maximum_step: f64,
        _: &mut [f64],
        options: &SolveOptions,
        _: &mut SolverStats,
    ) -> Result<f64, SolveError> {
        let mut scale = 0.0_f64;
        for (state, derivative) in state.iter().zip(&self.derivative) {
            let tolerance = options.absolute_tolerance + options.relative_tolerance * state.abs();
            scale = scale.max((derivative / tolerance).abs());
        }
        let estimate = if scale == 0.0 { 1.0e-3 } else { 0.01 / scale };
        Ok(estimate.clamp(f64::EPSILON, maximum_step))
    }

    fn attempt_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        options: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError> {
        compute_jacobian(
            problem,
            state,
            time,
            &self.derivative,
            &mut self.jacobian,
            stats,
        )?;
        let result = perform_step(
            self.scheme,
            problem,
            state,
            time,
            step,
            &self.derivative,
            &self.jacobian,
            self.previous_nonlinear.as_deref(),
            stats,
        )?;
        candidate.copy_from_slice(&result.state);
        self.pending_nonlinear = result.current_nonlinear;
        let error_norm = if options.adaptive {
            scaled_error_norm(&result.error, state, candidate, options)
        } else {
            0.0
        };
        Ok(StepEstimate::new(error_norm))
    }

    fn accept_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        _: &[f64],
        state: &[f64],
        time: f64,
        _: f64,
        _: bool,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        if self.scheme == Scheme::Etd2 {
            self.previous_nonlinear = Some(self.pending_nonlinear.clone());
        }
        evaluate(problem, &mut self.derivative, state, time, stats)
    }

    fn reject_step(&mut self) {}
}

fn compute_jacobian<F, P>(
    problem: &OdeProblem<F, P>,
    state: &[f64],
    time: f64,
    f0: &[f64],
    jacobian: &mut [f64],
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: crate::OdeFunction<P>,
{
    stats.jacobian_evaluations += 1;
    if problem.evaluate_jacobian(jacobian, state, time) {
        return jacobian
            .iter()
            .all(|v| v.is_finite())
            .then_some(())
            .ok_or(SolveError::NonFiniteDerivative);
    }
    let n = state.len();
    let mut perturbed = state.to_vec();
    let mut derivative = vec![0.0; n];
    for column in 0..n {
        let delta = f64::EPSILON.sqrt() * (1.0 + state[column].abs());
        perturbed[column] += delta;
        evaluate(problem, &mut derivative, &perturbed, time, stats)?;
        for row in 0..n {
            jacobian[row * n + column] = (derivative[row] - f0[row]) / delta;
        }
        perturbed[column] = state[column];
    }
    Ok(())
}

fn scaled_error_norm(
    error: &[f64],
    state: &[f64],
    candidate: &[f64],
    options: &SolveOptions,
) -> f64 {
    if error.is_empty() {
        return 0.0;
    }
    let sum = error
        .iter()
        .zip(state)
        .zip(candidate)
        .map(|((e, u), v)| {
            let scale =
                options.absolute_tolerance + options.relative_tolerance * u.abs().max(v.abs());
            (e / scale).powi(2)
        })
        .sum::<f64>();
    (sum / error.len() as f64).sqrt()
}
