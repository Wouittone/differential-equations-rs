//! Native regular-ODE QNDF1 implementation.
//!
//! This is the fixed-order one quasi-constant-step NDF method from the
//! pinned OrdinaryDiffEqBDF source. Singular mass matrices, residual DAEs,
//! split/IMEX paths, and variable-order QNDF are intentionally excluded.

use super::tableaux::{backward_differentiation, error_constant, ndf_kappa};
use crate::integrator::{
    ControllerConfig, KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::linear::{DenseLu, LinearError, StateLayout, factorize, solve_factorized};
use crate::tableau::LinearMultistepTableau;
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

const MAX_NEWTON_ITERATIONS: usize = 12;
const NEWTON_TOLERANCE: f64 = 1.0e-12;
// The shared driver's proportional controller uses a conservative safety
// factor while retaining QNDF's order-one error exponent and factor bounds.
const CONTROLLER: ControllerConfig = ControllerConfig::proportional(2, 0.9, 0.2, 10.0, 0.2);

/// Adaptive first-order quasi-constant-step NDF method for regular ODEs.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qndf1;

/// First-order quasi-constant-step BDF method (`QNDF1(kappa = 0)`).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qbdf1;

impl Qndf1 {
    /// Returns the shared BDF1 base formula with its NDF modifier.
    pub fn tableau(self) -> Result<&'static LinearMultistepTableau, SolveError> {
        backward_differentiation(1)
    }
}

impl Qbdf1 {
    /// Returns the shared BDF1 formula; this solver ignores its NDF modifier.
    pub fn tableau(self) -> Result<&'static LinearMultistepTableau, SolveError> {
        backward_differentiation(1)
    }
}

impl OdeAlgorithm for Qndf1 {
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
            Qndf1Kernel::new(problem.initial_state().len(), true)?,
        )
    }
}

impl OdeAlgorithm for Qbdf1 {
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
            Qndf1Kernel::new(problem.initial_state().len(), false)?,
        )
    }
}

struct Workspace {
    layout: StateLayout,
    current_derivative: Vec<f64>,
    evaluation_derivative: Vec<f64>,
    history_state: Vec<f64>,
    extrapolated_state: Vec<f64>,
    perturbed_state: Vec<f64>,
    perturbed_derivative: Vec<f64>,
    residual: Vec<f64>,
    correction: Vec<f64>,
    matrix: Vec<f64>,
    pivots: Vec<usize>,
    factorization: Option<DenseLu>,
    dense_active: bool,
    factorization_ready: bool,
    last_step: Option<f64>,
}

impl Workspace {
    fn new(dimension: usize) -> Self {
        let layout = StateLayout::for_validated_state(dimension);
        Self {
            layout,
            current_derivative: vec![0.0; dimension],
            evaluation_derivative: vec![0.0; dimension],
            history_state: vec![0.0; dimension],
            extrapolated_state: vec![0.0; dimension],
            perturbed_state: vec![0.0; dimension],
            perturbed_derivative: vec![0.0; dimension],
            residual: vec![0.0; dimension],
            correction: vec![0.0; dimension],
            matrix: vec![0.0; dimension * dimension],
            pivots: vec![0; dimension],
            factorization: None,
            dense_active: false,
            factorization_ready: false,
            last_step: None,
        }
    }
}

struct Qndf1Kernel {
    workspace: Workspace,
    kappa: f64,
    error_constant: f64,
}

impl Qndf1Kernel {
    fn new(dimension: usize, ndf: bool) -> Result<Self, SolveError> {
        let tableau = backward_differentiation(1)?;
        Ok(Self {
            workspace: Workspace::new(dimension),
            kappa: ndf_kappa(tableau, ndf)?,
            error_constant: error_constant(tableau, ndf)?,
        })
    }
}

impl<F, P> StepKernel<F, P> for Qndf1Kernel
where
    F: crate::OdeFunction<P>,
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::with_controller(true, CONTROLLER)
            .recover_nonlinear_and_singular_failures()
    }

    fn initialize(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        self.workspace.last_step = None;
        evaluate_checked(
            problem,
            &mut self.workspace.current_derivative,
            state,
            time,
            stats,
        )
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
        Ok(estimate_initial_step(
            state,
            &self.workspace.current_derivative,
            options,
            maximum_step,
        ))
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
        self.workspace.factorization_ready = false;
        let startup = self.workspace.last_step.is_none();
        let kappa = if startup { 0.0 } else { self.kappa };
        if startup {
            self.workspace.extrapolated_state.copy_from_slice(state);
        } else {
            let rho = step
                / self
                    .workspace
                    .last_step
                    .ok_or(SolveError::InvalidMultistepHistory)?;
            for ((out, &now), &previous) in self
                .workspace
                .extrapolated_state
                .iter_mut()
                .zip(state)
                .zip(&self.workspace.history_state)
            {
                *out = now - rho * (now - previous);
            }
        }
        // QNDF's predictor is uprev + sum(D); for order one this is the
        // reinterpolated backward difference represented by extrapolated_state.
        for ((value, &now), &extrapolated) in candidate
            .iter_mut()
            .zip(state)
            .zip(&self.workspace.extrapolated_state)
        {
            *value = now + (now - extrapolated);
        }
        newton_step(
            problem,
            state,
            candidate,
            time,
            step,
            kappa,
            &mut self.workspace,
            stats,
        )?;
        if !options.adaptive || startup {
            return Ok(StepEstimate::new(if options.adaptive { 1.0 } else { 0.0 }));
        }
        let rho = step
            / self
                .workspace
                .last_step
                .ok_or(SolveError::InvalidMultistepHistory)?;
        let difference = candidate
            .iter()
            .zip(state)
            .zip(&self.workspace.history_state)
            .map(|((&next, &now), &previous)| {
                let d = next - now - rho * (now - previous);
                self.error_constant * d
            });
        Ok(StepEstimate::new(rms_scaled(
            difference, candidate, state, options,
        )))
    }

    fn accept_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        previous_state: &[f64],
        state: &[f64],
        time: f64,
        accepted_step: f64,
        callback_applied: bool,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        if callback_applied {
            self.workspace.last_step = None;
        } else {
            self.workspace.history_state.copy_from_slice(previous_state);
            self.workspace.last_step = Some(accepted_step);
        }
        evaluate_checked(
            problem,
            &mut self.workspace.current_derivative,
            state,
            time,
            stats,
        )
    }

    fn reject_step(&mut self) {
        self.workspace.factorization_ready = false;
    }
}

#[allow(clippy::too_many_arguments)]
fn newton_step<F, P>(
    problem: &OdeProblem<F, P>,
    previous: &[f64],
    candidate: &mut [f64],
    time: f64,
    step: f64,
    kappa: f64,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: crate::OdeFunction<P>,
{
    let alpha_zero = 1.0 - kappa;
    let alpha_one = 1.0 - 2.0 * kappa;
    let alpha_two = kappa;
    let evaluation_time = time + step;
    for _ in 0..MAX_NEWTON_ITERATIONS {
        stats.nonlinear_iterations += 1;
        evaluate_checked(
            problem,
            &mut workspace.evaluation_derivative,
            candidate,
            evaluation_time,
            stats,
        )?;
        let mut residual_norm: f64 = 0.0;
        for index in 0..candidate.len() {
            workspace.residual[index] = alpha_zero * candidate[index]
                - alpha_one * previous[index]
                - alpha_two * workspace.extrapolated_state[index]
                - step * workspace.evaluation_derivative[index];
            if !workspace.residual[index].is_finite() {
                return Err(SolveError::NonFiniteDerivative);
            }
            residual_norm = residual_norm.max(workspace.residual[index].abs());
        }
        if residual_norm <= NEWTON_TOLERANCE * (1.0 + infinity_norm(candidate)) {
            return Ok(());
        }
        if !workspace.factorization_ready {
            build_factorization(
                problem,
                candidate,
                evaluation_time,
                step / alpha_zero,
                alpha_zero,
                workspace,
                stats,
            )?;
        }
        for (correction, &residual) in workspace.correction.iter_mut().zip(&workspace.residual) {
            *correction = -residual;
        }
        if workspace.dense_active {
            workspace
                .factorization
                .as_ref()
                .ok_or(SolveError::SingularLinearSystem)?
                .solve(&mut workspace.correction)
                .map_err(map_linear_error)?;
            workspace.dense_active = false;
        } else {
            solve_factorized(
                &workspace.matrix,
                &workspace.pivots,
                &mut workspace.correction,
                previous.len(),
            );
        }
        stats.linear_solves += 1;
        for (value, &correction) in candidate.iter_mut().zip(&workspace.correction) {
            *value += correction;
        }
    }
    Err(SolveError::NonlinearSolveFailed)
}

fn build_factorization<F, P>(
    problem: &OdeProblem<F, P>,
    state: &[f64],
    time: f64,
    derivative_scale: f64,
    diagonal_scale: f64,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: crate::OdeFunction<P>,
{
    let dimension = workspace.layout.dimension();
    if problem.evaluate_jacobian(&mut workspace.matrix, state, time) {
        for row in 0..dimension {
            for column in 0..dimension {
                let index = row * dimension + column;
                let derivative = workspace.matrix[index];
                if !derivative.is_finite() {
                    return Err(SolveError::NonFiniteDerivative);
                }
                workspace.matrix[index] =
                    diagonal_scale * f64::from(row == column) - derivative_scale * derivative;
            }
        }
    } else {
        for column in 0..dimension {
            workspace.perturbed_state.copy_from_slice(state);
            let perturbation = f64::EPSILON.sqrt() * state[column].abs().max(1.0);
            workspace.perturbed_state[column] += perturbation;
            evaluate_checked(
                problem,
                &mut workspace.perturbed_derivative,
                &workspace.perturbed_state,
                time,
                stats,
            )?;
            for row in 0..dimension {
                let derivative = (workspace.perturbed_derivative[row]
                    - workspace.evaluation_derivative[row])
                    / perturbation;
                if !derivative.is_finite() {
                    return Err(SolveError::NonFiniteDerivative);
                }
                workspace.matrix[row * dimension + column] =
                    diagonal_scale * f64::from(row == column) - derivative_scale * derivative;
            }
        }
    }
    stats.jacobian_evaluations += 1;
    stats.linear_factorizations += 1;
    let factorization = if workspace.factorization.is_none() {
        let dense =
            DenseLu::factorize(workspace.layout, &workspace.matrix).map_err(map_linear_error)?;
        factorize(&mut workspace.matrix, &mut workspace.pivots, dimension)
            .map_err(|_| SolveError::SingularLinearSystem)?;
        workspace.dense_active = true;
        dense
    } else {
        factorize(&mut workspace.matrix, &mut workspace.pivots, dimension)
            .map_err(|_| SolveError::SingularLinearSystem)?;
        workspace.dense_active = false;
        workspace
            .factorization
            .take()
            .ok_or(SolveError::NonlinearSolveFailed)?
    };
    workspace.factorization = Some(factorization);
    workspace.factorization_ready = true;
    Ok(())
}

fn rms_scaled<I>(values: I, candidate: &[f64], previous: &[f64], options: &SolveOptions) -> f64
where
    I: Iterator<Item = f64>,
{
    let mut squared = 0.0;
    for ((defect, &next), &old) in values.zip(candidate).zip(previous) {
        let scale =
            options.absolute_tolerance + options.relative_tolerance * next.abs().max(old.abs());
        squared += (defect / scale).powi(2);
    }
    (squared / candidate.len() as f64).sqrt()
}

fn estimate_initial_step(
    state: &[f64],
    derivative: &[f64],
    options: &SolveOptions,
    maximum_step: f64,
) -> f64 {
    let scale = state
        .iter()
        .zip(derivative)
        .map(|(&state, &derivative)| {
            derivative.abs()
                / (options.absolute_tolerance + options.relative_tolerance * state.abs())
        })
        .fold(0.0, f64::max);
    let estimate = if scale > 0.0 {
        (0.01 / scale).sqrt()
    } else {
        // A zero initial derivative does not imply a flat RHS (the common
        // stiff tracking test starts at its moving equilibrium). Keep the
        // first implicit step conservative so the NDF estimator can recover
        // its history before taking a long step.
        maximum_step.min(0.01)
    };
    estimate.max(f64::EPSILON).min(maximum_step)
}

fn evaluate_checked<F, P>(
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
    derivative
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteDerivative)
}

fn infinity_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

fn map_linear_error(error: LinearError) -> SolveError {
    match error {
        LinearError::Singular => SolveError::SingularLinearSystem,
        _ => SolveError::NonlinearSolveFailed,
    }
}
