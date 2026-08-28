//! Native regular-ODE QNDF2 implementation.
//!
//! Fixed-order two quasi-constant-step NDF with identity mass. Variable-order
//! QNDF, residual DAEs, singular mass matrices, and split/IMEX paths are out
//! of scope.

use crate::integrator::{
    ControllerConfig, KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::linear::{DenseLu, LinearError, StateLayout, factorize, solve_factorized};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

const DEFAULT_KAPPA: f64 = -1.0 / 9.0;
const MAX_NEWTON_ITERATIONS: usize = 12;
const NEWTON_TOLERANCE: f64 = 1.0e-12;
const CONTROLLER: ControllerConfig = ControllerConfig::proportional(3, 0.9, 0.2, 10.0, 0.2);

/// Adaptive second-order quasi-constant-step NDF method for regular ODEs.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qndf2;

/// Second-order quasi-constant-step BDF method (`QNDF2(kappa = 0)`).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qbdf2;

impl OdeAlgorithm for Qndf2 {
    fn solve_validated<F, P>(
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
            Qndf2Kernel::new(problem.initial_state().len(), DEFAULT_KAPPA),
        )
    }
}

impl OdeAlgorithm for Qbdf2 {
    fn solve_validated<F, P>(
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
            Qndf2Kernel::new(problem.initial_state().len(), 0.0),
        )
    }
}

struct Workspace {
    layout: StateLayout,
    current_derivative: Vec<f64>,
    evaluation_derivative: Vec<f64>,
    history_one: Vec<f64>,
    history_two: Vec<f64>,
    difference_one: Vec<f64>,
    difference_two: Vec<f64>,
    difference_three: Vec<f64>,
    raw_one: Vec<f64>,
    raw_two: Vec<f64>,
    extrapolated_state: Vec<f64>,
    forcing: Vec<f64>,
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
    prior_step: Option<f64>,
    history_count: usize,
}

impl Workspace {
    fn new(dimension: usize) -> Self {
        let layout = StateLayout::for_validated_state(dimension);
        Self {
            layout,
            current_derivative: vec![0.0; dimension],
            evaluation_derivative: vec![0.0; dimension],
            history_one: vec![0.0; dimension],
            history_two: vec![0.0; dimension],
            difference_one: vec![0.0; dimension],
            difference_two: vec![0.0; dimension],
            difference_three: vec![0.0; dimension],
            raw_one: vec![0.0; dimension],
            raw_two: vec![0.0; dimension],
            extrapolated_state: vec![0.0; dimension],
            forcing: vec![0.0; dimension],
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
            prior_step: None,
            history_count: 0,
        }
    }
}

struct Qndf2Kernel {
    workspace: Workspace,
    kappa: f64,
}

impl Qndf2Kernel {
    fn new(dimension: usize, kappa: f64) -> Self {
        Self {
            workspace: Workspace::new(dimension),
            kappa,
        }
    }
}

impl<F, P> StepKernel<F, P> for Qndf2Kernel
where
    F: Fn(&mut [f64], &[f64], &P, f64),
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
        self.workspace.history_count = 0;
        self.workspace.last_step = None;
        self.workspace.prior_step = None;
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
        let startup = self.workspace.history_count < 2;
        let (beta_zero, gamma_two) = if startup {
            (1.0, 1.0)
        } else {
            (1.0 / ((1.0 - self.kappa) * 1.5), 1.5)
        };
        build_differences(&mut self.workspace, state, step)?;
        for (((out, &now), &d1), &d2) in self
            .workspace
            .extrapolated_state
            .iter_mut()
            .zip(state)
            .zip(&self.workspace.difference_one)
            .zip(&self.workspace.difference_two)
        {
            *out = now + d1 + d2;
        }
        for (((out, &d1), &d2), beta) in self
            .workspace
            .forcing
            .iter_mut()
            .zip(&self.workspace.difference_one)
            .zip(&self.workspace.difference_two)
            .zip(std::iter::repeat(beta_zero))
        {
            *out = beta * (d1 + gamma_two * d2);
        }
        candidate.copy_from_slice(&self.workspace.extrapolated_state);
        newton_step(
            problem,
            candidate,
            time,
            step,
            beta_zero,
            &mut self.workspace,
            stats,
        )?;
        if !options.adaptive {
            return Ok(StepEstimate::new(0.0));
        }
        if startup {
            return Ok(StepEstimate::new(1.0));
        }
        let error = if self.workspace.history_count == 2 {
            for ((out, (&next, &now)), &d1) in self
                .workspace
                .difference_three
                .iter_mut()
                .zip(candidate.iter().zip(state))
                .zip(&self.workspace.difference_one)
            {
                *out = (self.kappa * gamma_two + 1.0 / 3.0) * ((next - now) - d1);
            }
            rms_scaled(
                self.workspace.difference_three.iter().copied(),
                candidate,
                state,
                options,
            )
        } else {
            1.0
        };
        Ok(StepEstimate::new(error))
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
            self.workspace.history_count = 0;
            self.workspace.last_step = None;
            self.workspace.prior_step = None;
        } else {
            self.workspace
                .history_two
                .copy_from_slice(&self.workspace.history_one);
            self.workspace.history_one.copy_from_slice(previous_state);
            self.workspace.prior_step = self.workspace.last_step;
            self.workspace.last_step = Some(accepted_step);
            self.workspace.history_count = (self.workspace.history_count + 1).min(2);
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

fn build_differences(
    workspace: &mut Workspace,
    state: &[f64],
    step: f64,
) -> Result<(), SolveError> {
    if workspace.history_count == 0 {
        workspace.difference_one.fill(0.0);
        workspace.difference_two.fill(0.0);
    } else if workspace.history_count == 1 {
        // QNDF2 deliberately takes two startup steps with κ = 0 and an
        // implicit-Euler initial guess; no order-two history is available yet.
        // The second accepted state seeds history_one/history_two for the
        // first genuine BDF2 step.
        workspace.difference_one.fill(0.0);
        workspace.difference_two.fill(0.0);
    } else {
        let dt_prev = workspace
            .last_step
            .ok_or(SolveError::InvalidMultistepHistory)?;
        let rho_one = step / dt_prev;
        let dt_prior = workspace.prior_step.unwrap_or(dt_prev);
        let rho_two = step / dt_prior;
        if (dt_prev - dt_prior).abs() > 1.0e-12 * dt_prev.abs().max(dt_prior.abs()).max(1.0) {
            for ((d1, &now), &old) in workspace
                .difference_one
                .iter_mut()
                .zip(state)
                .zip(&workspace.history_one)
            {
                *d1 = (now - old) * rho_one;
            }
            for ((d2, &d1), (&old, &older)) in workspace
                .difference_two
                .iter_mut()
                .zip(&workspace.difference_one)
                .zip(workspace.history_one.iter().zip(&workspace.history_two))
            {
                *d2 = d1 - (old - older) * rho_two;
            }
        } else {
            for (((a, b), &now), (&old, &older)) in workspace
                .raw_one
                .iter_mut()
                .zip(&mut workspace.raw_two)
                .zip(state)
                .zip(workspace.history_one.iter().zip(&workspace.history_two))
            {
                *a = now - old;
                *b = *a - (old - older);
            }
            if (rho_one - 1.0).abs() < 1.0e-12 {
                workspace.difference_one.copy_from_slice(&workspace.raw_one);
                workspace.difference_two.copy_from_slice(&workspace.raw_two);
            } else {
                for ((d1, d2), (&a, &b)) in workspace
                    .difference_one
                    .iter_mut()
                    .zip(&mut workspace.difference_two)
                    .zip(workspace.raw_one.iter().zip(&workspace.raw_two))
                {
                    *d1 = a * (-rho_one) + b * (-rho_one * (1.0 - rho_one) / 2.0);
                    *d2 = a * (-2.0 * rho_one) + b * (-rho_one * (1.0 - 2.0 * rho_one));
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn newton_step<F, P>(
    problem: &OdeProblem<F, P>,
    candidate: &mut [f64],
    time: f64,
    step: f64,
    beta_zero: f64,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
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
        for (((residual, &value), &extrapolated), (&forcing, &derivative)) in workspace
            .residual
            .iter_mut()
            .zip(candidate.iter())
            .zip(&workspace.extrapolated_state)
            .zip(
                workspace
                    .forcing
                    .iter()
                    .zip(&workspace.evaluation_derivative),
            )
        {
            *residual = value - extrapolated + forcing - step * beta_zero * derivative;
            if !residual.is_finite() {
                return Err(SolveError::NonFiniteDerivative);
            }
            residual_norm = residual_norm.max(residual.abs());
        }
        if residual_norm <= NEWTON_TOLERANCE * (1.0 + infinity_norm(candidate)) {
            return Ok(());
        }
        if !workspace.factorization_ready {
            build_factorization(
                problem,
                candidate,
                evaluation_time,
                step * beta_zero,
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
                candidate.len(),
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
    scale: f64,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
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
                workspace.matrix[index] = f64::from(row == column) - scale * derivative;
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
                    f64::from(row == column) - scale * derivative;
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
    for ((value, &next), &old) in values.zip(candidate).zip(previous) {
        let scale =
            options.absolute_tolerance + options.relative_tolerance * next.abs().max(old.abs());
        squared += (value / scale).powi(2);
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
        .map(|(&u, &f)| {
            f.abs() / (options.absolute_tolerance + options.relative_tolerance * u.abs())
        })
        .fold(0.0, f64::max);
    let estimate = if scale > 0.0 {
        (0.01 / scale).sqrt()
    } else {
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
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    (problem.rhs)(derivative, state, problem.parameters(), time);
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
