//! Fixed-step regular-ODE Modified Extended BDF2 (MEBDF2).
//!
//! This is the three-backward-Euler-correction construction used by the
//! pinned OrdinaryDiffEqBDF implementation. DAE residual and split/IMEX
//! behavior are intentionally outside this identity-mass ODE port.

use crate::integrator::{
    KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::linear::{DenseLu, LinearError, StateLayout, factorize, solve_factorized};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

const MAX_NEWTON_ITERATIONS: usize = 12;
const NEWTON_TOLERANCE: f64 = 1.0e-12;

/// Fixed-step second-order Modified Extended BDF method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Mebdf2;

impl OdeAlgorithm for Mebdf2 {
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
            Mebdf2Kernel::new(problem.initial_state().len()),
        )
    }
}

struct Workspace {
    layout: StateLayout,
    current_derivative: Vec<f64>,
    stage_one: Vec<f64>,
    stage_two: Vec<f64>,
    stage_three: Vec<f64>,
    stage_state: Vec<f64>,
    first_state: Vec<f64>,
    tmp_state: Vec<f64>,
    stage_derivative: Vec<f64>,
    perturbed_state: Vec<f64>,
    perturbed_derivative: Vec<f64>,
    residual: Vec<f64>,
    correction: Vec<f64>,
    matrix: Vec<f64>,
    pivots: Vec<usize>,
    factorization: Option<DenseLu>,
    dense_active: bool,
    factorization_ready: bool,
}

impl Workspace {
    fn new(dimension: usize) -> Self {
        let layout = StateLayout::for_validated_state(dimension);
        Self {
            layout,
            current_derivative: vec![0.0; dimension],
            stage_one: vec![0.0; dimension],
            stage_two: vec![0.0; dimension],
            stage_three: vec![0.0; dimension],
            stage_state: vec![0.0; dimension],
            first_state: vec![0.0; dimension],
            tmp_state: vec![0.0; dimension],
            stage_derivative: vec![0.0; dimension],
            perturbed_state: vec![0.0; dimension],
            perturbed_derivative: vec![0.0; dimension],
            residual: vec![0.0; dimension],
            correction: vec![0.0; dimension],
            matrix: vec![0.0; dimension * dimension],
            pivots: vec![0; dimension],
            factorization: None,
            dense_active: false,
            factorization_ready: false,
        }
    }
}

struct Mebdf2Kernel {
    workspace: Workspace,
}

impl Mebdf2Kernel {
    fn new(dimension: usize) -> Self {
        Self {
            workspace: Workspace::new(dimension),
        }
    }
}

impl<F, P> StepKernel<F, P> for Mebdf2Kernel
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::new(false, 2).recover_nonlinear_and_singular_failures()
    }

    fn initialize(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
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
        _: &[f64],
        _: f64,
        _: f64,
        _: f64,
        _: &mut [f64],
        _: &SolveOptions,
        _: &mut SolverStats,
    ) -> Result<f64, SolveError> {
        Err(SolveError::InitialStepRequired)
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
        for (z, &derivative) in self
            .workspace
            .stage_one
            .iter_mut()
            .zip(&self.workspace.current_derivative)
        {
            *z = step * derivative;
        }
        solve_correction(
            problem,
            state,
            time + step,
            step,
            0,
            0,
            &mut self.workspace,
            stats,
        )?;
        self.workspace
            .first_state
            .copy_from_slice(&self.workspace.stage_state);

        self.workspace
            .stage_two
            .copy_from_slice(&self.workspace.stage_one);
        solve_correction(
            problem,
            &[],
            time + 2.0 * step,
            step,
            1,
            1,
            &mut self.workspace,
            stats,
        )?;
        for (index, value) in self.workspace.tmp_state.iter_mut().enumerate() {
            *value = 0.5 * state[index] + self.workspace.first_state[index]
                - 0.5 * self.workspace.stage_state[index];
        }
        self.workspace
            .stage_three
            .copy_from_slice(&self.workspace.stage_two);
        solve_correction(
            problem,
            &[],
            time + step,
            step,
            2,
            2,
            &mut self.workspace,
            stats,
        )?;
        for (index, value) in candidate.iter_mut().enumerate() {
            *value = self.workspace.tmp_state[index] + self.workspace.stage_three[index];
        }
        Ok(StepEstimate::new(0.0))
    }

    fn accept_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        _: &[f64],
        state: &[f64],
        time: f64,
        _: f64,
        callback_applied: bool,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        if callback_applied {
            self.workspace.factorization = None;
        }
        self.workspace.factorization_ready = false;
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

#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
fn solve_correction<F, P>(
    problem: &OdeProblem<F, P>,
    base: &[f64],
    stage_time: f64,
    step: f64,
    base_kind: u8,
    increment_kind: u8,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    workspace.factorization_ready = false;
    for _ in 0..MAX_NEWTON_ITERATIONS {
        stats.nonlinear_iterations += 1;
        let dimension = workspace.layout.dimension();
        for index in 0..dimension {
            let base_value = match base_kind {
                0 => base[index],
                1 => workspace.first_state[index],
                _ => workspace.tmp_state[index],
            };
            let increment = match increment_kind {
                0 => workspace.stage_one[index],
                1 => workspace.stage_two[index],
                _ => workspace.stage_three[index],
            };
            workspace.stage_state[index] = base_value + increment;
        }
        evaluate_checked(
            problem,
            &mut workspace.stage_derivative,
            &workspace.stage_state,
            stage_time,
            stats,
        )?;
        let mut norm: f64 = 0.0;
        for index in 0..dimension {
            let increment = match increment_kind {
                0 => workspace.stage_one[index],
                1 => workspace.stage_two[index],
                _ => workspace.stage_three[index],
            };
            workspace.residual[index] = increment - step * workspace.stage_derivative[index];
            norm = norm.max(workspace.residual[index].abs());
        }
        if norm <= NEWTON_TOLERANCE * (1.0 + infinity_norm(&workspace.stage_state)) {
            return Ok(());
        }
        if !workspace.factorization_ready {
            build_factorization(problem, stage_time, step, workspace, stats)?;
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
                dimension,
            );
        }
        stats.linear_solves += 1;
        for index in 0..dimension {
            match increment_kind {
                0 => workspace.stage_one[index] += workspace.correction[index],
                1 => workspace.stage_two[index] += workspace.correction[index],
                _ => workspace.stage_three[index] += workspace.correction[index],
            }
        }
    }
    Err(SolveError::NonlinearSolveFailed)
}

fn build_factorization<F, P>(
    problem: &OdeProblem<F, P>,
    time: f64,
    scale: f64,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let dimension = workspace.layout.dimension();
    if problem.evaluate_jacobian(&mut workspace.matrix, &workspace.stage_state, time) {
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
            workspace
                .perturbed_state
                .copy_from_slice(&workspace.stage_state);
            let perturbation = f64::EPSILON.sqrt() * workspace.stage_state[column].abs().max(1.0);
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
                    - workspace.stage_derivative[row])
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
    if workspace.factorization.is_none() {
        workspace.factorization = Some(
            DenseLu::factorize(workspace.layout, &workspace.matrix).map_err(map_linear_error)?,
        );
        factorize(&mut workspace.matrix, &mut workspace.pivots, dimension)
            .map_err(|_| SolveError::SingularLinearSystem)?;
        workspace.dense_active = true;
    } else {
        factorize(&mut workspace.matrix, &mut workspace.pivots, dimension)
            .map_err(|_| SolveError::SingularLinearSystem)?;
        workspace.dense_active = false;
    }
    workspace.factorization_ready = true;
    Ok(())
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
