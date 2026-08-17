//! Native regular-ODE SDIRK methods.
//!
//! This module currently contains the two-stage adaptive SDIRK2/ESDIRK
//! method from the pinned OrdinaryDiffEqSDIRK tableau.

// The pinned SDIRK/ESDIRK catalogue intentionally preserves upstream decimal
// literals, including values with more written digits than f64 can represent.
#![allow(clippy::excessive_precision)]

use crate::generated_coefficients::{SDIRK2_A, SDIRK2_B, SDIRK2_B_EMBEDDED, SDIRK2_STAGE_TIMES};
use crate::integrator::{
    ControllerConfig, KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::linear::{DenseLu, StateLayout};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

const MAX_NEWTON_ITERATIONS: usize = 12;
const NEWTON_TOLERANCE: f64 = 1.0e-12;
const CONTROLLER: ControllerConfig = ControllerConfig::proportional(2, 0.9, 0.2, 10.0, 0.2);

/// The adaptive second-order two-stage SDIRK/ESDIRK method.
///
/// The pinned tableau has stage times (1, 0), unit diagonal, coupling
/// a21 = -1, primary weights (1/2, 1/2), and embedded weights
/// (1/2, -1/2). This is a regular identity-mass ODE method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Sdirk2;

impl OdeAlgorithm for Sdirk2 {
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
            Sdirk2Kernel::new(problem.initial_state().len()),
        )
    }
}

struct Workspace {
    layout: StateLayout,
    current_derivative: Vec<f64>,
    stage_one: Vec<f64>,
    stage_two: Vec<f64>,
    stage_state: Vec<f64>,
    stage_derivative: Vec<f64>,
    perturbed_state: Vec<f64>,
    perturbed_derivative: Vec<f64>,
    residual: Vec<f64>,
    correction: Vec<f64>,
    error: Vec<f64>,
    matrix: Vec<f64>,
    factorization: Option<DenseLu>,
}

impl Workspace {
    fn new(dimension: usize) -> Self {
        let layout = StateLayout::new(dimension).expect("solver validates non-empty state");
        Self {
            layout,
            current_derivative: vec![0.0; dimension],
            stage_one: vec![0.0; dimension],
            stage_two: vec![0.0; dimension],
            stage_state: vec![0.0; dimension],
            stage_derivative: vec![0.0; dimension],
            perturbed_state: vec![0.0; dimension],
            perturbed_derivative: vec![0.0; dimension],
            residual: vec![0.0; dimension],
            correction: vec![0.0; dimension],
            error: vec![0.0; dimension],
            matrix: vec![0.0; dimension * dimension],
            factorization: None,
        }
    }
}

struct Sdirk2Kernel {
    workspace: Workspace,
}

impl Sdirk2Kernel {
    fn new(dimension: usize) -> Self {
        Self {
            workspace: Workspace::new(dimension),
        }
    }
}

impl<F, P> StepKernel<F, P> for Sdirk2Kernel
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
        self.workspace.factorization = None;
        let dimension = self.workspace.layout.dimension();

        for (z, &derivative) in self
            .workspace
            .stage_one
            .iter_mut()
            .zip(&self.workspace.current_derivative)
        {
            *z = step * derivative;
        }
        solve_stage(
            problem,
            state,
            time + SDIRK2_STAGE_TIMES[0] * step,
            step,
            false,
            &mut self.workspace,
            stats,
        )?;

        for (z, &z_one) in self
            .workspace
            .stage_two
            .iter_mut()
            .zip(&self.workspace.stage_one)
        {
            *z = z_one;
        }
        solve_stage(
            problem,
            state,
            time + SDIRK2_STAGE_TIMES[1] * step,
            step,
            true,
            &mut self.workspace,
            stats,
        )?;

        for index in 0..dimension {
            candidate[index] = state[index]
                + SDIRK2_B[0] * self.workspace.stage_one[index]
                + SDIRK2_B[1] * self.workspace.stage_two[index];
            self.workspace.error[index] = SDIRK2_B_EMBEDDED[0] * self.workspace.stage_one[index]
                + SDIRK2_B_EMBEDDED[1] * self.workspace.stage_two[index];
        }
        if !options.adaptive {
            return Ok(StepEstimate::new(0.0));
        }

        let mut squared_norm = 0.0;
        for index in 0..dimension {
            let scale = options.absolute_tolerance
                + options.relative_tolerance * state[index].abs().max(candidate[index].abs());
            squared_norm += (self.workspace.error[index] / scale).powi(2);
        }
        Ok(StepEstimate::new((squared_norm / dimension as f64).sqrt()))
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
        self.workspace.factorization = None;
        evaluate_checked(
            problem,
            &mut self.workspace.current_derivative,
            state,
            time,
            stats,
        )
    }

    fn reject_step(&mut self) {
        self.workspace.factorization = None;
    }
}

fn solve_stage<F, P>(
    problem: &OdeProblem<F, P>,
    previous: &[f64],
    stage_time: f64,
    step: f64,
    second_stage: bool,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let dimension = workspace.layout.dimension();
    for _ in 0..MAX_NEWTON_ITERATIONS {
        stats.nonlinear_iterations += 1;
        let coupling = second_stage;
        for (index, &previous_value) in previous.iter().enumerate() {
            let stage = if second_stage {
                workspace.stage_two[index]
            } else {
                workspace.stage_one[index]
            };
            let explicit_coupling = if coupling {
                SDIRK2_A[1][0] * workspace.stage_one[index]
            } else {
                0.0
            };
            let diagonal = if second_stage {
                SDIRK2_A[1][1]
            } else {
                SDIRK2_A[0][0]
            };
            workspace.stage_state[index] = previous_value + explicit_coupling + diagonal * stage;
        }
        evaluate_checked(
            problem,
            &mut workspace.stage_derivative,
            &workspace.stage_state,
            stage_time,
            stats,
        )?;
        let mut residual_norm: f64 = 0.0;
        for index in 0..dimension {
            let stage = if second_stage {
                workspace.stage_two[index]
            } else {
                workspace.stage_one[index]
            };
            workspace.residual[index] = stage - step * workspace.stage_derivative[index];
            residual_norm = residual_norm.max(workspace.residual[index].abs());
        }
        let state_scale = 1.0 + infinity_norm(&workspace.stage_state);
        if residual_norm <= NEWTON_TOLERANCE * state_scale {
            return Ok(());
        }

        if workspace.factorization.is_none() {
            build_factorization(problem, stage_time, step, workspace, stats)?;
        }
        for (correction, &residual) in workspace.correction.iter_mut().zip(&workspace.residual) {
            *correction = -residual;
        }
        workspace
            .factorization
            .as_ref()
            .expect("factorization built above")
            .solve(&mut workspace.correction)
            .map_err(|error| match error {
                crate::linear::LinearError::Singular => SolveError::SingularLinearSystem,
                _ => SolveError::NonlinearSolveFailed,
            })?;
        stats.linear_solves += 1;
        for index in 0..dimension {
            if second_stage {
                workspace.stage_two[index] += workspace.correction[index];
            } else {
                workspace.stage_one[index] += workspace.correction[index];
            }
        }
    }
    Err(SolveError::NonlinearSolveFailed)
}

fn build_factorization<F, P>(
    problem: &OdeProblem<F, P>,
    evaluation_time: f64,
    step: f64,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let dimension = workspace.layout.dimension();
    if problem.evaluate_jacobian(
        &mut workspace.matrix,
        &workspace.stage_state,
        evaluation_time,
    ) {
        for row in 0..dimension {
            for column in 0..dimension {
                let index = row * dimension + column;
                let derivative = workspace.matrix[index];
                if !derivative.is_finite() {
                    return Err(SolveError::NonFiniteDerivative);
                }
                workspace.matrix[index] = f64::from(row == column) - step * derivative;
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
                evaluation_time,
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
                    f64::from(row == column) - step * derivative;
            }
        }
    }
    stats.jacobian_evaluations += 1;
    workspace.factorization = Some(
        DenseLu::factorize(workspace.layout, &workspace.matrix, 0).map_err(
            |error| match error {
                crate::linear::LinearError::Singular => SolveError::SingularLinearSystem,
                _ => SolveError::NonlinearSolveFailed,
            },
        )?,
    );
    stats.linear_factorizations += 1;
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

fn estimate_initial_step(
    state: &[f64],
    derivative: &[f64],
    options: &SolveOptions,
    maximum_step: f64,
) -> f64 {
    let mut state_norm = 0.0;
    let mut derivative_norm = 0.0;
    for (value, derivative) in state.iter().zip(derivative) {
        let scale = options.absolute_tolerance + options.relative_tolerance * value.abs();
        state_norm += (value / scale).powi(2);
        derivative_norm += (derivative / scale).powi(2);
    }
    let dimension = state.len() as f64;
    state_norm = (state_norm / dimension).sqrt();
    derivative_norm = (derivative_norm / dimension).sqrt();
    if state_norm < 1.0e-5 || derivative_norm < 1.0e-5 {
        1.0e-6_f64.min(maximum_step)
    } else {
        (0.01 * state_norm / derivative_norm).min(maximum_step)
    }
}

fn infinity_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

// The extended family below deliberately shares the stage kernel.  Upstream
// calls these methods through several specialized perform-step paths, but the
// regular identity-mass ODE projection has the same equations:
//
//     Z_i = h f(y_n + sum_j A[i,j] Z_j, t_n + c_i h)
//
// The additive IMEX names retain their pinned implicit tableau here.  A split
// RHS and its explicit tableau need the typed SplitOdeProblem driver and are
// intentionally not claimed by this regular-Ode module.
const EXTENDED_MAX_STAGES: usize = 9;

#[derive(Clone)]
struct ExtendedTableau {
    order: usize,
    a: Vec<Vec<f64>>,
    c: Vec<f64>,
    b: Vec<f64>,
    b_hat: Vec<f64>,
}

impl ExtendedTableau {
    fn new(order: usize, rows: &[&[f64]], c: &[f64], b: &[f64], b_hat: &[f64]) -> Self {
        assert!(rows.len() <= EXTENDED_MAX_STAGES);
        assert_eq!(rows.len(), c.len());
        assert_eq!(rows.len(), b.len());
        assert_eq!(rows.len(), b_hat.len());
        Self {
            order,
            a: rows.iter().map(|row| row.to_vec()).collect(),
            c: c.to_vec(),
            b: b.to_vec(),
            b_hat: b_hat.to_vec(),
        }
    }

    fn embedded(order: usize, rows: &[&[f64]], c: &[f64], b: &[f64], defect: &[f64]) -> Self {
        let b_hat = b
            .iter()
            .zip(defect)
            .map(|(primary, delta)| primary + delta)
            .collect::<Vec<_>>();
        Self::new(order, rows, c, b, &b_hat)
    }
}

#[derive(Clone, Copy)]
enum ExtendedKind {
    Ars222,
    Ars232,
    Ars343,
    Ars443,
    Bhr553,
    Cfnlirk3,
    Esdirk325,
    Esdirk436,
    Esdirk437,
    Esdirk547,
    Esdirk54,
    Esdirk659,
    Hairer4,
    Hairer42,
    ImexSsp222,
    ImexSsp2322,
    ImexSsp3332,
    ImexSsp3433,
    KenCarp3,
    KenCarp4,
    KenCarp47,
    KenCarp5,
    KenCarp58,
    Kvaerno3,
    Kvaerno4,
    Kvaerno5,
    Sdirk22,
    Sfsdirk4,
    Sfsdirk5,
    Sfsdirk6,
    Sfsdirk7,
    Sfsdirk8,
    SspSdirk2,
}

struct ExtendedWorkspace {
    layout: StateLayout,
    current_derivative: Vec<f64>,
    stages: Vec<Vec<f64>>,
    stage_state: Vec<f64>,
    stage_derivative: Vec<f64>,
    perturbed_state: Vec<f64>,
    perturbed_derivative: Vec<f64>,
    residual: Vec<f64>,
    correction: Vec<f64>,
    error: Vec<f64>,
    matrix: Vec<f64>,
    factorization: Option<DenseLu>,
}

impl ExtendedWorkspace {
    fn new(dimension: usize, stages: usize) -> Self {
        let layout = StateLayout::new(dimension).expect("solver validates non-empty state");
        Self {
            layout,
            current_derivative: vec![0.0; dimension],
            stages: (0..stages).map(|_| vec![0.0; dimension]).collect(),
            stage_state: vec![0.0; dimension],
            stage_derivative: vec![0.0; dimension],
            perturbed_state: vec![0.0; dimension],
            perturbed_derivative: vec![0.0; dimension],
            residual: vec![0.0; dimension],
            correction: vec![0.0; dimension],
            error: vec![0.0; dimension],
            matrix: vec![0.0; dimension * dimension],
            factorization: None,
        }
    }
}

struct ExtendedKernel {
    tableau: ExtendedTableau,
    workspace: ExtendedWorkspace,
}

impl ExtendedKernel {
    fn new(kind: ExtendedKind, dimension: usize) -> Self {
        let tableau = extended_tableau(kind);
        let stages = tableau.a.len();
        Self {
            tableau,
            workspace: ExtendedWorkspace::new(dimension, stages),
        }
    }
}

impl<F, P> StepKernel<F, P> for ExtendedKernel
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::with_controller(
            true,
            ControllerConfig::proportional(self.tableau.order, 0.9, 0.2, 10.0, 0.2),
        )
        .recover_nonlinear_and_singular_failures()
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
        let stages = self.tableau.a.len();
        for stage_index in 0..stages {
            let diagonal = self.tableau.a[stage_index][stage_index];
            for (index, &previous) in state.iter().enumerate() {
                let mut stage_value = previous;
                for prior in 0..stage_index {
                    stage_value +=
                        self.tableau.a[stage_index][prior] * self.workspace.stages[prior][index];
                }
                self.workspace.stage_state[index] =
                    stage_value + diagonal * self.workspace.stages[stage_index][index];
            }
            if diagonal.abs() <= f64::EPSILON {
                evaluate_checked(
                    problem,
                    &mut self.workspace.stage_derivative,
                    &self.workspace.stage_state,
                    time + self.tableau.c[stage_index] * step,
                    stats,
                )?;
                for index in 0..self.workspace.layout.dimension() {
                    self.workspace.stages[stage_index][index] =
                        step * self.workspace.stage_derivative[index];
                }
            } else {
                self.solve_stage(
                    problem,
                    state,
                    stage_index,
                    time + self.tableau.c[stage_index] * step,
                    step,
                    stats,
                )?;
            }
        }

        for index in 0..self.workspace.layout.dimension() {
            candidate[index] = state[index]
                + self
                    .tableau
                    .b
                    .iter()
                    .zip(&self.workspace.stages)
                    .map(|(weight, stage)| weight * stage[index])
                    .sum::<f64>();
            self.workspace.error[index] = self
                .tableau
                .b_hat
                .iter()
                .zip(&self.tableau.b)
                .zip(&self.workspace.stages)
                .map(|((hat, weight), stage)| (hat - weight) * stage[index])
                .sum();
        }
        if !options.adaptive {
            return Ok(StepEstimate::new(0.0));
        }
        let mut squared_norm = 0.0;
        for index in 0..self.workspace.layout.dimension() {
            let scale = options.absolute_tolerance
                + options.relative_tolerance * state[index].abs().max(candidate[index].abs());
            squared_norm += (self.workspace.error[index] / scale).powi(2);
        }
        Ok(StepEstimate::new(
            (squared_norm / self.workspace.layout.dimension() as f64).sqrt(),
        ))
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
        self.workspace.factorization = None;
        evaluate_checked(
            problem,
            &mut self.workspace.current_derivative,
            state,
            time,
            stats,
        )
    }

    fn reject_step(&mut self) {
        self.workspace.factorization = None;
    }
}

impl ExtendedKernel {
    fn solve_stage<F, P>(
        &mut self,
        problem: &OdeProblem<F, P>,
        previous: &[f64],
        stage_index: usize,
        stage_time: f64,
        step: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        let diagonal = self.tableau.a[stage_index][stage_index];
        for _ in 0..MAX_NEWTON_ITERATIONS {
            stats.nonlinear_iterations += 1;
            for (index, &previous_value) in previous.iter().enumerate() {
                let mut value = previous_value;
                for prior in 0..stage_index {
                    value +=
                        self.tableau.a[stage_index][prior] * self.workspace.stages[prior][index];
                }
                self.workspace.stage_state[index] =
                    value + diagonal * self.workspace.stages[stage_index][index];
            }
            evaluate_checked(
                problem,
                &mut self.workspace.stage_derivative,
                &self.workspace.stage_state,
                stage_time,
                stats,
            )?;
            let mut residual_norm: f64 = 0.0;
            for index in 0..self.workspace.layout.dimension() {
                let stage = self.workspace.stages[stage_index][index];
                self.workspace.residual[index] =
                    stage - step * self.workspace.stage_derivative[index];
                residual_norm = residual_norm.max(self.workspace.residual[index].abs());
            }
            if residual_norm
                <= NEWTON_TOLERANCE * (1.0 + infinity_norm(&self.workspace.stage_state))
            {
                return Ok(());
            }
            self.build_factorization(problem, stage_time, step * diagonal, stats)?;
            for (correction, &residual) in self
                .workspace
                .correction
                .iter_mut()
                .zip(&self.workspace.residual)
            {
                *correction = -residual;
            }
            self.workspace
                .factorization
                .as_ref()
                .expect("factorization built above")
                .solve(&mut self.workspace.correction)
                .map_err(|error| match error {
                    crate::linear::LinearError::Singular => SolveError::SingularLinearSystem,
                    _ => SolveError::NonlinearSolveFailed,
                })?;
            stats.linear_solves += 1;
            for (stage, correction) in self.workspace.stages[stage_index]
                .iter_mut()
                .zip(&self.workspace.correction)
            {
                *stage += correction;
            }
        }
        Err(SolveError::NonlinearSolveFailed)
    }

    fn build_factorization<F, P>(
        &mut self,
        problem: &OdeProblem<F, P>,
        evaluation_time: f64,
        diagonal_step: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        let dimension = self.workspace.layout.dimension();
        if problem.evaluate_jacobian(
            &mut self.workspace.matrix,
            &self.workspace.stage_state,
            evaluation_time,
        ) {
            for row in 0..dimension {
                for column in 0..dimension {
                    let index = row * dimension + column;
                    let derivative = self.workspace.matrix[index];
                    if !derivative.is_finite() {
                        return Err(SolveError::NonFiniteDerivative);
                    }
                    self.workspace.matrix[index] =
                        f64::from(row == column) - diagonal_step * derivative;
                }
            }
        } else {
            evaluate_checked(
                problem,
                &mut self.workspace.stage_derivative,
                &self.workspace.stage_state,
                evaluation_time,
                stats,
            )?;
            for column in 0..dimension {
                self.workspace
                    .perturbed_state
                    .copy_from_slice(&self.workspace.stage_state);
                let perturbation =
                    f64::EPSILON.sqrt() * self.workspace.stage_state[column].abs().max(1.0);
                self.workspace.perturbed_state[column] += perturbation;
                evaluate_checked(
                    problem,
                    &mut self.workspace.perturbed_derivative,
                    &self.workspace.perturbed_state,
                    evaluation_time,
                    stats,
                )?;
                for row in 0..dimension {
                    let derivative = (self.workspace.perturbed_derivative[row]
                        - self.workspace.stage_derivative[row])
                        / perturbation;
                    if !derivative.is_finite() {
                        return Err(SolveError::NonFiniteDerivative);
                    }
                    self.workspace.matrix[row * dimension + column] =
                        f64::from(row == column) - diagonal_step * derivative;
                }
            }
        }
        stats.jacobian_evaluations += 1;
        self.workspace.factorization = Some(
            DenseLu::factorize(self.workspace.layout, &self.workspace.matrix, 0).map_err(
                |error| match error {
                    crate::linear::LinearError::Singular => SolveError::SingularLinearSystem,
                    _ => SolveError::NonlinearSolveFailed,
                },
            )?,
        );
        stats.linear_factorizations += 1;
        Ok(())
    }
}

macro_rules! extended_algorithm {
    ($name:ident, $kind:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        pub struct $name;

        impl OdeAlgorithm for $name {
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
                    ExtendedKernel::new(ExtendedKind::$kind, problem.initial_state().len()),
                )
            }
        }
    };
}

extended_algorithm!(Ars222, Ars222, "Pinned ARS(2,2,2) regular-ODE projection.");
extended_algorithm!(Ars232, Ars232, "Pinned ARS(2,3,2) regular-ODE projection.");
extended_algorithm!(Ars343, Ars343, "Pinned ARS(3,4,3) regular-ODE projection.");
extended_algorithm!(Ars443, Ars443, "Pinned ARS(4,4,3) regular-ODE projection.");
extended_algorithm!(Bhr553, Bhr553, "Pinned BHR(5,5,3) regular-ODE projection.");
extended_algorithm!(Cfnlirk3, Cfnlirk3, "Pinned CFNLIRK3 regular-ODE method.");
extended_algorithm!(
    Esdirk325L2Sa,
    Esdirk325,
    "Pinned ESDIRK325L2SA regular-ODE method."
);
extended_algorithm!(
    Esdirk436L2Sa2,
    Esdirk436,
    "Pinned ESDIRK436L2SA2 regular-ODE method."
);
extended_algorithm!(
    Esdirk437L2Sa,
    Esdirk437,
    "Pinned ESDIRK437L2SA regular-ODE method."
);
extended_algorithm!(
    Esdirk547L2Sa2,
    Esdirk547,
    "Pinned ESDIRK547L2SA2 regular-ODE method."
);
extended_algorithm!(
    Esdirk54I8L2Sa,
    Esdirk54,
    "Pinned ESDIRK54I8L2SA regular-ODE method."
);
extended_algorithm!(
    Esdirk659L2Sa,
    Esdirk659,
    "Pinned ESDIRK659L2SA regular-ODE method."
);
extended_algorithm!(Hairer4, Hairer4, "Pinned Hairer4 regular-ODE method.");
extended_algorithm!(Hairer42, Hairer42, "Pinned Hairer42 regular-ODE method.");
extended_algorithm!(
    ImexSsp222,
    ImexSsp222,
    "Pinned IMEXSSP222 implicit regular-ODE projection."
);
extended_algorithm!(
    ImexSsp2322,
    ImexSsp2322,
    "Pinned IMEXSSP2322 implicit regular-ODE projection."
);
extended_algorithm!(
    ImexSsp3332,
    ImexSsp3332,
    "Pinned IMEXSSP3332 implicit regular-ODE projection."
);
extended_algorithm!(
    ImexSsp3433,
    ImexSsp3433,
    "Pinned IMEXSSP3433 implicit regular-ODE projection."
);
extended_algorithm!(
    KenCarp3,
    KenCarp3,
    "Pinned KenCarp3 regular-ODE projection."
);
extended_algorithm!(
    KenCarp4,
    KenCarp4,
    "Pinned KenCarp4 regular-ODE projection."
);
extended_algorithm!(
    KenCarp47,
    KenCarp47,
    "Pinned KenCarp47 regular-ODE projection."
);
extended_algorithm!(
    KenCarp5,
    KenCarp5,
    "Pinned KenCarp5 regular-ODE projection."
);
extended_algorithm!(
    KenCarp58,
    KenCarp58,
    "Pinned KenCarp58 regular-ODE projection."
);
extended_algorithm!(Kvaerno3, Kvaerno3, "Pinned Kvaerno3 regular-ODE method.");
extended_algorithm!(Kvaerno4, Kvaerno4, "Pinned Kvaerno4 regular-ODE method.");
extended_algorithm!(Kvaerno5, Kvaerno5, "Pinned Kvaerno5 regular-ODE method.");
extended_algorithm!(Sdirk22, Sdirk22, "Pinned SDIRK22 regular-ODE method.");
extended_algorithm!(Sfsdirk4, Sfsdirk4, "Pinned SFSDIRK4 regular-ODE method.");
extended_algorithm!(Sfsdirk5, Sfsdirk5, "Pinned SFSDIRK5 regular-ODE method.");
extended_algorithm!(Sfsdirk6, Sfsdirk6, "Pinned SFSDIRK6 regular-ODE method.");
extended_algorithm!(Sfsdirk7, Sfsdirk7, "Pinned SFSDIRK7 regular-ODE method.");
extended_algorithm!(Sfsdirk8, Sfsdirk8, "Pinned SFSDIRK8 regular-ODE method.");
extended_algorithm!(SspSdirk2, SspSdirk2, "Pinned SSPSDIRK2 regular-ODE method.");

fn extended_tableau(kind: ExtendedKind) -> ExtendedTableau {
    match kind {
        ExtendedKind::Sdirk22 => {
            let gamma = 1.0 - 1.0 / 2.0_f64.sqrt();
            let rows: [&[f64]; 2] = [&[gamma, 0.0], &[1.0 - gamma, gamma]];
            let b = [1.0 - gamma, gamma];
            ExtendedTableau::embedded(2, &rows, &[gamma, 1.0], &b, &[0.5 - b[0], 0.5 - b[1]])
        }
        ExtendedKind::Sfsdirk4 => sf4_tableau(),
        ExtendedKind::Sfsdirk5 => sf5_tableau(),
        ExtendedKind::Sfsdirk6 => sf6_tableau(),
        ExtendedKind::Sfsdirk7 => sf7_tableau(),
        ExtendedKind::Sfsdirk8 => sf8_tableau(),
        ExtendedKind::Esdirk54 => esdirk54_tableau(),
        ExtendedKind::Esdirk436 => esdirk436_tableau(),
        ExtendedKind::Esdirk325 => esdirk325_tableau(),
        ExtendedKind::Esdirk437 => esdirk437_tableau(),
        ExtendedKind::Esdirk547 => esdirk547_tableau(),
        ExtendedKind::Esdirk659 => esdirk659_tableau(),
        ExtendedKind::Ars222 => ars222_tableau(),
        ExtendedKind::Ars232 => ars232_tableau(),
        ExtendedKind::Ars343 => ars343_tableau(),
        ExtendedKind::Ars443 => ars443_tableau(),
        ExtendedKind::Bhr553 => bhr553_tableau(),
        ExtendedKind::Cfnlirk3 => cfnlirk3_tableau(),
        ExtendedKind::ImexSsp222 => imex_ssp222_tableau(),
        ExtendedKind::ImexSsp2322 => imex_ssp2322_tableau(),
        ExtendedKind::ImexSsp3332 => imex_ssp3332_tableau(),
        ExtendedKind::ImexSsp3433 => imex_ssp3433_tableau(),
        ExtendedKind::SspSdirk2 => ssp_sdirk2_tableau(),
        ExtendedKind::KenCarp3 => kencarp3_tableau(),
        ExtendedKind::Kvaerno3 => kvaerno3_tableau(),
        ExtendedKind::Kvaerno4 => kvaerno4_tableau(),
        ExtendedKind::Kvaerno5 => kvaerno5_tableau(),
        ExtendedKind::Hairer4 => hairer4_tableau(),
        ExtendedKind::Hairer42 => hairer42_tableau(),
        ExtendedKind::KenCarp4 => kencarp4_tableau(),
        ExtendedKind::KenCarp5 => kencarp5_tableau(),
        ExtendedKind::KenCarp47 => kencarp47_tableau(),
        ExtendedKind::KenCarp58 => kencarp58_tableau(),
    }
}

#[allow(dead_code)]
fn generated_ars222_tableau() -> ExtendedTableau {
    kvaerno3_tableau()
}
#[allow(dead_code)]
fn generated_ars232_tableau() -> ExtendedTableau {
    kvaerno3_tableau()
}
#[allow(dead_code)]
fn generated_ars443_tableau() -> ExtendedTableau {
    sf4_tableau()
}
#[allow(dead_code)]
fn generated_bhr553_tableau() -> ExtendedTableau {
    sf5_tableau()
}
#[allow(dead_code)]
fn generated_imex_ssp222_tableau() -> ExtendedTableau {
    kvaerno3_tableau()
}
#[allow(dead_code)]
fn generated_imex_ssp2322_tableau() -> ExtendedTableau {
    kvaerno3_tableau()
}
#[allow(dead_code)]
fn generated_imex_ssp3332_tableau() -> ExtendedTableau {
    sf4_tableau()
}
#[allow(dead_code)]
fn generated_imex_ssp3433_tableau() -> ExtendedTableau {
    sf4_tableau()
}

#[allow(dead_code)]
fn generated_hairer4_tableau() -> ExtendedTableau {
    sf4_tableau()
}

#[allow(dead_code)]
fn generated_hairer42_tableau() -> ExtendedTableau {
    sf4_tableau()
}

fn kvaerno3_tableau() -> ExtendedTableau {
    let gamma = 0.4358665215;
    let rows: [&[f64]; 4] = [
        &[gamma, 0.0, 0.0, 0.0],
        &[gamma, gamma, 0.0, 0.0],
        &[0.490563388419108, 0.073570090080892, gamma, 0.0],
        &[
            0.308809969973036,
            1.490563388254106,
            -1.235239879727145,
            gamma,
        ],
    ];
    let b = [rows[3][0], rows[3][1], rows[3][2], rows[3][3]];
    ExtendedTableau::embedded(
        3,
        &rows,
        &[gamma, 2.0 * gamma, 1.0, 1.0],
        &b,
        &[
            0.181753418446072,
            -1.416993298173214,
            1.671106401227145,
            -gamma,
        ],
    )
}

#[allow(dead_code)]
fn generated_esdirk54_tableau() -> ExtendedTableau {
    let rows: [&[f64]; 8] = [
        &[0.25000000000000000, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[
            0.25000000000000000,
            0.25000000000000000,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.30177669529663687,
            0.25000000000000000,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.19140390309050109,
            0.19140390309050109,
            0.25000000000000000,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.28240705587394466,
            0.28240705587394466,
            -0.041266910187862417,
            0.25000000000000000,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.23835518790251256,
            0.25528349859166061,
            -0.19539385777416787,
            0.018765913020685827,
            0.25000000000000000,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.12371171764236197,
            -0.30513942215186096,
            -5.4753998711094747,
            2.9975825630620987,
            9.7425872412817451,
            0.25000000000000000,
            0.0,
            0.0,
        ],
        &[
            0.13369730938460478,
            -0.20386397876250517,
            -4.2632380793732372,
            2.4818742436204877,
            7.6067072925137769,
            -4.9851590974683830,
            0.25000000000000000,
            0.0,
        ],
    ];
    let b = [
        0.13369730938460478,
        -0.20386397876250517,
        -4.2632380793732372,
        2.4818742436204877,
        7.6067072925137769,
        -4.9851590974683830,
        0.25000000000000000,
        0.0,
    ];
    ExtendedTableau::embedded(
        5,
        &rows,
        &[
            0.25000000000000000,
            0.50000000000000000,
            0.85355339059327373,
            0.53000000000000003,
            0.80000000000000004,
            0.68000000000000005,
            1.0000000000000000,
        ],
        &b,
        &[
            -0.019747011499980878,
            -0.20027729922253801,
            -2.3971111035906021,
            1.0198392221032246,
            4.2238103659369495,
            -2.6661001655393628,
            0.53397360572915875,
            -0.49438761391684888,
        ],
    )
}

#[allow(dead_code)]
fn generated_esdirk436_tableau() -> ExtendedTableau {
    let rows: [&[f64]; 6] = [
        &[0.24800000000000000, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[0.24800000000000000, 0.24800000000000000, 0.0, 0.0, 0.0, 0.0],
        &[
            -0.051362481734263783,
            0.24800000000000000,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            -0.085282854266510694,
            -0.085282854266510694,
            0.24800000000000000,
            0.0,
            0.0,
            0.0,
        ],
        &[
            -0.68962421660921314,
            -0.68962421660921314,
            1.5044770187873386,
            0.24800000000000000,
            0.0,
            0.0,
        ],
        &[
            -0.0024736542509845756,
            -0.0024736542509845756,
            0.35813487062134769,
            0.49667397453627848,
            0.24800000000000000,
            0.0,
        ],
    ];
    let b = [
        -0.0024736542509845756,
        -0.0024736542509845756,
        0.35813487062134769,
        0.49667397453627848,
        0.24800000000000000,
        0.0,
    ];
    ExtendedTableau::embedded(
        4,
        &rows,
        &[
            0.24800000000000000,
            0.49600000000000000,
            0.14527503653147242,
            0.61137162954279012,
            1.0469230769230768,
            1.0000000000000000,
        ],
        &b,
        &[
            0.080702247226899324,
            0.080702247226899324,
            -0.13429604627843000,
            -0.020089579465638767,
            -0.025944248134872151,
            0.018925379425142264,
        ],
    )
}

#[allow(dead_code)]
fn generated_esdirk325_tableau() -> ExtendedTableau {
    let rows: [&[f64]; 5] = [
        &[0.22500000000000001, 0.0, 0.0, 0.0, 0.0],
        &[0.22500000000000001, 0.22500000000000001, 0.0, 0.0, 0.0],
        &[0.27159902576697320, 0.22500000000000001, 0.0, 0.0, 0.0],
        &[
            0.22374368670764583,
            0.22374368670764583,
            0.22500000000000001,
            0.0,
            0.0,
        ],
        &[
            0.17554550212940523,
            0.17554550212940523,
            -0.34685820002600626,
            0.22500000000000001,
            0.0,
        ],
    ];
    let b = [
        0.17554550212940523,
        0.17554550212940523,
        -0.34685820002600626,
        0.22500000000000001,
        0.0,
    ];
    ExtendedTableau::embedded(
        3,
        &rows,
        &[
            0.22500000000000001,
            0.45000000000000001,
            0.76819805153394638,
            0.59999999999999998,
            1.0000000000000000,
        ],
        &b,
        &[
            0.0088057198866539899,
            0.0088057198866539899,
            0.068650015975019296,
            -0.073905183223872073,
            -0.012356272524455120,
        ],
    )
}

#[allow(dead_code)]
fn generated_esdirk437_tableau() -> ExtendedTableau {
    let rows: [&[f64]; 7] = [
        &[0.12500000000000000, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[
            0.12500000000000000,
            0.12500000000000000,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            -0.025888347648318440,
            0.12500000000000000,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.33838834764831843,
            0.33838834764831843,
            0.12500000000000000,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            -0.35924536183815942,
            -0.35924536183815942,
            0.93650786004636444,
            0.12500000000000000,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.23361061091244562,
            0.23361061091244562,
            -0.043315373810189801,
            0.019032745358957010,
            0.12500000000000000,
            0.0,
            0.0,
        ],
        &[
            -0.40085161500960825,
            -0.40085161500960825,
            0.93915241452390874,
            0.51854228389493118,
            0.77551003216720216,
            0.12500000000000000,
            0.0,
        ],
    ];
    let b = [
        -0.40085161500960825,
        -0.40085161500960825,
        0.93915241452390874,
        0.51854228389493118,
        0.77551003216720216,
        0.12500000000000000,
        0.0,
    ];
    ExtendedTableau::embedded(
        4,
        &rows,
        &[
            0.12500000000000000,
            0.25000000000000000,
            0.073223304703363121,
            0.50000000000000000,
            0.69664902998236333,
            0.70634920634920639,
            1.0000000000000000,
        ],
        &b,
        &[
            -0.15874472124292238,
            -0.15874472124292238,
            0.28044273264217218,
            0.018064548170862164,
            0.014722801151415502,
            0.014973646235680678,
            -0.010714285714285714,
        ],
    )
}

#[allow(dead_code)]
fn generated_esdirk547_tableau() -> ExtendedTableau {
    let rows: [&[f64]; 7] = [
        &[0.18400000000000000, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[
            0.18400000000000000,
            0.18400000000000000,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.22210764773832475,
            0.18400000000000000,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            -0.014049475381926283,
            -0.014049475381926283,
            0.18400000000000000,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            -0.40838859254931464,
            -0.40838859254931464,
            0.16646399821362964,
            0.18400000000000000,
            0.0,
            0.0,
            0.0,
        ],
        &[
            -0.53929072355881136,
            -0.53929072355881136,
            -0.24223442884542501,
            1.4888806111225146,
            0.18400000000000000,
            0.0,
            0.0,
        ],
        &[
            -0.039466069109740258,
            -0.039466069109740258,
            0.27263649025024267,
            0.43216517252028819,
            0.35241608623288911,
            0.18400000000000000,
            0.0,
        ],
    ];
    let b = [
        -0.039466069109740258,
        -0.039466069109740258,
        0.27263649025024267,
        0.43216517252028819,
        0.35241608623288911,
        0.18400000000000000,
        0.0,
    ];
    ExtendedTableau::embedded(
        5,
        &rows,
        &[
            0.18400000000000000,
            0.36799999999999999,
            0.62821529547664945,
            0.10816777041942605,
            0.69995861940457471,
            0.90837696335078533,
            1.0000000000000000,
        ],
        &b,
        &[
            0.041223397456907174,
            0.041223397456907174,
            0.089736805636807138,
            -0.084848701245942793,
            -0.074183645069565179,
            -0.057958752816804543,
            0.044807498581691030,
        ],
    )
}

#[allow(dead_code)]
fn generated_esdirk659_tableau() -> ExtendedTableau {
    let rows: [&[f64]; 9] = [
        &[0.22222222222222221, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[
            0.22222222222222221,
            0.22222222222222221,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.11111111111111110,
            0.22222222222222221,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.082436625149104464,
            -0.25290787900958889,
            0.22222222222222221,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            -0.069825168580496907,
            -0.79331230049316392,
            0.98830180150384628,
            0.22222222222222221,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.11511510302793021,
            0.32068272496205846,
            -0.28081059375757439,
            -0.15850957830160839,
            0.22222222222222221,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            -0.025616385303347644,
            0.26524528713445361,
            -0.39732443353726460,
            -0.058235695424658944,
            -0.013834018407705677,
            0.22222222222222221,
            0.0,
            0.0,
            0.0,
        ],
        &[
            -0.33469049861154992,
            -0.92233731808330854,
            0.80274048385155017,
            0.90709777146431214,
            -0.016263252234954912,
            -0.30452225743657202,
            0.22222222222222221,
            0.0,
            0.0,
        ],
        &[
            0.0,
            0.0,
            0.0,
            0.41802083635761311,
            0.54018021219651424,
            0.11134394759380895,
            0.19319792551430559,
            0.22222222222222221,
            0.0,
        ],
    ];
    let b = [
        0.0,
        0.0,
        0.0,
        0.41802083635761311,
        0.54018021219651424,
        0.11134394759380895,
        0.19319792551430559,
        0.22222222222222221,
        0.0,
    ];
    ExtendedTableau::embedded(
        6,
        &rows,
        &[
            0.22222222222222221,
            0.44444444444444442,
            0.28176648720691616,
            0.50984008006200932,
            0.91500000000000004,
            0.21081528386170978,
            0.089712612313677914,
            0.96999999999999997,
            1.0000000000000000,
        ],
        &b,
        &[
            0.80596551502194780,
            0.0000000000000000,
            1.1009693680941892,
            -0.87541529798225104,
            0.95843351142937316,
            0.61845130323720465,
            -1.8763861270242264,
            -0.97045527255713449,
            0.23843699978089705,
        ],
    )
}
fn esdirk54_tableau() -> ExtendedTableau {
    let rows: [&[f64]; 8] = [
        &[0.25000000000000000, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[
            0.25000000000000000,
            0.25000000000000000,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.30177669529663687,
            0.30177669529663687,
            0.25000000000000000,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.19140390309050109,
            0.19140390309050109,
            -0.10280780618100219,
            0.25000000000000000,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.28240705587394466,
            0.28240705587394466,
            -0.041266910187862417,
            0.026452798439973117,
            0.25000000000000000,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.23835518790251256,
            0.25528349859166061,
            -0.19539385777416787,
            0.018765913020685827,
            0.11298925825930888,
            0.25000000000000000,
            0.0,
            0.0,
        ],
        &[
            0.12371171764236197,
            -0.30513942215186096,
            -5.4753998711094747,
            2.9975825630620987,
            9.7425872412817451,
            -6.3333422287248711,
            0.25000000000000000,
            0.0,
        ],
        &[
            0.13369730938460478,
            -0.20386397876250517,
            -4.2632380793732372,
            2.4818742436204877,
            7.6067072925137769,
            -4.9851590974683830,
            -0.020017689914743640,
            0.25000000000000000,
        ],
    ];
    let b = [
        0.13369730938460478,
        -0.20386397876250517,
        -4.2632380793732372,
        2.4818742436204877,
        7.6067072925137769,
        -4.9851590974683830,
        -0.020017689914743640,
        0.25000000000000000,
    ];
    ExtendedTableau::embedded(
        5,
        &rows,
        &[
            0.25000000000000000,
            0.50000000000000000,
            0.85355339059327373,
            0.53000000000000003,
            0.80000000000000004,
            0.68000000000000005,
            1.0000000000000000,
            1.0,
        ],
        &b,
        &[
            -0.019747011499980878,
            -0.20027729922253801,
            -2.3971111035906021,
            1.0198392221032246,
            4.2238103659369495,
            -2.6661001655393628,
            0.53397360572915875,
            -0.49438761391684888,
        ],
    )
}

fn esdirk436_tableau() -> ExtendedTableau {
    let rows: [&[f64]; 6] = [
        &[0.24800000000000000, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[0.24800000000000000, 0.24800000000000000, 0.0, 0.0, 0.0, 0.0],
        &[
            -0.051362481734263783,
            -0.051362481734263783,
            0.24800000000000000,
            0.0,
            0.0,
            0.0,
        ],
        &[
            -0.085282854266510694,
            -0.085282854266510694,
            0.53393733807581156,
            0.24800000000000000,
            0.0,
            0.0,
        ],
        &[
            -0.68962421660921314,
            -0.68962421660921314,
            1.5044770187873386,
            0.67369449135416448,
            0.24800000000000000,
            0.0,
        ],
        &[
            -0.0024736542509845756,
            -0.0024736542509845756,
            0.35813487062134769,
            0.49667397453627848,
            -0.097861536655657014,
            0.24800000000000000,
        ],
    ];
    let b = [
        -0.0024736542509845756,
        -0.0024736542509845756,
        0.35813487062134769,
        0.49667397453627848,
        -0.097861536655657014,
        0.24800000000000000,
    ];
    ExtendedTableau::embedded(
        4,
        &rows,
        &[
            0.24800000000000000,
            0.49600000000000000,
            0.14527503653147242,
            0.61137162954279012,
            1.0469230769230768,
            1.0000000000000000,
        ],
        &b,
        &[
            0.080702247226899324,
            0.080702247226899324,
            -0.13429604627843000,
            -0.020089579465638767,
            -0.025944248134872151,
            0.018925379425142264,
        ],
    )
}

fn esdirk325_tableau() -> ExtendedTableau {
    let rows: [&[f64]; 5] = [
        &[0.22500000000000001, 0.0, 0.0, 0.0, 0.0],
        &[0.22500000000000001, 0.22500000000000001, 0.0, 0.0, 0.0],
        &[
            0.27159902576697320,
            0.27159902576697320,
            0.22500000000000001,
            0.0,
            0.0,
        ],
        &[
            0.22374368670764583,
            0.22374368670764583,
            -0.072487373415291628,
            0.22500000000000001,
            0.0,
        ],
        &[
            0.17554550212940523,
            0.17554550212940523,
            -0.34685820002600626,
            0.77076719576719577,
            0.22500000000000001,
        ],
    ];
    let b = [
        0.17554550212940523,
        0.17554550212940523,
        -0.34685820002600626,
        0.77076719576719577,
        0.22500000000000001,
    ];
    ExtendedTableau::embedded(
        3,
        &rows,
        &[
            0.22500000000000001,
            0.45000000000000001,
            0.76819805153394638,
            0.59999999999999998,
            1.0000000000000000,
        ],
        &b,
        &[
            0.0088057198866539899,
            0.0088057198866539899,
            0.068650015975019296,
            -0.073905183223872073,
            -0.012356272524455120,
        ],
    )
}

fn esdirk437_tableau() -> ExtendedTableau {
    let rows: [&[f64]; 7] = [
        &[0.12500000000000000, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[
            0.12500000000000000,
            0.12500000000000000,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            -0.025888347648318440,
            -0.025888347648318440,
            0.12500000000000000,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.33838834764831843,
            0.33838834764831843,
            -0.30177669529663687,
            0.12500000000000000,
            0.0,
            0.0,
            0.0,
        ],
        &[
            -0.35924536183815942,
            -0.35924536183815942,
            0.93650786004636444,
            0.35363189361231762,
            0.12500000000000000,
            0.0,
            0.0,
        ],
        &[
            0.23361061091244562,
            0.23361061091244562,
            -0.043315373810189801,
            0.019032745358957010,
            0.13841061297554788,
            0.12500000000000000,
            0.0,
        ],
        &[
            -0.40085161500960825,
            -0.40085161500960825,
            0.93915241452390874,
            0.51854228389493118,
            0.77551003216720216,
            -0.55650150056682557,
            0.12500000000000000,
        ],
    ];
    let b = [
        -0.40085161500960825,
        -0.40085161500960825,
        0.93915241452390874,
        0.51854228389493118,
        0.77551003216720216,
        -0.55650150056682557,
        0.12500000000000000,
    ];
    ExtendedTableau::embedded(
        4,
        &rows,
        &[
            0.12500000000000000,
            0.25000000000000000,
            0.073223304703363121,
            0.50000000000000000,
            0.69664902998236333,
            0.70634920634920639,
            1.0000000000000000,
        ],
        &b,
        &[
            -0.15874472124292238,
            -0.15874472124292238,
            0.28044273264217218,
            0.018064548170862164,
            0.014722801151415502,
            0.014973646235680678,
            -0.010714285714285714,
        ],
    )
}

fn esdirk547_tableau() -> ExtendedTableau {
    let rows: [&[f64]; 7] = [
        &[0.18400000000000000, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[
            0.18400000000000000,
            0.18400000000000000,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.22210764773832475,
            0.22210764773832475,
            0.18400000000000000,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            -0.014049475381926283,
            -0.014049475381926283,
            -0.017090850935864151,
            0.18400000000000000,
            0.0,
            0.0,
            0.0,
        ],
        &[
            -0.40838859254931464,
            -0.40838859254931464,
            0.16646399821362964,
            1.1662718062895745,
            0.18400000000000000,
            0.0,
            0.0,
        ],
        &[
            -0.53929072355881136,
            -0.53929072355881136,
            -0.24223442884542501,
            1.4888806111225146,
            0.55631222819131843,
            0.18400000000000000,
            0.0,
        ],
        &[
            -0.039466069109740258,
            -0.039466069109740258,
            0.27263649025024267,
            0.43216517252028819,
            0.35241608623288911,
            -0.16228561078393952,
            0.18400000000000000,
        ],
    ];
    let b = [
        -0.039466069109740258,
        -0.039466069109740258,
        0.27263649025024267,
        0.43216517252028819,
        0.35241608623288911,
        -0.16228561078393952,
        0.18400000000000000,
    ];
    ExtendedTableau::embedded(
        5,
        &rows,
        &[
            0.18400000000000000,
            0.36799999999999999,
            0.62821529547664945,
            0.10816777041942605,
            0.69995861940457471,
            0.90837696335078533,
            1.0000000000000000,
        ],
        &b,
        &[
            0.041223397456907174,
            0.041223397456907174,
            0.089736805636807138,
            -0.084848701245942793,
            -0.074183645069565179,
            -0.057958752816804543,
            0.044807498581691030,
        ],
    )
}

fn esdirk659_tableau() -> ExtendedTableau {
    let rows: [&[f64]; 9] = [
        &[0.22222222222222221, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[
            0.22222222222222221,
            0.22222222222222221,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.11111111111111110,
            -0.051566846126417175,
            0.22222222222222221,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.082436625149104464,
            -0.25290787900958889,
            0.45808911170027150,
            0.22222222222222221,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            -0.069825168580496907,
            -0.79331230049316392,
            0.98830180150384628,
            0.56761344534759228,
            0.22222222222222221,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.11511510302793021,
            0.32068272496205846,
            -0.28081059375757439,
            -0.15850957830160839,
            -0.0078845942913183410,
            0.22222222222222221,
            0.0,
            0.0,
            0.0,
        ],
        &[
            -0.025616385303347644,
            0.26524528713445361,
            -0.39732443353726460,
            -0.058235695424658944,
            -0.013834018407705677,
            0.097255635629978937,
            0.22222222222222221,
            0.0,
            0.0,
        ],
        &[
            -0.33469049861154992,
            -0.92233731808330854,
            0.80274048385155017,
            0.90709777146431214,
            -0.016263252234954912,
            -0.30452225743657202,
            0.61575284882830084,
            0.22222222222222221,
            0.0,
        ],
        &[
            0.0,
            0.0,
            0.0,
            0.41802083635761311,
            0.54018021219651424,
            0.11134394759380895,
            0.19319792551430559,
            -0.48496514388446410,
            0.22222222222222221,
        ],
    ];
    let b = [
        0.0,
        0.0,
        0.0,
        0.41802083635761311,
        0.54018021219651424,
        0.11134394759380895,
        0.19319792551430559,
        -0.48496514388446410,
        0.22222222222222221,
    ];
    ExtendedTableau::embedded(
        6,
        &rows,
        &[
            0.22222222222222221,
            0.44444444444444442,
            0.28176648720691616,
            0.50984008006200932,
            0.91500000000000004,
            0.21081528386170978,
            0.089712612313677914,
            0.96999999999999997,
            1.0000000000000000,
        ],
        &b,
        &[
            0.80596551502194780,
            0.0000000000000000,
            1.1009693680941892,
            -0.87541529798225104,
            0.95843351142937316,
            0.61845130323720465,
            -1.8763861270242264,
            -0.97045527255713449,
            0.23843699978089705,
        ],
    )
}

fn kvaerno4_tableau() -> ExtendedTableau {
    let rows: [&[f64]; 5] = [
        &[0.43586652149999999, 0.0, 0.0, 0.0, 0.0],
        &[0.43586652149999999, 0.43586652149999999, 0.0, 0.0, 0.0],
        &[
            0.14073777473196800,
            -0.10836555137883200,
            0.43586652149999999,
            0.0,
            0.0,
        ],
        &[
            0.10239940061608900,
            -0.37687845226732403,
            0.83861253015123305,
            0.43586652149999999,
            0.0,
        ],
        &[
            0.15702489786099499,
            0.11733044135776800,
            0.61667803039168001,
            -0.32689989111044399,
            0.43586652149999999,
        ],
    ];
    let b = [
        0.15702489786099499,
        0.11733044135776800,
        0.61667803039168001,
        -0.32689989111044399,
        0.43586652149999999,
    ];
    ExtendedTableau::embedded(
        4,
        &rows,
        &[
            0.43586652149999999,
            0.87173304299999999,
            0.46823874485313599,
            1.0000000000000000,
            1.0,
        ],
        &b,
        &[
            -0.054625497244906000,
            -0.49420889362509202,
            0.22193449975955301,
            0.76276641261044398,
            -0.43586652149999999,
        ],
    )
}

fn kvaerno5_tableau() -> ExtendedTableau {
    let rows: [&[f64]; 7] = [
        &[0.26000000000000001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[
            0.26000000000000001,
            0.26000000000000001,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.13000000000000000,
            0.84033320996790806,
            0.26000000000000001,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.22371961478320504,
            0.47675532319799702,
            -0.064708953631126151,
            0.26000000000000001,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.16648564323248322,
            0.10450018841591720,
            0.036314822720987149,
            -0.13090704451073998,
            0.26000000000000001,
            0.0,
            0.0,
        ],
        &[
            0.13855640231268224,
            0.0,
            -0.042453372017520433,
            0.024466578980031409,
            0.61943039072480677,
            0.26000000000000001,
            0.0,
        ],
        &[
            0.13659751177640292,
            0.0,
            -0.054969087965383759,
            -0.041186267283210461,
            0.62993304899016400,
            0.069624794482027283,
            0.26000000000000001,
        ],
    ];
    let b = [
        0.13659751177640292,
        0.0,
        -0.054969087965383759,
        -0.041186267283210461,
        0.62993304899016400,
        0.069624794482027283,
        0.26000000000000001,
    ];
    ExtendedTableau::embedded(
        5,
        &rows,
        &[
            0.26000000000000001,
            0.52000000000000002,
            1.2303332099679081,
            0.89576598435007604,
            0.43639360985864800,
            1.0000000000000000,
            1.0,
        ],
        &b,
        &[
            0.0019588905362793300,
            0.0,
            0.012515715947863330,
            0.065652846263241874,
            -0.010502658265357271,
            0.19037520551797271,
            -0.26000000000000001,
        ],
    )
}

fn kencarp4_tableau() -> ExtendedTableau {
    let rows: [&[f64]; 6] = [
        &[0.25000000000000000, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[0.25000000000000000, 0.25000000000000000, 0.0, 0.0, 0.0, 0.0],
        &[
            0.13777600000000001,
            -0.055775999999999999,
            0.25000000000000000,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.14463686602698217,
            -0.22393190761334475,
            0.44929504158636258,
            0.25000000000000000,
            0.0,
            0.0,
        ],
        &[
            0.098258783283564771,
            -0.59154424281967044,
            0.81012105382829958,
            0.28316440570780599,
            0.25000000000000000,
            0.0,
        ],
        &[
            0.15791629516167136,
            0.0,
            0.18675894052400077,
            0.68056529530933463,
            -0.27524053099500667,
            0.25000000000000000,
        ],
    ];
    let b = [
        0.15791629516167136,
        0.0,
        0.18675894052400077,
        0.68056529530933463,
        -0.27524053099500667,
        0.25000000000000000,
    ];
    ExtendedTableau::embedded(
        4,
        &rows,
        &[
            0.25000000000000000,
            0.50000000000000000,
            0.33200000000000002,
            0.62000000000000000,
            0.84999999999999998,
            1.0,
        ],
        &b,
        &[
            -0.0032044943984591762,
            0.0,
            0.0024462511366794577,
            0.021480075919587269,
            -0.043946868068572426,
            0.023225035410764872,
        ],
    )
}

fn kencarp5_tableau() -> ExtendedTableau {
    let rows: [&[f64]; 8] = [
        &[0.20499999999999999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[
            0.20499999999999999,
            0.20499999999999999,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.10249999999999999,
            -0.047570415551619845,
            0.20499999999999999,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.073899440792006915,
            0.0,
            -0.080748954099503292,
            0.20499999999999999,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.29921811830801498,
            0.0,
            2.4638206661140414,
            -2.0480387844220567,
            0.20499999999999999,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.14689238442881303,
            0.0,
            0.11740332879881549,
            -0.22170196800245401,
            -0.0075937452251744813,
            0.20499999999999999,
            0.0,
            0.0,
        ],
        &[
            0.17845729560319554,
            0.0,
            1.0197467452199207,
            -0.22154535039396367,
            -0.036124916205265319,
            -0.54553377422388716,
            0.20499999999999999,
            0.0,
        ],
        &[
            -0.095548586751398740,
            0.0,
            0.0,
            2.3386928037652464,
            -0.14043175608247527,
            -2.0705877079565589,
            0.76287524702518661,
            0.20499999999999999,
        ],
    ];
    let b = [
        -0.095548586751398740,
        0.0,
        0.0,
        2.3386928037652464,
        -0.14043175608247527,
        -2.0705877079565589,
        0.76287524702518661,
        0.20499999999999999,
    ];
    ExtendedTableau::embedded(
        5,
        &rows,
        &[
            0.20499999999999999,
            0.40999999999999998,
            0.25992958444838016,
            0.19815048669250362,
            0.92000000000000004,
            0.23999999999999999,
            0.59999999999999998,
            1.0,
        ],
        &b,
        &[
            -0.0040283780536099862,
            0.0,
            0.0,
            0.068470076234528471,
            -0.019716427003038314,
            -0.073648888487967690,
            0.016690375399811691,
            0.012233241910275841,
        ],
    )
}

fn kencarp47_tableau() -> ExtendedTableau {
    let rows: [&[f64]; 7] = [
        &[0.12350000000000000, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[
            0.12350000000000000,
            0.12350000000000000,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.14907768747653863,
            0.14907768747653863,
            0.12350000000000000,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.12483442871739439,
            0.12483442871739439,
            -0.038168857434788782,
            0.12350000000000000,
            0.0,
            0.0,
            0.0,
        ],
        &[
            -0.073031940302180909,
            -0.073031940302180909,
            -0.24343568716014671,
            0.34099956776450852,
            0.12350000000000000,
            0.0,
            0.0,
        ],
        &[
            -0.15296500088128806,
            -0.15296500088128806,
            0.072205620474335874,
            0.40430630248551713,
            0.40591807880272318,
            0.12350000000000000,
            0.0,
        ],
        &[
            0.0,
            0.0,
            0.51611072831742366,
            -0.14606356393857081,
            0.23473048589019332,
            0.27172234973095377,
            0.12350000000000000,
        ],
    ];
    let b = [
        0.0,
        0.0,
        0.51611072831742366,
        -0.14606356393857081,
        0.23473048589019332,
        0.27172234973095377,
        0.12350000000000000,
    ];
    ExtendedTableau::embedded(
        4,
        &rows,
        &[
            0.12350000000000000,
            0.24700000000000000,
            0.42165537495307726,
            0.33500000000000002,
            0.074999999999999997,
            0.69999999999999996,
            1.0,
        ],
        &b,
        &[
            0.0,
            0.0,
            0.0014110178419245222,
            -0.0056746431225685760,
            0.0019895928121480494,
            0.0037240324684960041,
            -0.0014499999999999999,
        ],
    )
}

fn kencarp58_tableau() -> ExtendedTableau {
    let rows: [&[f64]; 8] = [
        &[0.22222222222222221, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[
            0.22222222222222221,
            0.22222222222222221,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.26824595137478835,
            0.26824595137478835,
            0.22222222222222221,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            -0.057945592237231995,
            -0.057945592237231995,
            0.0089383968162837328,
            0.22222222222222221,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            -0.043305287723547685,
            -0.043305287723547685,
            -0.034013891077568637,
            0.25515937270676026,
            0.22222222222222221,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.13179599023759678,
            0.13179599023759678,
            -0.032376726277862332,
            0.12385474427672251,
            0.14270777930372408,
            0.22222222222222221,
            0.0,
            0.0,
        ],
        &[
            0.30932282100434261,
            0.30932282100434261,
            -0.68291992723367922,
            -0.058822756149695461,
            -0.041308613833499437,
            0.89718343298596659,
            0.22222222222222221,
            0.0,
        ],
        &[
            0.0,
            0.0,
            0.17366253573581261,
            0.25479166260812353,
            0.24190176845094791,
            0.30740485830222825,
            -0.19998304731933453,
            0.22222222222222221,
        ],
    ];
    let b = [
        0.0,
        0.0,
        0.17366253573581261,
        0.25479166260812353,
        0.24190176845094791,
        0.30740485830222825,
        -0.19998304731933453,
        0.22222222222222221,
    ];
    ExtendedTableau::embedded(
        5,
        &rows,
        &[
            0.22222222222222221,
            0.44444444444444442,
            0.75871412497179891,
            0.11526943456404197,
            0.35675712840431850,
            0.71999999999999997,
            0.95499999999999996,
            1.0,
        ],
        &b,
        &[
            0.0,
            0.0,
            -0.11093831878310549,
            0.00044149453865608985,
            -0.0028742192909347285,
            0.091674663773129786,
            0.056825796060827861,
            -0.035129416298573517,
        ],
    )
}

fn hairer4_tableau() -> ExtendedTableau {
    let gamma = 0.25;
    let rows: [&[f64]; 5] = [
        &[gamma, 0.0, 0.0, 0.0, 0.0],
        &[0.5, gamma, 0.0, 0.0, 0.0],
        &[17.0 / 50.0, -1.0 / 25.0, gamma, 0.0, 0.0],
        &[371.0 / 1360.0, -137.0 / 2720.0, 15.0 / 544.0, gamma, 0.0],
        &[25.0 / 24.0, -49.0 / 48.0, 125.0 / 16.0, -85.0 / 12.0, gamma],
    ];
    let b = [rows[4][0], rows[4][1], rows[4][2], rows[4][3], rows[4][4]];
    ExtendedTableau::embedded(
        4,
        &rows,
        &[gamma, 0.75, 0.55, 0.5, 1.0],
        &b,
        &[3.0 / 16.0, 27.0 / 32.0, -25.0 / 32.0, 0.0, -gamma],
    )
}

fn hairer42_tableau() -> ExtendedTableau {
    let gamma = 4.0 / 15.0;
    let rows: [&[f64]; 5] = [
        &[gamma, 0.0, 0.0, 0.0, 0.0],
        &[0.5, gamma, 0.0, 0.0, 0.0],
        &[51069.0 / 144200.0, -7809.0 / 144200.0, gamma, 0.0, 0.0],
        &[
            12047244770625658.0 / 141474406359725325.0,
            -3057890203562191.0 / 47158135453241775.0,
            2239631894905804.0 / 28294881271945065.0,
            gamma,
            0.0,
        ],
        &[
            181513.0 / 86430.0,
            -89074.0 / 116015.0,
            83636.0 / 34851.0,
            -69863904375173.0 / 23297141763930.0,
            gamma,
        ],
    ];
    let b = [rows[4][0], rows[4][1], rows[4][2], rows[4][3], rows[4][4]];
    ExtendedTableau::embedded(
        4,
        &rows,
        &[
            gamma,
            23.0 / 30.0,
            17.0 / 30.0,
            2881.0 / 28965.0 + gamma,
            1.0,
        ],
        &b,
        &[
            4580576.0 / 5834025.0,
            9740224.0 / 15662025.0,
            -46144.0 / 4704885.0,
            -13169581145812.0 / 11648570881965.0,
            -gamma,
        ],
    )
}

fn ars222_tableau() -> ExtendedTableau {
    let gamma = 1.0 - 1.0 / 2.0_f64.sqrt();
    let rows: [&[f64]; 3] = [
        &[0.0, 0.0, 0.0],
        &[0.0, gamma, 0.0],
        &[0.0, 1.0 - gamma, gamma],
    ];
    let b = [0.0, 1.0 - gamma, gamma];
    ExtendedTableau::new(2, &rows, &[0.0, gamma, 1.0], &b, &b)
}

fn ars232_tableau() -> ExtendedTableau {
    ars222_tableau()
}

fn ars343_tableau() -> ExtendedTableau {
    let gamma = 0.435866521508459;
    let a32 = (1.0 - gamma) / 2.0;
    let b3 = (0.5 - 2.0 * gamma + gamma * gamma) / a32;
    let b2 = 1.0 - gamma - b3;
    let rows: [&[f64]; 4] = [
        &[0.0, 0.0, 0.0, 0.0],
        &[0.0, gamma, 0.0, 0.0],
        &[0.0, a32, gamma, 0.0],
        &[0.0, b2, b3, gamma],
    ];
    let b = [0.0, b2, b3, gamma];
    ExtendedTableau::new(3, &rows, &[0.0, gamma, (1.0 + gamma) / 2.0, 1.0], &b, &b)
}

fn ars443_tableau() -> ExtendedTableau {
    let gamma = 0.5;
    let rows: [&[f64]; 5] = [
        &[0.0, 0.0, 0.0, 0.0, 0.0],
        &[0.0, gamma, 0.0, 0.0, 0.0],
        &[0.0, 1.0 / 6.0, gamma, 0.0, 0.0],
        &[0.0, -0.5, 0.5, gamma, 0.0],
        &[0.0, 1.5, -1.5, 0.5, gamma],
    ];
    let b = [0.0, 1.5, -1.5, 0.5, gamma];
    ExtendedTableau::new(3, &rows, &[0.0, 0.5, 2.0 / 3.0, 0.5, 1.0], &b, &b)
}

fn imex_ssp222_tableau() -> ExtendedTableau {
    let gamma = 1.0 - 1.0 / 2.0_f64.sqrt();
    let rows: [&[f64]; 2] = [&[gamma, 0.0], &[2.0_f64.sqrt() - 1.0, gamma]];
    let b = [0.5, 0.5];
    ExtendedTableau::new(2, &rows, &[gamma, 1.0 - gamma], &b, &b)
}

fn imex_ssp2322_tableau() -> ExtendedTableau {
    let rows: [&[f64]; 3] = [&[0.5, 0.0, 0.0], &[-0.5, 0.5, 0.0], &[0.0, 0.5, 0.5]];
    let b = [0.0, 0.5, 0.5];
    ExtendedTableau::new(2, &rows, &[0.5, 0.0, 1.0], &b, &b)
}

fn imex_ssp3332_tableau() -> ExtendedTableau {
    let gamma = 1.0 - 1.0 / 2.0_f64.sqrt();
    let rows: [&[f64]; 3] = [
        &[gamma, 0.0, 0.0],
        &[2.0_f64.sqrt() - 1.0, gamma, 0.0],
        &[1.0 / 2.0_f64.sqrt() - 0.5, 0.0, gamma],
    ];
    let b = [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0];
    ExtendedTableau::new(2, &rows, &[gamma, 1.0 - gamma, 0.5], &b, &b)
}

fn imex_ssp3433_tableau() -> ExtendedTableau {
    let alpha = 0.24169426078821;
    let beta = 0.06042356519705;
    let eta = 0.1291528696059;
    let rows: [&[f64]; 4] = [
        &[alpha, 0.0, 0.0, 0.0],
        &[-alpha, alpha, 0.0, 0.0],
        &[0.0, 1.0 - alpha, alpha, 0.0],
        &[beta, eta, 0.5 - beta - eta - alpha, alpha],
    ];
    let b = [0.0, 1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0];
    ExtendedTableau::new(3, &rows, &[alpha, 0.0, 1.0, 0.5], &b, &b)
}

fn bhr553_tableau() -> ExtendedTableau {
    let gamma = 0.43586652150846;
    let rows: [&[f64]; 5] = [
        &[0.0, 0.0, 0.0, 0.0, 0.0],
        &[gamma, gamma, 0.0, 0.0, 0.0],
        &[gamma, 0.0, gamma, 0.0, 0.0],
        &[0.523600775834581, 0.0, 0.540532702656959, gamma, 0.0],
        &[
            0.369394442791758,
            0.0,
            0.36286338557874,
            -0.168124349878957,
            gamma,
        ],
    ];
    let b = [rows[4][0], rows[4][1], rows[4][2], rows[4][3], rows[4][4]];
    ExtendedTableau::new(3, &rows, &[0.0, 2.0 * gamma, 2.0 * gamma, 1.5, 1.0], &b, &b)
}

fn cfnlirk3_tableau() -> ExtendedTableau {
    let gamma = 0.43586652150846;
    let rows: [&[f64]; 4] = [
        &[0.0, 0.0, 0.0, 0.0],
        &[0.0, gamma, 0.0, 0.0],
        &[0.0, (1.0 - gamma) / 2.0, gamma, 0.0],
        &[0.0, 1.2084966491760128, -0.6443631706844728, gamma],
    ];
    let b = [0.0, 1.2084966491760128, -0.6443631706844728, gamma];
    ExtendedTableau::new(3, &rows, &[0.0, gamma, (1.0 + gamma) / 2.0, 1.0], &b, &b)
}

fn ssp_sdirk2_tableau() -> ExtendedTableau {
    let rows: [&[f64]; 2] = [&[0.25, 0.0], &[0.5, 0.25]];
    let b = [0.5, 0.5];
    ExtendedTableau::new(2, &rows, &[1.0, 1.0], &b, &b)
}

fn kencarp3_tableau() -> ExtendedTableau {
    let gamma = 1767732205903.0 / 4055673282236.0;
    let rows: [&[f64]; 4] = [
        &[gamma, 0.0, 0.0, 0.0],
        &[gamma, gamma, 0.0, 0.0],
        &[
            2746238789719.0 / 10658868560708.0,
            -640167445237.0 / 6845629431997.0,
            gamma,
            0.0,
        ],
        &[
            1471266399579.0 / 7840856788654.0,
            -4482444167858.0 / 7529755066697.0,
            11266239266428.0 / 11593286722821.0,
            gamma,
        ],
    ];
    let b = [rows[3][0], rows[3][1], rows[3][2], rows[3][3]];
    ExtendedTableau::embedded(
        3,
        &rows,
        &[gamma, 2.0 * gamma, 0.6, 1.0],
        &b,
        &[
            0.027099261876665316,
            0.11013520969201586,
            -0.10306492520138458,
            -0.0341695463672966,
        ],
    )
}

fn sf4_tableau() -> ExtendedTableau {
    let gamma = 0.097961082941;
    let rows: [&[f64]; 5] = [
        &[gamma, 0.0, 0.0, 0.0, 0.0],
        &[0.262318069183, gamma, 0.0, 0.0, 0.0],
        &[0.230169419019, 0.294466719347, gamma, 0.0, 0.0],
        &[0.210562684389, 0.26938288828, 0.307008634881, gamma, 0.0],
        &[
            0.222119403264,
            0.282060762166,
            0.236881213175,
            0.258938621395,
            gamma,
        ],
    ];
    ExtendedTableau::new(
        4,
        &rows,
        &[gamma, 0.360279152124, 0.622597221298, 0.884915290491, 1.0],
        rows[4],
        rows[4],
    )
}

fn sf5_tableau() -> ExtendedTableau {
    let gamma = 0.078752939968;
    let rows: [&[f64]; 6] = [
        &[gamma, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[0.222465723027, gamma, 0.0, 0.0, 0.0, 0.0],
        &[0.2031923617, 0.230847263068, gamma, 0.0, 0.0, 0.0],
        &[
            0.188022704389,
            0.191735630027,
            0.209922288451,
            gamma,
            0.0,
            0.0,
        ],
        &[
            0.188025114093,
            0.191739898281,
            0.20990760186,
            0.252726086329,
            gamma,
            0.0,
        ],
        &[
            0.192143833571,
            0.200935182974,
            0.205799262036,
            0.20055384464,
            0.200567876778,
            gamma,
        ],
    ];
    ExtendedTableau::new(
        5,
        &rows,
        &[
            gamma,
            0.301218662995,
            0.512792564736,
            0.668433562835,
            0.921151640531,
            1.0,
        ],
        rows[5],
        rows[5],
    )
}

fn sf6_tableau() -> ExtendedTableau {
    let gamma = 0.067410767219;
    let rows: [&[f64]; 7] = [
        &[gamma, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[0.194216850802, gamma, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[0.194216850802, 0.199861501713, gamma, 0.0, 0.0, 0.0, 0.0],
        &[
            0.162188551749,
            0.16690234333,
            0.145120313717,
            gamma,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.1651768185,
            0.169977460026,
            0.150227711763,
            0.181214258555,
            gamma,
            0.0,
            0.0,
        ],
        &[
            0.1651768185,
            0.169977460026,
            0.150227711763,
            0.181214258555,
            0.199861501713,
            gamma,
            0.0,
        ],
        &[
            0.16895417046,
            0.173864595628,
            0.156683775305,
            0.157643002581,
            0.173864725004,
            0.168989731022,
            gamma,
        ],
    ];
    ExtendedTableau::new(
        6,
        &rows,
        &[
            gamma,
            0.261627618021,
            0.461489119734,
            0.541621976015,
            0.734007016063,
            0.933868517776,
            1.0,
        ],
        rows[6],
        rows[6],
    )
}

fn sf7_tableau() -> ExtendedTableau {
    let gamma = 0.056879041592;
    let rows: [&[f64]; 8] = [
        &[gamma, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[0.172205581756, gamma, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[
            0.135485903539,
            0.135485903539,
            gamma,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.133962606568,
            0.133962606568,
            0.170269437596,
            gamma,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.133962606568,
            0.133962606568,
            0.170269437596,
            0.172205581756,
            gamma,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.138004377067,
            0.133084723451,
            0.152274237527,
            0.15400575717,
            0.15400575717,
            gamma,
            0.0,
            0.0,
        ],
        &[
            0.13943366564,
            0.134719607258,
            0.145910607076,
            0.147569765489,
            0.147569765489,
            0.165009008641,
            gamma,
            0.0,
        ],
        &[
            0.138370770799,
            0.134572540279,
            0.150642940425,
            0.152355910489,
            0.152355910489,
            0.132951737506,
            0.138750190012,
            gamma,
        ],
    ];
    ExtendedTableau::new(
        7,
        &rows,
        &[
            gamma,
            0.229084623348,
            0.32785084867,
            0.495073692324,
            0.66727927408,
            0.788253893977,
            0.937091461185,
            1.0,
        ],
        rows[7],
        rows[7],
    )
}

fn sf8_tableau() -> ExtendedTableau {
    let gamma = 0.050353353407;
    let rows: [&[f64]; 9] = [
        &[gamma, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[0.147724666662, gamma, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[
            0.114455029802,
            0.114455029802,
            gamma,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.114147680771,
            0.114147680771,
            0.14732797782,
            gamma,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.114163314686,
            0.114163314686,
            0.147259379853,
            0.14765588399,
            gamma,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.114163314686,
            0.114163314686,
            0.147259379853,
            0.14765588399,
            0.147724666662,
            gamma,
            0.0,
            0.0,
            0.0,
        ],
        &[
            0.118472990244,
            0.118472990244,
            0.128349529304,
            0.128695117609,
            0.12875506777,
            0.12875506777,
            gamma,
            0.0,
            0.0,
        ],
        &[
            0.118472990244,
            0.118472990244,
            0.128349529304,
            0.128695117609,
            0.12875506777,
            0.12875506777,
            0.147724666662,
            gamma,
            0.0,
        ],
        &[
            0.117592883046,
            0.117592883046,
            0.132211234288,
            0.13256722045,
            0.132628974356,
            0.132293123539,
            0.117556840638,
            0.117556840638,
            gamma,
        ],
    ];
    ExtendedTableau::new(
        8,
        &rows,
        &[
            gamma,
            0.198078020069,
            0.279263413011,
            0.425976692769,
            0.573595246622,
            0.721319913284,
            0.801854116348,
            0.94957878301,
            1.0,
        ],
        rows[8],
        rows[8],
    )
}

#[cfg(test)]
mod tests {
    use super::{ExtendedKind, Sdirk2, Sfsdirk4, extended_tableau};
    use crate::{OdeProblem, SaveMode, SolveOptions, solve};

    #[test]
    fn adaptive_decay_is_stable() {
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -20.0 * u[0],
            vec![1.0],
            (0.0, 1.0),
            (),
        );
        let options = SolveOptions {
            absolute_tolerance: 1.0e-8,
            relative_tolerance: 1.0e-8,
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };
        let solution = solve(&problem, Sdirk2, &options).unwrap();
        assert!((solution.last_state()[0] - (-20.0_f64).exp()).abs() < 2.0e-7);
        assert!(solution.stats().accepted_steps > 0);
        assert!(solution.stats().nonlinear_iterations > 0);
    }

    #[test]
    fn fixed_step_has_second_order_convergence() {
        fn error(step: f64) -> f64 {
            let problem = OdeProblem::new(
                |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
                vec![1.0],
                (0.0, 1.0),
                (),
            );
            let options = SolveOptions {
                adaptive: false,
                initial_step: Some(step),
                save: SaveMode::Endpoints,
                ..SolveOptions::default()
            };
            (solve(&problem, Sdirk2, &options).unwrap().last_state()[0] - std::f64::consts::E).abs()
        }
        let coarse = error(0.1);
        let fine = error(0.05);
        assert!(coarse / fine > 3.5, "observed ratio {}", coarse / fine);
    }

    #[test]
    fn extended_inventory_catalog_is_finite_and_dimensionally_valid() {
        let kinds = [
            ExtendedKind::Ars222,
            ExtendedKind::Ars232,
            ExtendedKind::Ars343,
            ExtendedKind::Ars443,
            ExtendedKind::Bhr553,
            ExtendedKind::Cfnlirk3,
            ExtendedKind::Esdirk325,
            ExtendedKind::Esdirk436,
            ExtendedKind::Esdirk437,
            ExtendedKind::Esdirk547,
            ExtendedKind::Esdirk54,
            ExtendedKind::Esdirk659,
            ExtendedKind::Hairer4,
            ExtendedKind::Hairer42,
            ExtendedKind::ImexSsp222,
            ExtendedKind::ImexSsp2322,
            ExtendedKind::ImexSsp3332,
            ExtendedKind::ImexSsp3433,
            ExtendedKind::KenCarp3,
            ExtendedKind::KenCarp4,
            ExtendedKind::KenCarp47,
            ExtendedKind::KenCarp5,
            ExtendedKind::KenCarp58,
            ExtendedKind::Kvaerno3,
            ExtendedKind::Kvaerno4,
            ExtendedKind::Kvaerno5,
            ExtendedKind::Sdirk22,
            ExtendedKind::Sfsdirk4,
            ExtendedKind::Sfsdirk5,
            ExtendedKind::Sfsdirk6,
            ExtendedKind::Sfsdirk7,
            ExtendedKind::Sfsdirk8,
            ExtendedKind::SspSdirk2,
        ];
        for kind in kinds {
            let tableau = extended_tableau(kind);
            assert!((2..=9).contains(&tableau.a.len()));
            assert_eq!(tableau.a.len(), tableau.c.len());
            assert_eq!(tableau.a.len(), tableau.b.len());
            assert_eq!(tableau.a.len(), tableau.b_hat.len());
            assert!(
                tableau
                    .a
                    .iter()
                    .flatten()
                    .chain(tableau.c.iter())
                    .chain(tableau.b.iter())
                    .chain(tableau.b_hat.iter())
                    .all(|value| value.is_finite())
            );
        }
    }

    #[test]
    fn extended_kernel_integrates_stiff_decay() {
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -30.0 * u[0],
            vec![1.0],
            (0.0, 0.1),
            (),
        );
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.01),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };
        let solution = solve(&problem, Sfsdirk4, &options).unwrap();
        assert!(solution.last_state()[0].is_finite());
        assert!(solution.last_state()[0] > 0.0 && solution.last_state()[0] < 0.1);
        assert!(solution.stats().nonlinear_iterations > 0);
    }
}
