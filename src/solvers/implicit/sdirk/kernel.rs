use crate::integrator::{ControllerConfig, KernelCapabilities, StepEstimate, StepKernel};
use crate::linear::{DenseLu, StateLayout};
use crate::tableau::RungeKuttaTableau;
use crate::{OdeProblem, SolveError, SolveOptions, SolverStats};

const MAX_NEWTON_ITERATIONS: usize = 12;
const NEWTON_TOLERANCE: f64 = 1.0e-12;
const CONTROLLER: ControllerConfig = ControllerConfig::proportional(2, 0.9, 0.2, 10.0, 0.2);

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
        let layout = StateLayout::for_validated_state(dimension);
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

pub(super) struct Sdirk2Kernel {
    workspace: Workspace,
    tableau: &'static RungeKuttaTableau,
}

impl Sdirk2Kernel {
    pub(super) fn new(dimension: usize, tableau: &'static RungeKuttaTableau) -> Self {
        Self {
            workspace: Workspace::new(dimension),
            tableau,
        }
    }
}

impl<F, P> StepKernel<F, P> for Sdirk2Kernel
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
            (time + self.tableau.c()[0] * step, step),
            false,
            self.tableau,
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
            (time + self.tableau.c()[1] * step, step),
            true,
            self.tableau,
            &mut self.workspace,
            stats,
        )?;

        for index in 0..dimension {
            candidate[index] = state[index]
                + self.tableau.b()[0] * self.workspace.stage_one[index]
                + self.tableau.b()[1] * self.workspace.stage_two[index];
            self.workspace.error[index] = self.tableau.error().unwrap()[0]
                * self.workspace.stage_one[index]
                + self.tableau.error().unwrap()[1] * self.workspace.stage_two[index];
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
    time_and_step: (f64, f64),
    second_stage: bool,
    tableau: &RungeKuttaTableau,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: crate::OdeFunction<P>,
{
    let (stage_time, step) = time_and_step;
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
                tableau.a()[1][0] * workspace.stage_one[index]
            } else {
                0.0
            };
            let diagonal = if second_stage {
                tableau.a()[1][1]
            } else {
                tableau.a()[0][0]
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
            .ok_or(SolveError::NonlinearSolveFailed)?
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
    F: crate::OdeFunction<P>,
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
        DenseLu::factorize(workspace.layout, &workspace.matrix).map_err(|error| match error {
            crate::linear::LinearError::Singular => SolveError::SingularLinearSystem,
            _ => SolveError::NonlinearSolveFailed,
        })?,
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
        let layout = StateLayout::for_validated_state(dimension);
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

pub(super) struct ExtendedKernel {
    tableau: &'static RungeKuttaTableau,
    workspace: ExtendedWorkspace,
}

impl ExtendedKernel {
    pub(super) fn new(tableau: &'static RungeKuttaTableau, dimension: usize) -> Self {
        let stages = tableau.a().len();
        Self {
            tableau,
            workspace: ExtendedWorkspace::new(dimension, stages),
        }
    }
}

impl<F, P> StepKernel<F, P> for ExtendedKernel
where
    F: crate::OdeFunction<P>,
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::with_controller(
            self.tableau.error().is_some(),
            ControllerConfig::proportional(self.tableau.order(), 0.9, 0.2, 10.0, 0.2),
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
        let stages = self.tableau.a().len();
        for stage_index in 0..stages {
            let diagonal = self.tableau.a()[stage_index][stage_index];
            for (index, &previous) in state.iter().enumerate() {
                let mut stage_value = previous;
                for prior in 0..stage_index {
                    stage_value +=
                        self.tableau.a()[stage_index][prior] * self.workspace.stages[prior][index];
                }
                self.workspace.stage_state[index] =
                    stage_value + diagonal * self.workspace.stages[stage_index][index];
            }
            if diagonal.abs() <= f64::EPSILON {
                evaluate_checked(
                    problem,
                    &mut self.workspace.stage_derivative,
                    &self.workspace.stage_state,
                    time + self.tableau.c()[stage_index] * step,
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
                    time + self.tableau.c()[stage_index] * step,
                    step,
                    stats,
                )?;
            }
        }

        for index in 0..self.workspace.layout.dimension() {
            candidate[index] = state[index]
                + self
                    .tableau
                    .b()
                    .iter()
                    .zip(&self.workspace.stages)
                    .map(|(weight, stage)| weight * stage[index])
                    .sum::<f64>();
            self.workspace.error[index] = self.tableau.error().map_or(0.0, |weights| {
                weights
                    .iter()
                    .zip(&self.workspace.stages)
                    .map(|(weight, stage)| weight * stage[index])
                    .sum()
            });
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
        F: crate::OdeFunction<P>,
    {
        let diagonal = self.tableau.a()[stage_index][stage_index];
        for _ in 0..MAX_NEWTON_ITERATIONS {
            stats.nonlinear_iterations += 1;
            for (index, &previous_value) in previous.iter().enumerate() {
                let mut value = previous_value;
                for prior in 0..stage_index {
                    value +=
                        self.tableau.a()[stage_index][prior] * self.workspace.stages[prior][index];
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
                .ok_or(SolveError::NonlinearSolveFailed)?
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
        F: crate::OdeFunction<P>,
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
            DenseLu::factorize(self.workspace.layout, &self.workspace.matrix).map_err(|error| {
                match error {
                    crate::linear::LinearError::Singular => SolveError::SingularLinearSystem,
                    _ => SolveError::NonlinearSolveFailed,
                }
            })?,
        );
        stats.linear_factorizations += 1;
        Ok(())
    }
}
