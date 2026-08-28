//! Native regular-ODE SDIRK methods.
//!
//! This module currently contains the two-stage adaptive SDIRK2/ESDIRK
//! method from the pinned OrdinaryDiffEqSDIRK tableau.

// The pinned SDIRK/ESDIRK catalogue intentionally preserves upstream decimal
// literals, including values with more written digits than f64 can represent.
#![allow(clippy::excessive_precision)]

use crate::integrator::{
    ControllerConfig, KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::linear::{DenseLu, StateLayout};
use crate::tableau::{LazyTableau, RungeKuttaTableau, load_tableau};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};
use differential_equations_tableau_macros::define_implicit_rk_tableau_from_file;

define_implicit_rk_tableau_from_file!(pub(super) ARS222_TABLEAU, "Ars222", "src/tableau/resources/implicit/ars222.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) ARS232_TABLEAU, "Ars232", "src/tableau/resources/implicit/ars232.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) ARS343_TABLEAU, "Ars343", "src/tableau/resources/implicit/ars343.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) ARS443_TABLEAU, "Ars443", "src/tableau/resources/implicit/ars443.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) BHR553_TABLEAU, "Bhr553", "src/tableau/resources/implicit/bhr553.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) CFNLIRK3_TABLEAU, "Cfnlirk3", "src/tableau/resources/implicit/cfnlirk3.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) ESDIRK325_TABLEAU, "Esdirk325L2Sa", "src/tableau/resources/implicit/esdirk325_l2_sa.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) ESDIRK436_TABLEAU, "Esdirk436L2Sa2", "src/tableau/resources/implicit/esdirk436_l2_sa2.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) ESDIRK437_TABLEAU, "Esdirk437L2Sa", "src/tableau/resources/implicit/esdirk437_l2_sa.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) ESDIRK547_TABLEAU, "Esdirk547L2Sa2", "src/tableau/resources/implicit/esdirk547_l2_sa2.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) ESDIRK54_TABLEAU, "Esdirk54I8L2Sa", "src/tableau/resources/implicit/esdirk54_i8_l2_sa.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) ESDIRK659_TABLEAU, "Esdirk659L2Sa", "src/tableau/resources/implicit/esdirk659_l2_sa.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) HAIRER4_TABLEAU, "Hairer4", "src/tableau/resources/implicit/hairer4.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) HAIRER42_TABLEAU, "Hairer42", "src/tableau/resources/implicit/hairer42.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) IMEX_SSP222_TABLEAU, "ImexSsp222", "src/tableau/resources/implicit/imex_ssp222.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) IMEX_SSP2322_TABLEAU, "ImexSsp2322", "src/tableau/resources/implicit/imex_ssp2322.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) IMEX_SSP3332_TABLEAU, "ImexSsp3332", "src/tableau/resources/implicit/imex_ssp3332.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) IMEX_SSP3433_TABLEAU, "ImexSsp3433", "src/tableau/resources/implicit/imex_ssp3433.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) KENCARP3_TABLEAU, "KenCarp3", "src/tableau/resources/implicit/ken_carp3.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) KENCARP4_TABLEAU, "KenCarp4", "src/tableau/resources/implicit/ken_carp4.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) KENCARP47_TABLEAU, "KenCarp47", "src/tableau/resources/implicit/ken_carp47.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) KENCARP5_TABLEAU, "KenCarp5", "src/tableau/resources/implicit/ken_carp5.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) KENCARP58_TABLEAU, "KenCarp58", "src/tableau/resources/implicit/ken_carp58.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) KVAERNO3_TABLEAU, "Kvaerno3", "src/tableau/resources/implicit/kvaerno3.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) KVAERNO4_TABLEAU, "Kvaerno4", "src/tableau/resources/implicit/kvaerno4.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) KVAERNO5_TABLEAU, "Kvaerno5", "src/tableau/resources/implicit/kvaerno5.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) SDIRK22_TABLEAU, "Sdirk22", "src/tableau/resources/implicit/sdirk22.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) SFSDIRK4_TABLEAU, "Sfsdirk4", "src/tableau/resources/implicit/sfsdirk4.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) SFSDIRK5_TABLEAU, "Sfsdirk5", "src/tableau/resources/implicit/sfsdirk5.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) SFSDIRK6_TABLEAU, "Sfsdirk6", "src/tableau/resources/implicit/sfsdirk6.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) SFSDIRK7_TABLEAU, "Sfsdirk7", "src/tableau/resources/implicit/sfsdirk7.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) SFSDIRK8_TABLEAU, "Sfsdirk8", "src/tableau/resources/implicit/sfsdirk8.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) SSP_SDIRK2_TABLEAU, "SspSdirk2", "src/tableau/resources/implicit/ssp_sdirk2.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) SDIRK2_TABLEAU, "Sdirk2", "src/tableau/resources/implicit/sdirk2.json", crate = crate);

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

impl Sdirk2 {
    /// Returns this method's lazily materialized, validated tableau.
    pub fn tableau(self) -> Result<&'static RungeKuttaTableau, crate::tableau::TableauError> {
        load_tableau(&SDIRK2_TABLEAU)
    }
}

impl OdeAlgorithm for Sdirk2 {
    fn solve_validated<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        let tableau = self.tableau().map_err(|_| SolveError::InvalidTableau)?;
        drive_integration(
            problem,
            options,
            Sdirk2Kernel::new(problem.initial_state().len(), tableau),
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

struct Sdirk2Kernel {
    workspace: Workspace,
    tableau: &'static RungeKuttaTableau,
}

impl Sdirk2Kernel {
    fn new(dimension: usize, tableau: &'static RungeKuttaTableau) -> Self {
        Self {
            workspace: Workspace::new(dimension),
            tableau,
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
    F: Fn(&mut [f64], &[f64], &P, f64),
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

struct ExtendedKernel {
    tableau: &'static RungeKuttaTableau,
    workspace: ExtendedWorkspace,
}

impl ExtendedKernel {
    fn new(tableau: &'static RungeKuttaTableau, dimension: usize) -> Self {
        let stages = tableau.a().len();
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
            self.workspace.error[index] = self
                .tableau
                .error()
                .expect("compile-time validation requires an error estimator")
                .iter()
                .zip(&self.workspace.stages)
                .map(|(weight, stage)| weight * stage[index])
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

macro_rules! extended_algorithm {
    ($name:ident, $kind:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        pub struct $name;

        impl $name {
            /// Returns this method's lazily materialized, validated tableau.
            pub fn tableau(
                self,
            ) -> Result<&'static RungeKuttaTableau, crate::tableau::TableauError> {
                load_tableau(extended_resource(ExtendedKind::$kind))
            }
        }

        impl OdeAlgorithm for $name {
            fn solve_validated<F, P>(
                &self,
                problem: &OdeProblem<F, P>,
                options: &SolveOptions,
            ) -> Result<Solution, SolveError>
            where
                F: Fn(&mut [f64], &[f64], &P, f64),
            {
                let tableau = self.tableau().map_err(|_| SolveError::InvalidTableau)?;
                drive_integration(
                    problem,
                    options,
                    ExtendedKernel::new(tableau, problem.initial_state().len()),
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

fn extended_resource(kind: ExtendedKind) -> &'static LazyTableau {
    match kind {
        ExtendedKind::Ars222 => &ARS222_TABLEAU,
        ExtendedKind::Ars232 => &ARS232_TABLEAU,
        ExtendedKind::Ars343 => &ARS343_TABLEAU,
        ExtendedKind::Ars443 => &ARS443_TABLEAU,
        ExtendedKind::Bhr553 => &BHR553_TABLEAU,
        ExtendedKind::Cfnlirk3 => &CFNLIRK3_TABLEAU,
        ExtendedKind::Esdirk325 => &ESDIRK325_TABLEAU,
        ExtendedKind::Esdirk436 => &ESDIRK436_TABLEAU,
        ExtendedKind::Esdirk437 => &ESDIRK437_TABLEAU,
        ExtendedKind::Esdirk547 => &ESDIRK547_TABLEAU,
        ExtendedKind::Esdirk54 => &ESDIRK54_TABLEAU,
        ExtendedKind::Esdirk659 => &ESDIRK659_TABLEAU,
        ExtendedKind::Hairer4 => &HAIRER4_TABLEAU,
        ExtendedKind::Hairer42 => &HAIRER42_TABLEAU,
        ExtendedKind::ImexSsp222 => &IMEX_SSP222_TABLEAU,
        ExtendedKind::ImexSsp2322 => &IMEX_SSP2322_TABLEAU,
        ExtendedKind::ImexSsp3332 => &IMEX_SSP3332_TABLEAU,
        ExtendedKind::ImexSsp3433 => &IMEX_SSP3433_TABLEAU,
        ExtendedKind::KenCarp3 => &KENCARP3_TABLEAU,
        ExtendedKind::KenCarp4 => &KENCARP4_TABLEAU,
        ExtendedKind::KenCarp47 => &KENCARP47_TABLEAU,
        ExtendedKind::KenCarp5 => &KENCARP5_TABLEAU,
        ExtendedKind::KenCarp58 => &KENCARP58_TABLEAU,
        ExtendedKind::Kvaerno3 => &KVAERNO3_TABLEAU,
        ExtendedKind::Kvaerno4 => &KVAERNO4_TABLEAU,
        ExtendedKind::Kvaerno5 => &KVAERNO5_TABLEAU,
        ExtendedKind::Sdirk22 => &SDIRK22_TABLEAU,
        ExtendedKind::Sfsdirk4 => &SFSDIRK4_TABLEAU,
        ExtendedKind::Sfsdirk5 => &SFSDIRK5_TABLEAU,
        ExtendedKind::Sfsdirk6 => &SFSDIRK6_TABLEAU,
        ExtendedKind::Sfsdirk7 => &SFSDIRK7_TABLEAU,
        ExtendedKind::Sfsdirk8 => &SFSDIRK8_TABLEAU,
        ExtendedKind::SspSdirk2 => &SSP_SDIRK2_TABLEAU,
    }
}

#[cfg(test)]
mod tests {
    use super::{ExtendedKind, Sdirk2, Sfsdirk4, extended_resource};
    use crate::tableau::load_tableau;
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
            let tableau = load_tableau(extended_resource(kind)).unwrap();
            assert!((2..=9).contains(&tableau.a().len()));
            assert_eq!(tableau.a().len(), tableau.c().len());
            assert_eq!(tableau.a().len(), tableau.b().len());
            assert_eq!(tableau.a().len(), tableau.error().unwrap().len());
            assert!(
                tableau
                    .a()
                    .iter()
                    .flat_map(|row| row.iter())
                    .chain(tableau.c().iter())
                    .chain(tableau.b().iter())
                    .chain(tableau.error().unwrap().iter())
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
