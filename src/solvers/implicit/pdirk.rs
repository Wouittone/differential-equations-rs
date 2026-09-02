use crate::integrator::{KernelCapabilities, StepEstimate, StepKernel, integrate};
use crate::linear::{factorize, solve_factorized};
use crate::tableau::{RungeKuttaTableau, load_tableau};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};
use differential_equations_tableau_macros::define_implicit_rk_tableau_from_file;

define_implicit_rk_tableau_from_file!(
    pub(super) PDIRK44_TABLEAU,
    "Pdirk44",
    "src/tableau/resources/implicit/pdirk44.json",
    crate = crate
);

const MAX_NEWTON_ITERATIONS: usize = 12;
const NEWTON_TOLERANCE: f64 = 1.0e-12;

/// The fourth-order two-processor parallel diagonally implicit Runge--Kutta method.
///
/// The Rust kernel evaluates the two independent stages in each wave
/// sequentially. This preserves the PDIRK44 formula and deterministic solver
/// statistics while avoiding an internal thread-pool policy; independent
/// problem-level parallelism is provided by the ensemble API.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Pdirk44;

/// SciML-compatible type spelling for [`Pdirk44`].
pub type PDIRK44 = Pdirk44;

/// SciML-compatible value constructor for [`Pdirk44`].
#[allow(non_upper_case_globals)]
pub const PDIRK44: Pdirk44 = Pdirk44;

impl Pdirk44 {
    /// Returns this method's lazily materialized, validated tableau.
    pub fn tableau(self) -> Result<&'static RungeKuttaTableau, crate::tableau::TableauError> {
        load_tableau(&PDIRK44_TABLEAU)
    }
}

impl OdeAlgorithm for Pdirk44 {
    fn solve_validated<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        let tableau = self.tableau().map_err(|_| SolveError::InvalidTableau)?;
        integrate(
            problem,
            options,
            Pdirk44Kernel::new(problem.initial_state().len(), tableau)?,
        )
    }
}

struct Pdirk44Workspace {
    stage_base: Vec<f64>,
    stage_state: Vec<f64>,
    stage_derivative: Vec<f64>,
    perturbed_state: Vec<f64>,
    perturbed_derivative: Vec<f64>,
    residual: Vec<f64>,
    correction: Vec<f64>,
    jacobian: Vec<f64>,
    matrix: Vec<f64>,
    pivots: Vec<usize>,
    increments: [Vec<f64>; 4],
}

impl Pdirk44Workspace {
    fn new(dimension: usize) -> Self {
        Self {
            stage_base: vec![0.0; dimension],
            stage_state: vec![0.0; dimension],
            stage_derivative: vec![0.0; dimension],
            perturbed_state: vec![0.0; dimension],
            perturbed_derivative: vec![0.0; dimension],
            residual: vec![0.0; dimension],
            correction: vec![0.0; dimension],
            jacobian: vec![0.0; dimension * dimension],
            matrix: vec![0.0; dimension * dimension],
            pivots: vec![0; dimension],
            increments: std::array::from_fn(|_| vec![0.0; dimension]),
        }
    }
}

struct Pdirk44Kernel {
    workspace: Pdirk44Workspace,
    tableau: &'static RungeKuttaTableau,
}

impl Pdirk44Kernel {
    fn new(dimension: usize, tableau: &'static RungeKuttaTableau) -> Result<Self, SolveError> {
        if tableau.a().len() != 4 {
            return Err(SolveError::InvalidTableau);
        }
        Ok(Self {
            workspace: Pdirk44Workspace::new(dimension),
            tableau,
        })
    }
}

impl<F, P> StepKernel<F, P> for Pdirk44Kernel
where
    F: crate::OdeFunction<P>,
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::new(false, 4)
    }

    fn initialize(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: f64,
        _: &mut SolverStats,
    ) -> Result<(), SolveError> {
        Ok(())
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
        self.workspace.stage_base.copy_from_slice(state);
        solve_stage(
            problem,
            &mut self.workspace,
            0,
            time + self.tableau.c()[0] * step,
            step,
            self.tableau.a()[0][0],
            stats,
        )?;

        self.workspace.stage_base.copy_from_slice(state);
        solve_stage(
            problem,
            &mut self.workspace,
            1,
            time + self.tableau.c()[1] * step,
            step,
            self.tableau.a()[1][1],
            stats,
        )?;

        for (index, base) in self.workspace.stage_base.iter_mut().enumerate() {
            *base = state[index]
                + self.tableau.a()[2][0] * self.workspace.increments[0][index]
                + self.tableau.a()[2][1] * self.workspace.increments[1][index];
        }
        solve_stage(
            problem,
            &mut self.workspace,
            2,
            time + self.tableau.c()[2] * step,
            step,
            self.tableau.a()[2][2],
            stats,
        )?;

        for (index, base) in self.workspace.stage_base.iter_mut().enumerate() {
            *base = state[index]
                + self.tableau.a()[3][0] * self.workspace.increments[0][index]
                + self.tableau.a()[3][1] * self.workspace.increments[1][index]
                + self.tableau.a()[3][2] * self.workspace.increments[2][index];
        }
        solve_stage(
            problem,
            &mut self.workspace,
            3,
            time + self.tableau.c()[3] * step,
            step,
            self.tableau.a()[3][3],
            stats,
        )?;

        for (index, value) in candidate.iter_mut().enumerate() {
            *value = state[index];
            for (weight, increment) in self.tableau.b().iter().zip(&self.workspace.increments) {
                *value += weight * increment[index];
            }
        }
        Ok(StepEstimate::new(0.0))
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

#[allow(clippy::too_many_arguments)]
fn solve_stage<F, P>(
    problem: &OdeProblem<F, P>,
    workspace: &mut Pdirk44Workspace,
    stage: usize,
    stage_time: f64,
    step: f64,
    gamma: f64,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: crate::OdeFunction<P>,
{
    let dimension = workspace.stage_base.len();
    workspace.increments[stage].fill(0.0);
    for _ in 0..MAX_NEWTON_ITERATIONS {
        for (index, stage_state) in workspace.stage_state.iter_mut().enumerate() {
            *stage_state = workspace.stage_base[index] + gamma * workspace.increments[stage][index];
        }
        evaluate_checked(
            problem,
            &mut workspace.stage_derivative,
            &workspace.stage_state,
            stage_time,
            stats,
        )?;
        for (index, residual) in workspace.residual.iter_mut().enumerate() {
            *residual =
                workspace.increments[stage][index] - step * workspace.stage_derivative[index];
        }
        let residual_norm = infinity_norm(&workspace.residual);
        let increment_norm = infinity_norm(&workspace.increments[stage]);
        if residual_norm <= NEWTON_TOLERANCE * (1.0 + increment_norm) {
            return Ok(());
        }

        build_stage_jacobian(problem, workspace, stage_time, stats)?;
        for row in 0..dimension {
            for column in 0..dimension {
                workspace.matrix[row * dimension + column] =
                    -step * gamma * workspace.jacobian[row * dimension + column];
            }
            workspace.matrix[row * dimension + row] += 1.0;
            workspace.correction[row] = -workspace.residual[row];
        }
        factorize(&mut workspace.matrix, &mut workspace.pivots, dimension)?;
        stats.linear_factorizations += 1;
        solve_factorized(
            &workspace.matrix,
            &workspace.pivots,
            &mut workspace.correction,
            dimension,
        );
        stats.linear_solves += 1;
        for (increment, correction) in workspace.increments[stage]
            .iter_mut()
            .zip(&workspace.correction)
        {
            *increment += correction;
        }
        stats.nonlinear_iterations += 1;
    }
    Err(SolveError::NonlinearSolveFailed)
}

fn build_stage_jacobian<F, P>(
    problem: &OdeProblem<F, P>,
    workspace: &mut Pdirk44Workspace,
    time: f64,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: crate::OdeFunction<P>,
{
    stats.jacobian_evaluations += 1;
    if problem.evaluate_jacobian(&mut workspace.jacobian, &workspace.stage_state, time) {
        return workspace
            .jacobian
            .iter()
            .all(|value| value.is_finite())
            .then_some(())
            .ok_or(SolveError::NonFiniteDerivative);
    }

    let dimension = workspace.stage_state.len();
    workspace
        .perturbed_state
        .copy_from_slice(&workspace.stage_state);
    for column in 0..dimension {
        let original = workspace.stage_state[column];
        let delta = f64::EPSILON.sqrt() * original.abs().max(1.0);
        workspace.perturbed_state[column] = original + delta;
        evaluate_checked(
            problem,
            &mut workspace.perturbed_derivative,
            &workspace.perturbed_state,
            time,
            stats,
        )?;
        for row in 0..dimension {
            workspace.jacobian[row * dimension + column] =
                (workspace.perturbed_derivative[row] - workspace.stage_derivative[row]) / delta;
        }
        workspace.perturbed_state[column] = original;
    }
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

fn infinity_norm(values: &[f64]) -> f64 {
    values
        .iter()
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()))
}
