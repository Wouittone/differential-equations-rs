//! Native regular-ODE Cash4 singly-diagonally-implicit Runge--Kutta method.
//!
//! The coefficients and stage ordering are taken from the pinned
//! `OrdinaryDiffEqSDIRK` Cash4 tableau.  This implementation intentionally
//! supports only ordinary first-order problems; split and IMEX dispatch are
//! separate upstream algorithms.

use crate::integrator::{
    ControllerConfig, KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::linear::{DenseLu, StateLayout};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

const MAX_NEWTON_ITERATIONS: usize = 12;
const NEWTON_TOLERANCE: f64 = 1.0e-12;
const CONTROLLER: ControllerConfig = ControllerConfig::proportional(4, 0.9, 0.2, 10.0, 0.2);

// Cash4Tableau(::Type{T}, ::Type{T2}) at pinned revision
// 211142263781255a9aa2f910f6760b9f18ec29c8.
const GAMMA: f64 = 0.435866521508;
const A21: f64 = -1.1358665215;
const A31: f64 = 1.08543330679;
const A32: f64 = -0.721299828287;
const A41: f64 = 0.416349501547;
const A42: f64 = 0.190984004184;
const A43: f64 = -0.118643265417;
const A51: f64 = 0.896869652944;
const A52: f64 = 0.0182725272734;
const A53: f64 = -0.0845900310706;
const A54: f64 = -0.266418670647;
const C: [f64; 5] = [GAMMA, -0.7, 0.8, 0.924556761814, 1.0];
const B: [f64; 5] = [A51, A52, A53, A54, GAMMA];
// Cash4's `embedding=3` is the default in the pinned Julia constructor.
const BHAT2: [f64; 5] = [
    0.77669193291,
    0.0297472791484,
    -0.0267440239074,
    0.220304811849,
    0.0,
];
const ERROR: [f64; 5] = [
    BHAT2[0] - B[0],
    BHAT2[1] - B[1],
    BHAT2[2] - B[2],
    BHAT2[3] - B[3],
    BHAT2[4] - B[4],
];

/// The pinned fourth-order Cash SDIRK method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Cash4;

impl OdeAlgorithm for Cash4 {
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
            Cash4Kernel::new(problem.initial_state().len()),
        )
    }
}

struct Workspace {
    layout: StateLayout,
    current_derivative: Vec<f64>,
    stages: [Vec<f64>; 5],
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
            stages: std::array::from_fn(|_| vec![0.0; dimension]),
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

struct Cash4Kernel {
    workspace: Workspace,
}

impl Cash4Kernel {
    fn new(dimension: usize) -> Self {
        Self {
            workspace: Workspace::new(dimension),
        }
    }
}

impl<F, P> StepKernel<F, P> for Cash4Kernel
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
        // The first stage predictor is the explicit Euler increment.  Each
        // subsequent stage starts from the preceding stage increment.
        for (z, &derivative) in self.workspace.stages[0]
            .iter_mut()
            .zip(&self.workspace.current_derivative)
        {
            *z = step * derivative;
        }
        for stage in 0..5 {
            if stage > 0 {
                let (previous_stages, current_stages) = self.workspace.stages.split_at_mut(stage);
                current_stages[0].copy_from_slice(&previous_stages[stage - 1]);
            }
            // Nonautonomous Jacobians can vary with stage time, so only reuse
            // a factorization across Newton iterations of one stage.
            self.workspace.factorization = None;
            solve_stage(
                problem,
                state,
                time + C[stage] * step,
                step,
                stage,
                &mut self.workspace,
                stats,
            )?;
        }
        for (index, candidate_value) in candidate.iter_mut().enumerate() {
            *candidate_value = state[index]
                + B.iter()
                    .zip(&self.workspace.stages)
                    .map(|(&weight, stage)| weight * stage[index])
                    .sum::<f64>();
            self.workspace.error[index] = ERROR
                .iter()
                .zip(&self.workspace.stages)
                .map(|(&weight, stage)| weight * stage[index])
                .sum::<f64>();
        }
        if !options.adaptive {
            return Ok(StepEstimate::new(0.0));
        }
        let mut squared_norm = 0.0;
        for index in 0..candidate.len() {
            let scale = options.absolute_tolerance
                + options.relative_tolerance * state[index].abs().max(candidate[index].abs());
            squared_norm += (self.workspace.error[index] / scale).powi(2);
        }
        Ok(StepEstimate::new(
            (squared_norm / candidate.len() as f64).sqrt(),
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

fn solve_stage<F, P>(
    problem: &OdeProblem<F, P>,
    previous: &[f64],
    stage_time: f64,
    step: f64,
    stage_index: usize,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let dimension = workspace.layout.dimension();
    for _ in 0..MAX_NEWTON_ITERATIONS {
        stats.nonlinear_iterations += 1;
        for (index, &previous_value) in previous.iter().enumerate() {
            let coupling = match stage_index {
                0 => 0.0,
                1 => A21 * workspace.stages[0][index],
                2 => A31 * workspace.stages[0][index] + A32 * workspace.stages[1][index],
                3 => {
                    A41 * workspace.stages[0][index]
                        + A42 * workspace.stages[1][index]
                        + A43 * workspace.stages[2][index]
                }
                4 => {
                    A51 * workspace.stages[0][index]
                        + A52 * workspace.stages[1][index]
                        + A53 * workspace.stages[2][index]
                        + A54 * workspace.stages[3][index]
                }
                _ => return Err(SolveError::InvalidTableau),
            };
            workspace.stage_state[index] =
                previous_value + coupling + GAMMA * workspace.stages[stage_index][index];
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
            let stage = workspace.stages[stage_index][index];
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
        for (stage_value, correction) in workspace.stages[stage_index]
            .iter_mut()
            .zip(&workspace.correction)
        {
            *stage_value += correction;
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
                workspace.matrix[index] = f64::from(row == column) - step * GAMMA * derivative;
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
                    f64::from(row == column) - step * GAMMA * derivative;
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

#[cfg(test)]
mod tests {
    use super::Cash4;
    use crate::{OdeProblem, SaveMode, SolveOptions, solve};

    #[test]
    fn fixed_step_has_fourth_order_convergence() {
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
            (solve(&problem, Cash4, &options).unwrap().last_state()[0] - std::f64::consts::E).abs()
        }
        let coarse = error(0.1);
        let fine = error(0.05);
        assert!(coarse / fine > 10.0, "observed ratio {}", coarse / fine);
    }

    #[test]
    fn adaptive_stiff_decay() {
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
        let solution = solve(&problem, Cash4, &options).unwrap();
        assert!((solution.last_state()[0] - (-20.0_f64).exp()).abs() < 2.0e-7);
        assert!(solution.stats().accepted_steps > 0);
        assert!(solution.stats().nonlinear_iterations > 0);
    }
}
