//! Native regular-ODE SDIRK methods.
//!
//! This module currently contains the two-stage adaptive SDIRK2/ESDIRK
//! method from the pinned OrdinaryDiffEqSDIRK tableau.

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

#[cfg(test)]
mod tests {
    use super::Sdirk2;
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
}
