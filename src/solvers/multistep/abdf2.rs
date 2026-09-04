//! Native regular-ODE ABDF2 implementation.
//!
//! This is the fixed-leading-coefficient, adaptive two-step BDF method from
//! the pinned OrdinaryDiffEqBDF source.  The first accepted step uses an
//! implicit-Euler startup; subsequent steps use the variable-step ABDF2
//! coefficients.  DAE residual, split/IMEX, and variable-order paths are not
//! represented here.

use crate::integrator::{
    ControllerConfig, KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::linear::{DenseLu, LinearError, StateLayout, factorize, solve_factorized};
use crate::solvers::multistep::tableaux::ABDF2_TABLEAU;
use crate::tableau::{TableauError, VariableMultistepTableau, load_tableau};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

const MAX_NEWTON_ITERATIONS: usize = 12;
const NEWTON_TOLERANCE: f64 = 1.0e-12;
const CONTROLLER: ControllerConfig = ControllerConfig::proportional(2, 0.9, 0.2, 10.0, 0.2);

/// Adaptive order-two BDF with a fixed leading coefficient (ABDF2).
///
/// This constructor targets regular identity-mass initial-value ODEs.  The
/// variable-step coefficients are taken from the pinned OrdinaryDiffEqBDF
/// `bdf_perform_step.jl` ABDF2 path.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Abdf2;

impl Abdf2 {
    /// Returns the lazily parsed variable-step BDF2 tableau.
    pub fn tableau(&self) -> Result<&'static VariableMultistepTableau, TableauError> {
        load_tableau(&ABDF2_TABLEAU)
    }
}

impl OdeAlgorithm for Abdf2 {
    fn solve_validated<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        let tableau = self.tableau().map_err(|_| SolveError::InvalidTableau)?;
        drive_integration(
            problem,
            options,
            Abdf2Kernel::new(problem.initial_state().len(), tableau),
        )
    }
}

struct Workspace {
    layout: StateLayout,
    current_derivative: Vec<f64>,
    previous_derivative: Vec<f64>,
    previous_state: Vec<f64>,
    evaluation_derivative: Vec<f64>,
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
            previous_derivative: vec![0.0; dimension],
            previous_state: vec![0.0; dimension],
            evaluation_derivative: vec![0.0; dimension],
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

struct Abdf2Kernel {
    workspace: Workspace,
    tableau: &'static VariableMultistepTableau,
}

impl Abdf2Kernel {
    fn new(dimension: usize, tableau: &'static VariableMultistepTableau) -> Self {
        Self {
            workspace: Workspace::new(dimension),
            tableau,
        }
    }
}

impl<F, P> StepKernel<F, P> for Abdf2Kernel
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
        for ((value, &state_value), &derivative) in candidate
            .iter_mut()
            .zip(state)
            .zip(&self.workspace.current_derivative)
        {
            *value = state_value + step * derivative;
        }

        let startup = self.workspace.last_step.is_none();
        let rho = self
            .workspace
            .last_step
            .map_or(1.0, |previous| step / previous);
        let (alpha_one, alpha_two, beta_zero, beta_one, beta_two) = if startup {
            let alpha = self.tableau.startup_alpha();
            let beta = self.tableau.startup_beta();
            let leading = alpha[0];
            (
                -alpha[1] / leading,
                0.0,
                beta[0] / leading,
                beta[1] / leading,
                0.0,
            )
        } else {
            let alpha_zero = self
                .tableau
                .alpha(0, rho)
                .ok_or(SolveError::InvalidTableau)?;
            let alpha_one = self
                .tableau
                .alpha(1, rho)
                .ok_or(SolveError::InvalidTableau)?;
            let alpha_two = self
                .tableau
                .alpha(2, rho)
                .ok_or(SolveError::InvalidTableau)?;
            let beta_zero = self
                .tableau
                .beta(0, rho)
                .ok_or(SolveError::InvalidTableau)?;
            let beta_one = self
                .tableau
                .beta(1, rho)
                .ok_or(SolveError::InvalidTableau)?;
            let beta_two = self
                .tableau
                .beta(2, rho)
                .ok_or(SolveError::InvalidTableau)?;
            (
                -alpha_one / alpha_zero,
                -alpha_two / alpha_zero,
                beta_zero / alpha_zero,
                beta_one / alpha_zero,
                beta_two / alpha_zero,
            )
        };

        newton_step(
            problem,
            state,
            candidate,
            time,
            step,
            startup,
            alpha_one,
            alpha_two,
            beta_zero,
            beta_one,
            beta_two,
            &mut self.workspace,
            stats,
        )?;

        if !options.adaptive {
            return Ok(StepEstimate::new(0.0));
        }
        // Pinned ABDF2's fixed-leading-coefficient estimator is the derivative
        // defect declared by the canonical tableau. During implicit-Euler
        // startup, use its Euler predictor defect.
        let defect_scale = if startup {
            0.0
        } else {
            self.tableau
                .defect_scale(rho)
                .ok_or(SolveError::InvalidTableau)?
        };
        let defect_weights = if startup {
            [0.0; 3]
        } else {
            [
                self.tableau
                    .defect_weight(0, rho)
                    .ok_or(SolveError::InvalidTableau)?,
                self.tableau
                    .defect_weight(1, rho)
                    .ok_or(SolveError::InvalidTableau)?,
                self.tableau
                    .defect_weight(2, rho)
                    .ok_or(SolveError::InvalidTableau)?,
            ]
        };
        let mut squared_norm = 0.0;
        for index in 0..candidate.len() {
            let defect = if startup {
                candidate[index] - (state[index] + step * self.workspace.current_derivative[index])
            } else {
                step * defect_scale
                    * (defect_weights[0] * self.workspace.evaluation_derivative[index]
                        + defect_weights[1] * self.workspace.current_derivative[index]
                        + defect_weights[2] * self.workspace.previous_derivative[index])
            };
            let scale = options.absolute_tolerance
                + options.relative_tolerance * candidate[index].abs().max(state[index].abs());
            squared_norm += (defect / scale).powi(2);
        }
        Ok(StepEstimate::new(
            (squared_norm / candidate.len() as f64).sqrt(),
        ))
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
            self.workspace
                .previous_state
                .copy_from_slice(previous_state);
            self.workspace
                .previous_derivative
                .copy_from_slice(&self.workspace.current_derivative);
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
    startup: bool,
    alpha_one: f64,
    alpha_two: f64,
    beta_zero: f64,
    beta_one: f64,
    beta_two: f64,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: crate::OdeFunction<P>,
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
        for index in 0..candidate.len() {
            let history = if startup {
                0.0
            } else {
                alpha_two * workspace.previous_state[index]
            };
            let forcing = step
                * (beta_zero * workspace.evaluation_derivative[index]
                    + beta_one * workspace.current_derivative[index]
                    + if startup {
                        0.0
                    } else {
                        beta_two * workspace.previous_derivative[index]
                    });
            workspace.residual[index] =
                candidate[index] - alpha_one * previous[index] - history - forcing;
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
    derivative_scale: f64,
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
                workspace.matrix[index] = f64::from(row == column) - derivative_scale * derivative;
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
                    f64::from(row == column) - derivative_scale * derivative;
            }
        }
    }
    stats.jacobian_evaluations += 1;
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
        maximum_step
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
