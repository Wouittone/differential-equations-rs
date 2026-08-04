use crate::integrator::{
    ControllerConfig, KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::linear::{factorize, solve_factorized};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

const GAMMA: f64 = 1.0 / (2.0 + std::f64::consts::SQRT_2);
const C32: f64 = 6.0 + std::f64::consts::SQRT_2;
const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 6.0;

/// The adaptive Rosenbrock 2/3 W-method for stiff ODEs.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Rosenbrock23;

struct Workspace {
    current_derivative: Vec<f64>,
    perturbed_state: Vec<f64>,
    perturbed_derivative: Vec<f64>,
    time_derivative: Vec<f64>,
    midpoint_state: Vec<f64>,
    midpoint_derivative: Vec<f64>,
    candidate_derivative: Vec<f64>,
    k1: Vec<f64>,
    k2: Vec<f64>,
    k3: Vec<f64>,
    right_hand_side: Vec<f64>,
    jacobian: Vec<f64>,
    factorization: Vec<f64>,
    pivots: Vec<usize>,
    differentiation_valid: bool,
}

impl Workspace {
    fn new(dimension: usize) -> Self {
        Self {
            current_derivative: vec![0.0; dimension],
            perturbed_state: vec![0.0; dimension],
            perturbed_derivative: vec![0.0; dimension],
            time_derivative: vec![0.0; dimension],
            midpoint_state: vec![0.0; dimension],
            midpoint_derivative: vec![0.0; dimension],
            candidate_derivative: vec![0.0; dimension],
            k1: vec![0.0; dimension],
            k2: vec![0.0; dimension],
            k3: vec![0.0; dimension],
            right_hand_side: vec![0.0; dimension],
            jacobian: vec![0.0; dimension * dimension],
            factorization: vec![0.0; dimension * dimension],
            pivots: vec![0; dimension],
            differentiation_valid: false,
        }
    }
}

impl OdeAlgorithm for Rosenbrock23 {
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
            Rosenbrock23Kernel::new(problem.initial_state().len()),
        )
    }
}

struct Rosenbrock23Kernel {
    workspace: Workspace,
    candidate_derivative_valid: bool,
}

impl Rosenbrock23Kernel {
    fn new(dimension: usize) -> Self {
        Self {
            workspace: Workspace::new(dimension),
            candidate_derivative_valid: false,
        }
    }
}

impl<F, P> StepKernel<F, P> for Rosenbrock23Kernel
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::with_controller(
            true,
            ControllerConfig::proportional(3, SAFETY, MIN_FACTOR, MAX_FACTOR, MIN_FACTOR),
        )
    }

    fn initialize(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        evaluate(
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
        let error = perform_step(
            problem,
            state,
            time,
            step,
            candidate,
            options,
            &mut self.workspace,
            stats,
        )?;
        self.candidate_derivative_valid = options.adaptive;
        Ok(StepEstimate::new(error))
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
        self.workspace.differentiation_valid = false;
        if self.candidate_derivative_valid && !callback_applied {
            std::mem::swap(
                &mut self.workspace.current_derivative,
                &mut self.workspace.candidate_derivative,
            );
            Ok(())
        } else {
            evaluate(
                problem,
                &mut self.workspace.current_derivative,
                state,
                time,
                stats,
            )
        }
    }

    fn reject_step(&mut self) {}
}

#[allow(clippy::too_many_arguments)]
fn perform_step<F, P>(
    problem: &OdeProblem<F, P>,
    state: &[f64],
    time: f64,
    step: f64,
    candidate: &mut [f64],
    options: &SolveOptions,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<f64, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let dimension = state.len();
    if !workspace.differentiation_valid {
        differentiate(problem, state, time, workspace, stats)?;
        workspace.differentiation_valid = true;
    }
    for row in 0..dimension {
        for column in 0..dimension {
            workspace.factorization[row * dimension + column] = f64::from(row == column)
                - GAMMA * step * workspace.jacobian[row * dimension + column];
        }
    }
    factorize(
        &mut workspace.factorization,
        &mut workspace.pivots,
        dimension,
    )?;

    for index in 0..dimension {
        workspace.right_hand_side[index] =
            workspace.current_derivative[index] + GAMMA * step * workspace.time_derivative[index];
    }
    solve_factorized(
        &workspace.factorization,
        &workspace.pivots,
        &mut workspace.right_hand_side,
        dimension,
    );
    workspace.k1.copy_from_slice(&workspace.right_hand_side);
    stats.linear_solves += 1;

    for (index, &value) in state.iter().enumerate() {
        workspace.midpoint_state[index] = value + 0.5 * step * workspace.k1[index];
    }
    evaluate(
        problem,
        &mut workspace.midpoint_derivative,
        &workspace.midpoint_state,
        time + 0.5 * step,
        stats,
    )?;

    for index in 0..dimension {
        workspace.right_hand_side[index] =
            workspace.midpoint_derivative[index] - workspace.k1[index];
    }
    solve_factorized(
        &workspace.factorization,
        &workspace.pivots,
        &mut workspace.right_hand_side,
        dimension,
    );
    for (index, &value) in state.iter().enumerate() {
        workspace.k2[index] = workspace.right_hand_side[index] + workspace.k1[index];
        candidate[index] = value + step * workspace.k2[index];
    }
    stats.linear_solves += 1;

    if !options.adaptive {
        return Ok(0.0);
    }
    evaluate(
        problem,
        &mut workspace.candidate_derivative,
        candidate,
        time + step,
        stats,
    )?;
    for index in 0..dimension {
        workspace.right_hand_side[index] = workspace.candidate_derivative[index]
            - C32 * (workspace.k2[index] - workspace.midpoint_derivative[index])
            - 2.0 * (workspace.k1[index] - workspace.current_derivative[index])
            + step * workspace.time_derivative[index];
    }
    solve_factorized(
        &workspace.factorization,
        &workspace.pivots,
        &mut workspace.right_hand_side,
        dimension,
    );
    workspace.k3.copy_from_slice(&workspace.right_hand_side);
    stats.linear_solves += 1;

    let mut squared_norm = 0.0;
    for (index, &value) in state.iter().enumerate() {
        let local_error =
            (step / 6.0) * (workspace.k1[index] - 2.0 * workspace.k2[index] + workspace.k3[index]);
        let scale = options.absolute_tolerance
            + options.relative_tolerance * value.abs().max(candidate[index].abs());
        squared_norm += (local_error / scale).powi(2);
    }
    Ok((squared_norm / dimension as f64).sqrt())
}

fn differentiate<F, P>(
    problem: &OdeProblem<F, P>,
    state: &[f64],
    time: f64,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let dimension = state.len();
    if problem.evaluate_jacobian(&mut workspace.jacobian, state, time) {
        if workspace.jacobian.iter().any(|value| !value.is_finite()) {
            return Err(SolveError::NonFiniteDerivative);
        }
    } else {
        for column in 0..dimension {
            workspace.perturbed_state.copy_from_slice(state);
            let perturbation = f64::EPSILON.sqrt() * state[column].abs().max(1.0);
            workspace.perturbed_state[column] += perturbation;
            evaluate_unchecked(
                problem,
                &mut workspace.perturbed_derivative,
                &workspace.perturbed_state,
                time,
                stats,
            );
            for row in 0..dimension {
                let derivative = (workspace.perturbed_derivative[row]
                    - workspace.current_derivative[row])
                    / perturbation;
                if !derivative.is_finite() {
                    return Err(SolveError::NonFiniteDerivative);
                }
                workspace.jacobian[row * dimension + column] = derivative;
            }
        }
    }

    let time_perturbation = f64::EPSILON.sqrt() * time.abs().max(1.0);
    evaluate_unchecked(
        problem,
        &mut workspace.perturbed_derivative,
        state,
        time + time_perturbation,
        stats,
    );
    for index in 0..dimension {
        let derivative = (workspace.perturbed_derivative[index]
            - workspace.current_derivative[index])
            / time_perturbation;
        if !derivative.is_finite() {
            return Err(SolveError::NonFiniteDerivative);
        }
        workspace.time_derivative[index] = derivative;
    }
    stats.jacobian_evaluations += 1;
    Ok(())
}

fn evaluate_unchecked<F, P>(
    problem: &OdeProblem<F, P>,
    derivative: &mut [f64],
    state: &[f64],
    time: f64,
    stats: &mut SolverStats,
) where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    (problem.rhs)(derivative, state, problem.parameters(), time);
    stats.rhs_evaluations += 1;
}

fn evaluate<F, P>(
    problem: &OdeProblem<F, P>,
    derivative: &mut [f64],
    state: &[f64],
    time: f64,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    evaluate_unchecked(problem, derivative, state, time, stats);
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

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use crate::{CallbackAction, OdeProblem, Rosenbrock23, SaveMode, SolveOptions, solve};

    #[test]
    fn solves_a_stiff_nonautonomous_problem() {
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), time: f64| {
                du[0] = -1000.0 * (u[0] - time.cos()) - time.sin();
            },
            vec![1.0],
            (0.0, 1.0),
            (),
        );
        let options = SolveOptions {
            absolute_tolerance: 1.0e-7,
            relative_tolerance: 1.0e-7,
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        let solution = solve(&problem, Rosenbrock23, &options).unwrap();

        assert!((solution.last_state()[0] - 1.0_f64.cos()).abs() < 2.0e-6);
        assert!(solution.stats().linear_solves > 0);
        assert!(solution.stats().jacobian_evaluations > 0);
    }

    #[test]
    fn reuses_differentiation_after_a_rejected_step() {
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), time: f64| {
                du[0] = -1000.0 * (u[0] - time.cos()) - time.sin();
            },
            vec![1.0],
            (0.0, 1.0),
            (),
        );
        let options = SolveOptions {
            absolute_tolerance: 1.0e-7,
            relative_tolerance: 1.0e-7,
            initial_step: Some(1.0),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        let solution = solve(&problem, Rosenbrock23, &options).unwrap();
        let stats = solution.stats();

        assert!(stats.rejected_steps > 0);
        assert!(stats.jacobian_evaluations < stats.accepted_steps + stats.rejected_steps);
    }

    #[test]
    fn analytic_jacobian_avoids_state_differencing() {
        fn rhs(du: &mut [f64], u: &[f64], _: &(), time: f64) {
            du[0] = -1000.0 * (u[0] - time.cos()) - time.sin();
        }
        type TestRhs = fn(&mut [f64], &[f64], &(), f64);
        let numeric = OdeProblem::new(rhs as TestRhs, vec![1.0], (0.0, 0.1), ());
        let analytic = OdeProblem::new(rhs as TestRhs, vec![1.0], (0.0, 0.1), ()).with_jacobian(
            |jacobian: &mut [f64], _: &[f64], _: &(), _: f64| {
                jacobian[0] = -1000.0;
            },
        );
        let options = SolveOptions {
            absolute_tolerance: 1.0e-7,
            relative_tolerance: 1.0e-7,
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        let numeric = solve(&numeric, Rosenbrock23, &options).unwrap();
        let analytic = solve(&analytic, Rosenbrock23, &options).unwrap();

        assert!((numeric.last_state()[0] - analytic.last_state()[0]).abs() < 1.0e-12);
        assert!(analytic.stats().rhs_evaluations < numeric.stats().rhs_evaluations);
    }

    #[test]
    fn terminating_callback_skips_post_effect_rosenbrock_work() {
        let rhs_calls = Rc::new(Cell::new(0));
        let observed_calls = Rc::clone(&rhs_calls);
        let problem = OdeProblem::new(
            move |du: &mut [f64], u: &[f64], _: &(), _: f64| {
                observed_calls.set(observed_calls.get() + 1);
                du[0] = if u[0] == 12_345.0 { f64::NAN } else { -u[0] };
            },
            vec![1.0],
            (0.0, 1.0),
            (),
        )
        .with_discrete_callback(
            |_, _, time| time > 0.0,
            |state, _, _| {
                state[0] = 12_345.0;
                CallbackAction::Terminate
            },
        );
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.25),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        let solution = solve(&problem, Rosenbrock23, &options).unwrap();

        assert_eq!(solution.last_state()[0], 12_345.0);
        assert_eq!(rhs_calls.get(), solution.stats().rhs_evaluations);
        assert_eq!(solution.stats().accepted_steps, 1);
        assert_eq!(solution.stats().jacobian_evaluations, 1);
    }
}
