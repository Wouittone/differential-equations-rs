use crate::integrator::{
    KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::linear::{DenseLu, LinearError, StateLayout, factorize, solve_factorized};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

const MAX_NEWTON_ITERATIONS: usize = 12;
const NEWTON_TOLERANCE: f64 = 1.0e-12;

#[derive(Clone, Copy)]
enum ImplicitMethod {
    Euler,
    Midpoint,
    Trapezoid,
}

macro_rules! algorithm {
    ($name:ident, $documentation:literal, $method:ident) => {
        #[doc = $documentation]
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
                    ImplicitKernel::new(ImplicitMethod::$method, problem.initial_state().len()),
                )
            }
        }
    };
}

algorithm!(
    ImplicitEuler,
    "The fixed-step first-order implicit Euler method for stiff ODEs.",
    Euler
);
algorithm!(
    ImplicitMidpoint,
    "The fixed-step symmetric second-order implicit midpoint method.",
    Midpoint
);
algorithm!(
    Trapezoid,
    "The fixed-step second-order implicit trapezoid method.",
    Trapezoid
);

struct Workspace {
    layout: StateLayout,
    current_derivative: Vec<f64>,
    evaluation_state: Vec<f64>,
    base_derivative: Vec<f64>,
    perturbed_state: Vec<f64>,
    perturbed_derivative: Vec<f64>,
    residual: Vec<f64>,
    matrix: Vec<f64>,
    pivots: Vec<usize>,
    factorization: Option<DenseLu>,
    dense_active: bool,
    correction: Vec<f64>,
    factorization_scale: Option<f64>,
}

impl Workspace {
    fn new(dimension: usize) -> Self {
        let layout = StateLayout::new(dimension).expect("solver validates non-empty state");
        Self {
            layout,
            current_derivative: vec![0.0; dimension],
            evaluation_state: vec![0.0; dimension],
            base_derivative: vec![0.0; dimension],
            perturbed_state: vec![0.0; dimension],
            perturbed_derivative: vec![0.0; dimension],
            residual: vec![0.0; dimension],
            matrix: vec![0.0; dimension * dimension],
            pivots: vec![0; dimension],
            factorization: None,
            dense_active: false,
            correction: vec![0.0; dimension],
            factorization_scale: None,
        }
    }
}

struct ImplicitKernel {
    method: ImplicitMethod,
    workspace: Workspace,
}

impl ImplicitKernel {
    fn new(method: ImplicitMethod, dimension: usize) -> Self {
        Self {
            method,
            workspace: Workspace::new(dimension),
        }
    }
}

impl<F, P> StepKernel<F, P> for ImplicitKernel
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::new(false, 1)
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
        for ((candidate, value), derivative) in candidate
            .iter_mut()
            .zip(state)
            .zip(&self.workspace.current_derivative)
        {
            *candidate = value + step * derivative;
        }
        newton_step(
            problem,
            state,
            candidate,
            (time, step),
            self.method,
            &mut self.workspace,
            stats,
        )?;
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
            self.workspace.factorization_scale = None;
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
        self.workspace.factorization_scale = None;
    }
}

fn newton_step<F, P>(
    problem: &OdeProblem<F, P>,
    previous: &[f64],
    candidate: &mut [f64],
    time_and_step: (f64, f64),
    method: ImplicitMethod,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let (time, step) = time_and_step;
    let derivative_scale = match method {
        ImplicitMethod::Euler => step,
        ImplicitMethod::Midpoint | ImplicitMethod::Trapezoid => 0.5 * step,
    };
    let mut refresh_factorization = workspace.factorization_scale != Some(derivative_scale);
    for _ in 0..MAX_NEWTON_ITERATIONS {
        stats.nonlinear_iterations += 1;
        set_evaluation_state(&mut workspace.evaluation_state, previous, candidate, method);
        let evaluation_time = match method {
            ImplicitMethod::Midpoint => time + 0.5 * step,
            ImplicitMethod::Euler | ImplicitMethod::Trapezoid => time + step,
        };
        evaluate_unchecked(
            problem,
            &mut workspace.base_derivative,
            &workspace.evaluation_state,
            evaluation_time,
            stats,
        );
        set_residual_checked(
            &mut workspace.residual,
            previous,
            candidate,
            &workspace.current_derivative,
            &workspace.base_derivative,
            step,
            method,
        )?;
        let residual_norm = infinity_norm(&workspace.residual);
        let state_scale = 1.0 + infinity_norm(candidate);
        if residual_norm <= NEWTON_TOLERANCE * state_scale {
            return Ok(());
        }

        if refresh_factorization {
            build_factorization(problem, evaluation_time, derivative_scale, workspace, stats)?;
        }

        for (correction, residual) in workspace.correction.iter_mut().zip(&workspace.residual) {
            *correction = -*residual;
        }
        if workspace.dense_active {
            let mut correction = workspace
                .layout
                .state_mut(&mut workspace.correction)
                .map_err(map_linear_error)?;
            workspace
                .factorization
                .as_ref()
                .ok_or(SolveError::SingularLinearSystem)?
                .solve(correction.as_mut_slice())
                .map_err(map_linear_error)?;
            workspace.dense_active = false;
        } else {
            solve_factorized(
                &workspace.matrix,
                &workspace.pivots,
                &mut workspace.correction,
                previous.len(),
            );
        }
        stats.linear_solves += 1;

        for (candidate, correction) in candidate.iter_mut().zip(&workspace.correction) {
            *candidate += correction;
        }
        // A factorization reused from the previous step gets one chord-Newton
        // correction. If that is not enough, refresh it at the new state.
        refresh_factorization = true;
    }
    Err(SolveError::NonlinearSolveFailed)
}

fn set_evaluation_state(
    output: &mut [f64],
    previous: &[f64],
    candidate: &[f64],
    method: ImplicitMethod,
) {
    match method {
        ImplicitMethod::Midpoint => {
            for ((output, previous), candidate) in output.iter_mut().zip(previous).zip(candidate) {
                *output = 0.5 * (previous + candidate);
            }
        }
        ImplicitMethod::Euler | ImplicitMethod::Trapezoid => {
            output.copy_from_slice(candidate);
        }
    }
}

fn set_residual_checked(
    residual: &mut [f64],
    previous: &[f64],
    candidate: &[f64],
    previous_derivative: &[f64],
    implicit_derivative: &[f64],
    step: f64,
    method: ImplicitMethod,
) -> Result<(), SolveError> {
    for index in 0..residual.len() {
        if !implicit_derivative[index].is_finite() {
            return Err(SolveError::NonFiniteDerivative);
        }
        let increment = match method {
            ImplicitMethod::Euler | ImplicitMethod::Midpoint => step * implicit_derivative[index],
            ImplicitMethod::Trapezoid => {
                0.5 * step * (previous_derivative[index] + implicit_derivative[index])
            }
        };
        residual[index] = candidate[index] - previous[index] - increment;
        if !residual[index].is_finite() {
            return Err(SolveError::NonFiniteDerivative);
        }
    }
    Ok(())
}

fn build_factorization<F, P>(
    problem: &OdeProblem<F, P>,
    evaluation_time: f64,
    derivative_scale: f64,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let dimension = workspace.evaluation_state.len();
    workspace.factorization_scale = None;
    if problem.evaluate_jacobian(
        &mut workspace.matrix,
        &workspace.evaluation_state,
        evaluation_time,
    ) {
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
            workspace
                .perturbed_state
                .copy_from_slice(&workspace.evaluation_state);
            let perturbation =
                f64::EPSILON.sqrt() * workspace.evaluation_state[column].abs().max(1.0);
            workspace.perturbed_state[column] += perturbation;
            evaluate_unchecked(
                problem,
                &mut workspace.perturbed_derivative,
                &workspace.perturbed_state,
                evaluation_time,
                stats,
            );
            for row in 0..dimension {
                let derivative = (workspace.perturbed_derivative[row]
                    - workspace.base_derivative[row])
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
    if workspace.factorization.is_none() {
        let matrix = workspace
            .layout
            .matrix(&workspace.matrix)
            .map_err(map_linear_error)?;
        workspace.factorization = Some(
            DenseLu::factorize(
                workspace.layout,
                matrix.as_slice(),
                stats.jacobian_evaluations as u64,
            )
            .map_err(map_linear_error)?,
        );
        // Keep the preallocated legacy factors for subsequent refreshes. The
        // checked DenseLu path is exercised for the first solve while refreshes
        // remain allocation-free when a nonlinear step needs a new Jacobian.
        factorize(&mut workspace.matrix, &mut workspace.pivots, dimension)?;
        workspace.dense_active = true;
    } else {
        factorize(&mut workspace.matrix, &mut workspace.pivots, dimension)?;
        workspace.dense_active = false;
    }
    workspace.factorization_scale = Some(derivative_scale);
    Ok(())
}

fn map_linear_error(error: LinearError) -> SolveError {
    match error {
        LinearError::EmptyDimension
        | LinearError::DimensionOverflow { .. }
        | LinearError::LengthMismatch { .. }
        | LinearError::NonFiniteCoefficient
        | LinearError::Singular
        | LinearError::Unfactorized => SolveError::SingularLinearSystem,
    }
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
    evaluate_unchecked(problem, derivative, state, time, stats);
    derivative
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteDerivative)
}

fn infinity_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use crate::{
        CallbackAction, ImplicitEuler, ImplicitMidpoint, OdeProblem, SaveMode, SolveOptions,
        Trapezoid, solve,
    };

    type TestRhs = fn(&mut [f64], &[f64], &(), f64);

    fn stiff_decay() -> OdeProblem<TestRhs, ()> {
        fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
            du[0] = -100.0 * u[0];
        }
        OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
    }

    #[test]
    fn implicit_methods_stabilize_stiff_decay() {
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.05),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        let euler = solve(&stiff_decay(), ImplicitEuler, &options).unwrap();
        let midpoint = solve(&stiff_decay(), ImplicitMidpoint, &options).unwrap();
        let trapezoid = solve(&stiff_decay(), Trapezoid, &options).unwrap();

        assert!(euler.last_state()[0].abs() < 1.0e-12);
        assert!(midpoint.last_state()[0].abs() < 1.0e-7);
        assert!(trapezoid.last_state()[0].abs() < 1.0e-7);
        assert!(euler.stats().linear_solves > 0);
        assert!(euler.stats().jacobian_evaluations < euler.stats().accepted_steps);
        assert!(midpoint.stats().jacobian_evaluations < midpoint.stats().accepted_steps);
        assert!(trapezoid.stats().jacobian_evaluations < trapezoid.stats().accepted_steps);
    }

    #[test]
    fn analytic_jacobians_match_numerical_differentiation() {
        fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
            du[0] = -u[0] * u[0];
        }
        let numeric = OdeProblem::new(rhs as TestRhs, vec![1.0], (0.0, 1.0), ());
        let analytic = OdeProblem::new(rhs as TestRhs, vec![1.0], (0.0, 1.0), ()).with_jacobian(
            |jacobian: &mut [f64], state: &[f64], _: &(), _: f64| {
                jacobian[0] = -2.0 * state[0];
            },
        );
        assert!(!numeric.has_jacobian());
        assert!(analytic.has_jacobian());
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.01),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        let numeric = solve(&numeric, ImplicitMidpoint, &options).unwrap();
        let analytic = solve(&analytic, ImplicitMidpoint, &options).unwrap();

        assert!((numeric.last_state()[0] - analytic.last_state()[0]).abs() < 1.0e-12);
        assert!(analytic.stats().rhs_evaluations < numeric.stats().rhs_evaluations);
    }

    #[test]
    fn terminating_callback_does_not_run_post_effect_implicit_work() {
        let rhs_calls = Rc::new(Cell::new(0));
        let jacobian_calls = Rc::new(Cell::new(0));
        let rhs_at_effect = Rc::new(Cell::new(usize::MAX));
        let jacobian_at_effect = Rc::new(Cell::new(usize::MAX));
        let rhs_counter = Rc::clone(&rhs_calls);
        let jacobian_counter = Rc::clone(&jacobian_calls);
        let problem = OdeProblem::new(
            move |du: &mut [f64], state: &[f64], _: &(), _: f64| {
                rhs_counter.set(rhs_counter.get() + 1);
                du[0] = if state[0] == 42.0 { f64::NAN } else { state[0] };
            },
            vec![1.0],
            (0.0, 1.0),
            (),
        )
        .with_jacobian(move |jacobian: &mut [f64], _: &[f64], _: &(), _: f64| {
            jacobian_counter.set(jacobian_counter.get() + 1);
            jacobian[0] = 1.0;
        })
        .with_continuous_callback(|state, _, _| state[0] - 1.2, {
            let rhs_calls = Rc::clone(&rhs_calls);
            let jacobian_calls = Rc::clone(&jacobian_calls);
            let rhs_at_effect = Rc::clone(&rhs_at_effect);
            let jacobian_at_effect = Rc::clone(&jacobian_at_effect);
            move |state, _, _| {
                rhs_at_effect.set(rhs_calls.get());
                jacobian_at_effect.set(jacobian_calls.get());
                state[0] = 42.0;
                CallbackAction::Terminate
            }
        });
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.25),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        let solution = solve(&problem, ImplicitEuler, &options).unwrap();

        assert_eq!(solution.last_state(), &[42.0]);
        assert_eq!(rhs_calls.get(), rhs_at_effect.get());
        assert_eq!(jacobian_calls.get(), jacobian_at_effect.get());
        assert!(jacobian_calls.get() > 0);
    }
}
