use crate::{OdeAlgorithm, OdeProblem, SaveMode, Solution, SolveError, SolveOptions, SolverStats};

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
                integrate(problem, options, ImplicitMethod::$method)
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
    current_derivative: Vec<f64>,
    candidate: Vec<f64>,
    evaluation_state: Vec<f64>,
    base_derivative: Vec<f64>,
    perturbed_state: Vec<f64>,
    perturbed_derivative: Vec<f64>,
    residual: Vec<f64>,
    matrix: Vec<f64>,
    correction: Vec<f64>,
}

impl Workspace {
    fn new(dimension: usize) -> Self {
        Self {
            current_derivative: vec![0.0; dimension],
            candidate: vec![0.0; dimension],
            evaluation_state: vec![0.0; dimension],
            base_derivative: vec![0.0; dimension],
            perturbed_state: vec![0.0; dimension],
            perturbed_derivative: vec![0.0; dimension],
            residual: vec![0.0; dimension],
            matrix: vec![0.0; dimension * dimension],
            correction: vec![0.0; dimension],
        }
    }
}

fn integrate<F, P>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
    method: ImplicitMethod,
) -> Result<Solution, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    if options.adaptive {
        return Err(SolveError::AdaptiveStepUnsupported);
    }
    let fixed_step = options
        .initial_step
        .ok_or(SolveError::InitialStepRequired)?;
    let dimension = problem.initial_state().len();
    let (start, end) = problem.time_span();
    let direction = (end - start).signum();
    let maximum_step = options.max_step.min(fixed_step);
    let mut state = problem.initial_state().to_vec();
    let mut workspace = Workspace::new(dimension);
    let mut stats = SolverStats::default();
    evaluate(
        problem,
        &mut workspace.current_derivative,
        &state,
        start,
        &mut stats,
    )?;

    let mut time = start;
    let mut times = vec![start];
    let mut values = Vec::with_capacity(2 * dimension);
    values.extend_from_slice(&state);
    let mut steps = 0;

    while direction * (end - time) > 0.0 {
        if steps == options.max_steps {
            return Err(SolveError::MaxStepsExceeded);
        }
        steps += 1;
        let step = direction * maximum_step.min((end - time).abs());
        if time + step == time {
            return Err(SolveError::StepSizeUnderflow);
        }

        for ((candidate, value), derivative) in workspace
            .candidate
            .iter_mut()
            .zip(&state)
            .zip(&workspace.current_derivative)
        {
            *candidate = value + step * derivative;
        }
        newton_step(
            problem,
            &state,
            time,
            step,
            method,
            &mut workspace,
            &mut stats,
        )?;

        time += step;
        if direction * (end - time) <= 0.0 {
            time = end;
        }
        std::mem::swap(&mut state, &mut workspace.candidate);
        evaluate(
            problem,
            &mut workspace.current_derivative,
            &state,
            time,
            &mut stats,
        )?;
        stats.accepted_steps += 1;

        if options.save == SaveMode::EveryStep || time == end {
            times.push(time);
            values.extend_from_slice(&state);
        }
    }

    Ok(Solution::new(times, values, dimension, stats))
}

fn newton_step<F, P>(
    problem: &OdeProblem<F, P>,
    previous: &[f64],
    time: f64,
    step: f64,
    method: ImplicitMethod,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let dimension = previous.len();
    for _ in 0..MAX_NEWTON_ITERATIONS {
        stats.nonlinear_iterations += 1;
        set_evaluation_state(
            &mut workspace.evaluation_state,
            previous,
            &workspace.candidate,
            method,
        );
        let evaluation_time = match method {
            ImplicitMethod::Midpoint => time + 0.5 * step,
            ImplicitMethod::Euler | ImplicitMethod::Trapezoid => time + step,
        };
        evaluate(
            problem,
            &mut workspace.base_derivative,
            &workspace.evaluation_state,
            evaluation_time,
            stats,
        )?;
        set_residual(
            &mut workspace.residual,
            previous,
            &workspace.candidate,
            &workspace.current_derivative,
            &workspace.base_derivative,
            step,
            method,
        );
        let residual_norm = infinity_norm(&workspace.residual);
        let state_scale = 1.0 + infinity_norm(&workspace.candidate);
        if residual_norm <= NEWTON_TOLERANCE * state_scale {
            return Ok(());
        }

        for column in 0..dimension {
            workspace
                .perturbed_state
                .copy_from_slice(&workspace.candidate);
            let perturbation = f64::EPSILON.sqrt() * workspace.candidate[column].abs().max(1.0);
            workspace.perturbed_state[column] += perturbation;
            set_evaluation_state(
                &mut workspace.evaluation_state,
                previous,
                &workspace.perturbed_state,
                method,
            );
            evaluate(
                problem,
                &mut workspace.perturbed_derivative,
                &workspace.evaluation_state,
                evaluation_time,
                stats,
            )?;

            let derivative_factor = match method {
                ImplicitMethod::Euler | ImplicitMethod::Midpoint => step,
                ImplicitMethod::Trapezoid => 0.5 * step,
            };
            for row in 0..dimension {
                let derivative = (workspace.perturbed_derivative[row]
                    - workspace.base_derivative[row])
                    / perturbation;
                workspace.matrix[row * dimension + column] =
                    f64::from(row == column) - derivative_factor * derivative;
            }
        }
        stats.jacobian_evaluations += 1;

        for (correction, residual) in workspace.correction.iter_mut().zip(&workspace.residual) {
            *correction = -*residual;
        }
        solve_linear_system(&mut workspace.matrix, &mut workspace.correction, dimension)?;
        stats.linear_solves += 1;

        for (candidate, correction) in workspace.candidate.iter_mut().zip(&workspace.correction) {
            *candidate += correction;
        }
        if infinity_norm(&workspace.correction) <= NEWTON_TOLERANCE * state_scale {
            return Ok(());
        }
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

fn set_residual(
    residual: &mut [f64],
    previous: &[f64],
    candidate: &[f64],
    previous_derivative: &[f64],
    implicit_derivative: &[f64],
    step: f64,
    method: ImplicitMethod,
) {
    for index in 0..residual.len() {
        let increment = match method {
            ImplicitMethod::Euler | ImplicitMethod::Midpoint => step * implicit_derivative[index],
            ImplicitMethod::Trapezoid => {
                0.5 * step * (previous_derivative[index] + implicit_derivative[index])
            }
        };
        residual[index] = candidate[index] - previous[index] - increment;
    }
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

fn solve_linear_system(
    matrix: &mut [f64],
    right_hand_side: &mut [f64],
    dimension: usize,
) -> Result<(), SolveError> {
    for pivot_column in 0..dimension {
        let mut pivot_row = pivot_column;
        let mut pivot_magnitude = matrix[pivot_column * dimension + pivot_column].abs();
        for row in (pivot_column + 1)..dimension {
            let magnitude = matrix[row * dimension + pivot_column].abs();
            if magnitude > pivot_magnitude {
                pivot_magnitude = magnitude;
                pivot_row = row;
            }
        }
        if pivot_magnitude <= f64::EPSILON {
            return Err(SolveError::SingularLinearSystem);
        }
        if pivot_row != pivot_column {
            for column in 0..dimension {
                matrix.swap(
                    pivot_column * dimension + column,
                    pivot_row * dimension + column,
                );
            }
            right_hand_side.swap(pivot_column, pivot_row);
        }

        let pivot = matrix[pivot_column * dimension + pivot_column];
        for row in (pivot_column + 1)..dimension {
            let factor = matrix[row * dimension + pivot_column] / pivot;
            matrix[row * dimension + pivot_column] = 0.0;
            for column in (pivot_column + 1)..dimension {
                matrix[row * dimension + column] -=
                    factor * matrix[pivot_column * dimension + column];
            }
            right_hand_side[row] -= factor * right_hand_side[pivot_column];
        }
    }

    for row in (0..dimension).rev() {
        let mut value = right_hand_side[row];
        for column in (row + 1)..dimension {
            value -= matrix[row * dimension + column] * right_hand_side[column];
        }
        right_hand_side[row] = value / matrix[row * dimension + row];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{
        ImplicitEuler, ImplicitMidpoint, OdeProblem, SaveMode, SolveOptions, Trapezoid, solve,
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
    }

    #[test]
    fn pivoted_linear_solver_handles_row_exchange() {
        let mut matrix = vec![0.0, 2.0, 1.0, 1.0];
        let mut rhs = vec![4.0, 3.0];

        super::solve_linear_system(&mut matrix, &mut rhs, 2).unwrap();

        assert!((rhs[0] - 1.0).abs() < 1.0e-14);
        assert!((rhs[1] - 2.0).abs() < 1.0e-14);
    }
}
