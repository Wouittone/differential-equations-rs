use crate::{OdeAlgorithm, OdeProblem, SaveMode, Solution, SolveError, SolveOptions, SolverStats};

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
    candidate: Vec<f64>,
    candidate_derivative: Vec<f64>,
    k1: Vec<f64>,
    k2: Vec<f64>,
    k3: Vec<f64>,
    right_hand_side: Vec<f64>,
    jacobian: Vec<f64>,
    factorization: Vec<f64>,
    pivots: Vec<usize>,
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
            candidate: vec![0.0; dimension],
            candidate_derivative: vec![0.0; dimension],
            k1: vec![0.0; dimension],
            k2: vec![0.0; dimension],
            k3: vec![0.0; dimension],
            right_hand_side: vec![0.0; dimension],
            jacobian: vec![0.0; dimension * dimension],
            factorization: vec![0.0; dimension * dimension],
            pivots: vec![0; dimension],
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
        integrate(problem, options)
    }
}

fn integrate<F, P>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
) -> Result<Solution, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let dimension = problem.initial_state().len();
    let (start, end) = problem.time_span();
    let direction = (end - start).signum();
    let interval = (end - start).abs();
    let maximum_step = options.max_step.min(interval);
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

    let initial_step = match options.initial_step {
        Some(step) => step.min(maximum_step),
        None if !options.adaptive => return Err(SolveError::InitialStepRequired),
        None => estimate_initial_step(&state, &workspace.current_derivative, options, maximum_step),
    };
    let mut step = direction * initial_step;
    let mut time = start;
    let mut times = vec![start];
    let mut values = Vec::with_capacity(2 * dimension);
    values.extend_from_slice(&state);
    let mut attempts = 0;
    let mut previous_rejected = false;

    while direction * (end - time) > 0.0 {
        if attempts == options.max_steps {
            return Err(SolveError::MaxStepsExceeded);
        }
        attempts += 1;
        if direction * (time + step - end) > 0.0 {
            step = end - time;
        }
        if time + step == time {
            return Err(SolveError::StepSizeUnderflow);
        }

        let error = perform_step(
            problem,
            &state,
            time,
            step,
            options,
            &mut workspace,
            &mut stats,
        )?;
        if error <= 1.0 {
            time += step;
            if direction * (end - time) <= 0.0 {
                time = end;
            }
            std::mem::swap(&mut state, &mut workspace.candidate);
            if options.adaptive {
                std::mem::swap(
                    &mut workspace.current_derivative,
                    &mut workspace.candidate_derivative,
                );
            } else {
                evaluate(
                    problem,
                    &mut workspace.current_derivative,
                    &state,
                    time,
                    &mut stats,
                )?;
            }
            stats.accepted_steps += 1;

            if options.save == SaveMode::EveryStep || time == end {
                times.push(time);
                values.extend_from_slice(&state);
            }
            if options.adaptive {
                let mut factor = step_factor(error);
                if previous_rejected {
                    factor = factor.min(1.0);
                }
                step = direction * (step.abs() * factor).min(maximum_step);
            }
            previous_rejected = false;
        } else {
            stats.rejected_steps += 1;
            step *= step_factor(error).min(1.0);
            previous_rejected = true;
        }
    }

    Ok(Solution::new(times, values, dimension, stats))
}

fn perform_step<F, P>(
    problem: &OdeProblem<F, P>,
    state: &[f64],
    time: f64,
    step: f64,
    options: &SolveOptions,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<f64, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let dimension = state.len();
    numerical_differentiation(problem, state, time, workspace, stats)?;
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
    stats.jacobian_evaluations += 1;

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
        workspace.candidate[index] = value + step * workspace.k2[index];
    }
    stats.linear_solves += 1;

    if !options.adaptive {
        return Ok(0.0);
    }
    evaluate(
        problem,
        &mut workspace.candidate_derivative,
        &workspace.candidate,
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
            + options.relative_tolerance * value.abs().max(workspace.candidate[index].abs());
        squared_norm += (local_error / scale).powi(2);
    }
    Ok((squared_norm / dimension as f64).sqrt())
}

fn numerical_differentiation<F, P>(
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
    for column in 0..dimension {
        workspace.perturbed_state.copy_from_slice(state);
        let perturbation = f64::EPSILON.sqrt() * state[column].abs().max(1.0);
        workspace.perturbed_state[column] += perturbation;
        evaluate(
            problem,
            &mut workspace.perturbed_derivative,
            &workspace.perturbed_state,
            time,
            stats,
        )?;
        for row in 0..dimension {
            workspace.jacobian[row * dimension + column] = (workspace.perturbed_derivative[row]
                - workspace.current_derivative[row])
                / perturbation;
        }
    }

    let time_perturbation = f64::EPSILON.sqrt() * time.abs().max(1.0);
    evaluate(
        problem,
        &mut workspace.perturbed_derivative,
        state,
        time + time_perturbation,
        stats,
    )?;
    for index in 0..dimension {
        workspace.time_derivative[index] = (workspace.perturbed_derivative[index]
            - workspace.current_derivative[index])
            / time_perturbation;
    }
    Ok(())
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

fn step_factor(error: f64) -> f64 {
    if error == 0.0 {
        MAX_FACTOR
    } else if error.is_finite() {
        (SAFETY * error.powf(-1.0 / 3.0)).clamp(MIN_FACTOR, MAX_FACTOR)
    } else {
        MIN_FACTOR
    }
}

fn factorize(matrix: &mut [f64], pivots: &mut [usize], dimension: usize) -> Result<(), SolveError> {
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
        pivots[pivot_column] = pivot_row;
        if pivot_row != pivot_column {
            for column in 0..dimension {
                matrix.swap(
                    pivot_column * dimension + column,
                    pivot_row * dimension + column,
                );
            }
        }
        let pivot = matrix[pivot_column * dimension + pivot_column];
        for row in (pivot_column + 1)..dimension {
            let factor = matrix[row * dimension + pivot_column] / pivot;
            matrix[row * dimension + pivot_column] = factor;
            for column in (pivot_column + 1)..dimension {
                matrix[row * dimension + column] -=
                    factor * matrix[pivot_column * dimension + column];
            }
        }
    }
    Ok(())
}

fn solve_factorized(
    factorization: &[f64],
    pivots: &[usize],
    right_hand_side: &mut [f64],
    dimension: usize,
) {
    for (row, &pivot) in pivots.iter().enumerate() {
        if pivot != row {
            right_hand_side.swap(row, pivot);
        }
    }
    for row in 0..dimension {
        for column in 0..row {
            right_hand_side[row] -=
                factorization[row * dimension + column] * right_hand_side[column];
        }
    }
    for row in (0..dimension).rev() {
        for column in (row + 1)..dimension {
            right_hand_side[row] -=
                factorization[row * dimension + column] * right_hand_side[column];
        }
        right_hand_side[row] /= factorization[row * dimension + row];
    }
}

#[cfg(test)]
mod tests {
    use crate::{OdeProblem, Rosenbrock23, SaveMode, SolveOptions, solve};

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
}
