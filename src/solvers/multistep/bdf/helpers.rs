use super::{MAX_NEWTON_ITERATIONS, MAX_ORDER, NEWTON_TOLERANCE};
use crate::linear::{DenseLu, LinearError, StateLayout, factorize, solve_factorized};
use crate::{OdeProblem, SolveError, SolveOptions, SolverStats};

pub(super) struct NewtonWorkspace {
    layout: StateLayout,
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
}

impl NewtonWorkspace {
    pub(super) fn new(dimension: usize) -> Self {
        Self {
            layout: StateLayout::for_validated_state(dimension),
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
        }
    }

    pub(super) fn invalidate_factorization(&mut self) {
        self.factorization_ready = false;
    }
}

pub(super) fn reported_error(error: f64, order: usize) -> f64 {
    if error == 0.0 {
        0.0
    } else {
        error.powf(6.0 / (order + 1) as f64)
    }
}

pub(super) fn select_qndf_order(
    order: usize,
    error: f64,
    lower_error: f64,
    upper_error: f64,
) -> usize {
    let current_score = if error > 0.0 {
        error.powf(-1.0 / (order + 1) as f64) / 1.2
    } else {
        10.0
    };
    let lower_score = if order > 1 && lower_error > 0.0 {
        lower_error.powf(-1.0 / order as f64) / 1.3
    } else {
        0.0
    };
    let upper_score = if order < MAX_ORDER && upper_error > 0.0 {
        upper_error.powf(-1.0 / (order + 2) as f64) / 1.4
    } else {
        0.0
    };
    if upper_score > current_score && upper_score >= lower_score {
        order + 1
    } else if lower_score > current_score {
        order - 1
    } else {
        order
    }
}

pub(super) fn fixed_order_threshold(order: usize) -> usize {
    match order {
        1 => 0,
        2 => 2,
        3 => 5,
        4 => 9,
        _ => 14,
    }
}

pub(super) fn reinterpolate_differences(differences: &mut [Vec<f64>], order: usize, ratio: f64) {
    let mut u = [[0.0; MAX_ORDER]; MAX_ORDER];
    let mut r = [[0.0; MAX_ORDER]; MAX_ORDER];
    for column in 0..order {
        u[0][column] = -(column as f64 + 1.0);
        r[0][column] = -(column as f64 + 1.0) * ratio;
        for row in 1..order {
            u[row][column] =
                u[row - 1][column] * (row as f64 - (column as f64 + 1.0)) / (row as f64 + 1.0);
            r[row][column] = r[row - 1][column] * (row as f64 - (column as f64 + 1.0) * ratio)
                / (row as f64 + 1.0);
        }
    }
    let mut ru = [[0.0; MAX_ORDER]; MAX_ORDER];
    for row in 0..order {
        for column in 0..order {
            ru[row][column] = (0..order)
                .map(|middle| r[row][middle] * u[middle][column])
                .sum();
        }
    }
    let old = differences[..order].to_vec();
    for (column, difference) in differences.iter_mut().enumerate().take(order) {
        difference.fill(0.0);
        for row in 0..order {
            for (value, &delta) in difference.iter_mut().zip(&old[row]) {
                *value += delta * ru[row][column];
            }
        }
    }
}

pub(super) fn update_differences(differences: &mut [Vec<f64>], correction: &[f64], order: usize) {
    let (through_order, higher) = differences.split_at_mut(order + 1);
    for ((higher, &dd), &previous) in higher[0]
        .iter_mut()
        .zip(correction)
        .zip(&through_order[order])
    {
        *higher = dd - previous;
    }
    through_order[order].copy_from_slice(correction);
    for index in (0..order).rev() {
        let (lower, upper) = differences.split_at_mut(index + 1);
        for (value, &next) in lower[index].iter_mut().zip(&upper[0]) {
            *value += next;
        }
    }
}

pub(super) fn lagrange_state(
    output: &mut [f64],
    target: f64,
    times: &[f64],
    states: &[Vec<f64>],
    count: usize,
) {
    output.fill(0.0);
    if count == 0 {
        return;
    }
    for j in 0..count {
        let mut weight = 1.0;
        for m in 0..count {
            if m != j {
                weight *= (target - times[m]) / (times[j] - times[m]);
            }
        }
        for (value, &state) in output.iter_mut().zip(&states[j]) {
            *value += weight * state;
        }
    }
}

pub(super) fn fbdf_lte_scale(
    order: usize,
    evaluation_time: f64,
    step: f64,
    times: &[f64],
    coefficients: &[f64],
) -> f64 {
    if times.len() < order + 1 {
        return 1.0 / (order + 1) as f64;
    }
    let mut lte = -1.0 / (order + 1) as f64;
    for j in 2..=order {
        let mut r = 1.0 - j as f64;
        for i in 2..=order + 1 {
            r *= ((evaluation_time - j as f64 * step) - times[i - 1]) / (i as f64 * step);
        }
        lte -= coefficients[j - 1] * r;
    }
    let product = (1..=order + 1)
        .map(|j| j as f64 * step / (evaluation_time - times[j - 1]))
        .product::<f64>();
    lte * product
}

#[allow(clippy::too_many_arguments)]
pub(super) fn newton_solve<F, P>(
    problem: &OdeProblem<F, P>,
    candidate: &mut [f64],
    time: f64,
    derivative_scale: f64,
    forcing: &[f64],
    workspace: &mut NewtonWorkspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: crate::OdeFunction<P>,
{
    workspace.factorization_ready = false;
    for _ in 0..MAX_NEWTON_ITERATIONS {
        stats.nonlinear_iterations += 1;
        evaluate_checked(
            problem,
            &mut workspace.evaluation_derivative,
            candidate,
            time,
            stats,
        )?;
        let mut residual_norm: f64 = 0.0;
        for index in 0..candidate.len() {
            workspace.residual[index] = candidate[index]
                - derivative_scale * workspace.evaluation_derivative[index]
                - forcing[index];
            residual_norm = residual_norm.max(workspace.residual[index].abs());
        }
        if residual_norm <= NEWTON_TOLERANCE * (1.0 + infinity_norm(candidate)) {
            return Ok(());
        }
        if !workspace.factorization_ready {
            build_factorization(problem, candidate, time, derivative_scale, workspace, stats)?;
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
    workspace: &mut NewtonWorkspace,
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
                workspace.matrix[index] =
                    f64::from(row == column) - derivative_scale * workspace.matrix[index];
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
                workspace.matrix[row * dimension + column] =
                    f64::from(row == column) - derivative_scale * derivative;
            }
        }
    }
    stats.jacobian_evaluations += 1;
    stats.linear_factorizations += 1;
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

pub(super) fn rms_scaled<I>(
    values: I,
    candidate: &[f64],
    previous: &[f64],
    options: &SolveOptions,
) -> f64
where
    I: Iterator<Item = f64>,
{
    let mut squared = 0.0;
    for ((defect, &next), &old) in values.zip(candidate).zip(previous) {
        let scale =
            options.absolute_tolerance + options.relative_tolerance * next.abs().max(old.abs());
        squared += (defect / scale).powi(2);
    }
    (squared / candidate.len() as f64).sqrt()
}

pub(super) fn estimate_initial_step(
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
        maximum_step.min(0.01)
    };
    estimate.max(f64::EPSILON).min(maximum_step)
}

pub(super) fn relative_step_change(previous: f64, next: f64) -> f64 {
    (previous - next).abs() / previous.abs().max(next.abs()).max(f64::MIN_POSITIVE)
}

pub(super) fn evaluate_checked<F, P>(
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
