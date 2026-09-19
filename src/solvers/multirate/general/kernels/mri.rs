//! MRI-GARK explicit and implicit slow-stage kernels.

use super::super::evaluation::{evaluate_fast, evaluate_slow, finite};
use crate::linear::{factorize, solve_factorized};
use crate::tableau::MriTableau;
use crate::{SolveError, SolverStats, SplitOdeProblem};

const MAX_NEWTON_ITERATIONS: usize = 12;
const NEWTON_TOLERANCE: f64 = 1.0e-11;

#[allow(clippy::too_many_arguments)]
pub(in super::super) fn mri_step<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    state: &[f64],
    time: f64,
    step: f64,
    candidate: &mut [f64],
    m: usize,
    tableau: &MriTableau,
    stats: &mut SolverStats,
) -> Result<Vec<f64>, SolveError>
where
    FE: crate::OdeFunction<P>,
    FI: crate::OdeFunction<P>,
{
    let stages_count = tableau.dc().len();
    let dimension = state.len();
    let mut stages = vec![vec![0.0; dimension]; stages_count + 1];
    let mut slow = vec![vec![0.0; dimension]; stages_count];
    stages[0].copy_from_slice(state);
    let mut c_previous = 0.0;
    for stage in 0..stages_count {
        evaluate_slow(
            problem,
            &stages[stage],
            time + c_previous * step,
            &mut slow[stage],
            stats,
        )?;
        let gamma = tableau.gamma()[stage];
        if gamma == 0.0 {
            stages[stage + 1] = mri_substage(
                problem,
                &stages[stage],
                &slow,
                time,
                step,
                c_previous,
                tableau.dc()[stage],
                &tableau.w0()[stage],
                &tableau.w1()[stage],
                m,
                tableau.inner_order(),
                stats,
            )?;
        } else {
            let mut base = stages[stage].clone();
            for (j, slow_stage) in slow.iter().enumerate().take(stage + 1) {
                let weight = tableau.w0()[stage][j] + 0.5 * tableau.w1()[stage][j];
                for index in 0..dimension {
                    base[index] += step * weight * slow_stage[index];
                }
            }
            let target_time = time + (c_previous + tableau.dc()[stage]) * step;
            stages[stage + 1] = implicit_slow_endpoint(
                problem,
                &base,
                &stages[stage],
                target_time,
                gamma * step,
                stats,
            )?;
        }
        c_previous += tableau.dc()[stage];
    }
    candidate.copy_from_slice(&stages[stages_count]);
    let comparison = if let Some(weights0) = tableau.embedded0() {
        mri_substage(
            problem,
            &stages[stages_count - 1],
            &slow,
            time,
            step,
            c_previous - tableau.dc()[stages_count - 1],
            tableau.dc()[stages_count - 1],
            weights0,
            tableau.embedded1().ok_or(SolveError::InvalidTableau)?,
            m,
            tableau.inner_order(),
            stats,
        )?
    } else {
        stages[stages_count - 1].clone()
    };
    Ok(candidate
        .iter()
        .zip(comparison)
        .map(|(high, low)| high - low)
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn mri_substage<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    start: &[f64],
    slow: &[Vec<f64>],
    macro_time: f64,
    macro_step: f64,
    c_previous: f64,
    dc: f64,
    w0: &[f64],
    w1: &[f64],
    m: usize,
    inner_order: usize,
    stats: &mut SolverStats,
) -> Result<Vec<f64>, SolveError>
where
    FE: crate::OdeFunction<P>,
{
    let dimension = start.len();
    if dc == 0.0 {
        let mut value = start.to_vec();
        for (stage, derivative) in slow.iter().enumerate() {
            let weight =
                w0.get(stage).copied().unwrap_or(0.0) + 0.5 * w1.get(stage).copied().unwrap_or(0.0);
            for index in 0..dimension {
                value[index] += macro_step * weight * derivative[index];
            }
        }
        return Ok(value);
    }
    let h = 1.0 / m as f64;
    let mut value = start.to_vec();
    let mut k1 = vec![0.0; dimension];
    let mut k2 = vec![0.0; dimension];
    let mut k3 = vec![0.0; dimension];
    let mut k4 = vec![0.0; dimension];
    let mut temporary = vec![0.0; dimension];
    for micro in 0..m {
        let tau = micro as f64 * h;
        mri_rate(
            problem, &value, slow, macro_time, macro_step, c_previous, dc, w0, w1, tau, &mut k1,
            stats,
        )?;
        for index in 0..dimension {
            temporary[index] = value[index] + 0.5 * h * k1[index];
        }
        mri_rate(
            problem,
            &temporary,
            slow,
            macro_time,
            macro_step,
            c_previous,
            dc,
            w0,
            w1,
            tau + 0.5 * h,
            &mut k2,
            stats,
        )?;
        match inner_order {
            2 => {
                for index in 0..dimension {
                    value[index] += h * k2[index];
                }
            }
            3 => {
                for index in 0..dimension {
                    temporary[index] = value[index] - h * k1[index] + 2.0 * h * k2[index];
                }
                mri_rate(
                    problem,
                    &temporary,
                    slow,
                    macro_time,
                    macro_step,
                    c_previous,
                    dc,
                    w0,
                    w1,
                    tau + h,
                    &mut k3,
                    stats,
                )?;
                for index in 0..dimension {
                    value[index] += h * (k1[index] + 4.0 * k2[index] + k3[index]) / 6.0;
                }
            }
            _ => {
                for index in 0..dimension {
                    temporary[index] = value[index] + 0.5 * h * k2[index];
                }
                mri_rate(
                    problem,
                    &temporary,
                    slow,
                    macro_time,
                    macro_step,
                    c_previous,
                    dc,
                    w0,
                    w1,
                    tau + 0.5 * h,
                    &mut k3,
                    stats,
                )?;
                for index in 0..dimension {
                    temporary[index] = value[index] + h * k3[index];
                }
                mri_rate(
                    problem,
                    &temporary,
                    slow,
                    macro_time,
                    macro_step,
                    c_previous,
                    dc,
                    w0,
                    w1,
                    tau + h,
                    &mut k4,
                    stats,
                )?;
                for index in 0..dimension {
                    value[index] +=
                        h * (k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index]) / 6.0;
                }
            }
        }
    }
    Ok(value)
}

#[allow(clippy::too_many_arguments)]
fn mri_rate<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    state: &[f64],
    slow: &[Vec<f64>],
    macro_time: f64,
    macro_step: f64,
    c_previous: f64,
    dc: f64,
    w0: &[f64],
    w1: &[f64],
    tau: f64,
    output: &mut [f64],
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    FE: crate::OdeFunction<P>,
{
    evaluate_fast(
        problem,
        state,
        macro_time + (c_previous + tau * dc) * macro_step,
        output,
        stats,
    )?;
    for value in output.iter_mut() {
        *value *= macro_step * dc;
    }
    for (stage, derivative) in slow.iter().enumerate() {
        let weight =
            w0.get(stage).copied().unwrap_or(0.0) + tau * w1.get(stage).copied().unwrap_or(0.0);
        for (value, derivative) in output.iter_mut().zip(derivative) {
            *value += macro_step * weight * derivative;
        }
    }
    finite(output)
}

fn implicit_slow_endpoint<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    base: &[f64],
    predictor: &[f64],
    time: f64,
    scale: f64,
    stats: &mut SolverStats,
) -> Result<Vec<f64>, SolveError>
where
    FI: crate::OdeFunction<P>,
{
    let dimension = base.len();
    let mut value = predictor.to_vec();
    let mut derivative = vec![0.0; dimension];
    let mut perturbed_derivative = vec![0.0; dimension];
    let mut perturbed = value.clone();
    let mut residual = vec![0.0; dimension];
    let mut correction = vec![0.0; dimension];
    let mut matrix = vec![0.0; dimension * dimension];
    let mut pivots = vec![0usize; dimension];
    for _ in 0..MAX_NEWTON_ITERATIONS {
        evaluate_slow(problem, &value, time, &mut derivative, stats)?;
        let mut norm = 0.0_f64;
        for index in 0..dimension {
            residual[index] = value[index] - base[index] - scale * derivative[index];
            norm = norm.max(residual[index].abs());
        }
        if norm <= NEWTON_TOLERANCE * (1.0 + value.iter().fold(0.0_f64, |n, x| n.max(x.abs()))) {
            return Ok(value);
        }
        if !problem.evaluate_implicit_jacobian(&mut matrix, &value, time) {
            for column in 0..dimension {
                perturbed.copy_from_slice(&value);
                let delta = f64::EPSILON.sqrt() * (1.0 + value[column].abs());
                perturbed[column] += delta;
                evaluate_slow(problem, &perturbed, time, &mut perturbed_derivative, stats)?;
                for row in 0..dimension {
                    matrix[row * dimension + column] =
                        (perturbed_derivative[row] - derivative[row]) / delta;
                }
            }
        }
        stats.jacobian_evaluations += 1;
        for row in 0..dimension {
            for column in 0..dimension {
                matrix[row * dimension + column] *= -scale;
            }
            matrix[row * dimension + row] += 1.0;
            correction[row] = -residual[row];
        }
        factorize(&mut matrix, &mut pivots, dimension)?;
        stats.linear_factorizations += 1;
        solve_factorized(&matrix, &pivots, &mut correction, dimension);
        stats.linear_solves += 1;
        stats.nonlinear_iterations += 1;
        for (value, correction) in value.iter_mut().zip(&correction) {
            *value += correction;
        }
        finite(&value)?;
    }
    Err(SolveError::NonlinearSolveFailed)
}
