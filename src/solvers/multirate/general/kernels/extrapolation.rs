//! Extrapolation and Adams--Bashforth multirate steps.

use super::super::evaluation::{evaluate_fast, evaluate_slow};
use super::super::method::MultirateSequence;
use crate::solvers::multistep::{adams_bashforth, map_tableau_access_error};
use crate::{SolveError, SolverStats, SplitOdeProblem};

#[allow(clippy::too_many_arguments)]
pub(in super::super) fn mreef_step<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    state: &[f64],
    time: f64,
    step: f64,
    candidate: &mut [f64],
    m: usize,
    order: usize,
    sequence: MultirateSequence,
    stats: &mut SolverStats,
) -> Result<Vec<f64>, SolveError>
where
    FE: crate::OdeFunction<P>,
    FI: crate::OdeFunction<P>,
{
    let dimension = state.len();
    let ns: Vec<usize> = (1..=order)
        .map(|index| match sequence {
            MultirateSequence::Harmonic => index,
            MultirateSequence::Romberg => 1usize << (index - 1),
        })
        .collect();
    let mut table = Vec::with_capacity(order);
    for &macro_count in &ns {
        let mut value = state.to_vec();
        let macro_step = step / macro_count as f64;
        let micro_step = macro_step / m as f64;
        let mut slow = vec![0.0; dimension];
        let mut fast = vec![0.0; dimension];
        for macro_index in 0..macro_count {
            let macro_time = time + macro_index as f64 * macro_step;
            evaluate_slow(problem, &value, macro_time, &mut slow, stats)?;
            for micro_index in 0..m {
                let micro_time = macro_time + micro_index as f64 * micro_step;
                evaluate_fast(problem, &value, micro_time, &mut fast, stats)?;
                for ((value, fast), slow) in value.iter_mut().zip(&fast).zip(&slow) {
                    *value += micro_step * (fast + slow);
                }
            }
        }
        table.push(value);
    }
    for k in 1..order {
        for j in (k..order).rev() {
            let denominator = ns[j] as f64 / ns[j - k] as f64 - 1.0;
            let (left, right) = table.split_at_mut(j);
            let previous = &left[j - 1];
            let current = &mut right[0];
            for (current, previous) in current.iter_mut().zip(previous) {
                *current += (*current - previous) / denominator;
            }
        }
    }
    candidate.copy_from_slice(&table[order - 1]);
    Ok(candidate
        .iter()
        .zip(&table[order - 2])
        .map(|(high, low)| high - low)
        .collect())
}

#[allow(clippy::too_many_arguments)]
pub(in super::super) fn mrab_step<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    state: &[f64],
    time: f64,
    step: f64,
    candidate: &mut [f64],
    m: usize,
    order: usize,
    stats: &mut SolverStats,
) -> Result<Vec<f64>, SolveError>
where
    FE: crate::OdeFunction<P>,
    FI: crate::OdeFunction<P>,
{
    let dimension = state.len();
    let h = step / m as f64;
    let mut slow = vec![0.0; dimension];
    let mut fast = vec![0.0; dimension];
    evaluate_slow(problem, state, time, &mut slow, stats)?;
    let mut value = state.to_vec();
    // Upstream deliberately restarts AB-min(l,k) inside every macro step.
    let mut history: Vec<Vec<f64>> = Vec::with_capacity(order);
    for micro in 0..m {
        evaluate_fast(problem, &value, time + micro as f64 * h, &mut fast, stats)?;
        let combined: Vec<f64> = fast.iter().zip(&slow).map(|(a, b)| a + b).collect();
        history.insert(0, combined);
        history.truncate(order);
        let weights = &adams_bashforth(history.len().min(order))
            .map_err(map_tableau_access_error)?
            .beta()[1..];
        for index in 0..dimension {
            value[index] += h * weights
                .iter()
                .zip(history.iter())
                .map(|(weight, derivative)| weight * derivative[index])
                .sum::<f64>();
        }
    }
    candidate.copy_from_slice(&value);
    if order == 1 {
        return Ok(vec![0.0; dimension]);
    }
    let mut error = vec![0.0; dimension];
    if history.len() >= order {
        let high = &adams_bashforth(order)
            .map_err(map_tableau_access_error)?
            .beta()[1..];
        let low = &adams_bashforth(order - 1)
            .map_err(map_tableau_access_error)?
            .beta()[1..];
        for index in 0..dimension {
            error[index] = h
                * (high
                    .iter()
                    .zip(history.iter())
                    .map(|(weight, derivative)| weight * derivative[index])
                    .sum::<f64>()
                    - low
                        .iter()
                        .zip(history.iter())
                        .map(|(weight, derivative)| weight * derivative[index])
                        .sum::<f64>());
        }
    }
    Ok(error)
}
