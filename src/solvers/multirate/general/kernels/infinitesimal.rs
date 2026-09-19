//! Knoth--Wolke multirate infinitesimal-step kernel.

use super::super::evaluation::{evaluate_fast, evaluate_slow};
use crate::tableau::MisTableau;
use crate::{SolveError, SolverStats, SplitOdeProblem};

#[allow(clippy::too_many_arguments)]
pub(in super::super) fn mis_step<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    state: &[f64],
    time: f64,
    step: f64,
    candidate: &mut [f64],
    m: usize,
    tableau: &'static MisTableau,
    stats: &mut SolverStats,
) -> Result<Vec<f64>, SolveError>
where
    FE: crate::OdeFunction<P>,
    FI: crate::OdeFunction<P>,
{
    let stage_count = tableau.d().len();
    let dimension = state.len();
    let mut stages = vec![vec![0.0; dimension]; stage_count];
    let mut slow = vec![vec![0.0; dimension]; stage_count];
    stages[0].copy_from_slice(state);
    evaluate_slow(problem, state, time, &mut slow[0], stats)?;
    let mut fast = vec![0.0; dimension];
    let mut midpoint_fast = vec![0.0; dimension];
    let mut midpoint = vec![0.0; dimension];
    for stage in 1..stage_count {
        let mut value = state.to_vec();
        for (j, previous_stage) in stages.iter().take(stage).enumerate() {
            for index in 0..dimension {
                value[index] += tableau.alpha()[stage][j] * (previous_stage[index] - state[index]);
            }
        }
        let mut offset = vec![0.0; dimension];
        for (j, (previous_stage, previous_slow)) in stages.iter().zip(&slow).take(stage).enumerate()
        {
            for index in 0..dimension {
                offset[index] += tableau.gamma()[stage][j] / (tableau.d()[stage] * step)
                    * (previous_stage[index] - state[index])
                    + tableau.beta()[stage][j] / tableau.d()[stage] * previous_slow[index];
            }
        }
        let micro_count = ((m as f64 * tableau.d()[stage]).ceil() as usize).max(1);
        let h = tableau.d()[stage] * step / micro_count as f64;
        let slope = (tableau.c()[stage] - tableau.c_tilde()[stage]) / tableau.d()[stage];
        for micro in 0..micro_count {
            let tau = micro as f64 * h;
            evaluate_fast(
                problem,
                &value,
                time + tableau.c_tilde()[stage] * step + slope * tau,
                &mut fast,
                stats,
            )?;
            for index in 0..dimension {
                midpoint[index] = value[index] + 0.5 * h * (offset[index] + fast[index]);
            }
            evaluate_fast(
                problem,
                &midpoint,
                time + tableau.c_tilde()[stage] * step + slope * (tau + 0.5 * h),
                &mut midpoint_fast,
                stats,
            )?;
            for index in 0..dimension {
                value[index] += h * (offset[index] + midpoint_fast[index]);
            }
        }
        stages[stage] = value;
        evaluate_slow(
            problem,
            &stages[stage],
            time + tableau.c()[stage] * step,
            &mut slow[stage],
            stats,
        )?;
    }
    candidate.copy_from_slice(&stages[stage_count - 1]);
    Ok(stages[stage_count - 1]
        .iter()
        .zip(&stages[stage_count - 2])
        .map(|(high, low)| high - low)
        .collect())
}
