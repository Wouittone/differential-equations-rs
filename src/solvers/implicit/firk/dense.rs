use crate::SolveError;
use crate::solution::{DenseSegment, InterpolationError};

use super::tableau::Tableau;

#[allow(clippy::too_many_arguments)]
pub(super) fn interpolate_segment(
    tableau: &Tableau,
    dimension: usize,
    stages: &[f64],
    start_state: &[f64],
    start_time: f64,
    step: f64,
    time: f64,
    output: &mut [f64],
    weights: &mut [f64],
) -> Result<(), SolveError> {
    let theta = ((time - start_time) / step).clamp(0.0, 1.0);
    tableau.weights_at(theta, weights);
    output.copy_from_slice(start_state);
    for j in 0..tableau.stages {
        for component in 0..dimension {
            output[component] += step * weights[j] * stages[j * dimension + component];
        }
    }
    output
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteDerivative)
}

pub(super) struct CollocationAttemptSegment<'a> {
    pub(super) tableau: &'a Tableau,
    pub(super) dimension: usize,
    pub(super) start_state: &'a [f64],
    pub(super) midpoint_state: &'a [f64],
    pub(super) endpoint_state: &'a [f64],
    pub(super) full_stages: &'a [f64],
    pub(super) first_half_stages: &'a [f64],
    pub(super) second_half_stages: &'a [f64],
    pub(super) start_time: f64,
    pub(super) attempted_time: f64,
    pub(super) bound_time: f64,
    pub(super) adaptive: bool,
}

impl DenseSegment for CollocationAttemptSegment<'_> {
    fn interpolate(&self, time: f64, output: &mut [f64]) -> Result<(), InterpolationError> {
        if output.len() != self.dimension {
            return Err(InterpolationError::DimensionMismatch);
        }
        if time == self.bound_time {
            output.copy_from_slice(self.endpoint_state);
            return Ok(());
        }
        let mut weights = vec![0.0; self.tableau.stages];
        let step = self.attempted_time - self.start_time;
        let result = if self.adaptive {
            let half = 0.5 * step;
            if step.signum() * (time - (self.start_time + half)) <= 0.0 {
                interpolate_segment(
                    self.tableau,
                    self.dimension,
                    self.first_half_stages,
                    self.start_state,
                    self.start_time,
                    half,
                    time,
                    output,
                    &mut weights,
                )
            } else {
                interpolate_segment(
                    self.tableau,
                    self.dimension,
                    self.second_half_stages,
                    self.midpoint_state,
                    self.start_time + half,
                    half,
                    time,
                    output,
                    &mut weights,
                )
            }
        } else {
            interpolate_segment(
                self.tableau,
                self.dimension,
                self.full_stages,
                self.start_state,
                self.start_time,
                step,
                time,
                output,
                &mut weights,
            )
        };
        result.map_err(|_| InterpolationError::NonFiniteResult {
            context: "collocation",
        })
    }
}
