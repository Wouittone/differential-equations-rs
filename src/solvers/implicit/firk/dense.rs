use crate::SolveError;

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
