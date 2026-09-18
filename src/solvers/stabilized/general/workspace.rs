use super::family::{SPECTRAL_ITERATIONS, SPECTRAL_SAFETY, StabilizedFamily};
use crate::{OdeProblem, SolveError, SolverStats};

pub(super) struct StabilizedKernel {
    pub(super) family: StabilizedFamily,
    pub(super) first_derivative: Vec<f64>,
    pub(super) last_derivative: Vec<f64>,
    pub(super) derivative: Vec<f64>,
    pub(super) previous_two: Vec<f64>,
    pub(super) previous_one: Vec<f64>,
    pub(super) next_stage: Vec<f64>,
    pub(super) eigenvector: Vec<f64>,
    pub(super) perturbed_state: Vec<f64>,
    pub(super) perturbed_derivative: Vec<f64>,
    pub(super) previous_accepted_state: Option<Vec<f64>>,
    pub(super) previous_accepted_step: Option<f64>,
}

impl StabilizedKernel {
    pub(super) fn new(family: StabilizedFamily, dimension: usize) -> Self {
        Self {
            family,
            first_derivative: vec![0.0; dimension],
            last_derivative: vec![0.0; dimension],
            derivative: vec![0.0; dimension],
            previous_two: vec![0.0; dimension],
            previous_one: vec![0.0; dimension],
            next_stage: vec![0.0; dimension],
            eigenvector: vec![0.0; dimension],
            perturbed_state: vec![0.0; dimension],
            perturbed_derivative: vec![0.0; dimension],
            previous_accepted_state: None,
            previous_accepted_step: None,
        }
    }

    pub(super) fn evaluate<F, P>(
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
        finite(derivative)
    }

    pub(super) fn spectral_radius<F, P>(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<f64, SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        if vector_norm(&self.eigenvector) <= f64::MIN_POSITIVE {
            self.eigenvector.copy_from_slice(&self.first_derivative);
            if vector_norm(&self.eigenvector) <= f64::MIN_POSITIVE {
                self.eigenvector.fill(1.0);
            }
        }

        let perturbation = f64::EPSILON.sqrt() * vector_norm(state).max(1.0);
        let mut previous_radius = 0.0;
        let mut radius = 0.0;
        for iteration in 0..SPECTRAL_ITERATIONS {
            let direction_norm = vector_norm(&self.eigenvector);
            let scale = perturbation / direction_norm.max(f64::MIN_POSITIVE);
            for ((perturbed, value), direction) in self
                .perturbed_state
                .iter_mut()
                .zip(state)
                .zip(&self.eigenvector)
            {
                *perturbed = value + scale * direction;
            }
            Self::evaluate(
                problem,
                &mut self.perturbed_derivative,
                &self.perturbed_state,
                time,
                stats,
            )?;
            for ((direction, perturbed), base) in self
                .eigenvector
                .iter_mut()
                .zip(&self.perturbed_derivative)
                .zip(&self.first_derivative)
            {
                *direction = perturbed - base;
            }
            radius = vector_norm(&self.eigenvector) / perturbation;
            if radius == 0.0 {
                break;
            }
            if iteration > 0 && (radius - previous_radius).abs() <= 0.01 * radius.max(1.0) {
                break;
            }
            previous_radius = radius;
        }
        Ok(SPECTRAL_SAFETY * radius)
    }
}

fn vector_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

pub(super) fn finite(values: &[f64]) -> Result<(), SolveError> {
    values
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteDerivative)
}
