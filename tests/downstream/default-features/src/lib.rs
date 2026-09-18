//! Downstream smoke coverage for the default feature set.

use differential_equations::solvers::explicit::Tsit5;
use differential_equations::{OdeProblem, SaveMode, SolveOptions, solve_ensemble_parallel};

/// Solves independent exponential-decay problems through the default parallel API.
pub fn solve_parallel(initial_values: impl IntoIterator<Item = f64>) -> Vec<f64> {
    let options = SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(0.01)
        .with_save(SaveMode::Endpoints);
    solve_ensemble_parallel(
        initial_values,
        |initial| {
            OdeProblem::new(
                |derivative: &mut [f64], state: &[f64], _: &(), _: f64| {
                    derivative[0] = -state[0];
                },
                [initial],
                (0.0, 0.25),
                (),
            )
        },
        Tsit5,
        &options,
    )
    .into_iter()
    .map(|outcome| outcome.result.expect("downstream ensemble solve succeeds"))
    .map(|solution| solution.last_state()[0])
    .collect()
}

#[cfg(test)]
mod tests {
    use super::solve_parallel;

    #[test]
    fn default_features_expose_ordered_parallel_ensembles() {
        let endpoints = solve_parallel([1.0, 2.0, 4.0]);

        assert_eq!(endpoints.len(), 3);
        assert!((endpoints[1] - 2.0 * endpoints[0]).abs() < 1.0e-12);
        assert!((endpoints[2] - 4.0 * endpoints[0]).abs() < 1.0e-12);
    }
}
