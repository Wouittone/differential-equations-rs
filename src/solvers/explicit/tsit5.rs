crate::define_explicit_rk_from_file!(pub Tsit5, "src/tableau/resources/explicit/tsit5.json", crate = crate);

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::f64::consts::{E, TAU};

    use super::Tsit5;
    use crate::{OdeProblem, SaveMode, SolveError, SolveOptions, solve};

    #[test]
    fn solves_scalar_exponential_growth() {
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
            vec![1.0],
            (0.0, 1.0),
            (),
        );
        let options = SolveOptions {
            absolute_tolerance: 1.0e-11,
            relative_tolerance: 1.0e-11,
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        let solution = solve(&problem, Tsit5, &options).unwrap();

        assert!((solution.last_state()[0] - E).abs() < 2.0e-10);
        assert_eq!(solution.times(), &[0.0, 1.0]);
        assert!(solution.stats().accepted_steps > 0);
    }

    #[test]
    fn solves_a_vector_harmonic_oscillator() {
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| {
                du[0] = u[1];
                du[1] = -u[0];
            },
            vec![1.0, 0.0],
            (0.0, TAU),
            (),
        );
        let options = SolveOptions {
            absolute_tolerance: 1.0e-10,
            relative_tolerance: 1.0e-10,
            ..SolveOptions::default()
        };

        let solution = solve(&problem, Tsit5, &options).unwrap();

        assert!((solution.last_state()[0] - 1.0).abs() < 2.0e-9);
        assert!(solution.last_state()[1].abs() < 2.0e-9);
        assert_eq!(solution.times().len(), solution.stats().accepted_steps + 1);
    }

    #[test]
    fn supports_backward_integration() {
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
            vec![E],
            (1.0, 0.0),
            (),
        );
        let options = SolveOptions {
            absolute_tolerance: 1.0e-10,
            relative_tolerance: 1.0e-10,
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        let solution = solve(&problem, Tsit5, &options).unwrap();

        assert!((solution.last_state()[0] - 1.0).abs() < 2.0e-9);
        assert_eq!(solution.times(), &[1.0, 0.0]);
    }

    #[test]
    fn rejects_an_overly_large_initial_step() {
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
            vec![1.0],
            (0.0, 10.0),
            (),
        );
        let options = SolveOptions {
            absolute_tolerance: 1.0e-12,
            relative_tolerance: 1.0e-12,
            initial_step: Some(10.0),
            ..SolveOptions::default()
        };

        let solution = solve(&problem, Tsit5, &options).unwrap();

        assert!(solution.stats().rejected_steps > 0);
        assert_eq!(
            solution.stats().rhs_evaluations,
            1 + 6 * (solution.stats().accepted_steps + solution.stats().rejected_steps)
        );
    }

    #[test]
    fn reports_non_finite_derivatives() {
        let problem = OdeProblem::new(
            |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = f64::NAN,
            vec![1.0],
            (0.0, 1.0),
            (),
        );

        assert_eq!(
            solve(&problem, Tsit5, &SolveOptions::default()),
            Err(SolveError::NonFiniteDerivative)
        );
    }

    #[test]
    fn reports_a_non_finite_fsal_derivative() {
        let calls = Cell::new(0);
        let problem = OdeProblem::new(
            |du: &mut [f64], _: &[f64], _: &(), _: f64| {
                let call = calls.get();
                calls.set(call + 1);
                du[0] = if call == 6 { f64::NAN } else { 0.0 };
            },
            vec![1.0],
            (0.0, 1.0),
            (),
        );
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(1.0),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        assert_eq!(
            solve(&problem, Tsit5, &options),
            Err(SolveError::NonFiniteDerivative)
        );
    }
}
