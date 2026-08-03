use crate::explicit_rk::{ButcherTableau, ExplicitRungeKutta};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions};

const EMPTY: &[f64] = &[];
const A2: &[f64] = &[0.161];
const A3: &[f64] = &[-0.008_480_655_492_356_989, 0.335_480_655_492_357];
const A4: &[f64] = &[
    2.897_153_057_105_493_5,
    -6.359_448_489_975_075,
    4.362_295_432_869_581_5,
];
const A5: &[f64] = &[
    5.325_864_828_439_257,
    -11.748_883_564_062_828,
    7.495_539_342_889_836_5,
    -0.092_495_066_361_755_25,
];
const A6: &[f64] = &[
    5.861_455_442_946_42,
    -12.920_969_317_847_11,
    8.159_367_898_576_159,
    -0.071_584_973_281_401,
    -0.028_269_050_394_068_383,
];
const A7: &[f64] = &[
    0.096_460_766_818_065_23,
    0.01,
    0.479_889_650_414_499_6,
    1.379_008_574_103_742,
    -3.290_069_515_436_081,
    2.324_710_524_099_774,
];
const COEFFICIENTS: &[&[f64]] = &[EMPTY, A2, A3, A4, A5, A6, A7];
const NODES: &[f64] = &[0.0, 0.161, 0.327, 0.9, 0.980_025_540_904_509_7, 1.0, 1.0];
const WEIGHTS: &[f64] = &[
    0.096_460_766_818_065_23,
    0.01,
    0.479_889_650_414_499_6,
    1.379_008_574_103_742,
    -3.290_069_515_436_081,
    2.324_710_524_099_774,
    0.0,
];
const ERROR_WEIGHTS: &[f64] = &[
    -0.001_780_011_052_225_777,
    -0.000_816_434_459_656_746_9,
    0.007_880_878_010_261_995,
    -0.144_711_007_173_262_9,
    0.582_357_165_452_555_2,
    -0.458_082_105_929_186_97,
    0.015_151_515_151_515_152,
];

/// The Tsitouras 5/4 explicit Runge-Kutta method.
///
/// `Tsit5` is an adaptive, FSAL (first-same-as-last) method intended for
/// non-stiff ODEs at medium tolerances. It is a named facade over the shared
/// [`ExplicitRungeKutta`] kernel.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Tsit5;

impl ButcherTableau for Tsit5 {
    const NODES: &'static [f64] = NODES;
    const COEFFICIENTS: &'static [&'static [f64]] = COEFFICIENTS;
    const WEIGHTS: &'static [f64] = WEIGHTS;
    const ERROR_WEIGHTS: Option<&'static [f64]> = Some(ERROR_WEIGHTS);
    const ORDER: usize = 5;
    const FSAL: bool = true;
}

impl OdeAlgorithm for Tsit5 {
    fn solve<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        ExplicitRungeKutta::<Self>::new().solve(problem, options)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::f64::consts::{E, TAU};

    use crate::{OdeProblem, SaveMode, SolveError, SolveOptions, Tsit5, solve};

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
