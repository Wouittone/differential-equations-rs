//! Verner's efficient embedded explicit Runge--Kutta pairs.
//!
//! The coefficients are the compiled-`Float64` default-step coefficients from
//! `OrdinaryDiffEqVerner` at commit
//! `211142263781255a9aa2f910f6760b9f18ec29c8`. OrdinaryDiffEq's additional
//! stages for the method-specific dense interpolants are evaluated lazily only
//! when continuous output is requested.

use super::coefficient_data::{
    VERN6_A_ROWS, VERN6_B as GENERATED_VERN6_B, VERN6_DENSE, VERN6_E as GENERATED_VERN6_E,
    VERN6_EXTRA_STAGES, VERN6_STAGE_TIMES, VERN7_A_ROWS, VERN7_B as GENERATED_VERN7_B, VERN7_DENSE,
    VERN7_E as GENERATED_VERN7_E, VERN7_EXTRA_STAGES, VERN7_STAGE_TIMES, VERN8_A_ROWS,
    VERN8_B as GENERATED_VERN8_B, VERN8_DENSE, VERN8_E as GENERATED_VERN8_E, VERN8_EXTRA_STAGES,
    VERN8_STAGE_TIMES, VERN9_A_ROWS, VERN9_B as GENERATED_VERN9_B, VERN9_DENSE,
    VERN9_E as GENERATED_VERN9_E, VERN9_EXTRA_STAGES, VERN9_STAGE_TIMES,
};
use super::general::{ButcherTableau, ExplicitRungeKutta, LazyDenseStage};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions};

const VERN6_C: &[f64] = &VERN6_STAGE_TIMES;
const VERN6_A: &[&[f64]] = VERN6_A_ROWS;
const VERN6_B: &[f64] = &GENERATED_VERN6_B;
const VERN6_E: &[f64] = &GENERATED_VERN6_E;

const VERN7_C: &[f64] = &VERN7_STAGE_TIMES;
const VERN7_A: &[&[f64]] = VERN7_A_ROWS;
const VERN7_B: &[f64] = &GENERATED_VERN7_B;
const VERN7_E: &[f64] = &GENERATED_VERN7_E;

const VERN8_C: &[f64] = &VERN8_STAGE_TIMES;
const VERN8_A: &[&[f64]] = VERN8_A_ROWS;
const VERN8_B: &[f64] = &GENERATED_VERN8_B;
const VERN8_E: &[f64] = &GENERATED_VERN8_E;

const VERN9_C: &[f64] = &VERN9_STAGE_TIMES;
const VERN9_A: &[&[f64]] = VERN9_A_ROWS;
const VERN9_B: &[f64] = &GENERATED_VERN9_B;
const VERN9_E: &[f64] = &GENERATED_VERN9_E;

macro_rules! verner_algorithm {
    ($name:ident, $documentation:literal, $order:literal, $nodes:ident, $coefficients:ident, $weights:ident, $error:ident, $dense:ident, $extra:ident, $fsal:literal) => {
        #[doc = $documentation]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        pub struct $name;

        impl ButcherTableau for $name {
            const NODES: &'static [f64] = $nodes;
            const COEFFICIENTS: &'static [&'static [f64]] = $coefficients;
            const WEIGHTS: &'static [f64] = $weights;
            const ERROR_WEIGHTS: Option<&'static [f64]> = Some($error);
            const DENSE_COEFFICIENTS: Option<&'static [&'static [f64]]> = Some($dense);
            const LAZY_DENSE_STAGES: &'static [LazyDenseStage] = $extra;
            const ORDER: usize = $order;
            const FSAL: bool = $fsal;
        }

        impl OdeAlgorithm for $name {
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
    };
}

verner_algorithm!(
    Vern6,
    "Verner's efficient embedded order-6 explicit Runge--Kutta method.",
    6,
    VERN6_C,
    VERN6_A,
    VERN6_B,
    VERN6_E,
    VERN6_DENSE,
    VERN6_EXTRA_STAGES,
    true
);
verner_algorithm!(
    Vern7,
    "Verner's efficient embedded order-7 explicit Runge--Kutta method.",
    7,
    VERN7_C,
    VERN7_A,
    VERN7_B,
    VERN7_E,
    VERN7_DENSE,
    VERN7_EXTRA_STAGES,
    false
);
verner_algorithm!(
    Vern8,
    "Verner's efficient embedded order-8 explicit Runge--Kutta method.",
    8,
    VERN8_C,
    VERN8_A,
    VERN8_B,
    VERN8_E,
    VERN8_DENSE,
    VERN8_EXTRA_STAGES,
    false
);
verner_algorithm!(
    Vern9,
    "Verner's efficient embedded order-9 explicit Runge--Kutta method.",
    9,
    VERN9_C,
    VERN9_A,
    VERN9_B,
    VERN9_E,
    VERN9_DENSE,
    VERN9_EXTRA_STAGES,
    false
);

#[cfg(test)]
mod tests {
    use super::ButcherTableau;
    use super::{Vern6, Vern7, Vern8, Vern9};
    use crate::{OdeAlgorithm, OdeProblem, SaveMode, SolveOptions, solve};

    type TestProblem = OdeProblem<fn(&mut [f64], &[f64], &(), f64), ()>;

    fn exponential() -> TestProblem {
        fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
            du[0] = u[0];
        }
        OdeProblem::new(rhs, vec![1.0], (0.0, 2.0), ())
    }

    fn fixed_endpoint<A: OdeAlgorithm>(algorithm: A, step: f64) -> f64 {
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(step),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };
        solve(&exponential(), algorithm, &options)
            .unwrap()
            .last_state()[0]
    }

    fn convergence_ratio<A: OdeAlgorithm + Copy>(algorithm: A) -> f64 {
        let exact = 2.0_f64.exp();
        let coarse = (fixed_endpoint(algorithm, 0.5) - exact).abs();
        let fine = (fixed_endpoint(algorithm, 0.25) - exact).abs();
        coarse / fine
    }

    #[test]
    fn fixed_step_methods_have_their_expected_orders() {
        let ratios = [
            convergence_ratio(Vern6),
            convergence_ratio(Vern7),
            convergence_ratio(Vern8),
            convergence_ratio(Vern9),
        ];
        assert!(ratios[0] > 45.0);
        assert!(ratios[1] > 90.0);
        assert!(ratios[2] > 170.0);
        assert!(ratios[3] > 300.0);
    }

    #[test]
    fn adaptive_methods_reach_a_tight_tolerance() {
        let options = SolveOptions {
            absolute_tolerance: 1.0e-11,
            relative_tolerance: 1.0e-11,
            initial_step: Some(0.5),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };
        let exact = 2.0_f64.exp();

        for endpoint in [
            solve(&exponential(), Vern6, &options).unwrap().last_state()[0],
            solve(&exponential(), Vern7, &options).unwrap().last_state()[0],
            solve(&exponential(), Vern8, &options).unwrap().last_state()[0],
            solve(&exponential(), Vern9, &options).unwrap().last_state()[0],
        ] {
            assert!((endpoint - exact).abs() < 2.0e-9);
        }
    }

    fn assert_consistent<T: ButcherTableau>() {
        for (row, &node) in T::COEFFICIENTS.iter().zip(T::NODES) {
            assert!((row.iter().sum::<f64>() - node).abs() < 2.0e-12);
        }
        assert!((T::WEIGHTS.iter().sum::<f64>() - 1.0).abs() < 2.0e-13);
        assert!(T::ERROR_WEIGHTS.unwrap().iter().sum::<f64>().abs() < 2.0e-13);
    }

    fn is_fsal<T: ButcherTableau>() -> bool {
        T::FSAL
    }

    #[test]
    fn pinned_tableaus_are_internally_consistent() {
        assert_consistent::<Vern6>();
        assert_consistent::<Vern7>();
        assert_consistent::<Vern8>();
        assert_consistent::<Vern9>();
        assert!(is_fsal::<Vern6>());
        assert!(!is_fsal::<Vern7>() && !is_fsal::<Vern8>() && !is_fsal::<Vern9>());
    }
}
