//! Remaining high-order explicit Runge--Kutta tableaus from the pinned SciML source.
//!
//! Source: OrdinaryDiffEq.jl revision 211142263781255a9aa2f910f6760b9f18ec29c8.
//! The tableaus intentionally use the crate's reusable fixed/adaptive explicit
//! RK kernel; method-specific dense interpolation stages are not required for
//! the endpoint solver surface implemented here.

// These literals are the canonical binary64 values from the pinned upstream
// tableaus; shortening them would change the recovered methods.
#![allow(clippy::excessive_precision)]

use super::general::{ButcherTableau, ExplicitRungeKutta};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions};

mod coefficient_data {
    use differential_equations_tableau_macros::define_coefficients_from_file;

    define_coefficients_from_file!(
        pub(super),
        "coefficients/explicit/high_order.toml",
        crate = crate
    );
}

use coefficient_data::*;

// Compatibility reexports for the historical `explicit::high_order` façade.
pub use super::anas5::Anas5;
pub use super::frk65::Frk65;
pub use super::verner::{Vern6, Vern7, Vern8, Vern9};

// OrdinaryDiffEq stores this method's lower-order weights rather than the
// `b - b_hat` coefficients expected by the shared explicit RK kernel.
const PFRK87_E: &[f64] = &[
    PFRK87_B[0] - PFRK87_EMBEDDED_WEIGHTS[0],
    PFRK87_B[1] - PFRK87_EMBEDDED_WEIGHTS[1],
    PFRK87_B[2] - PFRK87_EMBEDDED_WEIGHTS[2],
    PFRK87_B[3] - PFRK87_EMBEDDED_WEIGHTS[3],
    PFRK87_B[4] - PFRK87_EMBEDDED_WEIGHTS[4],
    PFRK87_B[5] - PFRK87_EMBEDDED_WEIGHTS[5],
    PFRK87_B[6] - PFRK87_EMBEDDED_WEIGHTS[6],
    PFRK87_B[7] - PFRK87_EMBEDDED_WEIGHTS[7],
    PFRK87_B[8] - PFRK87_EMBEDDED_WEIGHTS[8],
    PFRK87_B[9] - PFRK87_EMBEDDED_WEIGHTS[9],
    PFRK87_B[10] - PFRK87_EMBEDDED_WEIGHTS[10],
    PFRK87_B[11] - PFRK87_EMBEDDED_WEIGHTS[11],
    PFRK87_B[12] - PFRK87_EMBEDDED_WEIGHTS[12],
];

macro_rules! high_order_algorithm {
    ($name:ident, $doc:literal, $nodes:ident, $a:ident, $b:ident, $e:ident, $e2:expr, $order:literal, $fsal:literal) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        pub struct $name;
        impl ButcherTableau for $name {
            const NODES: &'static [f64] = $nodes;
            const COEFFICIENTS: &'static [&'static [f64]] = $a;
            const WEIGHTS: &'static [f64] = $b;
            const ERROR_WEIGHTS: Option<&'static [f64]> = Some($e);
            const SECOND_ERROR_WEIGHTS: Option<&'static [f64]> = $e2;
            const ORDER: usize = $order;
            const FSAL: bool = $fsal;
        }
        impl OdeAlgorithm for $name {
            fn solve<F, P>(&self, problem: &OdeProblem<F, P>, options: &SolveOptions) -> Result<Solution, SolveError>
            where F: Fn(&mut [f64], &[f64], &P, f64),
            {
                ExplicitRungeKutta::<$name>::new().solve(problem, options)
            }
        }
    };
}

high_order_algorithm!(
    TanYam7,
    "Tanaka--Yamashita seventh-order explicit Runge--Kutta method.",
    TANYAM7_NODES,
    TANYAM7_A,
    TANYAM7_B,
    TANYAM7_E,
    None,
    7,
    false
);
high_order_algorithm!(
    TsitPap8,
    "Tsitouras--Papakostas eighth-order explicit Runge--Kutta method.",
    TSITPAP8_NODES,
    TSITPAP8_A,
    TSITPAP8_B,
    TSITPAP8_E,
    None,
    8,
    false
);
high_order_algorithm!(
    DP8,
    "Hairer--Norsett--Wanner Dormand--Prince eighth-order explicit Runge--Kutta method.",
    DP8_NODES,
    DP8_A,
    DP8_B,
    DP8_E,
    Some(DP8_E2),
    8,
    true
);
high_order_algorithm!(
    PFRK87,
    "Phase-fitted eighth-order (7) explicit Runge--Kutta pair (default phase estimate omega = 0).",
    PFRK87_NODES,
    PFRK87_A,
    PFRK87_B,
    PFRK87_E,
    None,
    8,
    false
);
high_order_algorithm!(
    Feagin10,
    "Feagin 10th-order explicit Runge--Kutta method.",
    FEAGIN10_NODES,
    FEAGIN10_A,
    FEAGIN10_B,
    FEAGIN10_E,
    None,
    10,
    false
);
high_order_algorithm!(
    Feagin12,
    "Feagin 12th-order explicit Runge--Kutta method.",
    FEAGIN12_NODES,
    FEAGIN12_A,
    FEAGIN12_B,
    FEAGIN12_E,
    None,
    12,
    false
);
high_order_algorithm!(
    Feagin14,
    "Feagin 14th-order explicit Runge--Kutta method.",
    FEAGIN14_NODES,
    FEAGIN14_A,
    FEAGIN14_B,
    FEAGIN14_E,
    None,
    14,
    false
);
high_order_algorithm!(
    RKV76IIa,
    "Verner RKV76.IIa seventh-order (sixth-order embedded) explicit Runge--Kutta pair.",
    RKV76IIA_NODES,
    RKV76IIA_A,
    RKV76IIA_B,
    RKV76IIA_E,
    None,
    7,
    false
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OdeProblem, SaveMode, SolveOptions, solve};
    type ScalarRhs = fn(&mut [f64], &[f64], &(), f64);

    fn exponential() -> OdeProblem<ScalarRhs, ()> {
        fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
            du[0] = u[0];
        }
        OdeProblem::new(rhs as ScalarRhs, vec![1.0], (0.0, 1.0), ())
    }
    #[test]
    fn pinned_tableaus_are_consistent() {
        macro_rules! check {
            ($t:ty) => {{
                for (row, &c) in <$t as ButcherTableau>::COEFFICIENTS
                    .iter()
                    .zip(<$t as ButcherTableau>::NODES)
                {
                    assert!(
                        (row.iter().sum::<f64>() - c).abs() < 5.0e-12,
                        "{} row sum mismatch: sum={:.17e} c={:.17e}",
                        std::any::type_name::<$t>(),
                        row.iter().sum::<f64>(),
                        c
                    );
                }
                let weight_sum = <$t as ButcherTableau>::WEIGHTS.iter().sum::<f64>();
                if (weight_sum - 1.0).abs() >= 5.0e-12 {
                    panic!(
                        "{} weights sum={:.17e}",
                        std::any::type_name::<$t>(),
                        weight_sum
                    );
                }
                let error_sum = <$t as ButcherTableau>::ERROR_WEIGHTS
                    .unwrap()
                    .iter()
                    .sum::<f64>();
                assert!(
                    error_sum.abs() < 5.0e-12,
                    "{} error weights sum={:.17e}",
                    std::any::type_name::<$t>(),
                    error_sum
                );
            }};
        }
        check!(TanYam7);
        check!(TsitPap8);
        check!(DP8);
        check!(PFRK87);
        check!(Feagin10);
        check!(Feagin12);
        check!(Feagin14);
        check!(RKV76IIa);
    }
    #[test]
    fn adaptive_endpoints_are_accurate() {
        let options = SolveOptions {
            absolute_tolerance: 1.0e-10,
            relative_tolerance: 1.0e-10,
            initial_step: Some(0.2),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };
        let exact = 1.0_f64.exp();
        macro_rules! check {
            ($a:expr) => {
                let solution = match solve(&exponential(), $a, &options) {
                    Ok(solution) => solution,
                    Err(error) => panic!("{} solve failed: {:?}", stringify!($a), error),
                };
                assert!(
                    (solution.last_state()[0] - exact).abs() < 2.0e-8,
                    "{} endpoint={:.17e}",
                    stringify!($a),
                    solution.last_state()[0]
                )
            };
        }
        check!(TanYam7);
        check!(TsitPap8);
        check!(DP8);
        check!(PFRK87);
        check!(Feagin10);
        check!(Feagin12);
        check!(Feagin14);
        check!(RKV76IIa);
    }

    #[test]
    fn algorithm_facades_dispatch_to_their_own_tableaus() {
        fn nonautonomous(du: &mut [f64], u: &[f64], _: &(), time: f64) {
            du[0] = (1.0 + time) * u[0] + time.sin();
        }

        let problem = OdeProblem::new(
            nonautonomous as fn(&mut [f64], &[f64], &(), f64),
            vec![0.75],
            (0.0, 1.0),
            (),
        );
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.25),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        macro_rules! check {
            ($algorithm:ident) => {{
                let facade = solve(&problem, $algorithm, &options).unwrap();
                let direct = ExplicitRungeKutta::<$algorithm>::new()
                    .solve(&problem, &options)
                    .unwrap();
                assert_eq!(
                    facade.last_state(),
                    direct.last_state(),
                    "{} substituted another tableau",
                    stringify!($algorithm)
                );
                assert_eq!(facade.stats(), direct.stats());
            }};
        }

        check!(TanYam7);
        check!(TsitPap8);
        check!(DP8);
        check!(PFRK87);
        check!(Feagin10);
        check!(Feagin12);
        check!(Feagin14);
        check!(RKV76IIa);
    }

    fn fixed_exponential_endpoint<A: OdeAlgorithm>(algorithm: A, step: f64) -> f64 {
        fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
            du[0] = u[0];
        }
        let problem = OdeProblem::new(
            rhs as fn(&mut [f64], &[f64], &(), f64),
            vec![1.0],
            (0.0, 2.0),
            (),
        );
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(step),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };
        solve(&problem, algorithm, &options).unwrap().last_state()[0]
    }

    fn convergence_ratio<A: OdeAlgorithm + Copy>(algorithm: A) -> f64 {
        let exact = 2.0_f64.exp();
        let coarse_error = (fixed_exponential_endpoint(algorithm, 0.5) - exact).abs();
        let fine_error = (fixed_exponential_endpoint(algorithm, 0.25) - exact).abs();
        coarse_error / fine_error
    }

    #[test]
    fn fixed_step_methods_exhibit_their_method_specific_orders() {
        for (name, ratio) in [
            ("TanYam7", convergence_ratio(TanYam7)),
            ("RKV76IIa", convergence_ratio(RKV76IIa)),
        ] {
            assert!(ratio > 50.0, "{name} convergence ratio={ratio:.6e}");
        }
        for (name, ratio) in [
            ("TsitPap8", convergence_ratio(TsitPap8)),
            ("DP8", convergence_ratio(DP8)),
            ("PFRK87", convergence_ratio(PFRK87)),
        ] {
            assert!(ratio > 100.0, "{name} convergence ratio={ratio:.6e}");
        }
        let feagin10_ratio = convergence_ratio(Feagin10);
        assert!(
            feagin10_ratio > 200.0,
            "Feagin10 convergence ratio={feagin10_ratio:.6e}"
        );
    }
}
