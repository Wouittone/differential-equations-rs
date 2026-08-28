//! Remaining high-order explicit Runge--Kutta tableaus from the pinned SciML source.
//!
//! Source: OrdinaryDiffEq.jl revision 211142263781255a9aa2f910f6760b9f18ec29c8.
//! The tableaus intentionally use the crate's reusable fixed/adaptive explicit
//! RK kernel; method-specific dense interpolation stages are not required for
//! the endpoint solver surface implemented here.

crate::define_explicit_rk_from_file!(pub TanYam7, "src/tableau/resources/explicit/tan_yam7.json", crate = crate);
crate::define_explicit_rk_from_file!(pub TsitPap8, "src/tableau/resources/explicit/tsit_pap8.json", crate = crate);
crate::define_explicit_rk_from_file!(pub DP8, "src/tableau/resources/explicit/dp8.json", crate = crate);
crate::define_explicit_rk_from_file!(pub PFRK87, "src/tableau/resources/explicit/pfrk87.json", crate = crate);
crate::define_explicit_rk_from_file!(pub Feagin10, "src/tableau/resources/explicit/feagin10.json", crate = crate);
crate::define_explicit_rk_from_file!(pub Feagin12, "src/tableau/resources/explicit/feagin12.json", crate = crate);
crate::define_explicit_rk_from_file!(pub Feagin14, "src/tableau/resources/explicit/feagin14.json", crate = crate);
crate::define_explicit_rk_from_file!(pub RKV76IIa, "src/tableau/resources/explicit/rkv76_iia.json", crate = crate);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OdeAlgorithm, OdeProblem, SaveMode, SolveOptions, solve};
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
            ($algorithm:expr) => {{
                let tableau = $algorithm.tableau().unwrap();
                for (row, &c) in tableau.a().iter().zip(tableau.c()) {
                    assert!(
                        (row.iter().sum::<f64>() - c).abs() < 5.0e-12,
                        "{} row sum mismatch: sum={:.17e} c={:.17e}",
                        tableau.name(),
                        row.iter().sum::<f64>(),
                        c
                    );
                }
                let weight_sum = tableau.b().iter().sum::<f64>();
                if (weight_sum - 1.0).abs() >= 5.0e-12 {
                    panic!("{} weights sum={:.17e}", tableau.name(), weight_sum);
                }
                let error_sum = tableau.error().unwrap().iter().sum::<f64>();
                assert!(
                    error_sum.abs() < 5.0e-12,
                    "{} error weights sum={:.17e}",
                    tableau.name(),
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
    fn algorithm_facades_reference_their_named_resources() {
        assert_eq!(TanYam7.tableau().unwrap().name(), "TanYam7");
        assert_eq!(TsitPap8.tableau().unwrap().name(), "TsitPap8");
        assert_eq!(DP8.tableau().unwrap().name(), "DP8");
        assert_eq!(PFRK87.tableau().unwrap().name(), "PFRK87");
        assert_eq!(Feagin10.tableau().unwrap().name(), "Feagin10");
        assert_eq!(Feagin12.tableau().unwrap().name(), "Feagin12");
        assert_eq!(Feagin14.tableau().unwrap().name(), "Feagin14");
        assert_eq!(RKV76IIa.tableau().unwrap().name(), "RKV76IIa");
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
