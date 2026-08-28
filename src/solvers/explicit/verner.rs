//! Verner's efficient embedded explicit Runge--Kutta pairs.
//!
//! The coefficients are the compiled-`Float64` default-step coefficients from
//! `OrdinaryDiffEqVerner` at commit
//! `211142263781255a9aa2f910f6760b9f18ec29c8`. OrdinaryDiffEq's additional
//! stages for the method-specific dense interpolants are evaluated lazily only
//! when continuous output is requested.

crate::define_explicit_rk_from_file!(pub Vern6, "tableaux/explicit/vern6.json", crate = crate);
crate::define_explicit_rk_from_file!(pub Vern7, "tableaux/explicit/vern7.json", crate = crate);
crate::define_explicit_rk_from_file!(pub Vern8, "tableaux/explicit/vern8.json", crate = crate);
crate::define_explicit_rk_from_file!(pub Vern9, "tableaux/explicit/vern9.json", crate = crate);

#[cfg(test)]
mod tests {
    use super::{Vern6, Vern7, Vern8, Vern9};
    use crate::tableau::RungeKuttaTableau;
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

    fn assert_consistent(tableau: &RungeKuttaTableau) {
        for (row, &node) in tableau.a().iter().zip(tableau.c()) {
            assert!((row.iter().sum::<f64>() - node).abs() < 2.0e-12);
        }
        assert!((tableau.b().iter().sum::<f64>() - 1.0).abs() < 2.0e-13);
        assert!(tableau.error().unwrap().iter().sum::<f64>().abs() < 2.0e-13);
    }

    #[test]
    fn pinned_tableaus_are_internally_consistent() {
        assert_consistent(Vern6.tableau().unwrap());
        assert_consistent(Vern7.tableau().unwrap());
        assert_consistent(Vern8.tableau().unwrap());
        assert_consistent(Vern9.tableau().unwrap());
        assert!(Vern6.tableau().unwrap().fsal());
        assert!(!Vern7.tableau().unwrap().fsal());
        assert!(!Vern8.tableau().unwrap().fsal());
        assert!(!Vern9.tableau().unwrap().fsal());
    }
}
