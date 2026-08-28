//! Kubatko--Yeager--Ketcheson (2014) three-stage, second-order SSP method.
//!
//! The tableau is the algebraic form of the pinned
//! `KYK2014DGSSPRK_3S2` Shu--Osher recurrence from
//! `OrdinaryDiffEqSSPRK/src/ssprk_perform_step.jl`.  The upstream method is
//! fixed-step and uses an SSP coefficient of `0.8417`.  This native facade
//! intentionally exposes the regular ODE state/update only; Julia's stage and
//! step limiter callbacks are outside the current `OdeProblem` interface.

crate::define_explicit_rk_from_file!(
    pub Kyk2014DgSsprk3S2,
    "src/tableau/resources/explicit/kyk2014_dg_ssprk3_s2.json",
    crate = crate
);

#[cfg(test)]
mod tests {
    use super::Kyk2014DgSsprk3S2;
    use crate::{CallbackAction, OdeProblem, SaveMode, SolveOptions, solve};

    type Rhs = fn(&mut [f64], &[f64], &(), f64);

    fn exponential() -> OdeProblem<Rhs, ()> {
        fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
            du[0] = u[0];
        }
        OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
    }

    fn fixed(step: f64) -> SolveOptions {
        SolveOptions {
            adaptive: false,
            initial_step: Some(step),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        }
    }

    #[test]
    fn converges_at_second_order() {
        let coarse = solve(&exponential(), Kyk2014DgSsprk3S2, &fixed(0.1))
            .unwrap()
            .last_state()[0];
        let fine = solve(&exponential(), Kyk2014DgSsprk3S2, &fixed(0.05))
            .unwrap()
            .last_state()[0];
        let observed =
            ((coarse - std::f64::consts::E).abs() / (fine - std::f64::consts::E).abs()).log2();
        assert!(observed > 1.85, "observed order was {observed}");
    }

    #[test]
    fn backward_save_at_and_callback_are_supported() {
        let backward = OdeProblem::new(
            (|du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0]) as Rhs,
            vec![std::f64::consts::E],
            (1.0, 0.0),
            (),
        );
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.02),
            save_at: vec![0.75, 0.5, 0.0],
            ..SolveOptions::default()
        };
        let solution = solve(&backward, Kyk2014DgSsprk3S2, &options).unwrap();
        assert_eq!(solution.times(), &[0.75, 0.5, 0.0]);
        assert!((solution.last_state()[0] - 1.0).abs() < 2.0e-4);

        let terminating = exponential()
            .with_continuous_callback(|_, _, time| time - 0.5, |_, _, _| CallbackAction::Terminate);
        let solution = solve(&terminating, Kyk2014DgSsprk3S2, &fixed(0.1)).unwrap();
        assert!((solution.times().last().unwrap() - 0.5).abs() < 1.0e-14);
        assert_eq!(solution.stats().callback_invocations, 1);
    }
}
