//! Kubatko--Yeager--Ketcheson (2014) three-stage, second-order SSP method.
//!
//! The tableau is the algebraic form of the pinned
//! `KYK2014DGSSPRK_3S2` Shu--Osher recurrence from
//! `OrdinaryDiffEqSSPRK/src/ssprk_perform_step.jl`.  The upstream method is
//! fixed-step and uses an SSP coefficient of `0.8417`.  This native facade
//! intentionally exposes the regular ODE state/update only; Julia's stage and
//! step limiter callbacks are outside the current `OdeProblem` interface.

#![allow(clippy::excessive_precision)]

use crate::explicit_rk::{ButcherTableau, ExplicitRungeKutta};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions};

const EMPTY: &[f64] = &[];

// Pinned upstream Shu--Osher constants.  Expanding the recurrence gives
//
//   u₁ = u₀ + dt β₁₀ f₀,
//   u₂ = u₀ + dt (α₂₁β₁₀ f₀ + β₂₁ f₁),
//   u₃ = u₀ + dt ((α₃₂α₂₁β₁₀ + β₃₀) f₀
//                    + α₃₂β₂₁ f₁ + β₃₂ f₂).
//
// Keeping the products as expressions makes the correspondence to the
// source recurrence explicit while retaining all supplied decimal digits.
const ALPHA_21: f64 = 0.912_646_880_140_844;
const ALPHA_32: f64 = 0.655_043_082_833_159;
const BETA_10: f64 = 0.528_005_024_856_522;
const BETA_21: f64 = 0.481_882_138_633_993;
const BETA_30: f64 = 0.022_826_837_460_491;
const BETA_32: f64 = 0.345_866_039_233_415;

const KYK2014_A2: &[f64] = &[BETA_10];
const KYK2014_A3: &[f64] = &[ALPHA_21 * BETA_10, BETA_21];
const KYK2014_A: &[&[f64]] = &[EMPTY, KYK2014_A2, KYK2014_A3];
const KYK2014_B: &[f64] = &[
    ALPHA_32 * ALPHA_21 * BETA_10 + BETA_30,
    ALPHA_32 * BETA_21,
    BETA_32,
];
const KYK2014_C: &[f64] = &[0.0, BETA_10, ALPHA_21 * BETA_10 + BETA_21];

/// Fixed-step KYK2014 discontinuous-Galerkin SSPRK(3,2).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Kyk2014DgSsprk3S2;

struct Kyk2014DgSsprk3S2Tableau;

impl ButcherTableau for Kyk2014DgSsprk3S2Tableau {
    const NODES: &'static [f64] = KYK2014_C;
    const COEFFICIENTS: &'static [&'static [f64]] = KYK2014_A;
    const WEIGHTS: &'static [f64] = KYK2014_B;
    const ERROR_WEIGHTS: Option<&'static [f64]> = None;
    const ORDER: usize = 2;
    const FSAL: bool = false;
}

impl OdeAlgorithm for Kyk2014DgSsprk3S2 {
    fn solve<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        ExplicitRungeKutta::<Kyk2014DgSsprk3S2Tableau>::new().solve(problem, options)
    }
}

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
