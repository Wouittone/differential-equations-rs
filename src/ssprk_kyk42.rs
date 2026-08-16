//! Kubatko--Yeager--Ketcheson optimal SSPRK(4,2).
//!
//! The pinned `OrdinaryDiffEqSSPRK` implementation stores this method in
//! Shu--Osher form.  The regular ODE facade expands that recurrence into its
//! equivalent four-stage Butcher tableau.  Stage and step limiter callbacks
//! are intentionally not exposed because they are not part of `OdeProblem`.

#![allow(clippy::excessive_precision)]

use crate::explicit_rk::{ButcherTableau, ExplicitRungeKutta};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions};

const EMPTY: &[f64] = &[];

// Pinned upstream Shu--Osher constants from
// OrdinaryDiffEqSSPRK/src/ssprk_caches.jl.  Expanding the recurrence keeps
// the coefficients in the same order as the source while allowing the shared
// explicit driver to provide save_at, callbacks, and backward integration.
const ALPHA_21: f64 = 0.605_193_558_660_171;
const ALPHA_32: f64 = 0.997_202_692_912_61;
const ALPHA_43: f64 = 0.747_139_090_645_627;
const BETA_10: f64 = 0.406_584_463_657_504;
const BETA_21: f64 = 0.246_062_298_456_822;
const BETA_30: f64 = 0.013_637_216_641_451;
const BETA_32: f64 = 0.405_447_122_055_692;
const BETA_40: f64 = 0.016_453_567_333_598;
const BETA_43: f64 = 0.303_775_146_447_707;

const KYKSSPRK42_A2: &[f64] = &[BETA_10];
const KYKSSPRK42_A3: &[f64] = &[ALPHA_21 * BETA_10, BETA_21];
const KYKSSPRK42_A4: &[f64] = &[
    ALPHA_32 * ALPHA_21 * BETA_10 + BETA_30,
    ALPHA_32 * BETA_21,
    BETA_32,
];
const KYKSSPRK42_A: &[&[f64]] = &[EMPTY, KYKSSPRK42_A2, KYKSSPRK42_A3, KYKSSPRK42_A4];
const KYKSSPRK42_B: &[f64] = &[
    ALPHA_43 * (ALPHA_32 * ALPHA_21 * BETA_10 + BETA_30) + BETA_40,
    ALPHA_43 * ALPHA_32 * BETA_21,
    ALPHA_43 * BETA_32,
    BETA_43,
];
const KYKSSPRK42_C: &[f64] = &[
    0.0,
    BETA_10,
    0.492_124_596_913_643_8,
    0.909_832_311_987_961_3,
];

struct KykSsprk42Tableau;

impl ButcherTableau for KykSsprk42Tableau {
    const NODES: &'static [f64] = KYKSSPRK42_C;
    const COEFFICIENTS: &'static [&'static [f64]] = KYKSSPRK42_A;
    const WEIGHTS: &'static [f64] = KYKSSPRK42_B;
    const ERROR_WEIGHTS: Option<&'static [f64]> = None;
    const ORDER: usize = 2;
    const FSAL: bool = false;
}

/// Fixed-step KYK optimal SSPRK(4,2).
///
/// As in the pinned Julia algorithm, no embedded error estimator is supplied;
/// requesting adaptive stepping therefore returns `AdaptiveStepUnsupported`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct KykSsprk42;

#[allow(non_camel_case_types)]
pub type KYKSSPRK42 = KykSsprk42;

impl OdeAlgorithm for KykSsprk42 {
    fn solve<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        ExplicitRungeKutta::<KykSsprk42Tableau>::new().solve(problem, options)
    }
}

#[cfg(test)]
mod tests {
    use super::KykSsprk42;
    use crate::{CallbackAction, OdeProblem, SaveMode, SolveError, SolveOptions, solve};

    type Rhs = fn(&mut [f64], &[f64], &(), f64);

    fn exponential(interval: (f64, f64)) -> OdeProblem<Rhs, ()> {
        fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
            du[0] = u[0];
        }
        OdeProblem::new(rhs, vec![1.0], interval, ())
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
        let coarse = solve(&exponential((0.0, 1.0)), KykSsprk42, &fixed(0.1))
            .unwrap()
            .last_state()[0];
        let fine = solve(&exponential((0.0, 1.0)), KykSsprk42, &fixed(0.05))
            .unwrap()
            .last_state()[0];
        let observed =
            ((coarse - std::f64::consts::E).abs() / (fine - std::f64::consts::E).abs()).log2();
        assert!(observed > 1.85, "observed order was {observed}");
    }

    #[test]
    fn fixed_only_and_backward_save_at() {
        let adaptive = SolveOptions {
            adaptive: true,
            ..fixed(0.1)
        };
        assert!(matches!(
            solve(&exponential((0.0, 1.0)), KykSsprk42, &adaptive),
            Err(SolveError::AdaptiveStepUnsupported)
        ));

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
        let solution = solve(&backward, KykSsprk42, &options).unwrap();
        assert_eq!(solution.times(), &[0.75, 0.5, 0.0]);
        assert!((solution.last_state()[0] - 1.0).abs() < 2.0e-4);
    }

    #[test]
    fn callback_terminates_without_losing_endpoint() {
        let problem = exponential((0.0, 1.0))
            .with_continuous_callback(|_, _, time| time - 0.5, |_, _, _| CallbackAction::Terminate);
        let solution = solve(&problem, KykSsprk42, &fixed(0.1)).unwrap();
        assert!((solution.times().last().unwrap() - 0.5).abs() < 1.0e-14);
        assert_eq!(solution.stats().callback_invocations, 1);
    }
}
