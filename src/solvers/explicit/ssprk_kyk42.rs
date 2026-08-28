//! Kubatko--Yeager--Ketcheson optimal SSPRK(4,2).
//!
//! The pinned `OrdinaryDiffEqSSPRK` implementation stores this method in
//! Shu--Osher form.  The regular ODE facade expands that recurrence into its
//! equivalent four-stage Butcher tableau.  Stage and step limiter callbacks
//! are intentionally not exposed because they are not part of `OdeProblem`.

crate::define_explicit_rk_from_file!(
    pub KykSsprk42,
    "tableaux/explicit/kyk_ssprk42.json",
    crate = crate
);

#[allow(non_camel_case_types)]
/// Exact OrdinaryDiffEq-compatible spelling alias for [`KykSsprk42`].
pub type KYKSSPRK42 = KykSsprk42;

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
