use super::{
    SspRk53, SspRk53H, SspRk53TwoN1, SspRk53TwoN2, SspRk54, SspRk63, SspRk73, SspRk83, SspRk104,
    SspRk432, SspRk932,
};
use crate::{CallbackAction, OdeAlgorithm, OdeProblem, SaveMode, SolveError, SolveOptions, solve};

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn exponential() -> OdeProblem<TestRhs, ()> {
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

fn endpoint<A: OdeAlgorithm>(algorithm: A, step: f64) -> f64 {
    solve(&exponential(), algorithm, &fixed(step))
        .unwrap()
        .last_state()[0]
}

fn observed_order<A: OdeAlgorithm + Copy>(algorithm: A) -> f64 {
    let coarse = (endpoint(algorithm, 0.1) - std::f64::consts::E).abs();
    let fine = (endpoint(algorithm, 0.05) - std::f64::consts::E).abs();
    (coarse / fine).log2()
}

#[test]
fn ssprk432_supports_fixed_and_adaptive_modes_at_third_order() {
    let fixed_order = observed_order(SspRk432);
    assert!(fixed_order > 2.9, "fixed observed order was {fixed_order}");

    let adaptive = SolveOptions {
        absolute_tolerance: 1.0e-9,
        relative_tolerance: 1.0e-9,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let endpoint = solve(&exponential(), SspRk432, &adaptive)
        .unwrap()
        .last_state()[0];
    assert!((endpoint - std::f64::consts::E).abs() < 2.0e-8);
}

#[test]
fn ssprk932_supports_fixed_adaptive_and_shared_output_features() {
    let fixed_order = observed_order(SspRk932);
    assert!(fixed_order > 2.9, "fixed observed order was {fixed_order}");

    let adaptive = SolveOptions {
        absolute_tolerance: 1.0e-9,
        relative_tolerance: 1.0e-9,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let endpoint = solve(&exponential(), SspRk932, &adaptive)
        .unwrap()
        .last_state()[0];
    assert!((endpoint - std::f64::consts::E).abs() < 2.0e-8);

    let backward = OdeProblem::new(
        (|du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0]) as TestRhs,
        vec![std::f64::consts::E],
        (1.0, 0.0),
        (),
    );
    let save_at_options = SolveOptions {
        save_at: vec![0.8, 0.5, 0.2],
        ..adaptive.clone()
    };
    let saved = solve(&backward, SspRk932, &save_at_options).unwrap();
    assert_eq!(saved.times(), &[0.8, 0.5, 0.2]);
    assert!((saved.last_state()[0] - 0.2f64.exp()).abs() < 2.0e-7);

    let terminating = exponential()
        .with_continuous_callback(|_, _, time| time - 0.5, |_, _, _| CallbackAction::Terminate);
    let terminated = solve(&terminating, SspRk932, &adaptive).unwrap();
    assert!((terminated.times().last().unwrap() - 0.5).abs() < 1.0e-12);
    assert_eq!(terminated.stats().callback_invocations, 1);
}

#[test]
fn ruuth_third_order_methods_converge_at_order_three() {
    for order in [
        observed_order(SspRk53),
        observed_order(SspRk63),
        observed_order(SspRk73),
        observed_order(SspRk83),
    ] {
        assert!(order > 2.9, "observed order was {order}");
    }
}

#[test]
fn low_storage_variants_converge_at_order_three() {
    for order in [
        observed_order(SspRk53TwoN1),
        observed_order(SspRk53TwoN2),
        observed_order(SspRk53H),
    ] {
        assert!(order > 2.9, "observed order was {order}");
    }
}

#[test]
fn fourth_order_methods_converge_at_order_four() {
    for order in [observed_order(SspRk54), observed_order(SspRk104)] {
        assert!(order > 3.85, "observed order was {order}");
    }
}

#[test]
fn fixed_step_methods_reject_adaptive_stepping() {
    assert_eq!(
        solve(&exponential(), SspRk104, &SolveOptions::default()),
        Err(SolveError::AdaptiveStepUnsupported)
    );
}

#[test]
fn positive_linear_decay_remains_nonnegative_at_ssp_steps() {
    fn decay(du: &mut [f64], u: &[f64], _: &(), _: f64) {
        du[0] = -u[0];
    }
    let problem = OdeProblem::new(decay as TestRhs, vec![1.0], (0.0, 6.0), ());
    for value in [
        solve(&problem, SspRk53, &fixed(0.5)).unwrap().last_state()[0],
        solve(&problem, SspRk54, &fixed(0.5)).unwrap().last_state()[0],
        solve(&problem, SspRk104, &fixed(0.5)).unwrap().last_state()[0],
    ] {
        assert!(value >= 0.0);
    }
}

#[test]
fn shared_output_and_callback_features_work_for_extended_methods() {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
        du[0] = u[0];
    }
    let backward = OdeProblem::new(rhs as TestRhs, vec![std::f64::consts::E], (1.0, 0.0), ());
    let backward_options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.01),
        save_at: vec![1.0, 0.5, 0.0],
        ..SolveOptions::default()
    };
    let solution = solve(&backward, SspRk104, &backward_options).unwrap();
    assert_eq!(solution.times(), &[1.0, 0.5, 0.0]);
    assert!((solution.last_state()[0] - 1.0).abs() < 1.0e-10);

    let terminating = exponential()
        .with_continuous_callback(|_, _, time| time - 0.5, |_, _, _| CallbackAction::Terminate);
    let solution = solve(&terminating, SspRk53, &fixed(0.1)).unwrap();
    assert!((solution.times().last().unwrap() - 0.5).abs() < 1.0e-14);
    assert_eq!(solution.stats().callback_invocations, 1);

    let adaptive_backward =
        OdeProblem::new(rhs as TestRhs, vec![std::f64::consts::E], (1.0, 0.0), ());
    let adaptive_options = SolveOptions {
        absolute_tolerance: 1.0e-8,
        relative_tolerance: 1.0e-8,
        save_at: vec![0.8, 0.5, 0.2],
        ..SolveOptions::default()
    };
    let adaptive_solution = solve(&adaptive_backward, SspRk432, &adaptive_options).unwrap();
    assert_eq!(adaptive_solution.times(), &[0.8, 0.5, 0.2]);
    assert!((adaptive_solution.last_state()[0] - 0.2f64.exp()).abs() < 2.0e-7);

    let terminating = exponential()
        .with_continuous_callback(|_, _, time| time - 0.5, |_, _, _| CallbackAction::Terminate);
    let callback_options = SolveOptions {
        save: SaveMode::Endpoints,
        save_at: Vec::new(),
        ..adaptive_options.clone()
    };
    let adaptive_solution = solve(&terminating, SspRk432, &callback_options).unwrap();
    assert!((adaptive_solution.times().last().unwrap() - 0.5).abs() < 1.0e-12);
    assert_eq!(adaptive_solution.stats().callback_invocations, 1);
}
