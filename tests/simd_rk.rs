use differential_equations::algorithms::simd::{MER5v2, MER6v2, RK6v4};
use differential_equations::{
    CallbackAction, OdeAlgorithm, OdeProblem, SaveMode, SolveOptions, solve,
};

type Rhs = fn(&mut [f64], &[f64], &(), f64);

fn exponential(derivative: &mut [f64], state: &[f64], _: &(), _: f64) {
    derivative[0] = state[0];
}

fn endpoint<A: OdeAlgorithm>(algorithm: A, step: f64) -> f64 {
    let problem = OdeProblem::new(exponential as Rhs, vec![1.0], (0.0, 1.0), ());
    solve(
        &problem,
        algorithm,
        &SolveOptions {
            adaptive: false,
            initial_step: Some(step),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        },
    )
    .unwrap()
    .last_state()[0]
}

fn error<A: OdeAlgorithm>(algorithm: A, step: f64) -> f64 {
    (endpoint(algorithm, step) - 1.0_f64.exp()).abs()
}

#[test]
fn pinned_simd_tableaus_recover_their_design_orders() {
    let mer5_ratio = error(MER5v2, 0.2) / error(MER5v2, 0.1);
    let mer6_ratio = error(MER6v2, 0.2) / error(MER6v2, 0.1);
    let rk6_ratio = error(RK6v4, 0.2) / error(RK6v4, 0.1);
    assert!(mer5_ratio > 20.0, "MER5v2 ratio {mer5_ratio}");
    assert!(mer6_ratio > 40.0, "MER6v2 ratio {mer6_ratio}");
    assert!(rk6_ratio > 40.0, "RK6v4 ratio {rk6_ratio}");
}

#[test]
fn adaptive_control_and_stage_metadata_are_method_specific() {
    assert_eq!(MER5v2.order(), 5);
    assert_eq!(MER5v2.stage_count(), 14);
    assert_eq!(MER6v2.order(), 6);
    assert_eq!(MER6v2.stage_count(), 15);
    assert_eq!(RK6v4.order(), 6);
    assert_eq!(RK6v4.stage_count(), 22);

    let problem = OdeProblem::new(exponential as Rhs, vec![1.0], (0.0, 1.0), ());
    for solution in [
        solve(&problem, MER5v2, &SolveOptions::default()).unwrap(),
        solve(&problem, MER6v2, &SolveOptions::default()).unwrap(),
        solve(&problem, RK6v4, &SolveOptions::default()).unwrap(),
    ] {
        assert!((solution.last_state()[0] - 1.0_f64.exp()).abs() < 2.0e-5);
        assert!(solution.stats().accepted_steps > 0);
    }
}

#[test]
fn shared_dense_callbacks_and_backward_time_remain_available() {
    let problem = OdeProblem::new(exponential as Rhs, vec![1.0], (0.0, 1.0), ())
        .with_continuous_callback(
            |state, _: &(), _: f64| state[0] - 1.5,
            |_: &mut [f64], _: &(), _: f64| CallbackAction::Terminate,
        );
    let solution = solve(
        &problem,
        MER6v2,
        &SolveOptions {
            adaptive: false,
            initial_step: Some(0.1),
            retain_dense_output: true,
            ..SolveOptions::default()
        },
    )
    .unwrap();
    let event_time = *solution.times().last().unwrap();
    assert!((event_time - 1.5_f64.ln()).abs() < 2.0e-5);
    assert_eq!(solution.interpolate(event_time).unwrap(), vec![1.5]);

    let backward_problem = OdeProblem::new(exponential as Rhs, vec![1.0_f64.exp()], (1.0, 0.0), ());
    let backward = solve(
        &backward_problem,
        RK6v4,
        &SolveOptions {
            adaptive: false,
            initial_step: Some(0.1),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        },
    )
    .unwrap();
    assert!((backward.last_state()[0] - 1.0).abs() < 2.0e-8);
}
