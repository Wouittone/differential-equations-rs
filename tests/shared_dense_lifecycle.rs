use differential_equations::solvers::explicit::low_storage_rk::CarpenterKennedy2N54;
use differential_equations::solvers::extrapolation::AitkenNeville;
use differential_equations::solvers::implicit::firk::RadauIIA5;
use differential_equations::solvers::implicit::general::ImplicitEuler;
use differential_equations::solvers::multistep::Ab3;
use differential_equations::solvers::stabilized::RKC;
use differential_equations::{
    CallbackAction, OdeAlgorithm, OdeProblem, SaveMode, SolveOptions, solve,
};

type Rhs = fn(&mut [f64], &[f64], &(), f64);

fn exponential(derivative: &mut [f64], state: &[f64], _: &(), _: f64) {
    derivative[0] = state[0];
}

fn retained<A: OdeAlgorithm>(algorithm: A) -> differential_equations::Solution {
    let problem = OdeProblem::new(exponential as Rhs, vec![1.0], (0.0, 1.0), ());
    solve(
        &problem,
        algorithm,
        &SolveOptions {
            adaptive: false,
            initial_step: Some(0.05),
            save: SaveMode::Endpoints,
            retain_dense_output: true,
            ..SolveOptions::default()
        },
    )
    .unwrap()
}

fn assert_queryable<A: OdeAlgorithm>(algorithm: A, tolerance: f64) {
    let solution = retained(algorithm);
    assert_eq!(
        solution.interpolate(0.0).unwrap(),
        solution.state(0).unwrap()
    );
    assert_eq!(solution.interpolate(1.0).unwrap(), solution.last_state());
    let sample = solution.interpolate(0.37).unwrap()[0];
    assert!(
        (sample - 0.37_f64.exp()).abs() < tolerance,
        "sample {sample}"
    );
}

#[test]
fn shared_hermite_fallback_covers_representative_step_kernels() {
    assert_queryable(ImplicitEuler, 2.0e-2);
    assert_queryable(Ab3, 2.0e-3);
    assert_queryable(RKC, 3.0e-3);
    assert_queryable(CarpenterKennedy2N54, 2.0e-5);
}

#[test]
fn firk_and_extrapolation_retain_their_own_extensions() {
    assert_queryable(RadauIIA5, 2.0e-6);
    assert_queryable(AitkenNeville::default(), 2.0e-5);
}

#[test]
fn callback_discontinuity_bounds_the_shared_left_segment() {
    let problem = OdeProblem::new(exponential as Rhs, vec![1.0], (0.0, 1.0), ())
        .with_continuous_callback(
            |state, _: &(), _: f64| state[0] - 1.5,
            |state, _: &(), _: f64| {
                state[0] = 10.0;
                CallbackAction::Continue
            },
        );
    let solution = solve(
        &problem,
        ImplicitEuler,
        &SolveOptions {
            adaptive: false,
            initial_step: Some(0.1),
            save: SaveMode::Endpoints,
            retain_dense_output: true,
            event_tolerance: 1.0e-12,
            ..SolveOptions::default()
        },
    )
    .unwrap();
    let event_time = solution.times()[1];
    assert_eq!(solution.interpolate(event_time).unwrap(), vec![10.0]);
    assert!(solution.interpolate(0.2).unwrap()[0] < 1.5);
    assert!(solution.interpolate(event_time + 0.1).unwrap()[0] > 10.0);
}
