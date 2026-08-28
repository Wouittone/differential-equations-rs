use differential_equations::solvers::explicit::{Dp5, OwrenZen3, OwrenZen4, OwrenZen5};
use differential_equations::{
    CallbackAction, OdeAlgorithm, OdeProblem, SaveMode, SolveOptions, solve,
};

type Rhs = fn(&mut [f64], &[f64], &(), f64);

fn exponential(derivative: &mut [f64], state: &[f64], _: &(), _: f64) {
    derivative[0] = state[0];
}

fn problem(initial: f64, span: (f64, f64)) -> OdeProblem<Rhs, ()> {
    OdeProblem::new(exponential as Rhs, vec![initial], span, ())
}

fn retained_with_step<A: OdeAlgorithm>(
    algorithm: A,
    initial: f64,
    span: (f64, f64),
    step: f64,
) -> differential_equations::Solution {
    solve(
        &problem(initial, span),
        algorithm,
        &SolveOptions {
            adaptive: false,
            initial_step: Some(step),
            save: SaveMode::Endpoints,
            retain_dense_output: true,
            ..SolveOptions::default()
        },
    )
    .unwrap()
}

fn retained_one_step<A: OdeAlgorithm>(
    algorithm: A,
    initial: f64,
    span: (f64, f64),
) -> differential_equations::Solution {
    retained_with_step(algorithm, initial, span, (span.1 - span.0).abs())
}

fn interpolation_error<A: OdeAlgorithm>(algorithm: A, step: f64) -> f64 {
    let solution = retained_one_step(algorithm, 1.0, (0.0, step));
    let sample = 0.37 * step;
    (solution.interpolate(sample).unwrap()[0] - sample.exp()).abs()
}

fn assert_dense_order<A: OdeAlgorithm + Copy>(algorithm: A, minimum_ratio: f64) {
    let coarse = interpolation_error(algorithm, 0.4);
    let fine = interpolation_error(algorithm, 0.2);
    assert!(
        coarse / fine > minimum_ratio,
        "dense error ratio {} <= {minimum_ratio} ({coarse:e}, {fine:e})",
        coarse / fine
    );
}

#[test]
fn free_continuous_extensions_converge_at_their_method_orders() {
    // Dense order p has one-step interpolation error O(h^(p+1)). These lower
    // bounds leave room for non-asymptotic terms while distinguishing the
    // continuous extensions from cubic Hermite or endpoint-linear sampling.
    assert_dense_order(Dp5, 20.0);
    assert_dense_order(OwrenZen3, 8.0);
    assert_dense_order(OwrenZen4, 16.0);
    assert_dense_order(OwrenZen5, 25.0);
}

fn assert_endpoints_and_direction<A: OdeAlgorithm + Copy>(algorithm: A) {
    let forward = retained_with_step(algorithm, 1.0, (0.0, 1.0), 0.25);
    assert_eq!(forward.interpolate(0.0).unwrap(), forward.state(0).unwrap());
    assert_eq!(forward.interpolate(1.0).unwrap(), forward.state(1).unwrap());

    let backward = retained_with_step(algorithm, 1.0_f64.exp(), (1.0, 0.0), 0.25);
    assert_eq!(
        backward.interpolate(1.0).unwrap(),
        backward.state(0).unwrap()
    );
    assert_eq!(
        backward.interpolate(0.0).unwrap(),
        backward.state(1).unwrap()
    );
    let backward_sample = backward.interpolate(0.37).unwrap()[0];
    assert!(
        (backward_sample - 0.37_f64.exp()).abs() < 1.0e-2,
        "backward sample {backward_sample}"
    );
}

#[test]
fn method_specific_segments_agree_at_endpoints_forward_and_backward() {
    assert_endpoints_and_direction(Dp5);
    assert_endpoints_and_direction(OwrenZen3);
    assert_endpoints_and_direction(OwrenZen4);
    assert_endpoints_and_direction(OwrenZen5);
}

fn assert_save_at_matches_retained_query<A: OdeAlgorithm + Copy>(algorithm: A) {
    let retained = retained_one_step(algorithm, 1.0, (0.0, 1.0));
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(1.0),
        save_at: vec![0.2, 0.55, 0.9],
        ..SolveOptions::default()
    };
    let sampled = solve(&problem(1.0, (0.0, 1.0)), algorithm, &options).unwrap();
    for (index, &time) in sampled.times().iter().enumerate() {
        assert_eq!(
            sampled.state(index).unwrap(),
            retained.interpolate(time).unwrap()
        );
    }
}

#[test]
fn save_at_and_post_solve_queries_share_each_continuous_extension() {
    assert_save_at_matches_retained_query(Dp5);
    assert_save_at_matches_retained_query(OwrenZen3);
    assert_save_at_matches_retained_query(OwrenZen4);
    assert_save_at_matches_retained_query(OwrenZen5);
}

fn assert_continuous_root_uses_dense_segment<A: OdeAlgorithm + Copy>(algorithm: A) {
    let event_problem = problem(1.0, (0.0, 1.0)).with_continuous_callback(
        |state, _: &(), _: f64| state[0] - 1.8,
        |_: &mut [f64], _: &(), _: f64| CallbackAction::Terminate,
    );
    let solution = solve(
        &event_problem,
        algorithm,
        &SolveOptions {
            adaptive: false,
            initial_step: Some(0.25),
            event_tolerance: 1.0e-13,
            retain_dense_output: true,
            ..SolveOptions::default()
        },
    )
    .unwrap();
    let event_time = *solution.times().last().unwrap();
    assert!(
        (event_time - 1.8_f64.ln()).abs() < 1.0e-2,
        "event time {event_time}"
    );
    assert!((solution.state(solution.times().len() - 1).unwrap()[0] - 1.8).abs() < 2.0e-12);
    assert!((solution.interpolate(event_time).unwrap()[0] - 1.8).abs() < 2.0e-12);
}

#[test]
fn continuous_event_localization_uses_method_specific_segments() {
    assert_continuous_root_uses_dense_segment(Dp5);
    assert_continuous_root_uses_dense_segment(OwrenZen3);
    assert_continuous_root_uses_dense_segment(OwrenZen4);
    assert_continuous_root_uses_dense_segment(OwrenZen5);
}

#[test]
fn callback_discontinuity_keeps_left_segment_and_right_endpoint() {
    let event_problem = problem(1.0, (0.0, 1.0)).with_continuous_callback(
        |state, _: &(), _: f64| state[0] - 1.8,
        |state: &mut [f64], _: &(), _: f64| {
            state[0] = 10.0;
            CallbackAction::Continue
        },
    );
    let solution = solve(
        &event_problem,
        Dp5,
        &SolveOptions {
            adaptive: false,
            initial_step: Some(1.0),
            save: SaveMode::Endpoints,
            event_tolerance: 1.0e-13,
            retain_dense_output: true,
            ..SolveOptions::default()
        },
    )
    .unwrap();
    let event_time = solution.times()[1];
    assert_eq!(solution.interpolate(event_time).unwrap(), vec![10.0]);
    assert!((solution.interpolate(0.5).unwrap()[0] - 0.5_f64.exp()).abs() < 2.0e-4);
    assert!(solution.interpolate(0.75).unwrap()[0] > 10.0);
}

fn assert_dense_sampling_has_no_rhs_cost<A: OdeAlgorithm + Copy>(algorithm: A) {
    let plain = solve(
        &problem(1.0, (0.0, 1.0)),
        algorithm,
        &SolveOptions {
            adaptive: false,
            initial_step: Some(1.0),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        },
    )
    .unwrap();
    let sampled = solve(
        &problem(1.0, (0.0, 1.0)),
        algorithm,
        &SolveOptions {
            adaptive: false,
            initial_step: Some(1.0),
            save_at: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            retain_dense_output: true,
            ..SolveOptions::default()
        },
    )
    .unwrap();
    assert_eq!(
        sampled.stats().rhs_evaluations,
        plain.stats().rhs_evaluations
    );
}

#[test]
fn free_dense_extensions_add_no_rhs_evaluations() {
    assert_dense_sampling_has_no_rhs_cost(Dp5);
    assert_dense_sampling_has_no_rhs_cost(OwrenZen3);
    assert_dense_sampling_has_no_rhs_cost(OwrenZen4);
    assert_dense_sampling_has_no_rhs_cost(OwrenZen5);
}
