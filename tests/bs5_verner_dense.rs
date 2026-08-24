use differential_equations::algorithms::explicit::general::Bs5;
use differential_equations::algorithms::explicit::high_order::{Vern6, Vern7, Vern8, Vern9};
use differential_equations::{
    CallbackAction, OdeAlgorithm, OdeProblem, SaveMode, Solution, SolveOptions, solve,
};

type Rhs = fn(&mut [f64], &[f64], &(), f64);

fn exponential(derivative: &mut [f64], state: &[f64], _: &(), _: f64) {
    derivative[0] = state[0];
}

fn problem(initial: f64, span: (f64, f64)) -> OdeProblem<Rhs, ()> {
    OdeProblem::new(exponential as Rhs, vec![initial], span, ())
}

fn retained<A: OdeAlgorithm>(algorithm: A, initial: f64, span: (f64, f64), step: f64) -> Solution {
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

fn interpolation_error<A: OdeAlgorithm>(algorithm: A, step: f64) -> f64 {
    let solution = retained(algorithm, 1.0, (0.0, step), step);
    let sample = 0.37 * step;
    (solution.interpolate(sample).unwrap()[0] - sample.exp()).abs()
}

fn assert_dense_order<A: OdeAlgorithm + Copy>(algorithm: A, minimum_ratio: f64) {
    let coarse = interpolation_error(algorithm, 0.8);
    let fine = interpolation_error(algorithm, 0.4);
    assert!(
        coarse / fine > minimum_ratio,
        "dense error ratio {} <= {minimum_ratio} ({coarse:e}, {fine:e})",
        coarse / fine
    );
}

#[test]
fn continuous_extensions_converge_above_cubic_hermite_order() {
    assert_dense_order(Bs5, 20.0);
    assert_dense_order(Vern6, 35.0);
    assert_dense_order(Vern7, 60.0);
    assert_dense_order(Vern8, 100.0);
    assert_dense_order(Vern9, 150.0);
}

fn assert_endpoints_and_backward<A: OdeAlgorithm + Copy>(algorithm: A) {
    let forward = retained(algorithm, 1.0, (0.0, 1.0), 0.25);
    assert_eq!(forward.interpolate(0.0).unwrap(), forward.state(0).unwrap());
    assert_eq!(forward.interpolate(1.0).unwrap(), forward.state(1).unwrap());

    let backward = retained(algorithm, 1.0_f64.exp(), (1.0, 0.0), 0.25);
    assert_eq!(
        backward.interpolate(1.0).unwrap(),
        backward.state(0).unwrap()
    );
    assert_eq!(
        backward.interpolate(0.0).unwrap(),
        backward.state(1).unwrap()
    );
    assert!((backward.interpolate(0.37).unwrap()[0] - 0.37_f64.exp()).abs() < 2.0e-7);
}

#[test]
fn retained_segments_agree_at_endpoints_and_support_backward_queries() {
    assert_endpoints_and_backward(Bs5);
    assert_endpoints_and_backward(Vern6);
    assert_endpoints_and_backward(Vern7);
    assert_endpoints_and_backward(Vern8);
    assert_endpoints_and_backward(Vern9);
}

fn assert_julia_samples<A: OdeAlgorithm + Copy>(algorithm: A, expected: [f64; 3]) {
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(1.0),
        save_at: vec![0.2, 0.55, 0.9],
        ..SolveOptions::default()
    };
    let sampled = solve(&problem(1.0, (0.0, 1.0)), algorithm, &options).unwrap();
    let retained = retained(algorithm, 1.0, (0.0, 1.0), 1.0);
    for ((index, &time), expected) in sampled.times().iter().enumerate().zip(expected) {
        let value = sampled.state(index).unwrap()[0];
        assert!(
            (value - expected).abs() < 3.0e-10,
            "time={time}, value={value}"
        );
        assert_eq!(
            sampled.state(index).unwrap(),
            retained.interpolate(time).unwrap()
        );
    }
}

#[test]
fn save_at_and_retained_queries_match_pinned_julia_samples() {
    assert_julia_samples(
        Bs5,
        [1.2214004531632976, 1.7332401956162267, 2.4595777580872276],
    );
    assert_julia_samples(
        Vern6,
        [1.2213906976267277, 1.7332688761483586, 2.459584705061208],
    );
    assert_julia_samples(
        Vern7,
        [1.2214027749928802, 1.733252953490458, 2.4596033527459955],
    );
    assert_julia_samples(
        Vern8,
        [1.2214025169262221, 1.733252914957541, 2.4596031147569155],
    );
    assert_julia_samples(
        Vern9,
        [1.2214027544790693, 1.7332529995165806, 2.4596031102113534],
    );
}

fn assert_root<A: OdeAlgorithm>(algorithm: A, expected: f64) {
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
        (event_time - expected).abs() < 7.0e-12,
        "event time {event_time}"
    );
    assert!((solution.last_state()[0] - 1.8).abs() < 2.0e-12);
    assert!((solution.interpolate(event_time).unwrap()[0] - 1.8).abs() < 2.0e-12);
}

#[test]
fn continuous_roots_match_pinned_julia() {
    assert_root(Bs5, 0.5877866641397586);
    assert_root(Vern6, 0.5877866649209265);
    assert_root(Vern7, 0.587786664891878);
    assert_root(Vern8, 0.5877866649024726);
    assert_root(Vern9, 0.5877866649021184);
}

#[test]
fn callback_discontinuity_retains_left_polynomial_and_right_state() {
    let event_problem = problem(1.0, (0.0, 1.0)).with_continuous_callback(
        |state, _: &(), _: f64| state[0] - 1.8,
        |state: &mut [f64], _: &(), _: f64| {
            state[0] = 10.0;
            CallbackAction::Continue
        },
    );
    let solution = solve(
        &event_problem,
        Vern9,
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
    assert!((solution.interpolate(0.5).unwrap()[0] - 0.5_f64.exp()).abs() < 2.0e-7);
    assert!(solution.interpolate(0.8).unwrap()[0] > 10.0);
}

fn assert_lazy_rhs_cost<A: OdeAlgorithm + Copy>(algorithm: A, extra_stages: usize) {
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
    let dense = retained(algorithm, 1.0, (0.0, 1.0), 1.0);
    assert_eq!(
        dense.stats().rhs_evaluations,
        plain.stats().rhs_evaluations + extra_stages
    );
}

#[test]
fn lazy_extensions_only_pay_their_documented_rhs_cost() {
    assert_lazy_rhs_cost(Bs5, 3);
    assert_lazy_rhs_cost(Vern6, 3);
    assert_lazy_rhs_cost(Vern7, 6);
    assert_lazy_rhs_cost(Vern8, 8);
    assert_lazy_rhs_cost(Vern9, 10);
}
