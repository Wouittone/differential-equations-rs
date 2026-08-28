use differential_equations::solvers::rosenbrock::*;
use differential_equations::{
    CallbackAction, OdeAlgorithm, OdeProblem, SaveMode, Solution, SolveOptions, solve,
};

type Rhs = fn(&mut [f64], &[f64], &(), f64);

fn exponential(du: &mut [f64], u: &[f64], _: &(), _: f64) {
    du[0] = u[0];
}

fn problem(initial: f64, span: (f64, f64)) -> OdeProblem<Rhs, ()> {
    OdeProblem::new(exponential as Rhs, vec![initial], span, ())
        .with_jacobian(|jacobian: &mut [f64], _: &[f64], _: &(), _: f64| jacobian[0] = 1.0)
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

fn assert_julia_samples<A: OdeAlgorithm + Copy>(algorithm: A, expected: [f64; 3]) {
    assert_julia_samples_with_step(algorithm, 1.0, expected);
}

fn assert_julia_samples_with_step<A: OdeAlgorithm + Copy>(
    algorithm: A,
    step: f64,
    expected: [f64; 3],
) {
    let solution = solve(
        &problem(1.0, (0.0, 1.0)),
        algorithm,
        &SolveOptions {
            adaptive: false,
            initial_step: Some(step),
            save_at: vec![0.2, 0.55, 0.9],
            ..SolveOptions::default()
        },
    )
    .unwrap();
    let retained = retained(algorithm, 1.0, (0.0, 1.0), step);
    for ((index, state), expected) in solution.values().chunks_exact(1).enumerate().zip(expected) {
        assert!(
            (state[0] - expected).abs() < 2.0e-9,
            "{} != {expected}",
            state[0]
        );
        assert_eq!(
            solution.state(index).unwrap(),
            retained.interpolate(solution.times()[index]).unwrap()
        );
    }
}

#[test]
fn method_specific_extensions_match_pinned_julia_samples() {
    assert_julia_samples(
        Rosenbrock23,
        [1.2056854249492381, 1.7581349186104047, 2.555584412271571],
    );
    assert_julia_samples(
        Rosenbrock32,
        [1.2056854249492381, 1.7581349186104047, 2.555584412271571],
    );
    assert_julia_samples(
        Rodas4,
        [1.2223065548277507, 1.7391385379449833, 2.4640010504603405],
    );
    assert_julia_samples(
        Rodas42,
        [1.2392812428729862, 1.7475844645860965, 2.463095029748041],
    );
    assert_julia_samples(
        Rodas4P,
        [1.221685790365529, 1.726319535221495, 2.454614379084964],
    );
    assert_julia_samples(
        Rodas4P2,
        [1.2232671935197534, 1.732844725762429, 2.4584704329313953],
    );
    assert_julia_samples(
        Rodas4PW,
        [1.22104930807622, 1.7293490093072716, 2.457175466256193],
    );
    assert_julia_samples(
        Rodas5,
        [1.221398669841696, 1.733705675156384, 2.459950411435234],
    );
    let rodas5p = [1.2214034175316406, 1.7331826623028603, 2.4594699855442346];
    assert_julia_samples(Rodas5P, rodas5p);
    assert_julia_samples(Rodas5Pe, rodas5p);
    assert_julia_samples(Rodas5Pr, rodas5p);
    assert_julia_samples_with_step(
        Rodas6P,
        0.05,
        [1.221402758160135, 1.7332530178672585, 2.4596031111566306],
    );
    assert_julia_samples_with_step(
        Rodas23W,
        0.05,
        [1.2214252392756328, 1.733340750389209, 2.4598068390510974],
    );
    assert_julia_samples_with_step(
        Rodas3P,
        0.05,
        [1.2214022274913472, 1.7332509469671944, 2.4595983022996055],
    );
    assert_julia_samples(
        Tsit5DA,
        [1.221229731357594, 1.732618106157463, 2.4593928127656737],
    );
}

fn assert_queryable<A: OdeAlgorithm + Copy>(algorithm: A) {
    let forward = retained(algorithm, 1.0, (0.0, 0.2), 0.1);
    assert_eq!(forward.interpolate(0.0).unwrap(), forward.state(0).unwrap());
    assert_eq!(forward.interpolate(0.2).unwrap(), forward.state(1).unwrap());
    assert!(forward.interpolate(0.075).unwrap()[0].is_finite());
    let backward = retained(algorithm, 0.2_f64.exp(), (0.2, 0.0), 0.1);
    assert_eq!(
        backward.interpolate(0.2).unwrap(),
        backward.state(0).unwrap()
    );
    assert_eq!(
        backward.interpolate(0.0).unwrap(),
        backward.state(1).unwrap()
    );
    assert!(backward.interpolate(0.075).unwrap()[0].is_finite());
}

#[test]
fn every_native_rosenbrock_dispatch_retains_forward_and_backward_segments() {
    macro_rules! check {
        ($($algorithm:expr),+ $(,)?) => { $(assert_queryable($algorithm);)+ };
    }
    check!(
        Rosenbrock23,
        Rosenbrock32,
        Ros2,
        Rodas3,
        Rodas3d,
        Ros3,
        Ros3Pr,
        Ros3Prl,
        Ros3Prl2,
        Ros3p,
        Ros34Prw,
        Ros34Pw3,
        Grk4a,
        Grk4t,
        Rok4a,
        Ros34Pw1b,
        Ros34Pw2,
        Rodas4,
        Rodas42,
        Rodas4P,
        Rodas4P2,
        Rodas4PW,
        Rodas5,
        Rodas5P,
        Rodas5Pe,
        Rodas5Pr,
        Rodas6P,
        RosenbrockW6S4OS,
        Rodas23W,
        Rodas3P,
        Ros2Pr,
        Ros2S,
        Ros34Pw1a,
        Ros4LStab,
        RosShamp4,
        Scholz4_7,
        Veldd4,
        Velds4,
        Tsit5DA,
    );
}

fn interpolation_error<A: OdeAlgorithm>(algorithm: A, step: f64) -> f64 {
    let solution = retained(algorithm, 1.0, (0.0, step), step);
    let time = 0.37 * step;
    (solution.interpolate(time).unwrap()[0] - time.exp()).abs()
}

#[test]
fn representative_dense_dispatches_converge_under_refinement() {
    for (coarse, fine) in [
        (
            interpolation_error(Rosenbrock23, 0.8),
            interpolation_error(Rosenbrock23, 0.4),
        ),
        (
            interpolation_error(Rodas4, 0.8),
            interpolation_error(Rodas4, 0.4),
        ),
        (
            interpolation_error(Rodas5P, 0.8),
            interpolation_error(Rodas5P, 0.4),
        ),
        (
            interpolation_error(Rodas6P, 0.8),
            interpolation_error(Rodas6P, 0.4),
        ),
        (
            interpolation_error(Ros2, 0.4),
            interpolation_error(Ros2, 0.2),
        ),
    ] {
        assert!(
            fine < coarse,
            "dense error did not decrease: {coarse:e} -> {fine:e}"
        );
    }
}

fn assert_root<A: OdeAlgorithm>(algorithm: A, expected: f64, tolerance: f64) {
    assert_root_with_step(algorithm, 0.25, expected, tolerance);
}

fn assert_root_with_step<A: OdeAlgorithm>(algorithm: A, step: f64, expected: f64, tolerance: f64) {
    let event_problem = problem(1.0, (0.0, 1.0)).with_continuous_callback(
        |state, _: &(), _: f64| state[0] - 1.8,
        |_: &mut [f64], _: &(), _: f64| CallbackAction::Terminate,
    );
    let solution = solve(
        &event_problem,
        algorithm,
        &SolveOptions {
            adaptive: false,
            initial_step: Some(step),
            event_tolerance: 1.0e-13,
            retain_dense_output: true,
            ..SolveOptions::default()
        },
    )
    .unwrap();
    let event_time = *solution.times().last().unwrap();
    assert!(
        (event_time - expected).abs() < tolerance,
        "event time {event_time}, expected {expected}"
    );
    assert!((solution.interpolate(event_time).unwrap()[0] - 1.8).abs() < 2.0e-12);
}

#[test]
fn continuous_roots_match_pinned_julia_for_each_dispatch_kind() {
    assert_root(Rosenbrock23, 0.5865779168719102, 8.0e-11);
    assert_root(Rodas4, 0.5877877517967383, 8.0e-11);
    assert_root(Rodas5P, 0.587786630367156, 8.0e-11);
    assert_root_with_step(Rodas6P, 0.05, 0.5877866649019579, 8.0e-11);
    assert_root(Ros2, 0.7767698682101478, 8.0e-11);
}

#[test]
fn callback_discontinuity_keeps_left_segment_and_right_state() {
    let event_problem = problem(1.0, (0.0, 1.0)).with_continuous_callback(
        |state, _: &(), _: f64| state[0] - 1.8,
        |state: &mut [f64], _: &(), _: f64| {
            state[0] = 10.0;
            CallbackAction::Continue
        },
    );
    let solution = solve(
        &event_problem,
        Rodas4,
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
    assert!(solution.interpolate(0.5).unwrap()[0] < 2.0);
    assert!(solution.interpolate(0.8).unwrap()[0] > 9.0);
}

fn rhs_count<A: OdeAlgorithm>(algorithm: A, dense: bool) -> usize {
    solve(
        &problem(1.0, (0.0, 0.5)),
        algorithm,
        &SolveOptions {
            adaptive: false,
            initial_step: Some(0.1),
            save: SaveMode::Endpoints,
            retain_dense_output: dense,
            ..SolveOptions::default()
        },
    )
    .unwrap()
    .stats()
    .rhs_evaluations
}

#[test]
fn dense_service_adds_no_rhs_evaluations() {
    assert_eq!(
        rhs_count(Rosenbrock23, false),
        rhs_count(Rosenbrock23, true)
    );
    assert_eq!(rhs_count(Rodas4, false), rhs_count(Rodas4, true));
    assert_eq!(rhs_count(Ros2, false), rhs_count(Ros2, true));
}
