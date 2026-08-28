use differential_equations::solvers::explicit::{
    Kyk2014DgSsprk3S2, KykSsprk42, Prrk22, Prrk33, Prrk54, SspRk22, SspRk33, SspRk43, SspRk53,
    SspRk53H, SspRk53TwoN1, SspRk53TwoN2, SspRk54, SspRk63, SspRk73, SspRk83, SspRk104, SspRk432,
    SspRk932, SspRkMsvs32, SspRkMsvs43,
};
use differential_equations::{
    CallbackAction, OdeAlgorithm, OdeProblem, SaveMode, Solution, SolveOptions, solve,
};

type Rhs = fn(&mut [f64], &[f64], &(), f64);

fn exponential(du: &mut [f64], u: &[f64], _: &(), _: f64) {
    du[0] = u[0];
}

fn exponential_problem(initial: f64, span: (f64, f64)) -> OdeProblem<Rhs, ()> {
    OdeProblem::new(exponential as Rhs, vec![initial], span, ())
}

fn retained<A: OdeAlgorithm>(algorithm: A, initial: f64, span: (f64, f64), step: f64) -> Solution {
    solve(
        &exponential_problem(initial, span),
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

fn special_error<A: OdeAlgorithm>(algorithm: A, step: f64) -> f64 {
    let solution = retained(algorithm, 1.0, (0.0, step), step);
    let time = 0.37 * step;
    (solution.interpolate(time).unwrap()[0] - time.exp()).abs()
}

fn assert_special_order<A: OdeAlgorithm + Copy>(algorithm: A) {
    let coarse = special_error(algorithm, 0.4);
    let fine = special_error(algorithm, 0.2);
    assert!(
        coarse / fine > 5.0,
        "SSP quadratic extension ratio {} ({coarse:e}, {fine:e})",
        coarse / fine
    );
}

#[test]
fn pinned_special_ssp_extensions_have_second_order_dense_convergence() {
    assert_special_order(SspRk22);
    assert_special_order(SspRk33);
    assert_special_order(SspRk43);
    assert_special_order(SspRk432);
}

fn assert_special_formula<A: OdeAlgorithm + Copy>(algorithm: A) {
    let solution = retained(algorithm, 1.0, (0.0, 1.0), 1.0);
    let endpoint = solution.last_state()[0];
    for theta in [0.2, 0.55, 0.9] {
        let expected = 1.0 + theta + theta * theta * (endpoint - 2.0);
        assert!((solution.interpolate(theta).unwrap()[0] - expected).abs() < 2.0e-14);
    }
    assert_eq!(
        solution.interpolate(0.0).unwrap(),
        solution.state(0).unwrap()
    );
    assert_eq!(
        solution.interpolate(1.0).unwrap(),
        solution.state(1).unwrap()
    );

    let backward = retained(algorithm, 1.0_f64.exp(), (1.0, 0.0), 0.05);
    assert_eq!(
        backward.interpolate(1.0).unwrap(),
        backward.state(0).unwrap()
    );
    assert_eq!(
        backward.interpolate(0.0).unwrap(),
        backward.state(1).unwrap()
    );
    assert!((backward.interpolate(0.37).unwrap()[0] - 0.37_f64.exp()).abs() < 5.0e-4);
}

#[test]
fn special_ssp_segments_match_the_pinned_formula_and_both_directions() {
    assert_special_formula(SspRk22);
    assert_special_formula(SspRk33);
    assert_special_formula(SspRk43);
    assert_special_formula(SspRk432);
}

fn positive_quadratic_root(endpoint: f64) -> f64 {
    let a = endpoint - 2.0;
    (-1.0 + (1.0 + 3.2 * a).sqrt()) / (2.0 * a)
}

fn assert_special_sampling_and_root<A: OdeAlgorithm + Copy>(algorithm: A) {
    let retained = retained(algorithm, 1.0, (0.0, 1.0), 1.0);
    let sampled = solve(
        &exponential_problem(1.0, (0.0, 1.0)),
        algorithm,
        &SolveOptions {
            adaptive: false,
            initial_step: Some(1.0),
            save_at: vec![0.2, 0.55, 0.9],
            ..SolveOptions::default()
        },
    )
    .unwrap();
    for (index, &time) in sampled.times().iter().enumerate() {
        assert_eq!(
            sampled.state(index).unwrap(),
            retained.interpolate(time).unwrap()
        );
    }

    let event_problem = exponential_problem(1.0, (0.0, 1.0)).with_continuous_callback(
        |state, _: &(), _: f64| state[0] - 1.8,
        |_: &mut [f64], _: &(), _: f64| CallbackAction::Terminate,
    );
    let event = solve(
        &event_problem,
        algorithm,
        &SolveOptions {
            adaptive: false,
            initial_step: Some(1.0),
            event_tolerance: 1.0e-13,
            retain_dense_output: true,
            ..SolveOptions::default()
        },
    )
    .unwrap();
    let expected_root = positive_quadratic_root(retained.last_state()[0]);
    assert!((event.times().last().unwrap() - expected_root).abs() < 1.0e-12);
    assert!((event.last_state()[0] - 1.8).abs() < 2.0e-12);
}

#[test]
fn special_ssp_save_at_roots_and_queries_use_one_segment() {
    assert_special_sampling_and_root(SspRk22);
    assert_special_sampling_and_root(SspRk33);
    assert_special_sampling_and_root(SspRk43);
    assert_special_sampling_and_root(SspRk432);
}

fn quadratic(du: &mut [f64], _: &[f64], _: &(), time: f64) {
    du[0] = 2.0 * time;
}

fn quadratic_problem(initial: f64, span: (f64, f64)) -> OdeProblem<Rhs, ()> {
    OdeProblem::new(quadratic as Rhs, vec![initial], span, ())
}

fn assert_generic_hermite<A: OdeAlgorithm>(algorithm: A) {
    let solution = solve(
        &quadratic_problem(0.0, (0.0, 1.0)),
        algorithm,
        &SolveOptions {
            adaptive: false,
            initial_step: Some(1.0),
            save: SaveMode::Endpoints,
            retain_dense_output: true,
            ..SolveOptions::default()
        },
    )
    .unwrap();
    assert!((solution.interpolate(0.3).unwrap()[0] - 0.09).abs() < 5.0e-13);
    assert!((solution.interpolate(0.7).unwrap()[0] - 0.49).abs() < 5.0e-13);
}

#[test]
fn generic_ssp_dispatches_retain_honest_hermite_segments() {
    assert_generic_hermite(SspRk53);
    assert_generic_hermite(Prrk22::default());
    assert_generic_hermite(SspRkMsvs32);
}

#[test]
fn every_implemented_generic_ssp_type_retains_a_queryable_segment() {
    fn check<A: OdeAlgorithm>(algorithm: A) {
        let solution = retained(algorithm, 1.0, (0.0, 1.0), 0.25);
        assert!(solution.interpolate(0.375).is_some());
    }

    check(SspRk53);
    check(SspRk53H);
    check(SspRk53TwoN1);
    check(SspRk53TwoN2);
    check(SspRk54);
    check(SspRk63);
    check(SspRk73);
    check(SspRk83);
    check(SspRk104);
    check(SspRk932);
    check(KykSsprk42);
    check(Kyk2014DgSsprk3S2);
    check(Prrk22::default());
    check(Prrk33::default());
    check(Prrk54::default());
    check(SspRkMsvs32);
    check(SspRkMsvs43);
}

#[test]
fn generic_hermite_drives_roots_and_preserves_callback_sides() {
    let problem = quadratic_problem(0.0, (0.0, 1.0)).with_continuous_callback(
        |state, _: &(), _: f64| state[0] - 0.36,
        |state: &mut [f64], _: &(), _: f64| {
            state[0] = 10.0;
            CallbackAction::Continue
        },
    );
    let solution = solve(
        &problem,
        SspRk53,
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
    assert!((event_time - 0.6).abs() < 1.0e-12);
    assert_eq!(solution.interpolate(event_time).unwrap(), vec![10.0]);
    assert!((solution.interpolate(0.3).unwrap()[0] - 0.09).abs() < 5.0e-13);
    assert!(solution.interpolate(0.8).unwrap()[0] > 10.0);
}

#[test]
fn special_extensions_are_free_and_generic_endpoint_work_is_not_duplicated() {
    fn counts<A: OdeAlgorithm + Copy>(algorithm: A, dense: bool) -> usize {
        solve(
            &exponential_problem(1.0, (0.0, 1.0)),
            algorithm,
            &SolveOptions {
                adaptive: false,
                initial_step: Some(0.25),
                save: SaveMode::Endpoints,
                save_at: if dense {
                    vec![0.125, 0.375, 0.625, 0.875]
                } else {
                    Vec::new()
                },
                retain_dense_output: dense,
                ..SolveOptions::default()
            },
        )
        .unwrap()
        .stats()
        .rhs_evaluations
    }

    assert_eq!(counts(SspRk22, true), counts(SspRk22, false));
    assert_eq!(counts(SspRk33, true), counts(SspRk33, false));
    assert_eq!(counts(SspRk43, true), counts(SspRk43, false));
    assert_eq!(counts(SspRk432, true), counts(SspRk432, false));
    assert_eq!(counts(SspRkMsvs32, true), counts(SspRkMsvs32, false));

    let plain = counts(SspRk53, false);
    let dense = counts(SspRk53, true);
    assert_eq!(
        dense - plain,
        4,
        "one endpoint derivative per accepted step"
    );
}
