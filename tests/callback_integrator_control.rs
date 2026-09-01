use std::cell::Cell;

use differential_equations::solvers::explicit::split_euler::{SplitEuler, solve_split};
use differential_equations::solvers::explicit::{Rk4, Tsit5};
use differential_equations::solvers::multirate::MRIGARKERK22a;
use differential_equations::solvers::multistep::IMEXEuler;
use differential_equations::solvers::second_order::{
    Dprkn4, Irkn3, NewmarkBeta, Nystrom4, PseudoVerletLeapfrog, SecondOrderOdeAlgorithm,
    SecondOrderOdeProblem, VelocityVerlet, solve_second_order, solve_symplectic,
};
use differential_equations::solvers::stabilized::IRKC;
use differential_equations::{
    CallbackAction, OdeProblem, SaveMode, SolveError, SolveOptions, SplitOdeProblem, solve,
};

type SplitRhs = fn(&mut [f64], &[f64], &(), f64);
type Acceleration = fn(&mut [f64], &[f64], &[f64], &(), f64);

fn options(step: f64, adaptive: bool) -> SolveOptions {
    SolveOptions::new()
        .with_adaptive(adaptive)
        .with_initial_step(step)
        .with_max_step(1.0)
        .with_save(SaveMode::EveryStep)
}

fn has_step_after(times: &[f64], event_time: f64, requested_step: f64) -> bool {
    times.windows(2).any(|window| {
        (window[0] - event_time).abs() < 1.0e-12
            && (window[1] - window[0] - requested_step).abs() < 1.0e-12
    })
}

#[derive(Default)]
struct MutableRate {
    value: Cell<f64>,
}

#[test]
fn callback_mutates_parameters_and_overrides_the_fixed_step() {
    let problem = OdeProblem::new(
        |derivative: &mut [f64], _: &[f64], rate: &MutableRate, _: f64| {
            derivative[0] = rate.value.get();
        },
        [0.0],
        (0.0, 1.0),
        MutableRate::default(),
    )
    .with_preset_time_callback([0.5], |_, rate, _| {
        rate.value.set(2.0);
        CallbackAction::ContinueWithStepSize(0.25)
    });

    let solution = solve(&problem, Tsit5, &options(0.5, false)).unwrap();

    assert!((solution.last_state()[0] - 1.0).abs() < 1.0e-12);
    assert!(has_step_after(solution.times(), 0.5, 0.25));
    assert_eq!(problem.parameters().value.get(), 2.0);
}

#[test]
fn callback_step_requests_override_the_adaptive_controller() {
    let problem = OdeProblem::new(
        |derivative: &mut [f64], _: &[f64], _: &(), _: f64| derivative[0] = 0.0,
        [0.0],
        (0.0, 1.0),
        (),
    )
    .with_preset_time_callback([0.5], |_, _, _| CallbackAction::ContinueWithStepSize(0.25));

    let solution = solve(&problem, Tsit5, &options(0.5, true)).unwrap();

    assert!(has_step_after(solution.times(), 0.5, 0.25));
}

#[test]
fn unmodified_continuous_events_still_invalidate_truncated_endpoint_caches() {
    let problem = OdeProblem::new(
        |derivative: &mut [f64], _: &[f64], _: &(), time| derivative[0] = time,
        [0.0],
        (0.0, 1.0),
        (),
    )
    .with_continuous_callback(
        |_, _, time| time - 0.5,
        |_, _, _| CallbackAction::ContinueUnmodified,
    );

    let solution = solve(&problem, Tsit5, &options(0.75, false)).unwrap();

    assert!((solution.last_state()[0] - 0.5).abs() < 1.0e-12);
    assert_eq!(solution.stats().callback_invocations, 1);
}

#[test]
fn callback_step_requests_follow_direction_and_maximum_bounds() {
    let backward = OdeProblem::new(
        |derivative: &mut [f64], _: &[f64], _: &(), _: f64| derivative[0] = 0.0,
        [0.0],
        (1.0, 0.0),
        (),
    )
    .with_preset_time_callback([0.75], |_, _, _| CallbackAction::ContinueWithStepSize(0.2));
    let backward_solution = solve(&backward, Rk4, &options(0.25, false)).unwrap();
    assert!(has_step_after(backward_solution.times(), 0.75, -0.2));

    let bounded = OdeProblem::new(
        |derivative: &mut [f64], _: &[f64], _: &(), _: f64| derivative[0] = 0.0,
        [0.0],
        (0.0, 1.0),
        (),
    )
    .with_preset_time_callback([0.25], |_, _, _| CallbackAction::ContinueWithStepSize(2.0));
    let bounded_solution = solve(&bounded, Rk4, &options(0.25, false).with_max_step(0.4)).unwrap();
    assert!(has_step_after(bounded_solution.times(), 0.25, 0.4));
}

#[test]
fn the_last_simultaneous_step_request_wins_and_initial_requests_apply_immediately() {
    let problem = OdeProblem::new(
        |derivative: &mut [f64], _: &[f64], _: &(), _: f64| derivative[0] = 0.0,
        [0.0],
        (0.0, 1.0),
        (),
    )
    .with_preset_time_callback([0.0], |_, _, _| CallbackAction::ContinueWithStepSize(0.1))
    .with_preset_time_callback([0.4], |_, _, _| CallbackAction::ContinueWithStepSize(0.2))
    .with_preset_time_callback([0.4], |_, _, _| CallbackAction::ContinueWithStepSize(0.3));

    let solution = solve(&problem, Rk4, &options(0.5, false)).unwrap();

    assert!(has_step_after(solution.times(), 0.0, 0.1));
    assert!(has_step_after(solution.times(), 0.4, 0.3));
}

#[test]
fn invalid_callback_step_requests_are_typed_errors() {
    for step in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let problem = OdeProblem::new(
            |derivative: &mut [f64], _: &[f64], _: &(), _: f64| derivative[0] = 0.0,
            [0.0],
            (0.0, 1.0),
            (),
        )
        .with_preset_time_callback([0.0], move |_, _, _| {
            CallbackAction::ContinueWithStepSize(step)
        });

        assert_eq!(
            solve(&problem, Rk4, &options(0.5, false)),
            Err(SolveError::InvalidCallbackStepSize)
        );
    }
}

fn zero(derivative: &mut [f64], _: &[f64], _: &(), _: f64) {
    derivative.fill(0.0);
}

fn split_problem() -> SplitOdeProblem<SplitRhs, SplitRhs, ()> {
    SplitOdeProblem::new(zero as SplitRhs, zero as SplitRhs, [0.0], (0.0, 1.0), ())
        .with_preset_time_callback([0.25], |_, _, _| CallbackAction::ContinueWithStepSize(0.2))
}

#[test]
fn every_split_driver_honors_callback_step_requests() {
    let fixed = options(0.25, false);
    let solutions = [
        solve_split(&split_problem(), SplitEuler, &fixed).unwrap(),
        solve_split(&split_problem(), MRIGARKERK22a::new(4), &fixed).unwrap(),
        solve_split(&split_problem(), IMEXEuler, &fixed).unwrap(),
        solve_split(&split_problem(), IRKC::default(), &fixed).unwrap(),
    ];

    for solution in solutions {
        assert!(has_step_after(solution.times(), 0.25, 0.2));
    }
}

fn zero_acceleration(acceleration: &mut [f64], _: &[f64], _: &[f64], _: &(), _: f64) {
    acceleration.fill(0.0);
}

fn second_order_problem() -> SecondOrderOdeProblem<Acceleration, ()> {
    SecondOrderOdeProblem::new(
        zero_acceleration as Acceleration,
        [0.0],
        [0.0],
        (0.0, 1.0),
        (),
    )
    .with_preset_time_callback([0.25], |_, _, _, _| {
        CallbackAction::ContinueWithStepSize(0.2)
    })
}

fn assert_second_order_step_request<A: SecondOrderOdeAlgorithm>(algorithm: A, adaptive: bool) {
    let solution =
        solve_second_order(&second_order_problem(), algorithm, &options(0.25, adaptive)).unwrap();
    assert!(has_step_after(solution.times(), 0.25, 0.2));
}

#[test]
fn every_second_order_driver_honors_callback_step_requests() {
    assert_second_order_step_request(NewmarkBeta::default(), false);
    assert_second_order_step_request(Nystrom4, false);
    assert_second_order_step_request(Dprkn4, true);
    assert_second_order_step_request(Irkn3, false);
    assert_second_order_step_request(VelocityVerlet, false);

    let solution = solve_symplectic(
        &second_order_problem(),
        PseudoVerletLeapfrog,
        &options(0.25, false),
    )
    .unwrap();
    assert!(has_step_after(solution.times(), 0.25, 0.2));
}
