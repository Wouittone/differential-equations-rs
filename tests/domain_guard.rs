use std::cell::Cell;
use std::rc::Rc;

use differential_equations::callbacks::DomainGuard;
use differential_equations::ndarray::{ArrayViewD, ArrayViewMutD, arr0, array};
use differential_equations::solvers::explicit::split_euler::{
    SplitEuler, SplitOdeAlgorithm, solve_split,
};
use differential_equations::solvers::explicit::{Euler, Tsit5};
use differential_equations::solvers::multirate::MRIGARKERK22a;
use differential_equations::solvers::multistep::IMEXEuler;
use differential_equations::solvers::second_order::{
    Dprkn4, Irkn3, NewmarkBeta, Nystrom4, PseudoVerletLeapfrog, SecondOrderOdeAlgorithm,
    SecondOrderOdeProblem, VelocityVerlet, solve_second_order, solve_symplectic,
};
use differential_equations::solvers::stabilized::IRKC;
use differential_equations::{
    CallbackAction, CallbackSet, ConfigurationError, OdeProblem, SaveMode, SolveError,
    SolveOptions, SplitOdeProblem, solve,
};

type SplitRhs = fn(&mut [f64], &[f64], &(), f64);
type Acceleration = fn(&mut [f64], &[f64], &[f64], &(), f64);

fn fixed(step: f64) -> SolveOptions {
    SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(step)
        .with_save(SaveMode::EveryStep)
}

fn zero(derivative: &mut [f64], _: &[f64], _: &(), _: f64) {
    derivative.fill(0.0);
}

fn zero_acceleration(acceleration: &mut [f64], _: &[f64], _: &[f64], _: &(), _: f64) {
    acceleration.fill(0.0);
}

fn reject_first_candidate<P>() -> CallbackSet<P> {
    let first = Cell::new(true);
    DomainGuard::new(move |_: &[f64], _: &P, time| time != 0.0 && first.replace(false))
        .into_callback_set()
        .unwrap()
}

#[test]
fn unsafe_candidates_are_retried_before_callbacks_or_saving() {
    let callback_calls = Rc::new(Cell::new(0));
    let calls = Rc::clone(&callback_calls);
    let callbacks = DomainGuard::new(|state: &[f64], _: &(), _| state[0] < 0.0)
        .into_callback_set()
        .unwrap()
        .append(CallbackSet::new().with_discrete_callback(
            |_, _, _| true,
            move |_, _, _| {
                calls.set(calls.get() + 1);
                CallbackAction::ContinueUnmodified
            },
        ));
    let problem = OdeProblem::new(
        |derivative: &mut [f64], state: &[f64], _: &(), _| derivative[0] = -10.0 * state[0],
        [1.0],
        (0.0, 0.2),
        (),
    )
    .with_callback_set(callbacks);

    let solution = solve(&problem, Euler, &fixed(0.2)).unwrap();

    assert_eq!(solution.stats().rejected_steps, 1);
    assert_eq!(solution.stats().accepted_steps, 2);
    assert_eq!(callback_calls.get(), solution.stats().accepted_steps + 1);
    assert!(solution.values().iter().all(|value| *value >= 0.0));
    assert_eq!(solution.times(), &[0.0, 0.1, 0.2]);
}

#[test]
fn guards_compose_with_the_most_conservative_retry_factor() {
    let first_larger_guard = Cell::new(true);
    let first_smaller_guard = Cell::new(true);
    let callbacks = DomainGuard::new(move |_: &[f64], _: &(), time| {
        time != 0.0 && first_larger_guard.replace(false)
    })
    .with_reduction_factor(0.8)
    .into_callback_set()
    .unwrap()
    .append(
        DomainGuard::new(move |_: &[f64], _: &(), time| {
            time != 0.0 && first_smaller_guard.replace(false)
        })
        .with_reduction_factor(0.25)
        .into_callback_set()
        .unwrap(),
    );
    let problem = OdeProblem::new(zero, [1.0], (0.0, 1.0), ()).with_callback_set(callbacks);
    let solution = solve(&problem, Euler, &fixed(1.0)).unwrap();

    assert_eq!(solution.stats().rejected_steps, 1);
    assert_eq!(solution.times()[1], 0.25);
}

#[test]
fn initialized_domain_failures_and_invalid_factors_are_typed() {
    let callbacks = DomainGuard::new(|state: &[f64], _: &(), _| state[0] < 0.0)
        .into_callback_set()
        .unwrap();
    let problem = OdeProblem::new(zero, [-1.0], (0.0, 1.0), ()).with_callback_set(callbacks);
    assert_eq!(
        solve(&problem, Euler, &fixed(0.1)),
        Err(SolveError::InitialStateOutOfDomain)
    );

    let callbacks = CallbackSet::new()
        .with_initialize(|state: &mut [f64], _: &(), _| state[0] = 0.0)
        .append(
            DomainGuard::new(|state: &[f64], _: &(), _| state[0] < 0.0)
                .into_callback_set()
                .unwrap(),
        );
    let corrected = OdeProblem::new(zero, [-1.0], (0.0, 0.1), ()).with_callback_set(callbacks);
    assert!(solve(&corrected, Euler, &fixed(0.1)).is_ok());

    for factor in [0.0, 1.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            DomainGuard::new(|_: &[f64], _: &(), _| false)
                .with_reduction_factor(factor)
                .into_callback_set(),
            Err(ConfigurationError::InvalidParameter {
                parameter: "domain-guard reduction factor",
                ..
            })
        ));
    }
}

#[test]
fn adaptive_and_ndarray_states_share_the_same_guard_semantics() {
    let problem =
        OdeProblem::new(zero, [1.0], (0.0, 0.4), ()).with_callback_set(reject_first_candidate());
    let adaptive = solve(
        &problem,
        Tsit5,
        &SolveOptions::new()
            .with_initial_step(0.4)
            .with_save(SaveMode::EveryStep),
    )
    .unwrap();
    assert_eq!(adaptive.stats().rejected_steps, 1);
    assert!((adaptive.times()[1] - 0.2).abs() < 1.0e-12);

    let first_backward_candidate = Cell::new(true);
    let callbacks = DomainGuard::new(move |_: &[f64], _: &(), time| {
        time != 1.0 && first_backward_candidate.replace(false)
    })
    .into_callback_set()
    .unwrap();
    let backward = OdeProblem::new(zero, [1.0], (1.0, 0.0), ()).with_callback_set(callbacks);
    let backward = solve(&backward, Euler, &fixed(0.4)).unwrap();
    assert_eq!(backward.stats().rejected_steps, 1);
    assert!((backward.times()[1] - 0.8).abs() < 1.0e-12);
    assert!(backward.times().windows(2).all(|times| times[1] < times[0]));

    let solve_array = |initial| {
        let callbacks = reject_first_candidate();
        let problem = OdeProblem::from_array(
            |mut derivative: ArrayViewMutD<'_, f64>, _: ArrayViewD<'_, f64>, _: &(), _| {
                derivative.fill(0.0);
            },
            initial,
            (0.0, 0.2),
            (),
        )
        .with_callback_set(callbacks);
        solve(&problem, Euler, &fixed(0.2)).unwrap()
    };

    let scalar = solve_array(arr0(1.0).into_dyn());
    let vector = solve_array(array![1.0, 2.0].into_dyn());
    let matrix = solve_array(array![[1.0, 2.0], [3.0, 4.0]].into_dyn());
    assert!(scalar.state_shape().is_empty());
    assert_eq!(vector.state_shape(), &[2]);
    assert_eq!(matrix.state_shape(), &[2, 2]);
    for solution in [&scalar, &vector, &matrix] {
        assert_eq!(solution.stats().rejected_steps, 1);
        assert!((solution.times()[1] - 0.1).abs() < 1.0e-12);
    }
}

fn assert_split_guard<A: SplitOdeAlgorithm + Copy>(algorithm: A) {
    let problem = SplitOdeProblem::new(zero as SplitRhs, zero as SplitRhs, [1.0], (0.0, 0.3), ())
        .with_callback_set(reject_first_candidate());
    let solution = solve_split(&problem, algorithm, &fixed(0.3)).unwrap();
    assert_eq!(solution.stats().rejected_steps, 1);
    assert!((solution.times()[1] - 0.15).abs() < 1.0e-12);
}

fn second_order_problem() -> SecondOrderOdeProblem<Acceleration, ()> {
    let first = Cell::new(true);
    let callbacks = DomainGuard::new(move |_: &[f64], _: &[f64], _: &(), time| {
        time != 0.0 && first.replace(false)
    })
    .into_second_order_callback_set()
    .unwrap();
    SecondOrderOdeProblem::new(
        zero_acceleration as Acceleration,
        [0.0],
        [1.0],
        (0.0, 0.3),
        (),
    )
    .with_callback_set(callbacks)
}

fn assert_second_order_guard<A: SecondOrderOdeAlgorithm + Copy>(algorithm: A, adaptive: bool) {
    let solution = solve_second_order(
        &second_order_problem(),
        algorithm,
        &fixed(0.3).with_adaptive(adaptive),
    )
    .unwrap();
    assert_eq!(solution.stats().rejected_steps, 1);
    assert!((solution.times()[1] - 0.15).abs() < 1.0e-12);
}

#[test]
fn guards_route_through_every_split_and_second_order_driver_family() {
    assert_split_guard(SplitEuler);
    assert_split_guard(MRIGARKERK22a::new(4));
    assert_split_guard(IMEXEuler);
    assert_split_guard(IRKC::default());

    assert_second_order_guard(NewmarkBeta::default(), false);
    assert_second_order_guard(Nystrom4, false);
    assert_second_order_guard(Dprkn4, true);
    assert_second_order_guard(Irkn3, false);
    assert_second_order_guard(VelocityVerlet, false);

    let symplectic =
        solve_symplectic(&second_order_problem(), PseudoVerletLeapfrog, &fixed(0.3)).unwrap();
    assert!((symplectic.times()[1] - 0.15).abs() < 1.0e-12);
}
