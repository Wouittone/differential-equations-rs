use std::cell::RefCell;
use std::rc::Rc;

use differential_equations::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD, arr0, array};
use differential_equations::solvers::explicit::Rk4;
use differential_equations::solvers::explicit::split_euler::{SplitEuler, solve_split};
use differential_equations::solvers::multirate::MRIGARKERK22a;
use differential_equations::solvers::multistep::IMEXEuler;
use differential_equations::solvers::second_order::{
    Dprkn4, Irkn3, NewmarkBeta, Nystrom4, PseudoVerletLeapfrog, SecondOrderCallbackSet,
    SecondOrderOdeAlgorithm, SecondOrderOdeProblem, VelocityVerlet, solve_second_order,
    solve_symplectic,
};
use differential_equations::solvers::stabilized::IRKC;
use differential_equations::{
    CallbackAction, CallbackSet, EventCrossing, OdeProblem, SaveMode, Solution, SolveError,
    SolveOptions, SplitOdeProblem, solve,
};

type SplitRhs = fn(&mut [f64], &[f64], &(), f64);
type Acceleration = fn(&mut [f64], &[f64], &[f64], &(), f64);

fn zero(derivative: &mut [f64], _: &[f64], _: &(), _: f64) {
    derivative.fill(0.0);
}

fn zero_acceleration(acceleration: &mut [f64], _: &[f64], _: &[f64], _: &(), _: f64) {
    acceleration.fill(0.0);
}

fn fixed_options() -> SolveOptions {
    SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(1.0)
        .with_max_step(1.0)
        .with_save(SaveMode::Endpoints)
}

#[test]
fn simultaneous_events_are_reported_once_with_crossing_directions() {
    let observed = Rc::new(RefCell::new(Vec::new()));
    let effect_observed = Rc::clone(&observed);
    let callbacks = CallbackSet::<()>::new().with_vector_continuous_callback(
        3,
        |conditions, _, _, time| {
            conditions[0] = time - 0.5;
            conditions[1] = 0.5 - time;
            conditions[2] = time - 0.75;
        },
        move |state, _, time, events| {
            effect_observed.borrow_mut().push((time, events.to_vec()));
            if time < 0.6 {
                state[0] += 1.0;
                CallbackAction::Continue
            } else {
                state[0] += 10.0;
                CallbackAction::Terminate
            }
        },
    );
    let problem = OdeProblem::new(zero, vec![0.0], (0.0, 1.0), ()).with_callback_set(callbacks);
    let solution = solve(&problem, Rk4, &fixed_options()).unwrap();

    let observed = observed.borrow();
    assert_eq!(observed.len(), 2);
    assert!((observed[0].0 - 0.5).abs() < 1.0e-12);
    assert_eq!(
        observed[0].1,
        [
            EventCrossing::Rising,
            EventCrossing::Falling,
            EventCrossing::None,
        ]
    );
    assert!((observed[1].0 - 0.75).abs() < 1.0e-12);
    assert_eq!(
        observed[1].1,
        [
            EventCrossing::None,
            EventCrossing::None,
            EventCrossing::Rising,
        ]
    );
    assert_eq!(solution.stats().callback_invocations, 2);
    assert_eq!(solution.last_state(), &[11.0]);
    assert!((solution.times().last().unwrap() - 0.75).abs() < 1.0e-12);
}

#[test]
fn earliest_root_wins_across_scalar_and_vector_callbacks() {
    let callbacks = CallbackSet::<()>::new()
        .with_vector_continuous_callback(
            1,
            |conditions, _, _, time| conditions[0] = time - 0.75,
            |state, _, _, _| {
                state[0] = 2.0;
                CallbackAction::Terminate
            },
        )
        .with_continuous_callback(
            |_, _, time| time - 0.25,
            |state, _, _| {
                state[0] = 1.0;
                CallbackAction::Terminate
            },
        );
    let problem = OdeProblem::new(zero, vec![0.0], (0.0, 1.0), ()).with_callback_set(callbacks);
    let solution = solve(&problem, Rk4, &fixed_options()).unwrap();

    assert!((solution.times().last().unwrap() - 0.25).abs() < 1.0e-12);
    assert_eq!(solution.last_state(), &[1.0]);
}

#[test]
fn backward_crossings_keep_direction_and_zero_surfaces_do_not_refire() {
    let observed = Rc::new(RefCell::new(Vec::new()));
    let effect_observed = Rc::clone(&observed);
    let backward = OdeProblem::new(zero, vec![0.0], (1.0, 0.0), ()).with_callback_set(
        CallbackSet::new().with_vector_continuous_callback(
            1,
            |conditions, _, _, time| conditions[0] = time - 0.5,
            move |_, _, _, events| {
                effect_observed.borrow_mut().extend_from_slice(events);
                CallbackAction::Terminate
            },
        ),
    );
    let backward_solution = solve(&backward, Rk4, &fixed_options()).unwrap();
    assert_eq!(&*observed.borrow(), &[EventCrossing::Falling]);
    assert!((backward_solution.times().last().unwrap() - 0.5).abs() < 1.0e-12);

    let sticking = OdeProblem::new(
        |derivative: &mut [f64], _: &[f64], _: &(), _: f64| derivative[0] = 1.0,
        vec![0.0],
        (0.0, 1.0),
        (),
    )
    .with_callback_set(CallbackSet::new().with_vector_continuous_callback(
        1,
        |conditions, state, _, _| conditions[0] = state[0] - 0.5,
        |state, _, _, _| {
            state[0] = 0.5;
            CallbackAction::Continue
        },
    ));
    let solution = solve(
        &sticking,
        Rk4,
        &fixed_options().with_initial_step(0.25).with_max_step(0.25),
    )
    .unwrap();
    assert_eq!(solution.stats().callback_invocations, 1);
}

#[test]
fn multiple_vector_callbacks_keep_independent_mask_lengths() {
    let first_masks = Rc::new(RefCell::new(Vec::new()));
    let second_masks = Rc::new(RefCell::new(Vec::new()));
    let first_observed = Rc::clone(&first_masks);
    let second_observed = Rc::clone(&second_masks);
    let callbacks = CallbackSet::<()>::new()
        .with_vector_continuous_callback(
            2,
            |conditions, state, _, _| {
                conditions[0] = state[0] - 3.0;
                conditions[1] = state[0] - 5.0;
            },
            move |_, _, _, events| {
                first_observed.borrow_mut().push(events.to_vec());
                CallbackAction::Continue
            },
        )
        .with_vector_continuous_callback(
            1,
            |conditions, state, _, _| conditions[0] = state[0] - 4.0,
            move |_, _, _, events| {
                second_observed.borrow_mut().push(events.to_vec());
                CallbackAction::Continue
            },
        );
    let problem = OdeProblem::new(
        |derivative: &mut [f64], _: &[f64], _: &(), _: f64| derivative[0] = 1.0,
        vec![0.0],
        (0.0, 6.0),
        (),
    )
    .with_callback_set(callbacks);
    let solution = solve(
        &problem,
        Rk4,
        &fixed_options().with_initial_step(6.0).with_max_step(6.0),
    )
    .unwrap();

    assert_eq!(
        &*first_masks.borrow(),
        &[
            vec![EventCrossing::Rising, EventCrossing::None],
            vec![EventCrossing::None, EventCrossing::Rising],
        ]
    );
    assert_eq!(&*second_masks.borrow(), &[vec![EventCrossing::Rising]]);
    assert_eq!(solution.stats().callback_invocations, 3);
}

#[test]
fn invalid_vector_callback_definitions_are_typed_errors() {
    let empty = OdeProblem::new(zero, vec![0.0], (0.0, 1.0), ()).with_callback_set(
        CallbackSet::new().with_vector_continuous_callback(
            0,
            |_, _, _, _| {},
            |_, _, _, _| CallbackAction::Continue,
        ),
    );
    assert_eq!(
        solve(&empty, Rk4, &fixed_options()),
        Err(SolveError::InvalidVectorCallbackLength)
    );

    let incomplete = OdeProblem::new(zero, vec![0.0], (0.0, 1.0), ()).with_callback_set(
        CallbackSet::new().with_vector_continuous_callback(
            2,
            |conditions, _, _, time| conditions[0] = time - 0.5,
            |_, _, _, _| CallbackAction::Continue,
        ),
    );
    assert_eq!(
        solve(&incomplete, Rk4, &fixed_options()),
        Err(SolveError::NonFiniteCallbackCondition)
    );
}

fn solve_array(initial_state: ArrayD<f64>) -> Solution {
    let expected_shape = initial_state.shape().to_vec();
    let problem = OdeProblem::from_array(
        |mut derivative: ArrayViewMutD<'_, f64>, _: ArrayViewD<'_, f64>, _: &(), _: f64| {
            derivative.fill(0.0);
        },
        initial_state,
        (0.0, 1.0),
        (),
    )
    .with_array_vector_continuous_callback(
        1,
        move |mut conditions, state, _, time| {
            assert_eq!(state.shape(), expected_shape);
            conditions[0] = time - 0.5;
        },
        |mut state, _, _, events| {
            assert_eq!(events, &[EventCrossing::Rising]);
            state.fill(2.0);
            CallbackAction::Terminate
        },
    );
    solve(&problem, Rk4, &fixed_options()).unwrap()
}

#[test]
fn ndarray_vector_callbacks_preserve_scalar_vector_and_matrix_shapes() {
    for (initial_state, expected_shape) in [
        (arr0(1.0).into_dyn(), vec![]),
        (array![1.0, 2.0].into_dyn(), vec![2]),
        (array![[1.0, 2.0], [3.0, 4.0]].into_dyn(), vec![2, 2]),
    ] {
        let solution = solve_array(initial_state);
        assert_eq!(solution.state_shape(), expected_shape);
        assert!(solution.last_state().iter().all(|value| *value == 2.0));
        assert!((solution.times().last().unwrap() - 0.5).abs() < 1.0e-12);
    }
}

fn split_problem() -> SplitOdeProblem<SplitRhs, SplitRhs, ()> {
    let callbacks = CallbackSet::<()>::new().with_vector_continuous_callback(
        1,
        |conditions, _, _, time| conditions[0] = time - 0.5,
        |state, _, _, events| {
            assert_eq!(events, &[EventCrossing::Rising]);
            state[0] = 2.0;
            CallbackAction::Terminate
        },
    );
    SplitOdeProblem::new(
        zero as SplitRhs,
        zero as SplitRhs,
        vec![1.0],
        (0.0, 1.0),
        (),
    )
    .with_callback_set(callbacks)
}

#[test]
fn every_split_driver_routes_vector_events() {
    let solutions = [
        solve_split(&split_problem(), SplitEuler, &fixed_options()).unwrap(),
        solve_split(&split_problem(), MRIGARKERK22a::new(4), &fixed_options()).unwrap(),
        solve_split(&split_problem(), IMEXEuler, &fixed_options()).unwrap(),
        solve_split(&split_problem(), IRKC::default(), &fixed_options()).unwrap(),
    ];
    for solution in solutions {
        assert!((solution.times().last().unwrap() - 0.5).abs() < 1.0e-12);
        assert_eq!(solution.last_state(), &[2.0]);
        assert_eq!(solution.stats().callback_invocations, 1);
    }
}

fn second_order_problem() -> SecondOrderOdeProblem<Acceleration, ()> {
    let callbacks = SecondOrderCallbackSet::<()>::new().with_vector_continuous_callback(
        2,
        |conditions, _, _, _, time| {
            conditions[0] = time - 0.5;
            conditions[1] = 0.5 - time;
        },
        |velocity, position, _, _, events| {
            assert_eq!(events, &[EventCrossing::Rising, EventCrossing::Falling]);
            velocity[0] = 2.0;
            position[0] = 3.0;
            CallbackAction::Terminate
        },
    );
    SecondOrderOdeProblem::new(
        zero_acceleration as Acceleration,
        vec![0.0],
        vec![1.0],
        (0.0, 1.0),
        (),
    )
    .with_callback_set(callbacks)
}

fn assert_second_order_vector_event<A: SecondOrderOdeAlgorithm>(algorithm: A, adaptive: bool) {
    let solution = solve_second_order(
        &second_order_problem(),
        algorithm,
        &fixed_options().with_adaptive(adaptive),
    )
    .unwrap();
    assert!((solution.times().last().unwrap() - 0.5).abs() < 1.0e-12);
    assert_eq!(solution.last_velocity(), &[2.0]);
    assert_eq!(solution.last_position(), &[3.0]);
    assert_eq!(solution.stats().callback_invocations, 1);
}

#[test]
fn every_second_order_driver_routes_vector_events() {
    assert_second_order_vector_event(NewmarkBeta::default(), false);
    assert_second_order_vector_event(Nystrom4, false);
    assert_second_order_vector_event(Dprkn4, true);
    assert_second_order_vector_event(Irkn3, false);
    assert_second_order_vector_event(VelocityVerlet, false);

    let solution = solve_symplectic(
        &second_order_problem(),
        PseudoVerletLeapfrog,
        &fixed_options(),
    )
    .unwrap();
    assert!((solution.times().last().unwrap() - 0.5).abs() < 1.0e-12);
    assert_eq!(solution.last_velocity(), &[2.0]);
    assert_eq!(solution.last_position(), &[3.0]);
}
