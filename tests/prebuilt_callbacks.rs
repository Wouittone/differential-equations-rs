use std::cell::{Cell, RefCell};
use std::rc::Rc;

use differential_equations::callbacks::{FunctionCallingCallback, PeriodicCallback};
use differential_equations::ndarray::{ArrayView2, ArrayViewMut2, array};
use differential_equations::solvers::explicit::split_euler::{
    SplitEuler, SplitOdeAlgorithm, solve_split,
};
use differential_equations::solvers::explicit::{Rk4, Tsit5};
use differential_equations::solvers::multirate::MRIGARKERK22a;
use differential_equations::solvers::multistep::IMEXEuler;
use differential_equations::solvers::second_order::{
    Dprkn4, Irkn3, NewmarkBeta, Nystrom4, PseudoVerletLeapfrog, SecondOrderOdeAlgorithm,
    SecondOrderOdeProblem, VelocityVerlet, solve_second_order, solve_symplectic,
};
use differential_equations::solvers::stabilized::IRKC;
use differential_equations::{
    CallbackAction, ConfigurationError, OdeProblem, SaveMode, SolveOptions, SplitOdeProblem, solve,
};

fn fixed(step: f64) -> SolveOptions {
    SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(step)
        .with_save(SaveMode::Endpoints)
}

fn zero(derivative: &mut [f64], _: &[f64], _: &(), _: f64) {
    derivative.fill(0.0);
}

fn zero_acceleration(acceleration: &mut [f64], _: &[f64], _: &[f64], _: &(), _: f64) {
    acceleration.fill(0.0);
}

fn assert_times(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "{actual:?}");
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "{actual} != {expected}"
        );
    }
}

#[test]
fn periodic_callbacks_follow_phase_boundaries_and_direction() {
    let forward_times = Rc::new(RefCell::new(Vec::new()));
    let observed = Rc::clone(&forward_times);
    let callbacks = PeriodicCallback::new(0.25)
        .with_phase(0.125)
        .with_initial_affect(true)
        .with_final_affect(true)
        .into_callback_set((0.0, 1.0), move |state, _: &(), time| {
            observed.borrow_mut().push(time);
            state[0] += 1.0;
            CallbackAction::Continue
        })
        .unwrap();
    let problem = OdeProblem::new(zero, [0.0], (0.0, 1.0), ()).with_callback_set(callbacks);
    let solution = solve(&problem, Rk4, &fixed(0.4)).unwrap();

    assert_times(
        &forward_times.borrow(),
        &[0.0, 0.125, 0.375, 0.625, 0.875, 1.0],
    );
    assert_eq!(solution.last_state(), &[6.0]);

    let decimal_times = Rc::new(RefCell::new(Vec::new()));
    let observed = Rc::clone(&decimal_times);
    let callbacks = PeriodicCallback::new(0.3)
        .with_phase(0.1)
        .with_final_affect(true)
        .into_callback_set((0.0, 1.0), move |_, _: &(), time| {
            observed.borrow_mut().push(time);
            CallbackAction::Continue
        })
        .unwrap();
    let problem = OdeProblem::new(zero, [0.0], (0.0, 1.0), ()).with_callback_set(callbacks);
    solve(&problem, Rk4, &fixed(0.8)).unwrap();
    assert_times(&decimal_times.borrow(), &[0.1, 0.4, 0.7, 1.0]);

    let backward_times = Rc::new(RefCell::new(Vec::new()));
    let observed = Rc::clone(&backward_times);
    let callbacks = PeriodicCallback::new(0.25)
        .into_callback_set((1.0, 0.0), move |_, _: &(), time| {
            observed.borrow_mut().push(time);
            CallbackAction::Continue
        })
        .unwrap();
    let problem = OdeProblem::new(zero, [0.0], (1.0, 0.0), ()).with_callback_set(callbacks);
    solve(&problem, Rk4, &fixed(0.4)).unwrap();

    assert_times(&backward_times.borrow(), &[0.75, 0.5, 0.25, 0.0]);
}

#[test]
fn periodic_policy_routes_through_split_and_second_order_drivers() {
    let split_callbacks = PeriodicCallback::new(0.2)
        .with_final_affect(true)
        .into_callback_set((0.0, 0.5), |state, _: &(), _| {
            state[0] += 1.0;
            CallbackAction::Continue
        })
        .unwrap();
    let split =
        SplitOdeProblem::new(zero, zero, [0.0], (0.0, 0.5), ()).with_callback_set(split_callbacks);
    let split_solution = solve_split(&split, SplitEuler, &fixed(0.3)).unwrap();
    assert_eq!(split_solution.last_state(), &[3.0]);

    let second_order_callbacks = PeriodicCallback::new(0.25)
        .into_second_order_callback_set((0.0, 0.5), |_, position, _: &(), _| {
            position[0] += 1.0;
            CallbackAction::Continue
        })
        .unwrap();
    let second_order = SecondOrderOdeProblem::new(zero_acceleration, [0.0], [0.0], (0.0, 0.5), ())
        .with_callback_set(second_order_callbacks);
    let second_order_solution =
        solve_second_order(&second_order, NewmarkBeta::default(), &fixed(0.4)).unwrap();
    assert_eq!(second_order_solution.last_position(), &[2.0]);
}

#[test]
fn periodic_policy_does_not_materialize_dense_schedules() {
    let callbacks = PeriodicCallback::new(1.0e-9)
        .into_callback_set((0.0, 1.0), |_, _: &(), _| CallbackAction::Terminate)
        .unwrap();
    let problem = OdeProblem::new(zero, [0.0], (0.0, 1.0), ()).with_callback_set(callbacks);
    let solution = solve(&problem, Rk4, &fixed(1.0)).unwrap();

    assert!((solution.times().last().unwrap() - 1.0e-9).abs() < f64::EPSILON);
    assert_eq!(solution.stats().callback_invocations, 1);

    let start = 1.0e16;
    let callbacks = PeriodicCallback::new(4.0)
        .into_callback_set((start, start + 16.0), |_, _: &(), _| {
            CallbackAction::Terminate
        })
        .unwrap();
    let problem =
        OdeProblem::new(zero, [0.0], (start, start + 16.0), ()).with_callback_set(callbacks);
    let solution = solve(&problem, Rk4, &fixed(8.0)).unwrap();
    assert_eq!(*solution.times().last().unwrap(), start + 4.0);
    assert_eq!(solution.stats().callback_invocations, 1);
}

#[test]
fn function_calling_callbacks_hit_exact_times_without_duplicates() {
    let times = Rc::new(RefCell::new(Vec::new()));
    let observed = Rc::clone(&times);
    let callbacks = FunctionCallingCallback::at_times([0.25, 0.75])
        .with_every_step(true)
        .with_start(false)
        .into_callback_set((0.0, 1.0), move |_, _: &(), time| {
            observed.borrow_mut().push(time);
        })
        .unwrap();
    let problem = OdeProblem::new(zero, [0.0], (0.0, 1.0), ()).with_callback_set(callbacks);
    let solution = solve(&problem, Rk4, &fixed(0.6)).unwrap();

    assert_times(&times.borrow(), &[0.25, 0.75, 1.0]);
    assert_eq!(solution.stats().callback_invocations, 3);
    assert_eq!(solution.times(), &[0.0, 1.0]);
}

#[test]
fn observation_only_callbacks_preserve_fsal_work() {
    let plain = OdeProblem::new(
        |derivative: &mut [f64], state: &[f64], _: &(), _| derivative[0] = -state[0],
        [1.0],
        (0.0, 1.0),
        (),
    );
    let plain_solution = solve(&plain, Tsit5, &fixed(0.1)).unwrap();

    let calls = Rc::new(Cell::new(0));
    let observed_calls = Rc::clone(&calls);
    let callbacks = FunctionCallingCallback::every_step()
        .into_callback_set((0.0, 1.0), move |_, _: &(), _| {
            observed_calls.set(observed_calls.get() + 1);
        })
        .unwrap();
    let observed = OdeProblem::new(
        |derivative: &mut [f64], state: &[f64], _: &(), _| derivative[0] = -state[0],
        [1.0],
        (0.0, 1.0),
        (),
    )
    .with_callback_set(callbacks);
    let observed_solution = solve(&observed, Tsit5, &fixed(0.1)).unwrap();

    assert_eq!(calls.get(), observed_solution.stats().accepted_steps + 1);
    assert_eq!(
        observed_solution.stats().rhs_evaluations,
        plain_solution.stats().rhs_evaluations
    );
}

type SplitRhs = fn(&mut [f64], &[f64], &(), f64);
type Acceleration = fn(&mut [f64], &[f64], &[f64], &(), f64);

fn split_problem(observed: bool) -> SplitOdeProblem<SplitRhs, SplitRhs, ()> {
    let problem = SplitOdeProblem::new(zero as SplitRhs, zero as SplitRhs, [0.0], (0.0, 1.0), ());
    if observed {
        problem.with_callback_set(
            FunctionCallingCallback::every_step()
                .into_callback_set((0.0, 1.0), |_, _: &(), _| {})
                .unwrap(),
        )
    } else {
        problem
    }
}

fn assert_split_observer_preserves_work<A: SplitOdeAlgorithm + Copy>(algorithm: A) {
    let plain = solve_split(&split_problem(false), algorithm, &fixed(0.1)).unwrap();
    let observed = solve_split(&split_problem(true), algorithm, &fixed(0.1)).unwrap();
    assert_eq!(
        observed.stats().rhs_evaluations,
        plain.stats().rhs_evaluations
    );
    assert_eq!(
        observed.stats().callback_invocations,
        observed.stats().accepted_steps + 1
    );
}

fn second_order_problem(observed: bool) -> SecondOrderOdeProblem<Acceleration, ()> {
    let problem = SecondOrderOdeProblem::new(
        zero_acceleration as Acceleration,
        [0.0],
        [0.0],
        (0.0, 1.0),
        (),
    );
    if observed {
        problem.with_callback_set(
            FunctionCallingCallback::every_step()
                .into_second_order_callback_set((0.0, 1.0), |_, _, _: &(), _| {})
                .unwrap(),
        )
    } else {
        problem
    }
}

fn assert_second_order_observer_preserves_work<A: SecondOrderOdeAlgorithm + Copy>(
    algorithm: A,
    adaptive: bool,
) {
    let options = fixed(0.1).with_adaptive(adaptive);
    let plain = solve_second_order(&second_order_problem(false), algorithm, &options).unwrap();
    let observed = solve_second_order(&second_order_problem(true), algorithm, &options).unwrap();
    assert_eq!(
        observed.stats().rhs_evaluations,
        plain.stats().rhs_evaluations
    );
    assert_eq!(
        observed.stats().callback_invocations,
        observed.stats().accepted_steps + 1
    );
}

#[test]
fn observation_only_callbacks_preserve_work_across_standalone_drivers() {
    assert_split_observer_preserves_work(SplitEuler);
    assert_split_observer_preserves_work(MRIGARKERK22a::new(4));
    assert_split_observer_preserves_work(IMEXEuler);
    assert_split_observer_preserves_work(IRKC::default());

    assert_second_order_observer_preserves_work(NewmarkBeta::default(), false);
    assert_second_order_observer_preserves_work(Nystrom4, false);
    assert_second_order_observer_preserves_work(Dprkn4, true);
    assert_second_order_observer_preserves_work(Irkn3, false);
    assert_second_order_observer_preserves_work(VelocityVerlet, false);

    let options = fixed(0.1);
    let plain =
        solve_symplectic(&second_order_problem(false), PseudoVerletLeapfrog, &options).unwrap();
    let observed =
        solve_symplectic(&second_order_problem(true), PseudoVerletLeapfrog, &options).unwrap();
    assert_eq!(observed.rhs_evaluations(), plain.rhs_evaluations());
}

#[test]
fn function_calling_callbacks_preserve_ndarray_state_shape() {
    let calls = Rc::new(Cell::new(0));
    let observed_calls = Rc::clone(&calls);
    let callbacks = FunctionCallingCallback::at_times([0.5])
        .with_start(false)
        .into_callback_set((0.0, 1.0), move |state, _: &(), time| {
            assert_eq!(state, &[1.0, 2.0, 3.0, 4.0]);
            assert_eq!(time, 0.5);
            observed_calls.set(observed_calls.get() + 1);
        })
        .unwrap();
    let problem = OdeProblem::from_array(
        |mut derivative: ArrayViewMut2<'_, f64>, _: ArrayView2<'_, f64>, _: &(), _| {
            derivative.fill(0.0);
        },
        array![[1.0, 2.0], [3.0, 4.0]],
        (0.0, 1.0),
        (),
    )
    .with_callback_set(callbacks);
    let solution = solve(&problem, Tsit5, &fixed(0.75)).unwrap();

    assert_eq!(calls.get(), 1);
    assert_eq!(solution.state_shape(), &[2, 2]);
    assert_eq!(
        solution.last_state_array(),
        array![[1.0, 2.0], [3.0, 4.0]].into_dyn()
    );
}

#[test]
fn prebuilt_callback_configuration_errors_are_typed() {
    for period in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            PeriodicCallback::new(period)
                .into_callback_set((0.0, 1.0), |_, _: &(), _| { CallbackAction::Continue }),
            Err(ConfigurationError::InvalidParameter {
                parameter: "callback period",
                ..
            })
        ));
    }
    assert!(matches!(
        PeriodicCallback::new(0.1)
            .with_phase(-0.1)
            .into_callback_set((0.0, 1.0), |_, _: &(), _| CallbackAction::Continue),
        Err(ConfigurationError::InvalidParameter {
            parameter: "callback phase",
            ..
        })
    ));
    assert!(matches!(
        FunctionCallingCallback::at_times([0.75, 0.25])
            .into_callback_set((0.0, 1.0), |_, _: &(), _| {}),
        Err(ConfigurationError::InvalidParameter {
            parameter: "function-calling times",
            ..
        })
    ));
    assert!(matches!(
        PeriodicCallback::new(1.0).into_callback_set((1.0e16, 1.0e16 + 10.0), |_, _: &(), _| {
            CallbackAction::Continue
        }),
        Err(ConfigurationError::InvalidParameter {
            parameter: "callback period",
            ..
        })
    ));
}
