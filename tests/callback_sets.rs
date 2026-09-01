use std::cell::{Cell, RefCell};
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
    CallbackAction, CallbackSave, CallbackSet, OdeProblem, SaveMode, Solution, SolveError,
    SolveOptions, SplitOdeProblem, solve,
};

fn fixed_options() -> SolveOptions {
    SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(0.25)
        .with_max_step(0.25)
        .with_save(SaveMode::Endpoints)
}

fn zero(derivative: &mut [f64], _: &[f64], _: &(), _: f64) {
    derivative.fill(0.0);
}

type SplitRhs = fn(&mut [f64], &[f64], &(), f64);
type Acceleration = fn(&mut [f64], &[f64], &[f64], &(), f64);

fn zero_acceleration(acceleration: &mut [f64], _: &[f64], _: &[f64], _: &(), _: f64) {
    acceleration.fill(0.0);
}

#[test]
fn callback_sets_compose_and_preserve_effect_order() {
    let first = CallbackSet::<()>::new().with_preset_time_callback([0.5], |state, _, _| {
        state[0] = 10.0 * state[0] + 1.0;
        CallbackAction::Continue
    });
    let second = CallbackSet::<()>::default().with_preset_time_callback_saving(
        [0.5],
        CallbackSave::After,
        |state, _, _| {
            state[0] = 10.0 * state[0] + 2.0;
            CallbackAction::Continue
        },
    );
    assert_eq!(first.len(), 1);
    assert!(!first.is_empty());

    let callbacks = first.append(second);
    assert_eq!(callbacks.len(), 2);
    let problem = OdeProblem::new(zero, vec![0.0], (0.0, 1.0), ()).with_callback_set(callbacks);
    let solution = solve(&problem, Rk4, &fixed_options()).unwrap();

    assert_eq!(solution.stats().callback_invocations, 2);
    assert_eq!(solution.last_state(), &[12.0]);
    assert_eq!(solution.times(), &[0.0, 0.5, 1.0]);
}

#[test]
fn lifecycle_hooks_wrap_initial_effects_and_callback_termination() {
    let order = Rc::new(RefCell::new(Vec::new()));
    let initialize_order = Rc::clone(&order);
    let effect_order = Rc::clone(&order);
    let finalize_order = Rc::clone(&order);
    let callbacks = CallbackSet::<()>::new()
        .with_initialize_saving(CallbackSave::Both, move |state, _, time| {
            initialize_order.borrow_mut().push(("initialize", time));
            state[0] = 2.0;
        })
        .with_discrete_callback(
            |state, _, time| time == 0.0 && state[0] == 2.0,
            move |state, _, time| {
                effect_order.borrow_mut().push(("effect", time));
                state[0] = 3.0;
                CallbackAction::Terminate
            },
        )
        .with_finalize(move |state, _, time| {
            finalize_order.borrow_mut().push(("finalize", time));
            state[0] = 4.0;
        });
    let problem = OdeProblem::new(zero, vec![1.0], (0.0, 1.0), ()).with_callback_set(callbacks);
    let solution = solve(&problem, Rk4, &fixed_options()).unwrap();

    assert_eq!(
        &*order.borrow(),
        &[("initialize", 0.0), ("effect", 0.0), ("finalize", 0.0)]
    );
    assert_eq!(solution.stats().callback_invocations, 1);
    assert_eq!(solution.times(), &[0.0, 0.0]);
    assert_eq!(solution.state(0), Some([1.0].as_slice()));
    assert_eq!(solution.state(1), Some([4.0].as_slice()));
}

#[test]
fn finalizers_run_only_for_successful_solves_and_do_not_invent_save_at_output() {
    let finalized = Rc::new(Cell::new(0));
    let successful_finalized = Rc::clone(&finalized);
    let successful = OdeProblem::new(zero, vec![1.0], (0.0, 1.0), ()).with_callback_set(
        CallbackSet::new().with_finalize(move |state, _, time| {
            successful_finalized.set(successful_finalized.get() + 1);
            assert_eq!(time, 1.0);
            state[0] = 9.0;
        }),
    );
    let options = fixed_options().with_save_at([0.5]);
    let solution = solve(&successful, Rk4, &options).unwrap();
    assert_eq!(finalized.get(), 1);
    assert_eq!(solution.times(), &[0.5]);
    assert_eq!(solution.last_state(), &[1.0]);

    let failed_finalized = Rc::new(Cell::new(0));
    let observed = Rc::clone(&failed_finalized);
    let failed = OdeProblem::new(
        |derivative: &mut [f64], _: &[f64], _: &(), _: f64| derivative[0] = f64::NAN,
        vec![1.0],
        (0.0, 1.0),
        (),
    )
    .with_callback_set(CallbackSet::new().with_finalize(move |_, _, _| {
        observed.set(observed.get() + 1);
    }));
    assert_eq!(
        solve(&failed, Rk4, &fixed_options()),
        Err(SolveError::NonFiniteDerivative)
    );
    assert_eq!(failed_finalized.get(), 0);
}

#[test]
fn lifecycle_hooks_reject_nonfinite_states() {
    let invalid_initial = OdeProblem::new(zero, vec![1.0], (0.0, 1.0), ())
        .with_callback_set(CallbackSet::new().with_initialize(|state, _, _| state[0] = f64::NAN));
    assert_eq!(
        solve(&invalid_initial, Rk4, &fixed_options()),
        Err(SolveError::NonFiniteCallbackState)
    );

    let invalid_final = OdeProblem::new(zero, vec![1.0], (0.0, 1.0), ()).with_callback_set(
        CallbackSet::new().with_finalize(|state, _, _| state[0] = f64::INFINITY),
    );
    assert_eq!(
        solve(&invalid_final, Rk4, &fixed_options()),
        Err(SolveError::NonFiniteCallbackState)
    );
}

#[test]
fn one_set_combines_continuous_and_discrete_callbacks() {
    let callbacks = CallbackSet::<()>::new()
        .with_continuous_callback(
            |state, _, _| state[0] - 0.5,
            |state, _, _| {
                state[0] = 2.0;
                CallbackAction::Continue
            },
        )
        .with_discrete_callback(
            |state, _, time| (time - 0.5).abs() < 1.0e-12 && state[0] == 2.0,
            |state, _, _| {
                state[0] = 3.0;
                CallbackAction::Continue
            },
        );
    let problem = OdeProblem::new(
        |derivative: &mut [f64], _: &[f64], _: &(), _: f64| derivative[0] = 1.0,
        vec![0.0],
        (0.0, 1.0),
        (),
    )
    .with_callback_set(callbacks);
    let solution = solve(&problem, Rk4, &fixed_options()).unwrap();

    assert_eq!(solution.stats().callback_invocations, 2);
    assert_eq!(solution.times().len(), 4);
    assert!((solution.times()[1] - 0.5).abs() < 1.0e-12);
    assert_eq!(solution.times()[1], solution.times()[2]);
    assert!((solution.state(1).unwrap()[0] - 0.5).abs() < 1.0e-12);
    assert_eq!(solution.state(2), Some([3.0].as_slice()));
}

#[test]
fn callback_sets_work_for_split_problems() {
    let callbacks = CallbackSet::<()>::new().with_preset_time_callback([0.5], |state, _, _| {
        state[0] = 4.0;
        CallbackAction::Continue
    });
    let problem =
        SplitOdeProblem::new(zero, zero, vec![1.0], (0.0, 1.0), ()).with_callback_set(callbacks);
    let solution = solve_split(&problem, SplitEuler, &fixed_options()).unwrap();

    assert_eq!(solution.stats().callback_invocations, 1);
    assert_eq!(solution.last_state(), &[4.0]);
}

fn split_lifecycle_problem() -> SplitOdeProblem<SplitRhs, SplitRhs, ()> {
    let callbacks = CallbackSet::<()>::new()
        .with_initialize(|state, _, _| state[0] = 2.0)
        .with_finalize(|state, _, _| state[0] = 3.0);
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
fn every_split_driver_routes_lifecycle_hooks() {
    let split_euler =
        solve_split(&split_lifecycle_problem(), SplitEuler, &fixed_options()).unwrap();
    let multirate = solve_split(
        &split_lifecycle_problem(),
        MRIGARKERK22a::new(4),
        &fixed_options(),
    )
    .unwrap();
    let multistep = solve_split(&split_lifecycle_problem(), IMEXEuler, &fixed_options()).unwrap();
    let irkc = solve_split(
        &split_lifecycle_problem(),
        IRKC::default(),
        &fixed_options(),
    )
    .unwrap();

    for solution in [split_euler, multirate, multistep, irkc] {
        assert_eq!(solution.times(), &[0.0, 1.0]);
        assert_eq!(solution.state(0), Some([2.0].as_slice()));
        assert_eq!(solution.last_state(), &[3.0]);
        assert_eq!(solution.stats().callback_invocations, 0);
    }
}

fn solve_array(initial: ArrayD<f64>) -> Solution {
    let callbacks = CallbackSet::<()>::new()
        .with_initialize(|state, _, _| state.fill(1.5))
        .with_preset_time_callback([0.5], |state, _, _| {
            state.fill(2.0);
            CallbackAction::Continue
        })
        .with_finalize(|state, _, _| state.fill(3.0));
    let problem = OdeProblem::from_array(
        |mut derivative: ArrayViewMutD<'_, f64>, _: ArrayViewD<'_, f64>, _: &(), _: f64| {
            derivative.fill(0.0);
        },
        initial,
        (0.0, 1.0),
        (),
    )
    .with_callback_set(callbacks);
    solve(&problem, Rk4, &fixed_options()).unwrap()
}

#[test]
fn callback_sets_preserve_scalar_vector_and_matrix_shapes() {
    for (initial, expected_shape) in [
        (arr0(1.0).into_dyn(), &[][..]),
        (array![1.0, 1.0].into_dyn(), &[2][..]),
        (array![[1.0, 1.0], [1.0, 1.0]].into_dyn(), &[2, 2][..]),
    ] {
        let solution = solve_array(initial);
        assert_eq!(solution.state_shape(), expected_shape);
        assert!(solution.state(0).unwrap().iter().all(|value| *value == 1.5));
        assert!(solution.last_state().iter().all(|value| *value == 3.0));
    }
}

#[test]
fn second_order_callback_sets_keep_partitions_separate() {
    let callbacks = SecondOrderCallbackSet::<()>::new().with_preset_time_callback_saving(
        [0.5],
        CallbackSave::Both,
        |velocity, position, _, _| {
            velocity[0] = 2.0;
            position[0] = 3.0;
            CallbackAction::Continue
        },
    );
    let problem = SecondOrderOdeProblem::new(
        |acceleration: &mut [f64], _: &[f64], _: &[f64], _: &(), _: f64| {
            acceleration.fill(0.0);
        },
        vec![0.0],
        vec![1.0],
        (0.0, 1.0),
        (),
    )
    .with_callback_set(callbacks);
    let solution = solve_second_order(&problem, VelocityVerlet, &fixed_options()).unwrap();

    let event = solution
        .times()
        .windows(2)
        .position(|times| times == [0.5, 0.5])
        .unwrap();
    assert_eq!(solution.velocity(event), Some([0.0].as_slice()));
    assert_eq!(solution.position(event), Some([1.0].as_slice()));
    assert_eq!(solution.velocity(event + 1), Some([2.0].as_slice()));
    assert_eq!(solution.position(event + 1), Some([3.0].as_slice()));
}

fn second_order_lifecycle_problem() -> SecondOrderOdeProblem<Acceleration, ()> {
    let callbacks = SecondOrderCallbackSet::<()>::new()
        .with_initialize(|velocity, position, _, _| {
            velocity[0] = 0.0;
            position[0] = 2.0;
        })
        .with_finalize(|velocity, position, _, _| {
            velocity[0] = 3.0;
            position[0] = 4.0;
        });
    SecondOrderOdeProblem::new(
        zero_acceleration as Acceleration,
        vec![1.0],
        vec![1.0],
        (0.0, 1.0),
        (),
    )
    .with_callback_set(callbacks)
}

fn assert_second_order_lifecycle<A: SecondOrderOdeAlgorithm>(algorithm: A, adaptive: bool) {
    let solution = solve_second_order(
        &second_order_lifecycle_problem(),
        algorithm,
        &fixed_options().with_adaptive(adaptive),
    )
    .unwrap();
    assert_eq!(solution.velocity(0), Some([0.0].as_slice()));
    assert_eq!(solution.position(0), Some([2.0].as_slice()));
    assert_eq!(solution.last_velocity(), &[3.0]);
    assert_eq!(solution.last_position(), &[4.0]);
    assert_eq!(solution.stats().callback_invocations, 0);
}

#[test]
fn every_second_order_driver_routes_lifecycle_hooks() {
    assert_second_order_lifecycle(NewmarkBeta::default(), false);
    assert_second_order_lifecycle(Nystrom4, false);
    assert_second_order_lifecycle(Dprkn4, true);
    assert_second_order_lifecycle(Irkn3, false);
    assert_second_order_lifecycle(VelocityVerlet, false);

    let solution = solve_symplectic(
        &second_order_lifecycle_problem(),
        PseudoVerletLeapfrog,
        &fixed_options(),
    )
    .unwrap();
    assert_eq!(solution.velocity(0), Some([0.0].as_slice()));
    assert_eq!(solution.position(0), Some([2.0].as_slice()));
    assert_eq!(solution.last_velocity(), &[3.0]);
    assert_eq!(solution.last_position(), &[4.0]);
}
