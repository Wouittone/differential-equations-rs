use differential_equations::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD, arr0, array};
use differential_equations::solvers::explicit::Rk4;
use differential_equations::solvers::explicit::split_euler::{SplitEuler, solve_split};
use differential_equations::solvers::second_order::{
    SecondOrderCallbackSet, SecondOrderOdeProblem, VelocityVerlet, solve_second_order,
};
use differential_equations::{
    CallbackAction, CallbackSave, CallbackSet, OdeProblem, SaveMode, Solution, SolveOptions,
    SplitOdeProblem, solve,
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

fn solve_array(initial: ArrayD<f64>) -> Solution {
    let callbacks = CallbackSet::<()>::new().with_preset_time_callback([0.5], |state, _, _| {
        state.fill(2.0);
        CallbackAction::Continue
    });
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
        assert!(solution.last_state().iter().all(|value| *value == 2.0));
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
