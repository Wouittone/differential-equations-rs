use std::cell::RefCell;
use std::rc::Rc;

use differential_equations::ndarray::{ArrayView2, ArrayViewMut2, ArrayViewMutD, array};
use differential_equations::solvers::explicit::Rk4;
use differential_equations::solvers::explicit::split_euler::{SplitEuler, solve_split};
use differential_equations::solvers::multirate::MRIGARKERK22a;
use differential_equations::solvers::multistep::IMEXEuler;
use differential_equations::solvers::rosenbrock::Rodas5P;
use differential_equations::solvers::second_order::{
    Dprkn4, Irkn3, NewmarkBeta, Nystrom4, PseudoVerletLeapfrog, SecondOrderOdeAlgorithm,
    SecondOrderOdeProblem, SecondOrderSolveError, SymplecticSolveError, VelocityVerlet,
    solve_second_order, solve_symplectic,
};
use differential_equations::solvers::stabilized::IRKC;
use differential_equations::{
    CallbackAction, OdeAlgorithm, OdeProblem, SaveMode, SolveError, SolveOptions, SplitOdeProblem,
    solve,
};

fn fixed(step: f64) -> SolveOptions {
    SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(step)
        .with_max_step(step)
        .with_save(SaveMode::EveryStep)
}

type SplitRhs = fn(&mut [f64], &[f64], &(), f64);
type Acceleration = fn(&mut [f64], &[f64], &[f64], &(), f64);

fn zero_split_rhs(derivative: &mut [f64], _: &[f64], _: &(), _: f64) {
    derivative[0] = 0.0;
}

fn zero_acceleration(acceleration: &mut [f64], _: &[f64], _: &[f64], _: &(), _: f64) {
    acceleration[0] = 0.0;
}

fn split_preset_problem() -> SplitOdeProblem<SplitRhs, SplitRhs, ()> {
    SplitOdeProblem::new(
        zero_split_rhs as SplitRhs,
        zero_split_rhs as SplitRhs,
        vec![0.0],
        (0.0, 1.0),
        (),
    )
    .with_preset_time_callback([0.25, 0.5], |state, _, _| {
        state[0] += 1.0;
        CallbackAction::Continue
    })
}

fn assert_first_order_preset_times<A: OdeAlgorithm>(algorithm: A) {
    let observed = Rc::new(RefCell::new(Vec::new()));
    let callback_observed = Rc::clone(&observed);
    let problem = OdeProblem::new(
        |derivative: &mut [f64], _: &[f64], _: &(), _: f64| derivative[0] = 1.0,
        vec![0.0],
        (0.0, 1.0),
        (),
    )
    .with_preset_time_callback([0.0, 0.3, 0.55, 0.9, 1.0], move |state, _, time| {
        callback_observed.borrow_mut().push(time);
        state[0] += 1.0;
        CallbackAction::Continue
    });

    let solution = solve(&problem, algorithm, &fixed(0.4)).unwrap();

    assert_eq!(&*observed.borrow(), &[0.0, 0.3, 0.55, 0.9, 1.0]);
    assert_eq!(solution.stats().callback_invocations, 5);
    assert!((solution.last_state()[0] - 6.0).abs() < 1.0e-11);
}

#[test]
fn explicit_and_stiff_solvers_hit_preset_times_without_option_time_stops() {
    assert_first_order_preset_times(Rk4);
    assert_first_order_preset_times(Rodas5P);
}

#[test]
fn backward_preset_times_follow_integration_direction() {
    let observed = Rc::new(RefCell::new(Vec::new()));
    let callback_observed = Rc::clone(&observed);
    let problem = OdeProblem::new(
        |derivative: &mut [f64], _: &[f64], _: &(), _: f64| derivative[0] = 0.0,
        vec![0.0],
        (1.0, 0.0),
        (),
    )
    .with_preset_time_callback([1.0, 0.7, 0.45, 0.1, 0.0], move |state, _, time| {
        callback_observed.borrow_mut().push(time);
        state[0] += 1.0;
        CallbackAction::Continue
    });

    let solution = solve(&problem, Rk4, &fixed(0.4)).unwrap();

    assert_eq!(&*observed.borrow(), &[1.0, 0.7, 0.45, 0.1, 0.0]);
    assert_eq!(solution.last_state(), &[5.0]);
}

#[test]
fn callbacks_at_the_same_preset_time_keep_insertion_order() {
    let problem = OdeProblem::new(
        |derivative: &mut [f64], _: &[f64], _: &(), _: f64| derivative[0] = 0.0,
        vec![0.0],
        (0.0, 1.0),
        (),
    )
    .with_preset_time_callback([0.5], |state, _, _| {
        state[0] = 10.0 * state[0] + 1.0;
        CallbackAction::Continue
    })
    .with_preset_time_callback([0.5], |state, _, _| {
        state[0] = 10.0 * state[0] + 2.0;
        CallbackAction::Continue
    });

    let solution = solve(&problem, Rk4, &fixed(0.4)).unwrap();

    assert_eq!(solution.last_state(), &[12.0]);
    assert_eq!(solution.stats().callback_invocations, 2);
}

#[test]
fn invalid_preset_time_sequences_are_reported_before_integration() {
    for times in [
        vec![0.5, 0.25],
        vec![-0.1],
        vec![1.1],
        vec![f64::NAN],
        vec![0.5, 0.5],
    ] {
        let problem = OdeProblem::new(
            |derivative: &mut [f64], _: &[f64], _: &(), _: f64| derivative[0] = 0.0,
            vec![0.0],
            (0.0, 1.0),
            (),
        )
        .with_preset_time_callback(times, |_, _, _| CallbackAction::Continue);

        assert_eq!(
            solve(&problem, Rk4, &fixed(0.4)),
            Err(SolveError::InvalidPresetTimes)
        );
    }

    let split = SplitOdeProblem::new(
        zero_split_rhs as SplitRhs,
        zero_split_rhs as SplitRhs,
        vec![0.0],
        (0.0, 1.0),
        (),
    )
    .with_preset_time_callback([0.75, 0.25], |_, _, _| CallbackAction::Continue);
    assert_eq!(
        solve_split(&split, SplitEuler, &fixed(0.4)),
        Err(SolveError::InvalidPresetTimes)
    );

    let partitioned = SecondOrderOdeProblem::new(
        zero_acceleration as Acceleration,
        vec![0.0],
        vec![0.0],
        (0.0, 1.0),
        (),
    )
    .with_preset_time_callback([0.75, 0.25], |_, _, _, _| CallbackAction::Continue);
    assert_eq!(
        solve_second_order(&partitioned, VelocityVerlet, &fixed(0.4)),
        Err(SecondOrderSolveError::Solve(SolveError::InvalidPresetTimes))
    );
    assert_eq!(
        solve_symplectic(&partitioned, PseudoVerletLeapfrog, &fixed(0.4)),
        Err(SymplecticSolveError::Solve(SolveError::InvalidPresetTimes))
    );
}

#[test]
fn ndarray_preset_callbacks_preserve_matrix_indexing() {
    let problem = OdeProblem::from_array(
        |mut derivative: ArrayViewMut2<'_, f64>, _: ArrayView2<'_, f64>, _: &(), _: f64| {
            derivative.fill(0.0);
        },
        array![[1.0, 2.0], [3.0, 4.0]],
        (0.0, 1.0),
        (),
    )
    .with_array_preset_time_callback([0.25], |mut state: ArrayViewMutD<'_, f64>, _, _| {
        state[[1, 0]] = 9.0;
        CallbackAction::Continue
    });

    let solution = solve(&problem, Rk4, &fixed(0.4)).unwrap();

    assert_eq!(solution.state_shape(), &[2, 2]);
    assert_eq!(solution.last_state_array()[[1, 0]], 9.0);
}

#[test]
fn every_split_driver_uses_problem_owned_preset_stops() {
    let split_euler = solve_split(&split_preset_problem(), SplitEuler, &fixed(0.4)).unwrap();
    let multirate =
        solve_split(&split_preset_problem(), MRIGARKERK22a::new(4), &fixed(0.4)).unwrap();
    let multistep = solve_split(&split_preset_problem(), IMEXEuler, &fixed(0.4)).unwrap();
    let irkc = solve_split(&split_preset_problem(), IRKC::default(), &fixed(0.4)).unwrap();

    for solution in [split_euler, multirate, multistep, irkc] {
        assert_eq!(solution.stats().callback_invocations, 2);
        assert!((solution.last_state()[0] - 2.0).abs() < 1.0e-11);
        assert!(solution.times().contains(&0.25));
        assert!(solution.times().contains(&0.5));
    }
}

fn partitioned_problem() -> SecondOrderOdeProblem<Acceleration, ()> {
    SecondOrderOdeProblem::new(
        zero_acceleration as Acceleration,
        vec![0.0],
        vec![0.0],
        (0.0, 1.0),
        (),
    )
    .with_preset_time_callback([0.25, 0.5], |_, position, _, _| {
        position[0] += 1.0;
        CallbackAction::Continue
    })
}

fn assert_second_order_preset_times<A: SecondOrderOdeAlgorithm>(algorithm: A, adaptive: bool) {
    let options = fixed(0.4).with_adaptive(adaptive);
    let solution = solve_second_order(&partitioned_problem(), algorithm, &options).unwrap();

    assert_eq!(solution.stats().callback_invocations, 2);
    assert_eq!(solution.last_position(), &[2.0]);
    assert!(solution.times().contains(&0.25));
    assert!(solution.times().contains(&0.5));
}

#[test]
fn second_order_and_symplectic_drivers_apply_preset_callbacks() {
    assert_second_order_preset_times(NewmarkBeta::default(), false);
    assert_second_order_preset_times(Nystrom4, false);
    assert_second_order_preset_times(Dprkn4, true);
    assert_second_order_preset_times(Irkn3, false);
    assert_second_order_preset_times(VelocityVerlet, false);

    let solution =
        solve_symplectic(&partitioned_problem(), PseudoVerletLeapfrog, &fixed(0.4)).unwrap();
    assert_eq!(solution.last_position(), &[2.0]);
}
