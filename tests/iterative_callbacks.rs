use std::cell::{Cell, RefCell};
use std::rc::Rc;

use differential_equations::callbacks::IterativeCallback;
use differential_equations::ndarray::{ArrayViewD, ArrayViewMutD, arr0, array};
use differential_equations::solvers::explicit::split_euler::{SplitEuler, solve_split};
use differential_equations::solvers::explicit::{Euler, Tsit5};
use differential_equations::solvers::multirate::MRIGARKERK22a;
use differential_equations::solvers::multistep::IMEXEuler;
use differential_equations::solvers::second_order::{
    Dprkn4, Irkn3, NewmarkBeta, Nystrom4, SecondOrderOdeAlgorithm, SecondOrderOdeProblem,
    SecondOrderSolveError, VelocityVerlet, solve_second_order,
};
use differential_equations::solvers::stabilized::IRKC;
use differential_equations::{
    CallbackAction, CallbackSave, CallbackSet, OdeProblem, SaveMode, SolveError, SolveOptions,
    SplitOdeProblem, solve,
};

fn options() -> SolveOptions {
    SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(0.8)
        .with_save(SaveMode::Endpoints)
}

fn zero(du: &mut [f64], _: &[f64], _: &(), _: f64) {
    du.fill(0.0);
}

#[test]
fn state_dependent_times_are_exact_in_both_directions_and_restart_each_solve() {
    for span in [(0.0_f64, 1.0_f64), (1.0, 0.0)] {
        let direction = (span.1 - span.0).signum();
        let calls = Rc::new(RefCell::new(Vec::new()));
        let observed = Rc::clone(&calls);
        let callbacks = IterativeCallback::new(move |u: &[f64], _: &(), t| {
            Some(t + direction * 0.125 * (u[0] + 1.0))
        })
        .into_callback_set(span, move |u, _, t| {
            observed.borrow_mut().push(t);
            u[0] += 1.0;
            CallbackAction::Continue
        })
        .unwrap();
        let problem = OdeProblem::new(zero, [0.0], span, ()).with_callback_set(callbacks);
        for adaptive in [false, true] {
            calls.borrow_mut().clear();
            let solution = solve(&problem, Tsit5, &options().with_adaptive(adaptive)).unwrap();
            let expected = [0.125, 0.375, 0.75].map(|offset| span.0 + direction * offset);
            assert_eq!(calls.borrow().as_slice(), expected);
            assert_eq!(solution.stats().callback_invocations, 3);
            assert_eq!(solution.last_state(), &[3.0]);
            assert_eq!(*solution.times().last().unwrap(), span.1);
        }
    }
}

#[test]
fn initial_scheduling_does_not_fire_or_save_an_effect() {
    let calls = Rc::new(Cell::new(0));
    let scheduled = Rc::clone(&calls);
    let callbacks = IterativeCallback::new(move |_: &[f64], _: &(), _| {
        scheduled.set(scheduled.get() + 1);
        None
    })
    .into_callback_set((0.0, 1.0), |_, _, _| panic!("no effect was scheduled"))
    .unwrap();
    let problem = OdeProblem::new(zero, [0.0], (0.0, 1.0), ()).with_callback_set(callbacks);
    let solution = solve(&problem, Euler, &options()).unwrap();
    assert_eq!(calls.get(), 1);
    assert_eq!(solution.stats().callback_invocations, 0);
    assert_eq!(solution.times(), &[0.0, 1.0]);
}

#[test]
fn initial_effect_and_termination_control_the_next_scheduling_call() {
    let calls = Rc::new(Cell::new(0));
    let scheduled = Rc::clone(&calls);
    let callbacks = IterativeCallback::new(move |u: &[f64], _: &(), _| {
        scheduled.set(scheduled.get() + 1);
        assert_eq!(u[0], 1.0);
        Some(0.5)
    })
    .with_initial_affect(true)
    .into_callback_set((0.0, 1.0), |u, _, t| {
        u[0] += 1.0;
        if t == 0.0 {
            CallbackAction::Continue
        } else {
            CallbackAction::Terminate
        }
    })
    .unwrap();
    let problem = OdeProblem::new(zero, [0.0], (0.0, 1.0), ()).with_callback_set(callbacks);
    let solution = solve(&problem, Euler, &options()).unwrap();
    assert_eq!(calls.get(), 1);
    assert_eq!(solution.last_state(), &[2.0]);
    assert_eq!(*solution.times().last().unwrap(), 0.5);
    assert_eq!(solution.stats().callback_invocations, 2);
}

#[test]
fn callback_saves_capture_both_sides_without_rescheduling_after_the_endpoint() {
    let calls = Rc::new(Cell::new(0));
    let scheduled = Rc::clone(&calls);
    let callbacks = IterativeCallback::new(move |_: &[f64], _: &(), t| {
        scheduled.set(scheduled.get() + 1);
        Some(t + 0.5)
    })
    .with_save(CallbackSave::Both)
    .into_callback_set((0.0, 1.0), |u, _, _| {
        u[0] += 1.0;
        CallbackAction::Continue
    })
    .unwrap();
    let problem = OdeProblem::new(zero, [0.0], (0.0, 1.0), ()).with_callback_set(callbacks);
    let solution = solve(&problem, Euler, &options()).unwrap();
    assert_eq!(calls.get(), 2);
    assert_eq!(solution.times(), &[0.0, 0.5, 0.5, 1.0, 1.0]);
    assert_eq!(solution.values(), &[0.0, 0.0, 1.0, 1.0, 2.0]);
}

#[test]
fn initializers_and_prior_effects_are_visible_to_scheduling() {
    let initializer = CallbackSet::new().with_initialize(|u, _: &(), _| u[0] = 0.25);
    let earlier = CallbackSet::new().with_discrete_callback(
        |_, _, t| t == 0.0,
        |u, _, _| {
            u[0] = 0.5;
            CallbackAction::Continue
        },
    );
    let callbacks = IterativeCallback::new(|u: &[f64], _: &(), _| Some(u[0]))
        .into_callback_set((0.0, 1.0), |_, _, _| CallbackAction::Terminate)
        .unwrap();
    let problem = OdeProblem::new(zero, [0.0], (0.0, 1.0), ())
        .with_callback_set(initializer.append(earlier).append(callbacks));
    let solution = solve(&problem, Euler, &options()).unwrap();
    assert_eq!(*solution.times().last().unwrap(), 0.5);
}

#[test]
fn iterative_effects_preserve_scalar_vector_and_matrix_shapes() {
    let run = |initial| {
        let callbacks = IterativeCallback::new(|_: &[f64], _: &(), t| Some(t + 0.5))
            .into_callback_set((0.0, 1.0), |u, _, _| {
                u.iter_mut().for_each(|value| *value += 1.0);
                CallbackAction::Continue
            })
            .unwrap();
        let problem = OdeProblem::from_array(
            |mut du: ArrayViewMutD<'_, f64>, _: ArrayViewD<'_, f64>, _: &(), _| du.fill(0.0),
            initial,
            (0.0, 1.0),
            (),
        )
        .with_callback_set(callbacks);
        solve(&problem, Euler, &options()).unwrap()
    };
    let solutions = [
        run(arr0(0.0).into_dyn()),
        run(array![0.0, 0.0].into_dyn()),
        run(array![[0.0, 0.0], [0.0, 0.0]].into_dyn()),
    ];
    for (solution, shape) in solutions.iter().zip([&[][..], &[2][..], &[2, 2][..]]) {
        assert_eq!(solution.state_shape(), shape);
        assert!(solution.last_state().iter().all(|value| *value == 2.0));
    }
}

#[test]
fn all_split_drivers_observe_dynamic_stops() {
    for span in [(0.0_f64, 1.0_f64), (1.0, 0.0)] {
        let direction = (span.1 - span.0).signum();
        let problem = SplitOdeProblem::new(zero, zero, [0.0], span, ()).with_callback_set(
            IterativeCallback::new(move |_: &[f64], _: &(), t| Some(t + direction * 0.25))
                .into_callback_set(span, |u, _, _| {
                    u[0] += 1.0;
                    CallbackAction::Continue
                })
                .unwrap(),
        );
        let solutions = [
            solve_split(&problem, SplitEuler, &options()).unwrap(),
            solve_split(&problem, MRIGARKERK22a::new(4), &options()).unwrap(),
            solve_split(&problem, IMEXEuler, &options()).unwrap(),
            solve_split(&problem, IRKC::default(), &options()).unwrap(),
        ];
        for solution in solutions {
            assert!((solution.last_state()[0] - 4.0).abs() < 1.0e-12);
            assert_eq!(solution.stats().callback_invocations, 4);
            assert_eq!(
                solution.times(),
                [
                    span.0,
                    span.0 + direction * 0.25,
                    span.0 + direction * 0.5,
                    span.0 + direction * 0.75,
                    span.1
                ]
            );
        }
    }
}

fn assert_second_order<A: SecondOrderOdeAlgorithm + Copy>(algorithm: A, adaptive: bool) {
    for span in [(0.0_f64, 1.0_f64), (1.0, 0.0)] {
        let direction = (span.1 - span.0).signum();
        let callbacks = IterativeCallback::new(move |v: &[f64], _: &[f64], _: &(), t| {
            Some(t + direction * 0.125 * (v[0] + 1.0))
        })
        .into_second_order_callback_set(span, |v, _, _, _| {
            v[0] += 1.0;
            CallbackAction::Continue
        })
        .unwrap();
        let problem = SecondOrderOdeProblem::new(
            |a: &mut [f64], _: &[f64], _: &[f64], _: &(), _| a.fill(0.0),
            [0.0],
            [0.0],
            span,
            (),
        )
        .with_callback_set(callbacks);
        for _ in 0..2 {
            let solution =
                solve_second_order(&problem, algorithm, &options().with_adaptive(adaptive))
                    .unwrap();
            assert_eq!(solution.last_velocity(), &[3.0]);
            assert_eq!(solution.stats().callback_invocations, 3);
            assert_eq!(
                solution.times(),
                [
                    span.0,
                    span.0 + direction * 0.125,
                    span.0 + direction * 0.375,
                    span.0 + direction * 0.75,
                    span.1
                ]
            );
        }
    }
}

#[test]
fn all_second_order_drivers_observe_dynamic_stops() {
    assert_second_order(NewmarkBeta::default(), false);
    assert_second_order(Nystrom4, false);
    assert_second_order(Dprkn4, true);
    assert_second_order(Irkn3, false);
    assert_second_order(VelocityVerlet, false);
}

#[test]
fn invalid_times_are_typed_errors_and_beyond_end_times_stop_scheduling() {
    for next in [0.0, f64::EPSILON, -0.5, f64::NAN, f64::INFINITY] {
        let callbacks = IterativeCallback::new(move |_: &[f64], _: &(), _| Some(next))
            .into_callback_set((0.0, 1.0), |_, _, _| CallbackAction::ContinueUnmodified)
            .unwrap();
        let problem = OdeProblem::new(zero, [0.0], (0.0, 1.0), ()).with_callback_set(callbacks);
        assert_eq!(
            solve(&problem, Euler, &options()),
            Err(SolveError::InvalidIterativeCallbackTime)
        );
    }
    let callbacks = IterativeCallback::new(|_: &[f64], _: &(), _| Some(2.0))
        .into_callback_set((0.0, 1.0), |_, _, _| panic!("beyond endpoint"))
        .unwrap();
    let problem = OdeProblem::new(zero, [0.0], (0.0, 1.0), ()).with_callback_set(callbacks);
    assert_eq!(
        solve(&problem, Euler, &options())
            .unwrap()
            .stats()
            .callback_invocations,
        0
    );

    let callbacks = IterativeCallback::new(|_: &[f64], _: &(), _| Some(0.5))
        .into_callback_set((0.0, 1.0), |_, _, _| CallbackAction::ContinueUnmodified)
        .unwrap();
    let problem = OdeProblem::new(zero, [0.0], (0.0, 1.0), ()).with_callback_set(callbacks);
    assert_eq!(
        solve(&problem, Euler, &options()),
        Err(SolveError::InvalidIterativeCallbackTime)
    );

    let callbacks = IterativeCallback::new(|_: &[f64], _: &[f64], _: &(), _| Some(0.0))
        .into_second_order_callback_set((0.0, 1.0), |_, _, _, _| CallbackAction::ContinueUnmodified)
        .unwrap();
    let problem = SecondOrderOdeProblem::new(
        |a: &mut [f64], _: &[f64], _: &[f64], _: &(), _| a.fill(0.0),
        [0.0],
        [0.0],
        (0.0, 1.0),
        (),
    )
    .with_callback_set(callbacks);
    assert_eq!(
        solve_second_order(&problem, Nystrom4, &options()).unwrap_err(),
        SecondOrderSolveError::Solve(SolveError::InvalidIterativeCallbackTime)
    );
}

#[test]
fn invalid_configuration_and_mismatched_start_are_rejected() {
    for span in [
        (0.0, 0.0),
        (f64::NAN, 1.0),
        (0.0, f64::INFINITY),
        (-f64::MAX, f64::MAX),
    ] {
        assert!(
            IterativeCallback::new(|_: &[f64], _: &(), _| None)
                .into_callback_set(span, |_, _, _| CallbackAction::ContinueUnmodified)
                .is_err()
        );
        assert!(
            IterativeCallback::new(|_: &[f64], _: &[f64], _: &(), _| None)
                .into_second_order_callback_set(span, |_, _, _, _| {
                    CallbackAction::ContinueUnmodified
                })
                .is_err()
        );
    }
    let callbacks = IterativeCallback::new(|_: &[f64], _: &(), _| None)
        .into_callback_set((1.0, 2.0), |_, _, _| CallbackAction::ContinueUnmodified)
        .unwrap();
    let problem = OdeProblem::new(zero, [0.0], (0.0, 2.0), ()).with_callback_set(callbacks);
    assert_eq!(
        solve(&problem, Euler, &options()),
        Err(SolveError::InvalidIterativeCallbackTime)
    );
}

#[test]
fn failed_effects_do_not_request_another_time() {
    for nonfinite_state in [false, true] {
        let callbacks = IterativeCallback::new(|_: &[f64], _: &(), _| {
            panic!("invalid effect must stop scheduling")
        })
        .with_initial_affect(true)
        .into_callback_set((0.0, 1.0), move |u, _, _| {
            if nonfinite_state {
                u[0] = f64::NAN;
                CallbackAction::Continue
            } else {
                CallbackAction::ContinueWithStepSize(0.0)
            }
        })
        .unwrap();
        let problem = OdeProblem::new(zero, [0.0], (0.0, 1.0), ()).with_callback_set(callbacks);
        let error = if nonfinite_state {
            SolveError::NonFiniteCallbackState
        } else {
            SolveError::InvalidCallbackStepSize
        };
        assert_eq!(solve(&problem, Euler, &options()), Err(error));
    }
}

#[test]
fn schedules_restart_after_a_failed_solve() {
    let fail = Rc::new(Cell::new(true));
    let choice_fail = Rc::clone(&fail);
    let callbacks = IterativeCallback::new(move |_: &[f64], _: &(), t| {
        Some(if t > 0.0 && choice_fail.get() {
            t
        } else {
            t + 0.5
        })
    })
    .into_callback_set((0.0, 1.0), |u, _, _| {
        u[0] += 1.0;
        CallbackAction::Continue
    })
    .unwrap();
    let problem = OdeProblem::new(zero, [0.0], (0.0, 1.0), ()).with_callback_set(callbacks);
    assert_eq!(
        solve(&problem, Euler, &options()),
        Err(SolveError::InvalidIterativeCallbackTime)
    );
    fail.set(false);
    assert_eq!(
        solve(&problem, Euler, &options()).unwrap().last_state(),
        &[2.0]
    );
}

#[test]
fn pending_times_survive_continuous_roots_and_other_exact_stops() {
    let choices = Rc::new(Cell::new(0));
    let observed_choices = Rc::clone(&choices);
    let effects = Rc::new(RefCell::new(Vec::new()));
    let observed_effects = Rc::clone(&effects);
    let callbacks = IterativeCallback::new(move |_: &[f64], _: &(), t| {
        observed_choices.set(observed_choices.get() + 1);
        Some(t + 0.5)
    })
    .into_callback_set((0.0, 1.0), move |_, _, t| {
        observed_effects.borrow_mut().push(t);
        CallbackAction::ContinueUnmodifiedWithStepSize(0.8)
    })
    .unwrap();
    let roots = Rc::new(Cell::new(0));
    let observed_roots = Rc::clone(&roots);
    let problem = OdeProblem::new(zero, [0.0], (0.0, 1.0), ())
        .with_continuous_callback(
            |_, _, t| t - 0.25,
            move |_, _, _| {
                observed_roots.set(observed_roots.get() + 1);
                CallbackAction::ContinueUnmodified
            },
        )
        .with_callback_set(callbacks);
    let solution = solve(
        &problem,
        Tsit5,
        &options().with_time_stops(vec![0.125, 0.75]),
    )
    .unwrap();
    assert_eq!(roots.get(), 1);
    assert_eq!(choices.get(), 2);
    assert_eq!(*effects.borrow(), [0.5, 1.0]);
    assert_eq!(solution.stats().callback_invocations, 3);
}

#[test]
fn second_order_rescheduling_errors_propagate_after_effects() {
    let calls = Rc::new(Cell::new(0));
    let observed = Rc::clone(&calls);
    let callbacks = IterativeCallback::new(|_: &[f64], _: &[f64], _: &(), _| Some(0.5))
        .into_second_order_callback_set((0.0, 1.0), move |_, _, _, _| {
            observed.set(observed.get() + 1);
            CallbackAction::ContinueUnmodified
        })
        .unwrap();
    let problem = SecondOrderOdeProblem::new(
        |a: &mut [f64], _: &[f64], _: &[f64], _: &(), _| a.fill(0.0),
        [0.0],
        [0.0],
        (0.0, 1.0),
        (),
    )
    .with_callback_set(callbacks);
    assert_eq!(
        solve_second_order(&problem, Nystrom4, &options()).unwrap_err(),
        SecondOrderSolveError::Solve(SolveError::InvalidIterativeCallbackTime)
    );
    assert_eq!(calls.get(), 1);
}
