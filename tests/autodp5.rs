use std::cell::{Cell, RefCell};
use std::rc::Rc;

use differential_equations::callbacks::DomainGuard;
use differential_equations::ndarray::{
    ArrayView0, ArrayView1, ArrayView2, ArrayViewMut0, ArrayViewMut1, ArrayViewMut2, arr0, array,
};
use differential_equations::solvers::automatic::{AutoDp5, AutoSwitchConfig, AutomaticBranch};
use differential_equations::solvers::explicit::Dp5;
use differential_equations::solvers::rosenbrock::Rodas5P;
use differential_equations::{
    CallbackAction, CallbackSave, CallbackSet, OdeProblem, SaveMode, SolveOptions, solve,
};

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn decay(derivative: &mut [f64], state: &[f64], _: &(), _: f64) {
    derivative[0] = -state[0];
}

fn decay_problem(initial: f64, span: (f64, f64)) -> OdeProblem<TestRhs, ()> {
    OdeProblem::new(decay as TestRhs, [initial], span, ())
}

fn fixed_options(step: f64) -> SolveOptions {
    SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(step)
        .with_max_step(step)
        .with_save(SaveMode::Endpoints)
}

fn forced_switch_config() -> AutoSwitchConfig {
    AutoSwitchConfig::new()
        .with_stiffness_thresholds(1.0e-14, 1.0e-12)
        .unwrap()
        .with_stiff_confirmations(1)
        .unwrap()
        .with_minimum_residence_steps(1)
        .unwrap()
        .with_switch_step_factor(1.0)
        .unwrap()
        .with_allow_switch_back(false)
}

fn forced_auto() -> AutoDp5<Rodas5P> {
    AutoDp5::new(Rodas5P)
        .with_switch_config(forced_switch_config())
        .unwrap()
}

fn assert_one_in_flight_switch(solution: &differential_equations::Solution) {
    let stats = solution.stats();
    assert_eq!(stats.algorithm_switches, 1);
    assert_eq!(stats.nonstiff_accepted_steps, 1);
    assert!(stats.stiff_accepted_steps > 0);
    assert_eq!(
        stats.nonstiff_accepted_steps + stats.stiff_accepted_steps,
        stats.accepted_steps
    );
    assert_eq!(stats.final_automatic_branch, Some(AutomaticBranch::Stiff));
}

#[test]
fn nonstiff_run_matches_dp5_without_switching_and_preserves_work_stats() {
    let config = AutoSwitchConfig::new()
        .with_stiffness_thresholds(1.0e100, 1.0e200)
        .unwrap();
    let options = fixed_options(0.05);
    let problem = decay_problem(1.0, (0.0, 1.0));

    let automatic = solve(
        &problem,
        AutoDp5::new(Rodas5P).with_switch_config(config).unwrap(),
        &options,
    )
    .unwrap();
    let dp5 = solve(&problem, Dp5, &options).unwrap();

    assert_eq!(automatic.times(), dp5.times());
    for index in 0..automatic.times().len() {
        assert_eq!(automatic.state(index), dp5.state(index));
    }
    assert_eq!(
        automatic.stats().rhs_evaluations,
        dp5.stats().rhs_evaluations
    );
    assert_eq!(automatic.stats().accepted_steps, dp5.stats().accepted_steps);
    assert_eq!(automatic.stats().rejected_steps, dp5.stats().rejected_steps);
    assert_eq!(automatic.stats().algorithm_switches, 0);
    assert_eq!(
        automatic.stats().nonstiff_accepted_steps,
        automatic.stats().accepted_steps
    );
    assert_eq!(automatic.stats().stiff_accepted_steps, 0);
    assert_eq!(
        automatic.stats().final_automatic_branch,
        Some(AutomaticBranch::NonStiff)
    );
}

#[test]
fn equilibria_and_conserved_components_do_not_create_false_stiffness() {
    let options = fixed_options(0.02);
    let equilibrium = OdeProblem::new(
        |derivative: &mut [f64], _: &[f64], _: &(), _: f64| derivative.fill(0.0),
        [2.0, -3.0],
        (0.0, 1.0),
        (),
    );
    let equilibrium_solution = solve(&equilibrium, AutoDp5::new(Rodas5P), &options).unwrap();
    assert_eq!(equilibrium_solution.last_state(), &[2.0, -3.0]);
    assert_eq!(equilibrium_solution.stats().algorithm_switches, 0);
    assert_eq!(
        equilibrium_solution.stats().final_automatic_branch,
        Some(AutomaticBranch::NonStiff)
    );

    let mixed = OdeProblem::new(
        |derivative: &mut [f64], state: &[f64], _: &(), _: f64| {
            derivative[0] = -state[0];
            derivative[1] = 0.0;
        },
        [1.0, 7.0],
        (0.0, 1.0),
        (),
    );
    let automatic = solve(&mixed, AutoDp5::new(Rodas5P), &options).unwrap();
    let explicit = solve(&mixed, Dp5, &options).unwrap();
    assert_eq!(automatic.times(), explicit.times());
    assert_eq!(automatic.last_state(), explicit.last_state());
    assert_eq!(automatic.stats().algorithm_switches, 0);
    assert_eq!(automatic.last_state()[1], 7.0);
}

#[test]
fn switches_after_the_first_accepted_step_forward_and_backward() {
    let forward = solve(
        &decay_problem(1.0, (0.0, 1.0)),
        forced_auto(),
        &fixed_options(0.05),
    )
    .unwrap();
    assert_one_in_flight_switch(&forward);
    assert!((forward.last_state()[0] - (-1.0_f64).exp()).abs() < 2.0e-8);

    let backward = solve(
        &decay_problem((-1.0_f64).exp(), (1.0, 0.0)),
        forced_auto(),
        &fixed_options(0.05),
    )
    .unwrap();
    assert_one_in_flight_switch(&backward);
    assert!((backward.last_state()[0] - 1.0).abs() < 2.0e-8);
}

#[derive(Default)]
struct MutableRate {
    value: Cell<f64>,
    evaluations_after_change: Cell<usize>,
}

#[test]
fn switch_does_not_replay_lifecycle_or_lose_callback_and_parameter_state() {
    let initialized = Rc::new(Cell::new(0));
    let affected = Rc::new(Cell::new(0));
    let finalized = Rc::new(Cell::new(0));
    let initialize_count = Rc::clone(&initialized);
    let affect_count = Rc::clone(&affected);
    let finalize_count = Rc::clone(&finalized);
    let callbacks = CallbackSet::<MutableRate>::new()
        .with_initialize(move |state, _, _| {
            initialize_count.set(initialize_count.get() + 1);
            state[0] = 2.0;
        })
        .with_preset_time_callback([0.5], move |state, rate, _| {
            affect_count.set(affect_count.get() + 1);
            state[0] += 3.0;
            rate.value.set(2.0);
            CallbackAction::Continue
        })
        .with_finalize(move |state, _, _| {
            finalize_count.set(finalize_count.get() + 1);
            state[0] += 5.0;
        });
    let problem = OdeProblem::new(
        |derivative: &mut [f64], state: &[f64], rate: &MutableRate, _: f64| {
            if rate.value.get() == 2.0 {
                rate.evaluations_after_change
                    .set(rate.evaluations_after_change.get() + 1);
            }
            derivative[0] = rate.value.get() * state[0];
        },
        [1.0],
        (0.0, 1.0),
        MutableRate {
            value: Cell::new(1.0),
            evaluations_after_change: Cell::new(0),
        },
    )
    .with_callback_set(callbacks);

    let solution = solve(&problem, forced_auto(), &fixed_options(0.025)).unwrap();
    let expected = (2.0 * 0.5_f64.exp() + 3.0) * 1.0_f64.exp() + 5.0;

    assert_one_in_flight_switch(&solution);
    assert_eq!(initialized.get(), 1);
    assert_eq!(affected.get(), 1);
    assert_eq!(finalized.get(), 1);
    assert_eq!(solution.stats().callback_invocations, 1);
    assert_eq!(problem.parameters().value.get(), 2.0);
    assert!(problem.parameters().evaluations_after_change.get() > 0);
    assert!((solution.last_state()[0] - expected).abs() < 2.0e-6);
}

#[test]
fn time_stops_save_at_and_retained_dense_output_span_the_switch() {
    let stop_hits = Rc::new(Cell::new(0));
    let observed_hits = Rc::clone(&stop_hits);
    let problem = decay_problem(1.0, (0.0, 1.0)).with_discrete_callback_saving(
        CallbackSave::None,
        |_, _, time| (time - 0.35).abs() < 1.0e-14,
        move |_, _, _| {
            observed_hits.set(observed_hits.get() + 1);
            CallbackAction::ContinueUnmodified
        },
    );
    let options = fixed_options(0.2)
        .with_time_stops([0.35, 0.6])
        .with_save_at([0.0, 0.1, 0.35, 0.55, 0.9, 1.0])
        .with_dense_output(true);

    let solution = solve(&problem, forced_auto(), &options).unwrap();

    assert_one_in_flight_switch(&solution);
    assert_eq!(stop_hits.get(), 1);
    assert_eq!(solution.times(), &[0.0, 0.1, 0.35, 0.55, 0.9, 1.0]);
    for time in [0.05, 0.1, 0.21, 0.35, 0.7, 0.95] {
        let interpolated = solution.interpolate(time).unwrap()[0];
        assert!(
            (interpolated - (-time).exp()).abs() < 2.0e-5,
            "dense output at {time}: {interpolated}"
        );
    }
}

#[test]
fn callback_can_terminate_after_the_switch_without_restarting() {
    let invocations = Rc::new(Cell::new(0));
    let observed = Rc::clone(&invocations);
    let problem =
        decay_problem(1.0, (0.0, 1.0)).with_preset_time_callback([0.4], move |state, _, _| {
            observed.set(observed.get() + 1);
            state[0] = 42.0;
            CallbackAction::Terminate
        });

    let solution = solve(&problem, forced_auto(), &fixed_options(0.1)).unwrap();

    assert_one_in_flight_switch(&solution);
    assert_eq!(invocations.get(), 1);
    assert_eq!(solution.stats().callback_invocations, 1);
    assert_eq!(solution.times().last().copied(), Some(0.4));
    assert_eq!(solution.last_state(), &[42.0]);
}

#[test]
fn switched_array_problems_preserve_scalar_vector_and_matrix_shapes() {
    let options = fixed_options(0.05).with_dense_output(true);
    let scalar = OdeProblem::from_array(
        |mut derivative: ArrayViewMut0<'_, f64>, state: ArrayView0<'_, f64>, _: &(), _: f64| {
            derivative[[]] = -state[[]];
        },
        arr0(1.0),
        (0.0, 0.2),
        (),
    );
    let vector = OdeProblem::from_array(
        |mut derivative: ArrayViewMut1<'_, f64>, state: ArrayView1<'_, f64>, _: &(), _: f64| {
            derivative.zip_mut_with(&state, |derivative, state| *derivative = -*state);
        },
        array![1.0, 2.0],
        (0.0, 0.2),
        (),
    );
    let matrix = OdeProblem::from_array(
        |mut derivative: ArrayViewMut2<'_, f64>, state: ArrayView2<'_, f64>, _: &(), _: f64| {
            derivative.zip_mut_with(&state, |derivative, state| *derivative = -*state);
        },
        array![[1.0, 2.0], [3.0, 4.0]],
        (0.0, 0.2),
        (),
    );

    let scalar_solution = solve(&scalar, forced_auto(), &options).unwrap();
    let vector_solution = solve(&vector, forced_auto(), &options).unwrap();
    let matrix_solution = solve(&matrix, forced_auto(), &options).unwrap();

    for solution in [&scalar_solution, &vector_solution, &matrix_solution] {
        assert_one_in_flight_switch(solution);
    }
    assert!(scalar_solution.state_shape().is_empty());
    assert!(
        scalar_solution
            .interpolate_array(0.125)
            .unwrap()
            .shape()
            .is_empty()
    );
    assert_eq!(vector_solution.state_shape(), &[2]);
    assert_eq!(vector_solution.last_state_array().shape(), &[2]);
    assert_eq!(matrix_solution.state_shape(), &[2, 2]);
    assert_eq!(
        matrix_solution.interpolate_array(0.125).unwrap().shape(),
        &[2, 2]
    );

    let scale = (-0.2_f64).exp();
    for (actual, initial) in vector_solution
        .last_state()
        .iter()
        .chain(matrix_solution.last_state())
        .zip([1.0, 2.0, 1.0, 2.0, 3.0, 4.0])
    {
        assert!((*actual - initial * scale).abs() < 2.0e-8);
    }
}

#[test]
fn initially_stiff_configuration_uses_only_the_stiff_branch() {
    let config = AutoSwitchConfig::new()
        .with_initial_branch(AutomaticBranch::Stiff)
        .with_allow_switch_back(false);
    let algorithm = AutoDp5::new(Rodas5P).with_switch_config(config).unwrap();

    let solution = solve(
        &decay_problem(1.0, (0.0, 0.5)),
        algorithm,
        &fixed_options(0.025),
    )
    .unwrap();

    assert_eq!(solution.stats().algorithm_switches, 0);
    assert_eq!(solution.stats().nonstiff_accepted_steps, 0);
    assert_eq!(
        solution.stats().stiff_accepted_steps,
        solution.stats().accepted_steps
    );
    assert_eq!(
        solution.stats().final_automatic_branch,
        Some(AutomaticBranch::Stiff)
    );
    assert!((solution.last_state()[0] - (-0.5_f64).exp()).abs() < 2.0e-9);
}

#[test]
fn disappearing_stiffness_switches_back_without_replaying_time() {
    let visited_after_transition = Rc::new(Cell::new(0));
    let observed_after_transition = Rc::clone(&visited_after_transition);
    let problem = OdeProblem::new(
        move |derivative: &mut [f64], state: &[f64], _: &(), time: f64| {
            if time >= 0.3 {
                observed_after_transition.set(observed_after_transition.get() + 1);
            }
            let rate = if time < 0.3 { -100.0 } else { -1.0 };
            derivative[0] = rate * state[0];
        },
        [1.0],
        (0.0, 0.6),
        (),
    )
    .with_jacobian(|jacobian, _, _, time| {
        jacobian[0] = if time < 0.3 { -100.0 } else { -1.0 };
    });
    let config = AutoSwitchConfig::new()
        .with_stiffness_thresholds(0.1, 0.5)
        .unwrap()
        .with_nonstiff_confirmations(1)
        .unwrap()
        .with_minimum_residence_steps(1)
        .unwrap()
        .with_switch_step_factor(1.0)
        .unwrap()
        .with_initial_branch(AutomaticBranch::Stiff);

    let solution = solve(
        &problem,
        AutoDp5::new(Rodas5P).with_switch_config(config).unwrap(),
        &fixed_options(0.01),
    )
    .unwrap();

    assert_eq!(solution.stats().algorithm_switches, 1);
    assert!(solution.stats().stiff_accepted_steps >= 30);
    assert!(solution.stats().nonstiff_accepted_steps > 0);
    assert_eq!(
        solution.stats().final_automatic_branch,
        Some(AutomaticBranch::NonStiff)
    );
    assert!(visited_after_transition.get() > 0);
}

#[test]
fn repeated_branch_entry_preserves_dense_output_and_callback_lifecycle() {
    let callback_invocations = Rc::new(Cell::new(0));
    let observed_invocations = Rc::clone(&callback_invocations);
    let problem = OdeProblem::new(
        |derivative: &mut [f64], state: &[f64], _: &(), time: f64| {
            let rate = if (0.2..0.4).contains(&time) {
                100.0
            } else {
                1.0
            };
            derivative[0] = -rate * (state[0] - time.cos()) - time.sin();
        },
        [1.0],
        (0.0, 0.6),
        (),
    )
    .with_jacobian(|jacobian, _, _, time| {
        jacobian[0] = if (0.2..0.4).contains(&time) {
            -100.0
        } else {
            -1.0
        };
    })
    .with_preset_time_callback_saving([0.5], CallbackSave::None, move |_, _, _| {
        observed_invocations.set(observed_invocations.get() + 1);
        CallbackAction::ContinueUnmodified
    });
    let config = AutoSwitchConfig::new()
        .with_stiffness_thresholds(0.1, 0.2)
        .unwrap()
        .with_stiff_confirmations(1)
        .unwrap()
        .with_nonstiff_confirmations(1)
        .unwrap()
        .with_minimum_residence_steps(1)
        .unwrap()
        .with_switch_step_factor(1.0)
        .unwrap();
    let options = fixed_options(0.01)
        .with_time_stops([0.2, 0.4])
        .with_save_at([0.0, 0.1, 0.25, 0.45, 0.5, 0.55, 0.6])
        .with_dense_output(true);

    let solution = solve(
        &problem,
        AutoDp5::new(Rodas5P).with_switch_config(config).unwrap(),
        &options,
    )
    .unwrap();

    assert_eq!(solution.stats().algorithm_switches, 2);
    assert!(solution.stats().nonstiff_accepted_steps > 0);
    assert!(solution.stats().stiff_accepted_steps > 0);
    assert_eq!(
        solution.stats().nonstiff_accepted_steps + solution.stats().stiff_accepted_steps,
        solution.stats().accepted_steps
    );
    assert_eq!(
        solution.stats().final_automatic_branch,
        Some(AutomaticBranch::NonStiff)
    );
    assert_eq!(callback_invocations.get(), 1);
    assert_eq!(solution.stats().callback_invocations, 1);
    for time in [0.1, 0.25, 0.35, 0.45, 0.5, 0.55] {
        let interpolated = solution.interpolate(time).unwrap()[0];
        assert!(
            (interpolated - time.cos()).abs() < 2.0e-5,
            "dense output at {time}: {interpolated}"
        );
    }
}

#[test]
fn explicit_error_rejections_can_trigger_a_stiff_handoff() {
    let problem = OdeProblem::new(
        |derivative: &mut [f64], state: &[f64], _: &(), _: f64| {
            derivative[0] = -10.0 * state[0];
        },
        [1.0],
        (0.0, 1.0),
        (),
    )
    .with_jacobian(|jacobian, _, _, _| jacobian[0] = -10.0);
    let config = AutoSwitchConfig::new()
        .with_stiffness_thresholds(1.0e100, 1.0e200)
        .unwrap()
        .with_rejection_confirmations(1)
        .unwrap()
        .with_minimum_residence_steps(100)
        .unwrap()
        .with_switch_step_factor(1.0)
        .unwrap()
        .with_allow_switch_back(false);
    let options = SolveOptions::new()
        .with_tolerances(1.0e-10, 1.0e-10)
        .with_initial_step(0.5)
        .with_max_step(0.5)
        .with_save(SaveMode::Endpoints);

    let solution = solve(
        &problem,
        AutoDp5::new(Rodas5P).with_switch_config(config).unwrap(),
        &options,
    )
    .unwrap();

    assert_eq!(solution.stats().algorithm_switches, 1);
    assert!(solution.stats().rejected_steps >= 1);
    assert_eq!(solution.stats().nonstiff_accepted_steps, 0);
    assert!(solution.stats().stiff_accepted_steps > 0);
    assert_eq!(
        solution.stats().final_automatic_branch,
        Some(AutomaticBranch::Stiff)
    );
    assert!((solution.last_state()[0] - (-10.0_f64).exp()).abs() < 2.0e-9);
}

#[test]
fn domain_rejections_do_not_count_as_stiffness_evidence() {
    let reject_first = Cell::new(true);
    let callbacks =
        DomainGuard::new(move |_: &[f64], _: &(), time| time != 0.0 && reject_first.replace(false))
            .into_callback_set()
            .unwrap();
    let problem = decay_problem(1.0, (0.0, 0.2)).with_callback_set(callbacks);
    let config = AutoSwitchConfig::new()
        .with_stiffness_thresholds(1.0e100, 1.0e200)
        .unwrap()
        .with_rejection_confirmations(1)
        .unwrap();

    let solution = solve(
        &problem,
        AutoDp5::new(Rodas5P).with_switch_config(config).unwrap(),
        &fixed_options(0.1),
    )
    .unwrap();

    assert_eq!(solution.stats().rejected_steps, 1);
    assert_eq!(solution.stats().algorithm_switches, 0);
    assert_eq!(
        solution.stats().final_automatic_branch,
        Some(AutomaticBranch::NonStiff)
    );
}

#[test]
fn continuous_callback_localization_uses_the_branch_that_produced_the_step() {
    let invocations = Rc::new(Cell::new(0));
    let observed = Rc::clone(&invocations);
    let problem = decay_problem(1.0, (0.0, 1.0)).with_continuous_callback(
        |state, _, _| state[0] - 0.5,
        move |state, _, _| {
            observed.set(observed.get() + 1);
            state[0] = 42.0;
            CallbackAction::Terminate
        },
    );

    let solution = solve(&problem, forced_auto(), &fixed_options(0.1)).unwrap();

    assert_one_in_flight_switch(&solution);
    assert_eq!(invocations.get(), 1);
    assert_eq!(solution.last_state(), &[42.0]);
    assert!((solution.times().last().unwrap() - std::f64::consts::LN_2).abs() < 2.0e-5);
}

#[test]
fn nonfinite_explicit_stage_can_handoff_instead_of_terminating() {
    let problem = OdeProblem::new(
        |derivative: &mut [f64], state: &[f64], _: &(), _: f64| {
            derivative[0] = if state[0].abs() > 5.0 {
                f64::NAN
            } else {
                -100.0 * state[0]
            };
        },
        [1.0],
        (0.0, 0.1),
        (),
    )
    .with_jacobian(|jacobian, _, _, _| jacobian[0] = -100.0);
    let config = AutoSwitchConfig::new()
        .with_stiffness_thresholds(1.0e100, 1.0e200)
        .unwrap()
        .with_rejection_confirmations(1)
        .unwrap()
        .with_switch_step_factor(0.1)
        .unwrap()
        .with_allow_switch_back(false);
    let options = SolveOptions::new()
        .with_tolerances(1.0e-8, 1.0e-8)
        .with_initial_step(0.1)
        .with_max_step(0.1)
        .with_save(SaveMode::Endpoints);

    let solution = solve(
        &problem,
        AutoDp5::new(Rodas5P).with_switch_config(config).unwrap(),
        &options,
    )
    .unwrap();

    assert_eq!(solution.stats().algorithm_switches, 1);
    assert!(solution.stats().rejected_steps >= 1);
    assert_eq!(solution.stats().nonstiff_accepted_steps, 0);
    assert_eq!(
        solution.stats().final_automatic_branch,
        Some(AutomaticBranch::Stiff)
    );
    assert!(solution.last_state()[0].is_finite());
}

#[test]
fn rejection_transitions_respect_max_step_in_both_directions() {
    fn solve_direction(span: (f64, f64)) -> differential_equations::Solution {
        let calls = Rc::new(Cell::new(0));
        let observed_calls = Rc::clone(&calls);
        let problem = OdeProblem::new(
            move |derivative: &mut [f64], state: &[f64], _: &(), _: f64| {
                let call = observed_calls.get();
                observed_calls.set(call + 1);
                derivative[0] = if call < 7 { -100.0 * state[0] } else { 0.0 };
            },
            [1.0],
            span,
            (),
        )
        .with_jacobian(|jacobian, _, _, _| jacobian[0] = 0.0);
        let config = AutoSwitchConfig::new()
            .with_stiffness_thresholds(1.0e100, 1.0e200)
            .unwrap()
            .with_rejection_confirmations(1)
            .unwrap()
            .with_switch_step_factor(100.0)
            .unwrap()
            .with_allow_switch_back(false);
        let options = SolveOptions::new()
            .with_tolerances(1.0e-8, 1.0e-8)
            .with_initial_step(0.1)
            .with_max_step(0.1)
            .with_save(SaveMode::EveryStep);

        solve(
            &problem,
            AutoDp5::new(Rodas5P).with_switch_config(config).unwrap(),
            &options,
        )
        .unwrap()
    }

    for solution in [solve_direction((0.0, 0.5)), solve_direction((0.5, 0.0))] {
        assert_eq!(solution.stats().algorithm_switches, 1);
        assert!(solution.stats().rejected_steps >= 1);
        for times in solution.times().windows(2) {
            assert!((times[1] - times[0]).abs() <= 0.1 + 8.0 * f64::EPSILON);
        }
    }
}

#[test]
fn accepted_intervals_and_effects_are_never_replayed() {
    let accepted_times = Rc::new(RefCell::new(Vec::new()));
    let observed_times = Rc::clone(&accepted_times);
    let problem = decay_problem(1.0, (0.0, 1.0)).with_discrete_callback(
        |_, _, time| time > 0.0,
        move |_, _, time| {
            observed_times.borrow_mut().push(time);
            CallbackAction::ContinueUnmodified
        },
    );

    let solution = solve(&problem, forced_auto(), &fixed_options(0.1)).unwrap();
    let accepted_times = accepted_times.borrow();

    assert_one_in_flight_switch(&solution);
    assert_eq!(accepted_times.len(), solution.stats().accepted_steps);
    assert_eq!(
        solution.stats().callback_invocations,
        solution.stats().accepted_steps
    );
    assert!(accepted_times.windows(2).all(|times| times[0] < times[1]));
    assert!((accepted_times[0] - 0.1).abs() < 8.0 * f64::EPSILON);
    assert!((accepted_times.last().unwrap() - 1.0).abs() < 8.0 * f64::EPSILON);
}

#[test]
fn backward_time_stops_and_dense_output_span_the_switch() {
    let options = fixed_options(0.2)
        .with_time_stops([0.65, 0.4])
        .with_save_at([1.0, 0.9, 0.65, 0.45, 0.1, 0.0])
        .with_dense_output(true);
    let solution = solve(
        &decay_problem((-1.0_f64).exp(), (1.0, 0.0)),
        forced_auto(),
        &options,
    )
    .unwrap();

    assert_one_in_flight_switch(&solution);
    assert_eq!(solution.times(), &[1.0, 0.9, 0.65, 0.45, 0.1, 0.0]);
    for time in [0.95, 0.8, 0.65, 0.42, 0.2, 0.05] {
        let interpolated = solution.interpolate(time).unwrap()[0];
        assert!(
            (interpolated - (-time).exp()).abs() < 2.0e-5,
            "backward dense output at {time}: {interpolated}"
        );
    }
}
