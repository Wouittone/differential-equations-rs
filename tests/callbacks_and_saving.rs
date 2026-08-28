use differential_equations::solvers::{explicit::*, implicit::*};
use differential_equations::solvers::{multistep::Ab3, rosenbrock::Rosenbrock23};
use differential_equations::*;

fn unit_rate(du: &mut [f64], _: &[f64], _: &(), _: f64) {
    du[0] = 1.0;
}

#[test]
fn continuous_callback_localizes_and_terminates() {
    let problem = OdeProblem::new(unit_rate, vec![0.0], (0.0, 2.0), ()).with_continuous_callback(
        |state, _: &(), _| state[0] - 0.75,
        |state, _: &(), _| {
            state[0] = 42.0;
            CallbackAction::Terminate
        },
    );
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.5),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };

    let solution = solve(&problem, Rk4, &options).unwrap();

    assert!((solution.times()[1] - 0.75).abs() < 1.0e-14);
    assert_eq!(solution.last_state(), &[42.0]);
    assert_eq!(solution.stats().callback_invocations, 1);
    assert_eq!(solution.stats().accepted_steps, 2);
}

#[test]
fn continuous_callback_uses_the_configured_event_tolerance() {
    let root = 0.123_456_789;
    let problem = OdeProblem::new(unit_rate, vec![0.0], (0.0, 1.0), ()).with_continuous_callback(
        move |state, _: &(), _| state[0] - root,
        |_, _: &(), _| CallbackAction::Terminate,
    );
    let options = SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(1.0)
        .with_event_tolerance(1.0e-8)
        .with_save(SaveMode::Endpoints);

    let solution = solve(&problem, Rk4, &options).unwrap();

    assert!((solution.times()[1] - root).abs() <= options.event_tolerance);
}

#[test]
fn direction_filter_ignores_the_opposite_crossing() {
    let problem = OdeProblem::new(unit_rate, vec![0.0], (0.0, 1.0), ())
        .with_continuous_callback_direction(
            EventDirection::Falling,
            |state, _: &(), _| state[0] - 0.5,
            |_, _: &(), _| CallbackAction::Terminate,
        );
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.25),
        ..SolveOptions::default()
    };

    let solution = solve(&problem, Rk4, &options).unwrap();

    assert_eq!(solution.times().last(), Some(&1.0));
    assert_eq!(solution.stats().callback_invocations, 0);
}

#[test]
fn initial_discrete_callback_can_change_state_and_terminate() {
    let problem = OdeProblem::new(unit_rate, vec![0.0], (0.0, 1.0), ()).with_discrete_callback(
        |_, _: &(), time| time == 0.0,
        |state, _: &(), _| {
            state[0] = 7.0;
            CallbackAction::Terminate
        },
    );

    let solution = solve(&problem, Tsit5, &SolveOptions::default()).unwrap();

    assert_eq!(solution.times(), &[0.0]);
    assert_eq!(solution.last_state(), &[7.0]);
    assert_eq!(solution.stats().rhs_evaluations, 0);
    assert_eq!(solution.stats().callback_invocations, 1);
}

fn assert_state_change_invalidates_cache<A: OdeAlgorithm>(algorithm: A, options: SolveOptions) {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    )
    .with_discrete_callback(
        |state, _: &(), time| time >= 0.5 && state[0] < 5.0,
        |state, _: &(), _| {
            state[0] += 10.0;
            CallbackAction::Continue
        },
    );

    let solution = solve(&problem, algorithm, &options).unwrap();

    assert_eq!(solution.stats().callback_invocations, 1);
    assert!(solution.last_state()[0] > 5.0);
}

#[test]
fn callbacks_work_across_solver_families() {
    let fixed = SolveOptions {
        adaptive: false,
        initial_step: Some(0.1),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    assert_state_change_invalidates_cache(Rk4, fixed.clone());
    assert_state_change_invalidates_cache(Ab3, fixed.clone());
    assert_state_change_invalidates_cache(ImplicitEuler, fixed);

    let adaptive = SolveOptions {
        initial_step: Some(0.1),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    assert_state_change_invalidates_cache(Tsit5, adaptive.clone());
    assert_state_change_invalidates_cache(Rosenbrock23, adaptive);
}

fn assert_termination_does_not_evaluate_affected_state<A: OdeAlgorithm>(
    algorithm: A,
    options: SolveOptions,
) {
    let problem = OdeProblem::new(
        |du: &mut [f64], state: &[f64], _: &(), _: f64| {
            du[0] = if state[0] >= 40.0 { f64::NAN } else { 1.0 };
        },
        vec![0.0],
        (0.0, 1.0),
        (),
    )
    .with_continuous_callback(
        |state, _: &(), _| state[0] - 0.5,
        |state, _: &(), _| {
            state[0] = 42.0;
            CallbackAction::Terminate
        },
    );

    let solution = solve(&problem, algorithm, &options).unwrap();
    assert_eq!(solution.last_state(), &[42.0]);
}

#[test]
fn terminating_callbacks_do_not_evaluate_the_affected_state() {
    let fixed = SolveOptions {
        adaptive: false,
        initial_step: Some(0.25),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    assert_termination_does_not_evaluate_affected_state(Rk4, fixed.clone());
    assert_termination_does_not_evaluate_affected_state(Ab3, fixed.clone());
    assert_termination_does_not_evaluate_affected_state(ImplicitEuler, fixed);

    let adaptive = SolveOptions {
        initial_step: Some(0.25),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    assert_termination_does_not_evaluate_affected_state(Tsit5, adaptive.clone());
    assert_termination_does_not_evaluate_affected_state(Rosenbrock23, adaptive);
}

#[test]
fn continuing_callback_forces_the_affected_state_to_be_saved() {
    let problem = OdeProblem::new(unit_rate, vec![0.0], (0.0, 1.0), ()).with_continuous_callback(
        |state, _: &(), _| state[0] - 0.5,
        |state, _: &(), _| {
            state[0] = 10.0;
            CallbackAction::Continue
        },
    );
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.25),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };

    let solution = solve(&problem, Rk4, &options).unwrap();

    assert_eq!(solution.times().len(), 3);
    assert!((solution.times()[1] - 0.5).abs() < 2.0e-15);
    assert_eq!(solution.times()[0], 0.0);
    assert_eq!(solution.times()[2], 1.0);
    assert_eq!(solution.state(1), Some([10.0].as_slice()));
}

#[test]
fn save_at_samples_forward_and_backward_trajectories() {
    let forward = OdeProblem::new(unit_rate, vec![0.0], (0.0, 1.0), ());
    let forward_options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.3),
        save_at: vec![0.2, 0.5, 0.8],
        ..SolveOptions::default()
    };
    let solution = solve(&forward, Rk4, &forward_options).unwrap();
    assert_eq!(solution.times(), &[0.2, 0.5, 0.8]);
    for (&time, state) in solution.times().iter().zip(solution.values()) {
        assert!((time - state).abs() < 2.0e-15);
    }

    let backward = OdeProblem::new(unit_rate, vec![1.0], (1.0, 0.0), ());
    let backward_options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.3),
        save_at: vec![0.8, 0.5, 0.2],
        ..SolveOptions::default()
    };
    let solution = solve(&backward, Rk4, &backward_options).unwrap();
    assert_eq!(solution.times(), &[0.8, 0.5, 0.2]);
    for (&time, state) in solution.times().iter().zip(solution.values()) {
        assert!((time - state).abs() < 2.0e-15);
    }
}

#[test]
fn callback_effects_do_not_change_earlier_save_at_samples() {
    let continuous = OdeProblem::new(unit_rate, vec![0.0], (0.0, 1.0), ())
        .with_continuous_callback(
            |state, _: &(), _| state[0] - 0.75,
            |state, _: &(), _| {
                state[0] = 42.0;
                CallbackAction::Terminate
            },
        );
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(1.0),
        save_at: vec![0.25, 0.5],
        ..SolveOptions::default()
    };
    let solution = solve(&continuous, Rk4, &options).unwrap();

    assert_eq!(solution.times().len(), 3);
    assert_eq!(&solution.times()[..2], &[0.25, 0.5]);
    assert!((solution.times()[2] - 0.75).abs() < 2.0e-15);
    assert!((solution.state(0).unwrap()[0] - 0.25).abs() < 2.0e-15);
    assert!((solution.state(1).unwrap()[0] - 0.5).abs() < 2.0e-15);
    assert_eq!(solution.state(2), Some([42.0].as_slice()));

    let discrete = OdeProblem::new(unit_rate, vec![0.0], (0.0, 1.0), ()).with_discrete_callback(
        |_, _: &(), time| time >= 0.6,
        |state, _: &(), _| {
            state[0] = 10.0;
            CallbackAction::Terminate
        },
    );
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.6),
        save_at: vec![0.2, 0.5],
        ..SolveOptions::default()
    };
    let solution = solve(&discrete, Rk4, &options).unwrap();

    assert_eq!(solution.times(), &[0.2, 0.5, 0.6]);
    assert!((solution.state(0).unwrap()[0] - 0.2).abs() < 2.0e-15);
    assert!((solution.state(1).unwrap()[0] - 0.5).abs() < 2.0e-15);
    assert_eq!(solution.state(2), Some([10.0].as_slice()));
}

#[test]
fn save_at_validation_rejects_out_of_order_or_out_of_span_times() {
    let problem = OdeProblem::new(unit_rate, vec![0.0], (0.0, 1.0), ());
    for save_at in [vec![0.5, 0.25], vec![-0.1], vec![f64::NAN]] {
        let options = SolveOptions {
            save_at,
            ..SolveOptions::default()
        };
        assert_eq!(
            solve(&problem, Tsit5, &options),
            Err(SolveError::InvalidSaveAt)
        );
    }
}

#[test]
fn non_finite_callback_outputs_are_reported() {
    let condition_problem = OdeProblem::new(unit_rate, vec![0.0], (0.0, 1.0), ())
        .with_continuous_callback(
            |_, _: &(), _| f64::NAN,
            |_, _: &(), _| CallbackAction::Continue,
        );
    assert_eq!(
        solve(&condition_problem, Tsit5, &SolveOptions::default()),
        Err(SolveError::NonFiniteCallbackCondition)
    );

    let state_problem = OdeProblem::new(unit_rate, vec![0.0], (0.0, 1.0), ())
        .with_discrete_callback(
            |_, _: &(), _| true,
            |state, _: &(), _| {
                state[0] = f64::INFINITY;
                CallbackAction::Continue
            },
        );
    assert_eq!(
        solve(&state_problem, Tsit5, &SolveOptions::default()),
        Err(SolveError::NonFiniteCallbackState)
    );
}
