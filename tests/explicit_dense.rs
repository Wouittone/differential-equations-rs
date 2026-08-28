use differential_equations::solvers::explicit::*;
use differential_equations::*;

fn cubic_rate(derivative: &mut [f64], _: &[f64], _: &(), time: f64) {
    derivative[0] = 3.0 * time * time;
}

fn fixed_cubic(span: (f64, f64), initial: f64, step: f64, save_at: Vec<f64>) -> Vec<f64> {
    let problem = OdeProblem::new(cubic_rate, vec![initial], span, ());
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        save: SaveMode::Endpoints,
        save_at,
        ..SolveOptions::default()
    };
    solve(&problem, Rk4, &options).unwrap().values().to_vec()
}

#[test]
fn rk4_save_at_uses_endpoint_hermite_forward_and_backward() {
    let forward = fixed_cubic((0.0, 1.0), 0.0, 1.0, vec![0.25, 0.75]);
    assert!((forward[0] - 0.015625).abs() < 1.0e-14);
    assert!((forward[1] - 0.421875).abs() < 1.0e-14);

    let backward = fixed_cubic((1.0, 0.0), 1.0, 1.0, vec![0.75, 0.25]);
    assert!((backward[0] - 0.421875).abs() < 1.0e-14);
    assert!((backward[1] - 0.015625).abs() < 1.0e-14);
}

#[test]
fn rk4_dense_sampling_preserves_exact_endpoints() {
    let problem = OdeProblem::new(cubic_rate, vec![0.0], (0.0, 1.0), ());
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(1.0),
        save_at: vec![0.0, 1.0],
        ..SolveOptions::default()
    };
    let solution = solve(&problem, Rk4, &options).unwrap();
    assert_eq!(solution.times(), &[0.0, 1.0]);
    assert_eq!(solution.values(), &[0.0, 1.0]);
}

#[test]
fn rejected_explicit_attempts_do_not_emit_dense_samples() {
    let problem = OdeProblem::new(
        |derivative: &mut [f64], state: &[f64], _: &(), _: f64| {
            derivative[0] = state[0];
        },
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions {
        initial_step: Some(1.0),
        absolute_tolerance: 1.0e-12,
        relative_tolerance: 1.0e-12,
        save_at: vec![0.25, 0.5, 0.75],
        ..SolveOptions::default()
    };
    let solution = solve(&problem, Tsit5, &options).unwrap();
    assert!(solution.stats().rejected_steps > 0);
    assert_eq!(solution.times(), &[0.25, 0.5, 0.75]);
    assert_eq!(solution.values().len(), 3);
}

#[test]
fn tsit5_save_at_uses_pinned_method_specific_continuous_extension() {
    let problem = OdeProblem::new(
        |derivative: &mut [f64], state: &[f64], _: &(), _: f64| {
            derivative[0] = state[0];
        },
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(1.0),
        save_at: vec![0.25, 0.75],
        ..SolveOptions::default()
    };
    let solution = solve(&problem, Tsit5, &options).unwrap();

    assert!((solution.values()[0] - 1.284_013_054_169_605_8).abs() < 2.0e-14);
    assert!((solution.values()[1] - 2.116_634_262_977_034_3).abs() < 2.0e-14);
    assert_eq!(solution.stats().rhs_evaluations, 7);
}

type ExponentialRhs = fn(&mut [f64], &[f64], &(), f64);

fn exponential_rhs(derivative: &mut [f64], state: &[f64], _: &(), _: f64) {
    derivative[0] = state[0];
}

fn exponential_problem(initial: f64, span: (f64, f64)) -> OdeProblem<ExponentialRhs, ()> {
    OdeProblem::new(exponential_rhs as ExponentialRhs, vec![initial], span, ())
}

#[test]
fn tsit5_continuous_callback_and_pre_root_save_at_share_the_full_step_extension() {
    let problem = exponential_problem(1.0, (0.0, 1.0)).with_continuous_callback(
        |state, _: &(), _: f64| state[0] * state[0] - 3.24,
        |_: &mut [f64], _: &(), _: f64| CallbackAction::Terminate,
    );
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(1.0),
        event_tolerance: 1.0e-13,
        save_at: vec![0.25, 0.5],
        ..SolveOptions::default()
    };
    let solution = solve(&problem, Tsit5, &options).unwrap();

    assert_eq!(solution.stats().callback_invocations, 1);
    assert_eq!(&solution.times()[..2], &[0.25, 0.5]);
    assert!((solution.values()[0] - 1.284_013_054_169_605_8).abs() < 2.0e-14);
    assert!((solution.values()[1] - 1.648_457_727_049_976_3).abs() < 2.0e-14);
    let root_index = solution.times().len() - 1;
    assert!(solution.times()[root_index] > 0.55);
    assert!((solution.state(root_index).unwrap()[0] - 1.8).abs() < 2.0e-12);
}

#[test]
fn retained_tsit5_segments_drive_post_solve_interpolation_forward_and_backward() {
    let forward_options = SolveOptions {
        adaptive: false,
        initial_step: Some(1.0),
        save: SaveMode::Endpoints,
        retain_dense_output: true,
        ..SolveOptions::default()
    };
    let forward = solve(
        &exponential_problem(1.0, (0.0, 1.0)),
        Tsit5,
        &forward_options,
    )
    .unwrap();
    assert_eq!(forward.times(), &[0.0, 1.0]);
    assert_eq!(forward.interpolate(0.0).unwrap(), forward.state(0).unwrap());
    assert_eq!(forward.interpolate(1.0).unwrap(), forward.state(1).unwrap());
    assert!((forward.interpolate(0.25).unwrap()[0] - 1.284_013_054_169_605_8).abs() < 2.0e-14);
    assert!((forward.interpolate(0.75).unwrap()[0] - 2.116_634_262_977_034_3).abs() < 2.0e-14);

    let backward = solve(
        &exponential_problem(1.0_f64.exp(), (1.0, 0.0)),
        Tsit5,
        &forward_options,
    )
    .unwrap();
    assert_eq!(backward.times(), &[1.0, 0.0]);
    assert!((backward.interpolate(0.75).unwrap()[0] - 2.116_000_526_129_777_6).abs() < 2.0e-14);
    assert!((backward.interpolate(0.25).unwrap()[0] - 1.283_736_699_170_017).abs() < 2.0e-14);
}

#[test]
fn dense_retention_is_opt_in_and_callback_endpoints_are_post_effect_states() {
    let default_options = SolveOptions {
        adaptive: false,
        initial_step: Some(1.0),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let plain = solve(
        &exponential_problem(1.0, (0.0, 1.0)),
        Tsit5,
        &default_options,
    )
    .unwrap();
    let linear_midpoint = 0.5 * (plain.state(0).unwrap()[0] + plain.state(1).unwrap()[0]);
    assert_eq!(plain.interpolate(0.5).unwrap(), vec![linear_midpoint]);

    let problem = exponential_problem(1.0, (0.0, 1.0)).with_continuous_callback(
        |state, _: &(), _: f64| state[0] * state[0] - 3.24,
        |state: &mut [f64], _: &(), _: f64| {
            state[0] = 10.0;
            CallbackAction::Continue
        },
    );
    let retained_options = SolveOptions {
        retain_dense_output: true,
        event_tolerance: 1.0e-13,
        ..default_options
    };
    let solution = solve(&problem, Tsit5, &retained_options).unwrap();
    assert_eq!(solution.stats().callback_invocations, 1);
    let event_time = solution.times()[1];
    assert_eq!(solution.interpolate(event_time).unwrap(), vec![10.0]);
    assert!((solution.interpolate(0.5).unwrap()[0] - 1.648_457_727_049_976_3).abs() < 2.0e-14);
    assert!(solution.interpolate(0.75).unwrap()[0] > 10.0);
}
