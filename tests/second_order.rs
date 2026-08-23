use differential_equations::algorithms::*;
use differential_equations::*;
use std::error::Error as _;

type Acceleration = fn(&mut [f64], &[f64], &[f64], &(), f64);

fn oscillator() -> SecondOrderOdeProblem<Acceleration, ()> {
    fn acceleration(output: &mut [f64], _: &[f64], position: &[f64], _: &(), _: f64) {
        output[0] = -position[0];
    }
    SecondOrderOdeProblem::new(acceleration, vec![0.0], vec![1.0], (0.0, 1.0), ())
}

fn fixed_options(step: f64) -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

fn endpoint_error<A: SecondOrderOdeAlgorithm>(algorithm: A, step: f64) -> f64 {
    let solution = solve_second_order(&oscillator(), algorithm, &fixed_options(step)).unwrap();
    (solution.last_position()[0] - 1.0_f64.cos()).hypot(solution.last_velocity()[0] + 1.0_f64.sin())
}

#[test]
fn expected_convergence_orders() {
    let first_order_ratio =
        endpoint_error(SymplecticEuler, 0.05) / endpoint_error(SymplecticEuler, 0.025);
    assert!(first_order_ratio > 1.8 && first_order_ratio < 2.2);

    for ratio in [
        endpoint_error(VelocityVerlet, 0.1) / endpoint_error(VelocityVerlet, 0.05),
        endpoint_error(VerletLeapfrog, 0.1) / endpoint_error(VerletLeapfrog, 0.05),
        endpoint_error(LeapfrogDriftKickDrift, 0.1) / endpoint_error(LeapfrogDriftKickDrift, 0.05),
    ] {
        assert!(ratio > 3.7 && ratio < 4.3, "ratio was {ratio}");
    }
}

#[test]
fn verlet_has_bounded_long_time_energy_error() {
    let problem = SecondOrderOdeProblem::new(
        |output: &mut [f64], _: &[f64], position: &[f64], _: &(), _| {
            output[0] = -position[0];
        },
        vec![0.0],
        vec![1.0],
        (0.0, 2_000.0),
        (),
    );
    let solution = solve_second_order(
        &problem,
        VelocityVerlet,
        &SolveOptions {
            adaptive: false,
            initial_step: Some(0.1),
            save: SaveMode::EveryStep,
            ..SolveOptions::default()
        },
    )
    .unwrap();

    let maximum_error = solution
        .position_values()
        .iter()
        .zip(solution.velocity_values())
        .map(|(position, velocity)| (0.5 * (position * position + velocity * velocity) - 0.5).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        maximum_error < 0.0013,
        "maximum energy error was {maximum_error}"
    );
}

#[test]
fn fixed_steps_work_backward_and_honor_save_at() {
    let problem = SecondOrderOdeProblem::new(
        |output: &mut [f64], _: &[f64], position: &[f64], _: &(), _| {
            output[0] = -position[0];
        },
        vec![-1.0_f64.sin()],
        vec![1.0_f64.cos()],
        (1.0, 0.0),
        (),
    );
    let solution = solve_second_order(
        &problem,
        VerletLeapfrog,
        &SolveOptions {
            adaptive: false,
            initial_step: Some(0.01),
            save_at: vec![0.75, 0.5, 0.0],
            ..SolveOptions::default()
        },
    )
    .unwrap();

    assert_eq!(solution.times(), &[0.75, 0.5, 0.0]);
    assert!((solution.last_position()[0] - 1.0).abs() < 1.0e-5);
    assert!(solution.last_velocity()[0].abs() < 1.0e-5);
}

#[test]
fn partitioned_callbacks_can_modify_and_terminate_both_states() {
    let problem = oscillator().with_continuous_callback(
        |_, _, _, time| time - 0.5,
        |velocity, position, _, _| {
            velocity[0] = 2.0;
            position[0] = 3.0;
            CallbackAction::Terminate
        },
    );
    let solution = solve_second_order(&problem, VelocityVerlet, &fixed_options(0.2)).unwrap();

    assert!((solution.times().last().unwrap() - 0.5).abs() < 1.0e-14);
    assert_eq!(solution.last_velocity(), &[2.0]);
    assert_eq!(solution.last_position(), &[3.0]);
    assert_eq!(solution.stats().callback_invocations, 1);
}

#[test]
fn callbacks_reinitialize_cached_acceleration() {
    let problem = oscillator().with_discrete_callback(
        |_, _, _, time| (time - 0.5).abs() < 1.0e-14,
        |velocity, position, _, _| {
            velocity[0] = 0.0;
            position[0] = 2.0;
            CallbackAction::Continue
        },
    );
    let solution = solve_second_order(&problem, VelocityVerlet, &fixed_options(0.1)).unwrap();
    let expected_position = 2.0 * 0.5_f64.cos();
    let expected_velocity = -2.0 * 0.5_f64.sin();

    assert!((solution.last_position()[0] - expected_position).abs() < 0.003);
    assert!((solution.last_velocity()[0] - expected_velocity).abs() < 0.003);
    assert_eq!(solution.stats().callback_invocations, 1);
}

#[test]
fn invalid_partition_and_fixed_step_options_are_reported() {
    let mismatch = SecondOrderOdeProblem::new(
        |_: &mut [f64], _: &[f64], _: &[f64], _: &(), _| {},
        vec![0.0, 1.0],
        vec![0.0],
        (0.0, 1.0),
        (),
    );
    assert_eq!(
        solve_second_order(&mismatch, VelocityVerlet, &fixed_options(0.1)),
        Err(SecondOrderSolveError::StateDimensionMismatch)
    );
    assert_eq!(
        solve_second_order(&oscillator(), VelocityVerlet, &SolveOptions::default()),
        Err(SecondOrderSolveError::Solve(
            SolveError::AdaptiveStepUnsupported
        ))
    );
}

#[test]
fn wrapped_second_order_errors_preserve_their_source() {
    let error = SecondOrderSolveError::from(SolveError::InvalidInitialStep);

    assert_eq!(
        error.to_string(),
        "the initial step must be finite and positive"
    );
    assert_eq!(
        error.source().map(ToString::to_string),
        Some("the initial step must be finite and positive".to_owned())
    );
}
