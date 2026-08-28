use differential_equations::solvers::second_order::{
    Dprkn6, NewmarkBeta, SecondOrderOdeProblem, SecondOrderSolution, VelocityVerlet,
    solve_second_order,
};
use differential_equations::{CallbackAction, SaveMode, SolveOptions};

type Acceleration = fn(&mut [f64], &[f64], &[f64], &(), f64);

fn oscillator(span: (f64, f64)) -> SecondOrderOdeProblem<Acceleration, ()> {
    fn acceleration(output: &mut [f64], _: &[f64], position: &[f64], _: &(), _: f64) {
        output[0] = -position[0];
    }
    SecondOrderOdeProblem::new(
        acceleration,
        vec![-span.0.sin()],
        vec![span.0.cos()],
        span,
        (),
    )
}

fn dense_options(step: f64) -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        save: SaveMode::Endpoints,
        retain_dense_output: true,
        ..SolveOptions::default()
    }
}

fn assert_oscillator_sample(solution: &SecondOrderSolution, time: f64, tolerance: f64) {
    let (velocity, position) = solution.interpolate(time).unwrap();
    assert!((position[0] - time.cos()).abs() < tolerance, "{position:?}");
    assert!((velocity[0] + time.sin()).abs() < tolerance, "{velocity:?}");
}

#[test]
fn retained_segments_cover_rkn_symplectic_and_structural_solvers() {
    let problem = oscillator((0.0, 1.0));
    let options = dense_options(0.05);

    let rkn = solve_second_order(&problem, Dprkn6, &options).unwrap();
    let symplectic = solve_second_order(&problem, VelocityVerlet, &options).unwrap();
    let structural = solve_second_order(&problem, NewmarkBeta::default(), &options).unwrap();

    assert_oscillator_sample(&rkn, 0.375, 4.0e-4);
    assert_oscillator_sample(&symplectic, 0.375, 4.0e-4);
    assert_oscillator_sample(&structural, 0.375, 4.0e-4);
    assert!(rkn.interpolate(-0.1).is_none());
    assert!(rkn.interpolate(1.1).is_none());
}

#[test]
fn retained_segments_work_backward() {
    let solution = solve_second_order(
        &oscillator((1.0, 0.0)),
        VelocityVerlet,
        &dense_options(0.05),
    )
    .unwrap();

    assert_oscillator_sample(&solution, 0.375, 4.0e-4);
}

#[test]
fn save_at_uses_partition_aware_position_interpolation() {
    let problem = SecondOrderOdeProblem::new(
        |output: &mut [f64], _: &[f64], _: &[f64], _: &(), _: f64| output[0] = 1.0,
        vec![0.0],
        vec![0.0],
        (0.0, 1.0),
        (),
    );
    let solution = solve_second_order(
        &problem,
        VelocityVerlet,
        &SolveOptions {
            adaptive: false,
            initial_step: Some(1.0),
            save_at: vec![0.5, 1.0],
            ..SolveOptions::default()
        },
    )
    .unwrap();

    assert_eq!(solution.times(), &[0.5, 1.0]);
    assert!((solution.velocity(0).unwrap()[0] - 0.5).abs() < 1.0e-14);
    assert!((solution.position(0).unwrap()[0] - 0.125).abs() < 1.0e-14);
}

#[test]
fn callback_discontinuities_bound_retained_segments() {
    let problem = oscillator((0.0, 1.0)).with_continuous_callback(
        |_, _, _, time| time - 0.5,
        |velocity, position, _, _| {
            velocity[0] = 0.0;
            position[0] = 4.0;
            CallbackAction::Continue
        },
    );
    let solution = solve_second_order(&problem, VelocityVerlet, &dense_options(0.2)).unwrap();

    let (event_velocity, event_position) = solution.interpolate(0.5).unwrap();
    assert_eq!(event_velocity, vec![0.0]);
    assert_eq!(event_position, vec![4.0]);
    let (_, before) = solution.interpolate(0.49).unwrap();
    let (_, after) = solution.interpolate(0.51).unwrap();
    assert!(
        (before[0] - 0.49_f64.cos()).abs() < 0.01,
        "pre-event sample was {before:?}"
    );
    assert!(after[0] > 3.99);
}
