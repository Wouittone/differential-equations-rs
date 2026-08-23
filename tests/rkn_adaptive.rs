use differential_equations::{
    CallbackAction, DPRKN4, DPRKN5, DPRKN6, DPRKN6FM, DPRKN8, DPRKN12, Dprkn4, Dprkn5, Dprkn6,
    Dprkn6Fm, Dprkn8, Dprkn12, ERKN4, ERKN5, ERKN7, Erkn4, Erkn5, Erkn7, FineRKN4, FineRKN5,
    FineRkn4, FineRkn5, IRKN3, IRKN4, Irkn3, Irkn4, SaveMode, SecondOrderOdeAlgorithm,
    SecondOrderOdeProblem, SolveOptions, solve_second_order,
};

type Acceleration = fn(&mut [f64], &[f64], &[f64], &(), f64);

fn oscillator() -> SecondOrderOdeProblem<Acceleration, ()> {
    fn acceleration(output: &mut [f64], _: &[f64], position: &[f64], _: &(), _: f64) {
        output[0] = -position[0];
    }
    SecondOrderOdeProblem::new(
        acceleration as Acceleration,
        vec![0.0],
        vec![1.0],
        (0.0, 10.0),
        (),
    )
}

fn adaptive_error<A: SecondOrderOdeAlgorithm>(algorithm: A) -> (f64, usize) {
    let solution = solve_second_order(
        &oscillator(),
        algorithm,
        &SolveOptions {
            absolute_tolerance: 1.0e-10,
            relative_tolerance: 1.0e-10,
            initial_step: Some(1.0),
            max_step: 1.0,
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        },
    )
    .unwrap();
    let error = (solution.last_position()[0] - 10.0_f64.cos())
        .hypot(solution.last_velocity()[0] + 10.0_f64.sin());
    (error, solution.stats().rejected_steps)
}

#[test]
fn embedded_rkn_methods_adapt_to_tight_tolerances() {
    let methods = [
        ("DPRKN4", adaptive_error(Dprkn4).0),
        ("DPRKN5", adaptive_error(Dprkn5).0),
        ("DPRKN6", adaptive_error(Dprkn6).0),
        ("DPRKN6FM", adaptive_error(Dprkn6Fm).0),
        ("DPRKN8", adaptive_error(Dprkn8).0),
        ("DPRKN12", adaptive_error(Dprkn12).0),
        ("ERKN4", adaptive_error(Erkn4).0),
        ("ERKN5", adaptive_error(Erkn5).0),
        ("ERKN7", adaptive_error(Erkn7).0),
    ];
    for (name, error) in methods {
        assert!(error < 2.0e-8, "{name} endpoint error was {error:e}");
    }
}

fn damped_error<A: SecondOrderOdeAlgorithm>(algorithm: A) -> f64 {
    fn acceleration(output: &mut [f64], velocity: &[f64], position: &[f64], _: &(), _: f64) {
        output[0] = -position[0] - 0.2 * velocity[0];
    }
    let problem = SecondOrderOdeProblem::new(
        acceleration as Acceleration,
        vec![0.25],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let solution = solve_second_order(
        &problem,
        algorithm,
        &SolveOptions {
            absolute_tolerance: 1.0e-10,
            relative_tolerance: 1.0e-10,
            initial_step: Some(0.5),
            max_step: 0.5,
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        },
    )
    .unwrap();
    let omega = 0.99_f64.sqrt();
    let b = 0.35 / omega;
    let sine = omega.sin();
    let cosine = omega.cos();
    let decay = (-0.1_f64).exp();
    let position = decay * (cosine + b * sine);
    let velocity = decay * (-0.1 * (cosine + b * sine) - omega * sine + b * omega * cosine);
    (solution.last_position()[0] - position).hypot(solution.last_velocity()[0] - velocity)
}

#[test]
fn fine_rkn_methods_retain_velocity_dependent_order() {
    assert!(damped_error(FineRkn4) < 2.0e-8);
    assert!(damped_error(FineRkn5) < 2.0e-8);
}

#[test]
fn dprkn6_dense_output_drives_save_at_and_continuous_roots() {
    fn acceleration(output: &mut [f64], _: &[f64], position: &[f64], _: &(), _: f64) {
        output[0] = -position[0];
    }
    let problem = SecondOrderOdeProblem::new(
        acceleration as Acceleration,
        vec![0.0],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(1.0),
        save_at: vec![0.0, 0.5, 1.0],
        ..SolveOptions::default()
    };
    let solution = solve_second_order(&problem, Dprkn6, &options).unwrap();
    assert_eq!(solution.times(), &[0.0, 0.5, 1.0]);
    assert!((solution.position(1).unwrap()[0] - 0.5_f64.cos()).abs() < 2.0e-5);
    assert!((solution.velocity(1).unwrap()[0] + 0.5_f64.sin()).abs() < 2.0e-5);

    let problem = problem.with_continuous_callback(
        |_, position, _, _| position[0] - 0.5_f64.cos(),
        |_, _, _, _| CallbackAction::Terminate,
    );
    let solution = solve_second_order(
        &problem,
        Dprkn6,
        &SolveOptions {
            adaptive: false,
            initial_step: Some(1.0),
            save: SaveMode::Endpoints,
            event_tolerance: 1.0e-10,
            ..SolveOptions::default()
        },
    )
    .unwrap();
    assert!((solution.times().last().unwrap() - 0.5).abs() < 2.0e-5);
}

fn irkn_fixed_error<A: SecondOrderOdeAlgorithm>(algorithm: A, step: f64) -> f64 {
    fn acceleration(output: &mut [f64], _: &[f64], position: &[f64], _: &(), _: f64) {
        output[0] = -position[0];
    }
    let problem = SecondOrderOdeProblem::new(
        acceleration as Acceleration,
        vec![0.0],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let solution = solve_second_order(
        &problem,
        algorithm,
        &SolveOptions {
            adaptive: false,
            initial_step: Some(step),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        },
    )
    .unwrap();
    (solution.last_position()[0] - 1.0_f64.cos()).hypot(solution.last_velocity()[0] + 1.0_f64.sin())
}

#[test]
fn irkn_history_kernels_reach_their_pinned_orders() {
    let third = irkn_fixed_error(Irkn3, 0.1) / irkn_fixed_error(Irkn3, 0.05);
    let fourth = irkn_fixed_error(Irkn4, 0.1) / irkn_fixed_error(Irkn4, 0.05);
    assert!(third > 6.0 && third < 10.0, "IRKN3 ratio was {third}");
    assert!(fourth > 12.0, "IRKN4 ratio was {fourth}");
}

#[test]
fn irkn_history_rebootstraps_after_callback_discontinuities() {
    fn acceleration(output: &mut [f64], _: &[f64], position: &[f64], _: &(), _: f64) {
        output[0] = -position[0];
    }
    let problem = SecondOrderOdeProblem::new(
        acceleration as Acceleration,
        vec![0.0],
        vec![1.0],
        (0.0, 1.0),
        (),
    )
    .with_discrete_callback(
        |_, _, _, time| time == 0.5,
        |velocity, _, _, _| {
            velocity[0] += 0.1;
            CallbackAction::Continue
        },
    );
    let solution = solve_second_order(
        &problem,
        Irkn4,
        &SolveOptions {
            adaptive: false,
            initial_step: Some(0.125),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        },
    )
    .unwrap();
    let expected_position = 1.0_f64.cos() + 0.1 * 0.5_f64.sin();
    let expected_velocity = -1.0_f64.sin() + 0.1 * 0.5_f64.cos();
    assert!((solution.last_position()[0] - expected_position).abs() < 2.0e-5);
    assert!((solution.last_velocity()[0] - expected_velocity).abs() < 2.0e-5);
    assert_eq!(solution.stats().callback_invocations, 1);
}

#[test]
fn rejected_attempts_do_not_advance_the_partitioned_state() {
    let (error, rejected) = adaptive_error(Dprkn4);
    assert!(
        rejected > 0,
        "the deliberately large first step must be rejected"
    );
    assert!(error < 2.0e-8);
}

#[test]
fn adaptive_rkn_methods_also_support_requested_fixed_steps_and_backward_time() {
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.01),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let forward = SecondOrderOdeProblem::new(
        |out: &mut [f64], _: &[f64], q: &[f64], _: &(), _: f64| out[0] = -q[0],
        vec![0.0],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let backward = SecondOrderOdeProblem::new(
        |out: &mut [f64], _: &[f64], q: &[f64], _: &(), _: f64| out[0] = -q[0],
        vec![-1.0_f64.sin()],
        vec![1.0_f64.cos()],
        (1.0, 0.0),
        (),
    );
    let forward = solve_second_order(&forward, Dprkn12, &options).unwrap();
    let backward = solve_second_order(&backward, Erkn7, &options).unwrap();
    assert!((forward.last_position()[0] - 1.0_f64.cos()).abs() < 2.0e-12);
    assert!((forward.last_velocity()[0] + 1.0_f64.sin()).abs() < 2.0e-12);
    assert!((backward.last_position()[0] - 1.0).abs() < 2.0e-11);
    assert!(backward.last_velocity()[0].abs() < 2.0e-11);
}

#[test]
fn sciml_spellings_and_second_order_namespace_are_public() {
    let _: Dprkn4 = DPRKN4;
    let _: Dprkn5 = DPRKN5;
    let _: Dprkn6 = DPRKN6;
    let _: Dprkn6Fm = DPRKN6FM;
    let _: Dprkn8 = DPRKN8;
    let _: Dprkn12 = DPRKN12;
    let _: Erkn4 = ERKN4;
    let _: Erkn5 = ERKN5;
    let _: Erkn7 = ERKN7;
    let _: FineRkn4 = FineRKN4;
    let _: FineRkn5 = FineRKN5;
    let _: Irkn3 = IRKN3;
    let _: Irkn4 = IRKN4;
    let _: differential_equations::algorithms::second_order::Dprkn12 =
        differential_equations::algorithms::second_order::DPRKN12;
}
