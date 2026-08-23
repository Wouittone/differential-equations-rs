use differential_equations::{
    Nystrom4, Nystrom4VelocityIndependent, Nystrom5VelocityIndependent, Rkn4, SaveMode,
    SecondOrderOdeAlgorithm, SecondOrderOdeProblem, SecondOrderSolveError, SolveError,
    SolveOptions, solve_second_order,
};

type Acceleration = fn(&mut [f64], &[f64], &[f64], &(), f64);

fn oscillator(
    span: (f64, f64),
    velocity: f64,
    position: f64,
) -> SecondOrderOdeProblem<Acceleration, ()> {
    fn acceleration(output: &mut [f64], _: &[f64], position: &[f64], _: &(), _: f64) {
        output[0] = -position[0];
    }
    SecondOrderOdeProblem::new(acceleration, vec![velocity], vec![position], span, ())
}

fn fixed(step: f64) -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

fn endpoint_error<A: SecondOrderOdeAlgorithm>(algorithm: A, step: f64) -> f64 {
    let solution =
        solve_second_order(&oscillator((0.0, 1.0), 0.0, 1.0), algorithm, &fixed(step)).unwrap();
    (solution.last_position()[0] - 1.0_f64.cos()).hypot(solution.last_velocity()[0] + 1.0_f64.sin())
}

#[test]
fn velocity_independent_nystrom_methods_reach_their_pinned_orders() {
    let fourth = endpoint_error(Nystrom4VelocityIndependent, 0.1)
        / endpoint_error(Nystrom4VelocityIndependent, 0.05);
    let fifth = endpoint_error(Nystrom5VelocityIndependent, 0.1)
        / endpoint_error(Nystrom5VelocityIndependent, 0.05);
    assert!(
        fourth > 14.0 && fourth < 18.0,
        "fourth-order ratio was {fourth}"
    );
    assert!(
        fifth > 27.0 && fifth < 37.0,
        "fifth-order ratio was {fifth}"
    );
}

#[test]
fn velocity_dependent_nystrom4_reaches_fourth_order() {
    fn acceleration(output: &mut [f64], velocity: &[f64], position: &[f64], _: &(), _: f64) {
        output[0] = -position[0] - 0.2 * velocity[0];
    }
    fn exact() -> (f64, f64) {
        let omega = 0.99_f64.sqrt();
        let b = 0.35 / omega;
        let sine = omega.sin();
        let cosine = omega.cos();
        let decay = (-0.1_f64).exp();
        let position = decay * (cosine + b * sine);
        let velocity = decay * (-0.1 * (cosine + b * sine) - omega * sine + b * omega * cosine);
        (velocity, position)
    }
    let problem = SecondOrderOdeProblem::new(
        acceleration as Acceleration,
        vec![0.25],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let error = |step| {
        let solution = solve_second_order(&problem, Nystrom4, &fixed(step)).unwrap();
        let (velocity, position) = exact();
        (solution.last_position()[0] - position).hypot(solution.last_velocity()[0] - velocity)
    };
    let ratio = error(0.1) / error(0.05);
    assert!(ratio > 14.0 && ratio < 18.0, "ratio was {ratio}");
}

#[test]
fn rkn4_is_fourth_order_on_the_linear_oscillator_class() {
    let ratio = endpoint_error(Rkn4, 0.1) / endpoint_error(Rkn4, 0.05);
    assert!(ratio > 14.0 && ratio < 18.0, "ratio was {ratio}");
}

#[test]
fn fixed_rkn_methods_support_backward_time_and_reject_adaptivity() {
    let backward = oscillator((1.0, 0.0), -1.0_f64.sin(), 1.0_f64.cos());
    let solution =
        solve_second_order(&backward, Nystrom5VelocityIndependent, &fixed(0.02)).unwrap();
    assert!((solution.last_position()[0] - 1.0).abs() < 2.0e-8);
    assert!(solution.last_velocity()[0].abs() < 2.0e-8);

    assert_eq!(
        solve_second_order(
            &oscillator((0.0, 1.0), 0.0, 1.0),
            Nystrom4,
            &SolveOptions::default(),
        ),
        Err(SecondOrderSolveError::Solve(
            SolveError::AdaptiveStepUnsupported
        ))
    );
}
