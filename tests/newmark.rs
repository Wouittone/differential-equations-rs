use differential_equations::solvers::second_order::{
    GeneralizedAlpha, NewmarkBeta, SecondOrderOdeProblem, solve_second_order,
};
use differential_equations::{CallbackAction, SaveMode, SolveOptions};

fn fixed_options(step: f64) -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

type SecondOrderRhs = fn(&mut [f64], &[f64], &[f64], &(), f64);

fn harmonic(
    span: (f64, f64),
    velocity: f64,
    position: f64,
) -> SecondOrderOdeProblem<SecondOrderRhs, ()> {
    fn acceleration(output: &mut [f64], _: &[f64], position: &[f64], _: &(), _: f64) {
        output[0] = -position[0];
    }
    SecondOrderOdeProblem::new(acceleration, vec![velocity], vec![position], span, ())
}

fn endpoint_error(step: f64) -> f64 {
    let solution = solve_second_order(
        &harmonic((0.0, 1.0), 1.0, 0.0),
        NewmarkBeta::default(),
        &fixed_options(step),
    )
    .unwrap();
    (solution.last_velocity()[0] - 1.0_f64.cos())
        .abs()
        .max((solution.last_position()[0] - 1.0_f64.sin()).abs())
}

#[test]
fn newmark_beta_is_second_order() {
    let coarse = endpoint_error(0.2);
    let fine = endpoint_error(0.1);
    assert!(coarse / fine > 3.7, "{coarse} {fine}");
    assert!(fine < 8.0e-4, "{fine}");
}

#[test]
fn generalized_alpha_rho_one_is_undamped_and_second_order() {
    let problem = harmonic((0.0, 2.0), 1.0, 0.0);
    let options = fixed_options(0.05);
    let newmark = solve_second_order(&problem, NewmarkBeta::default(), &options).unwrap();
    let generalized = solve_second_order(
        &problem,
        GeneralizedAlpha::from_spectral_radius(1.0).unwrap(),
        &options,
    )
    .unwrap();
    assert_eq!(newmark.times(), generalized.times());
    assert!((newmark.last_velocity()[0] - 2.0_f64.cos()).abs() < 5.0e-4);
    assert!((newmark.last_position()[0] - 2.0_f64.sin()).abs() < 5.0e-4);
    assert!((generalized.last_velocity()[0] - 2.0_f64.cos()).abs() < 5.0e-4);
    assert!((generalized.last_position()[0] - 2.0_f64.sin()).abs() < 5.0e-4);
}

#[test]
fn generalized_alpha_damps_high_frequency_response() {
    let problem = SecondOrderOdeProblem::new(
        |output: &mut [f64], _: &[f64], position: &[f64], _: &(), _: f64| {
            output[0] = -100.0 * position[0];
        },
        vec![1.0],
        vec![0.0],
        (0.0, 1.0),
        (),
    );
    let options = fixed_options(0.1);
    let damped = solve_second_order(
        &problem,
        GeneralizedAlpha::from_spectral_radius(0.5).unwrap(),
        &options,
    )
    .unwrap();
    let undamped = solve_second_order(
        &problem,
        GeneralizedAlpha::from_spectral_radius(1.0).unwrap(),
        &options,
    )
    .unwrap();
    let damped_norm = damped.last_velocity()[0].hypot(10.0 * damped.last_position()[0]);
    let undamped_norm = undamped.last_velocity()[0].hypot(10.0 * undamped.last_position()[0]);
    assert!(damped_norm < undamped_norm, "{damped_norm} {undamped_norm}");
}

#[test]
fn structural_methods_support_adaptivity_backward_time_and_callbacks() {
    let adaptive = SolveOptions::default().with_tolerances(1.0e-7, 1.0e-7);
    let forward = solve_second_order(
        &harmonic((0.0, 1.0), 1.0, 0.0),
        NewmarkBeta::default(),
        &adaptive,
    )
    .unwrap();
    assert!((forward.last_position()[0] - 1.0_f64.sin()).abs() < 2.0e-5);
    assert!(forward.stats().linear_solves > 0);
    assert!(forward.stats().rejected_steps < forward.stats().accepted_steps);

    let backward = solve_second_order(
        &harmonic((1.0, 0.0), 1.0_f64.cos(), 1.0_f64.sin()),
        GeneralizedAlpha::default(),
        &fixed_options(0.02),
    )
    .unwrap();
    assert!(backward.last_position()[0].abs() < 5.0e-5);
    assert!((backward.last_velocity()[0] - 1.0).abs() < 5.0e-5);

    let event_problem = harmonic((0.0, 2.0), 1.0, 0.0).with_continuous_callback(
        |_, position, _, _| position[0] - 0.5,
        |_, _, _, _| CallbackAction::Terminate,
    );
    let event =
        solve_second_order(&event_problem, NewmarkBeta::default(), &fixed_options(0.05)).unwrap();
    assert!((event.last_position()[0] - 0.5).abs() < 2.0e-12);
    assert_eq!(event.stats().callback_invocations, 1);
}

#[test]
fn constructors_validate_stability_ranges() {
    assert!(NewmarkBeta::new(-0.1, 0.5).is_err());
    assert!(NewmarkBeta::new(0.25, 1.1).is_err());
    assert!(GeneralizedAlpha::from_spectral_radius(-0.1).is_err());
    assert!(GeneralizedAlpha::from_spectral_radius(1.1).is_err());
    assert!(GeneralizedAlpha::from_hht_alpha(-0.4).is_err());
    assert!(GeneralizedAlpha::new(0.3, 0.1, 0.25, 0.5).is_err());
}
