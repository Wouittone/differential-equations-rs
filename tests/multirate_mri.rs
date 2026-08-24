use differential_equations::algorithms::multirate::{
    MIS, MRAB, MREEF, MRIGARKERK22a, MRIGARKERK22b, MRIGARKERK33a, MRIGARKERK45a, MRIGARKESDIRK34a,
    MRIGARKIRK21a,
};
use differential_equations::{
    CallbackAction, SaveMode, SolveOptions, SplitOdeAlgorithm, SplitOdeProblem, solve_split,
};

fn endpoint<A: SplitOdeAlgorithm>(algorithm: A, step: f64) -> f64 {
    let problem = SplitOdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -12.0 * u[0],
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = 11.0 * u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    )
    .with_implicit_jacobian(|jacobian, _: &[f64], _: &(), _: f64| jacobian[0] = 11.0);
    let options = SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(step)
        .with_save(SaveMode::Endpoints);
    solve_split(&problem, algorithm, &options)
        .unwrap()
        .last_state()[0]
}

#[test]
fn all_nine_inventory_names_solve_a_separated_timescale_problem() {
    let values = [
        ("MIS", endpoint(MIS::new(8), 0.01)),
        ("MRAB", endpoint(MRAB::new(3, 8), 0.01)),
        ("MREEF", endpoint(MREEF::default(), 0.01)),
        ("ERK22a", endpoint(MRIGARKERK22a::new(8), 0.01)),
        ("ERK22b", endpoint(MRIGARKERK22b::new(8), 0.01)),
        ("ERK33a", endpoint(MRIGARKERK33a::new(8), 0.01)),
        ("ERK45a", endpoint(MRIGARKERK45a::new(8), 0.01)),
        ("ESDIRK34a", endpoint(MRIGARKESDIRK34a::new(8), 0.01)),
        ("IRK21a", endpoint(MRIGARKIRK21a::new(8), 0.01)),
    ];
    let exact = (-1.0_f64).exp();
    for (name, value) in values {
        assert!(
            (value - exact).abs() < 0.03,
            "{name}: endpoint {value}, exact {exact}"
        );
    }
}

#[test]
fn refinement_reduces_error_for_representative_families() {
    let exact = (-1.0_f64).exp();
    for (coarse, fine) in [
        (endpoint(MIS::new(8), 0.1), endpoint(MIS::new(8), 0.05)),
        (
            endpoint(MRAB::new(3, 8), 0.1),
            endpoint(MRAB::new(3, 8), 0.05),
        ),
        (
            endpoint(MREEF::default(), 0.1),
            endpoint(MREEF::default(), 0.05),
        ),
        (
            endpoint(MRIGARKERK45a::new(8), 0.1),
            endpoint(MRIGARKERK45a::new(8), 0.05),
        ),
        (
            endpoint(MRIGARKESDIRK34a::new(8), 0.1),
            endpoint(MRIGARKESDIRK34a::new(8), 0.05),
        ),
    ] {
        assert!(
            (fine - exact).abs() < (coarse - exact).abs(),
            "{coarse} -> {fine}"
        );
    }
}

#[test]
fn split_lifecycle_supports_backward_callbacks_save_at_and_retained_dense_output() {
    let problem = SplitOdeProblem::new(
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = 0.75,
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = 0.25,
        vec![0.0],
        (0.0, 1.0),
        (),
    )
    .with_continuous_callback(
        |state, _: &(), _: f64| state[0] - 0.6,
        |state, _: &(), _: f64| {
            state[0] = 2.0;
            CallbackAction::Terminate
        },
    );
    let options = SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(0.25)
        .with_save_at([0.0, 0.2, 0.4])
        .with_dense_output(true)
        .with_event_tolerance(1.0e-10);
    let solution = solve_split(&problem, MRIGARKERK33a::new(4), &options).unwrap();
    assert!((solution.times().last().unwrap() - 0.6).abs() < 1.0e-9);
    assert_eq!(solution.last_state(), &[2.0]);
    assert!((solution.interpolate(0.3).unwrap()[0] - 0.3).abs() < 1.0e-9);

    let backward = SplitOdeProblem::new(
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = 0.75,
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = 0.25,
        vec![1.0],
        (1.0, 0.0),
        (),
    );
    let fixed = SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(0.1)
        .with_save(SaveMode::Endpoints);
    assert!(
        solve_split(&backward, MRIGARKERK22a::new(4), &fixed)
            .unwrap()
            .last_state()[0]
            .abs()
            < 1.0e-12
    );
}

#[test]
fn adaptive_and_implicit_statistics_are_deterministic() {
    let problem = SplitOdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -2.0 * u[0],
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -3.0 * u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    )
    .with_implicit_jacobian(|jacobian, _: &[f64], _: &(), _: f64| jacobian[0] = -3.0);
    let options = SolveOptions::new()
        .with_tolerances(1.0e-6, 1.0e-6)
        .with_max_step(0.2)
        .with_save(SaveMode::Endpoints);
    let first = solve_split(&problem, MRIGARKESDIRK34a::new(4), &options).unwrap();
    let second = solve_split(&problem, MRIGARKESDIRK34a::new(4), &options).unwrap();
    assert_eq!(first.stats(), second.stats());
    assert!(first.stats().nonlinear_iterations > 0);
    assert!(first.stats().linear_solves > 0);
    assert!((first.last_state()[0] - (-5.0_f64).exp()).abs() < 2.0e-4);
}
