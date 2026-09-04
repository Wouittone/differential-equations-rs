use differential_equations::solvers::explicit::{SplitOdeAlgorithm, solve_split};
use differential_equations::solvers::multirate::{
    MIS, MRAB, MREEF, MRIGARKERK22a, MRIGARKERK22b, MRIGARKERK33a, MRIGARKERK45a, MRIGARKESDIRK34a,
    MRIGARKIRK21a,
};
use differential_equations::tableau::{define_mri_tableau_from_file, load_tableau};
use differential_equations::{CallbackAction, SaveMode, SolveOptions, SplitOdeProblem};

use differential_equations as renamed;

define_mri_tableau_from_file!(pub DOWNSTREAM_ERK22A, "MRIGARKERK22a",
    "src/tableau/resources/mri/erk22a.json");
define_mri_tableau_from_file!(pub RENAMED_ERK22A, "MRIGARKERK22a",
    "src/tableau/resources/mri/erk22a.json", crate = renamed);
differential_equations::tableau::define_mis_tableau_from_file!(pub DOWNSTREAM_MIS, "MIS",
    "src/tableau/resources/mri/mis.json");

#[test]
fn mri_resources_are_inspectable_and_support_renamed_dependencies() {
    let built_in = MRIGARKERK22a::new(4).tableau().unwrap();
    assert_eq!(built_in.name(), "MRIGARKERK22a");
    assert_eq!(built_in.order(), 2);
    assert_eq!(built_in.inner_order(), 2);
    assert_eq!(built_in.dc(), &[0.5, 0.5]);
    assert_eq!(built_in.w0()[1], [-0.5, 1.0]);
    assert_eq!(load_tableau(&DOWNSTREAM_ERK22A).unwrap(), built_in);
    assert_eq!(load_tableau(&RENAMED_ERK22A).unwrap(), built_in);

    let embedded = MRIGARKERK45a::new(4).tableau().unwrap();
    assert_eq!(embedded.order(), 4);
    assert_eq!(embedded.w0().len(), 5);
    assert!(embedded.embedded0().is_some());
    assert!(embedded.embedded1().is_some());

    let implicit = MRIGARKESDIRK34a::new(4).tableau().unwrap();
    assert_eq!(implicit.gamma().len(), 6);
    assert!(implicit.gamma().iter().any(|value| *value != 0.0));

    let mis = MIS::new(4).tableau().unwrap();
    assert_eq!(mis.name(), "MIS");
    assert_eq!(mis.order(), 3);
    assert_eq!(mis.alpha().len(), 4);
    assert_eq!(mis.c(), &[0.0, 0.126848494553, 0.7404635564785064, 1.0]);
    assert_eq!(load_tableau(&DOWNSTREAM_MIS).unwrap(), mis);
}

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

#[test]
fn multirate_driver_hits_exact_time_stops_and_resumes_fixed_steps() {
    let problem = SplitOdeProblem::new(
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = 0.75,
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = 0.25,
        vec![0.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(0.4)
        .with_save(SaveMode::EveryStep)
        .with_time_stops([0.25, 0.5]);

    let solution = solve_split(&problem, MRIGARKERK22a::new(4), &options).unwrap();

    assert_eq!(solution.times(), &[0.0, 0.25, 0.5, 0.9, 1.0]);
    assert!((solution.last_state()[0] - 1.0).abs() < 1.0e-12);

    let backward = SplitOdeProblem::new(
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = 0.75,
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = 0.25,
        vec![1.0],
        (1.0, 0.0),
        (),
    );
    let backward_options = SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(0.4)
        .with_save(SaveMode::EveryStep)
        .with_time_stops([0.75, 0.5]);
    let backward_solution =
        solve_split(&backward, MRIGARKERK22a::new(4), &backward_options).unwrap();

    assert_eq!(backward_solution.times()[..3], [1.0, 0.75, 0.5]);
    assert!(backward_solution.last_state()[0].abs() < 1.0e-12);
}
