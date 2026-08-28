use differential_equations::tableau::define_explicit_rk_from_file;
use differential_equations::{OdeProblem, SaveMode, SolveOptions, solve};

define_explicit_rk_from_file!(pub FileHeun, "tests/resources/file_heun.json");

mod renamed_dependency {
    use diffeq::tableau::define_explicit_rk_from_file;
    use differential_equations as diffeq;

    define_explicit_rk_from_file!(
        pub FileHeun,
        "tests/resources/file_heun.json",
        crate = diffeq
    );
}

#[test]
fn resource_tableau_defines_a_lazily_loaded_solver_method() {
    let tableau = FileHeun.tableau().unwrap();
    assert_eq!(tableau.name(), "FileHeun");
    assert_eq!(tableau.a(), &[vec![0.0, 0.0], vec![1.0, 0.0]]);

    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions {
        absolute_tolerance: 1.0e-8,
        relative_tolerance: 1.0e-8,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };

    let solution = solve(&problem, FileHeun, &options).unwrap();

    assert!((solution.last_state()[0] - (-1.0_f64).exp()).abs() < 2.0e-6);
    assert!(solution.stats().accepted_steps > 0);
}

#[test]
fn resource_tableau_supports_a_renamed_dependency_path() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -u[0],
        [1.0],
        (0.0, 0.1),
        (),
    );
    let options = SolveOptions::default()
        .with_tolerances(1.0e-8, 1.0e-8)
        .with_save(SaveMode::Endpoints);

    let solution = solve(&problem, renamed_dependency::FileHeun, &options).unwrap();

    assert!((solution.last_state()[0] - (-0.1_f64).exp()).abs() < 2.0e-6);
}
