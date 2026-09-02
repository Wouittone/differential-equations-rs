use std::sync::LazyLock;

use differential_equations::SolveOptions;
use differential_equations::solvers::second_order::{SecondOrderOdeProblem, solve_symplectic};
use differential_equations::tableau::{
    LazySymplecticTableau, define_symplectic_from_file, load_tableau, parse_symplectic_tableau,
};

define_symplectic_from_file!(pub FileDriftKick, "tests/resources/file_drift_kick.json");

mod renamed_dependency {
    use diffeq::tableau::define_symplectic_from_file;
    use differential_equations as diffeq;
    define_symplectic_from_file!(pub FileDriftKick, "tests/resources/file_drift_kick.json", crate = diffeq);
}

#[test]
fn downstream_resources_solve_with_original_and_renamed_dependencies() {
    let problem = SecondOrderOdeProblem::new(
        |a: &mut [f64], _: &[f64], q: &[f64], _: &(), _| a[0] = -q[0],
        [0.0],
        [1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(0.01);
    let original = solve_symplectic(&problem, FileDriftKick, &options).unwrap();
    let renamed = solve_symplectic(&problem, renamed_dependency::FileDriftKick, &options).unwrap();
    assert_eq!(original, renamed);
    assert!((original.last_position()[0] - 1.0_f64.cos()).abs() < 1e-4);
    assert!((original.last_velocity()[0] + 1.0_f64.sin()).abs() < 1e-4);
}

#[test]
fn invalid_runtime_resources_remain_cached_errors_without_poisoning() {
    static INVALID: LazySymplecticTableau =
        LazyLock::new(|| parse_symplectic_tableau("{}", "Missing"));
    let first = load_tableau(&INVALID).unwrap_err();
    assert_eq!(first, load_tableau(&INVALID).unwrap_err());
}
