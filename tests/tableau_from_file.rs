use differential_equations::{
    OdeProblem, SaveMode, SolveOptions, define_explicit_rk_from_file, solve,
};

define_explicit_rk_from_file!(pub FileHeun, "tests/resources/file_heun.toml");

#[test]
fn resource_tableau_defines_a_zero_overhead_solver_method() {
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
