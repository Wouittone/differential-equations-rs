use differential_equations::algorithms::*;
use differential_equations::*;

#[test]
fn nonautonomous_forward_and_save_at() {
    let problem = OdeProblem::new(
        |du: &mut [f64], _: &[f64], _: &(), t: f64| du[0] = t,
        vec![0.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.02),
        save: SaveMode::Endpoints,
        save_at: vec![0.25, 0.5, 0.75],
        ..SolveOptions::default()
    };
    let solution = solve(&problem, Cash4, &options).unwrap();
    assert!((solution.last_state()[0] - 0.28125).abs() < 1.0e-4);
    assert!((solution.state(1).unwrap()[0] - 0.125).abs() < 1.0e-8);
    assert!(
        solution
            .times()
            .iter()
            .any(|time| (*time - 0.5).abs() < 1.0e-12)
    );
}

#[test]
fn analytic_jacobian_path_and_callback() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -30.0 * u[0],
        vec![1.0],
        (0.0, 0.1),
        (),
    )
    .with_jacobian(|jac: &mut [f64], _: &[f64], _: &(), _: f64| jac[0] = -30.0);
    let options = SolveOptions {
        absolute_tolerance: 1.0e-8,
        relative_tolerance: 1.0e-8,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let solution = solve(&problem, Cash4, &options).unwrap();
    assert!((solution.last_state()[0] - (-3.0_f64).exp()).abs() < 2.0e-7);
    assert!(solution.stats().jacobian_evaluations > 0);
    assert!(solution.stats().linear_factorizations > 0);
}
