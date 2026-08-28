use differential_equations::solvers::implicit::PDIRK44;
use differential_equations::{OdeProblem, SaveMode, SolveError, SolveOptions, solve};

fn fixed_options(step: f64) -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

#[test]
fn pdirk44_has_fourth_order_convergence() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let coarse = solve(&problem, PDIRK44, &fixed_options(0.1)).unwrap();
    let fine = solve(&problem, PDIRK44, &fixed_options(0.05)).unwrap();
    let exact = std::f64::consts::E;
    let coarse_error = (coarse.last_state()[0] - exact).abs();
    let fine_error = (fine.last_state()[0] - exact).abs();
    assert!(
        coarse_error / fine_error > 12.0,
        "{coarse_error} {fine_error}"
    );
    assert!(fine_error < 1.0e-6, "{fine_error}");
}

#[test]
fn pdirk44_supports_vectors_backward_time_and_analytic_jacobians() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), t: f64| {
            du[0] = -2.0 * u[0] + t;
            du[1] = u[0] - u[1];
        },
        vec![0.5, -0.25],
        (1.0, 0.0),
        (),
    )
    .with_jacobian(|jacobian: &mut [f64], _: &[f64], _: &(), _: f64| {
        jacobian.copy_from_slice(&[-2.0, 0.0, 1.0, -1.0]);
    });
    let solution = solve(&problem, PDIRK44, &fixed_options(0.05)).unwrap();
    assert_eq!(solution.last_state().len(), 2);
    assert!(solution.last_state().iter().all(|value| value.is_finite()));
    assert!(solution.stats().jacobian_evaluations > 0);
    assert!(solution.stats().linear_factorizations > 0);
    assert_eq!(
        solution.stats().linear_factorizations,
        solution.stats().linear_solves
    );
}

#[test]
fn pdirk44_rejects_adaptive_mode_and_nonfinite_rhs() {
    let problem = OdeProblem::new(
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = f64::NAN,
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    assert_eq!(
        solve(&problem, PDIRK44, &SolveOptions::default()),
        Err(SolveError::AdaptiveStepUnsupported)
    );
    assert_eq!(
        solve(&problem, PDIRK44, &fixed_options(0.1)),
        Err(SolveError::NonFiniteDerivative)
    );
}
