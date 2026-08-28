use differential_equations::solvers::multistep::{AN5, JVODE, JVODE_Adams, JVODE_BDF};
use differential_equations::{
    CallbackAction, OdeAlgorithm, OdeProblem, SaveMode, SolveOptions, solve,
};

fn endpoint<A: OdeAlgorithm>(algorithm: A, step: f64) -> f64 {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(step)
        .with_save(SaveMode::Endpoints);
    solve(&problem, algorithm, &options).unwrap().last_state()[0]
}

#[test]
fn an5_has_fifth_order_fixed_step_convergence() {
    let exact = std::f64::consts::E;
    let coarse = (endpoint(AN5, 0.1) - exact).abs();
    let fine = (endpoint(AN5, 0.05) - exact).abs();
    assert!(fine < coarse / 20.0, "coarse={coarse}, fine={fine}");
}

#[test]
fn jvode_aliases_are_genuine_configured_modes() {
    let options = SolveOptions::new()
        .with_tolerances(1.0e-7, 1.0e-7)
        .with_max_step(0.2)
        .with_save(SaveMode::Endpoints);
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -2.0 * u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let adams_alias = solve(&problem, JVODE_Adams::default(), &options).unwrap();
    let adams_mode = solve(&problem, JVODE::adams(), &options).unwrap();
    let bdf_alias = solve(&problem, JVODE_BDF::default(), &options).unwrap();
    let bdf_mode = solve(&problem, JVODE::bdf(), &options).unwrap();
    assert_eq!(adams_alias.last_state(), adams_mode.last_state());
    assert_eq!(bdf_alias.last_state(), bdf_mode.last_state());
    assert!((adams_alias.last_state()[0] - (-2.0_f64).exp()).abs() < 2.0e-5);
    assert!((bdf_alias.last_state()[0] - (-2.0_f64).exp()).abs() < 2.0e-5);
}

#[test]
fn bdf_mode_uses_analytic_and_finite_difference_jacobians() {
    let analytic_problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -40.0 * u[0],
        vec![1.0],
        (0.0, 0.5),
        (),
    )
    .with_jacobian(|jacobian, _: &[f64], _: &(), _: f64| jacobian[0] = -40.0);
    let finite_difference_problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -40.0 * u[0],
        vec![1.0],
        (0.0, 0.5),
        (),
    );
    let options = SolveOptions::new()
        .with_tolerances(1.0e-7, 1.0e-7)
        .with_initial_step(0.01)
        .with_max_step(0.05)
        .with_save(SaveMode::Endpoints);
    let analytic = solve(&analytic_problem, JVODE_BDF::default(), &options).unwrap();
    let finite_difference =
        solve(&finite_difference_problem, JVODE_BDF::default(), &options).unwrap();
    assert!(analytic.stats().jacobian_evaluations > 0);
    assert!(analytic.stats().linear_solves > 0);
    assert!(finite_difference.stats().rhs_evaluations > analytic.stats().rhs_evaluations);
    assert!((analytic.last_state()[0] - (-20.0_f64).exp()).abs() < 1.0e-5);
}

#[test]
fn lifecycle_supports_backward_callbacks_save_at_and_retained_dense_output() {
    let problem = OdeProblem::new(
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = 1.0,
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
        .with_initial_step(0.2)
        .with_save_at([0.0, 0.25, 0.5])
        .with_dense_output(true)
        .with_event_tolerance(1.0e-10);
    let solution = solve(&problem, AN5, &options).unwrap();
    assert!((solution.times().last().unwrap() - 0.6).abs() < 1.0e-9);
    assert_eq!(solution.last_state(), &[2.0]);
    assert!((solution.interpolate(0.3).unwrap()[0] - 0.3).abs() < 1.0e-8);

    let backward = OdeProblem::new(
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = 1.0,
        vec![1.0],
        (1.0, 0.0),
        (),
    );
    let fixed = SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(0.05)
        .with_save(SaveMode::Endpoints);
    assert!(
        solve(&backward, JVODE_Adams::default(), &fixed)
            .unwrap()
            .last_state()[0]
            .abs()
            < 1.0e-10
    );
}
