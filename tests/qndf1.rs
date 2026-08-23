use differential_equations::algorithms::*;
use differential_equations::*;

fn fixed(step: f64) -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

#[test]
fn fixed_step_has_first_order_convergence() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let coarse = solve(&problem, Qndf1, &fixed(0.1)).unwrap();
    let fine = solve(&problem, Qndf1, &fixed(0.05)).unwrap();
    let exact = (-1.0f64).exp();
    let coarse_error = (coarse.last_state()[0] - exact).abs();
    let fine_error = (fine.last_state()[0] - exact).abs();
    assert!(
        fine_error < coarse_error * 0.65,
        "errors {coarse_error} and {fine_error}"
    );
}

#[test]
fn adaptive_stiff_decay_and_backward_callback_work() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), t: f64| {
            du[0] = -15.0 * (u[0] - t.cos()) - t.sin();
        },
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions {
        absolute_tolerance: 1.0e-7,
        relative_tolerance: 1.0e-7,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let solution = solve(&problem, Qndf1, &options).unwrap();
    assert!((solution.last_state()[0] - 1.0f64.cos()).abs() < 2.0e-4);
    assert!(solution.stats().rejected_steps > 0);

    let backward = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -u[0],
        vec![(-1.0f64).exp()],
        (1.0, 0.0),
        (),
    )
    .with_discrete_callback(
        |_, _, t| (t - 0.5).abs() < 1.0e-12,
        |_u, _, _| CallbackAction::Continue,
    );
    let result = solve(&backward, Qndf1, &fixed(0.05)).unwrap();
    assert!((result.last_state()[0] - 1.0).abs() < 2.0e-2);
}

#[test]
fn reports_nonfinite_rhs_and_singular_systems() {
    let bad = OdeProblem::new(
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = f64::NAN,
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    assert_eq!(
        solve(&bad, Qndf1, &fixed(0.1)),
        Err(SolveError::NonFiniteDerivative)
    );

    let singular = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    )
    .with_jacobian(|j, _, _, _| j[0] = 1.0);
    assert_eq!(
        solve(&singular, Qndf1, &fixed(1.0)),
        Err(SolveError::SingularLinearSystem)
    );
}

#[test]
fn qbdf1_uses_zero_kappa_in_fixed_and_adaptive_modes() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), t: f64| {
            du[0] = -15.0 * (u[0] - t.cos()) - t.sin();
        },
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let qbdf_fixed = solve(&problem, Qbdf1, &fixed(0.01)).unwrap();
    let qndf_fixed = solve(&problem, Qndf1, &fixed(0.01)).unwrap();
    assert!((qbdf_fixed.last_state()[0] - 0.540_103_568_471_148_3).abs() < 2.0e-12);
    assert!((qbdf_fixed.last_state()[0] - qndf_fixed.last_state()[0]).abs() > 5.0e-5);

    let adaptive = SolveOptions {
        absolute_tolerance: 1.0e-8,
        relative_tolerance: 1.0e-8,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let qbdf_adaptive = solve(&problem, Qbdf1, &adaptive).unwrap();
    assert!((qbdf_adaptive.last_state()[0] - 1.0f64.cos()).abs() < 5.0e-6);
    assert!(qbdf_adaptive.stats().accepted_steps > 0);
}
