use differential_equations::algorithms::*;
use differential_equations::*;

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn exponential() -> OdeProblem<TestRhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
        du[0] = u[0];
    }
    OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
}

#[test]
fn generated_vern6_has_sixth_order_fixed_convergence() {
    let options = |step| SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let coarse = solve(&exponential(), Vern6, &options(0.2)).unwrap();
    let fine = solve(&exponential(), Vern6, &options(0.1)).unwrap();
    let exact = 1.0f64.exp();
    let e1 = (coarse.last_state()[0] - exact).abs();
    let e2 = (fine.last_state()[0] - exact).abs();
    assert!(e2 < e1 / 30.0, "errors {e1} and {e2}");
}

#[test]
fn adaptive_backward_and_callback_paths_remain_safe() {
    let options = SolveOptions {
        absolute_tolerance: 1.0e-10,
        relative_tolerance: 1.0e-10,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let adaptive = solve(&exponential(), Vern6, &options).unwrap();
    assert!((adaptive.last_state()[0] - 1.0f64.exp()).abs() < 1.0e-9);

    let backward = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
        vec![1.0f64.exp()],
        (1.0, 0.0),
        (),
    )
    .with_discrete_callback(
        |_, _, t| (t - 0.5).abs() < 1.0e-12,
        |_u, _, _| CallbackAction::Continue,
    );
    let result = solve(&backward, Vern6, &options).unwrap();
    assert!((result.last_state()[0] - 1.0).abs() < 1.0e-8);
}
