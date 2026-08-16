use differential_equations::{OdeProblem, Rok4a, SaveMode, SolveOptions, solve};

fn options(adaptive: bool, step: Option<f64>) -> SolveOptions {
    SolveOptions {
        adaptive,
        absolute_tolerance: 1.0e-9,
        relative_tolerance: 1.0e-9,
        initial_step: step,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

#[test]
fn rok4a_fixed_and_adaptive_regular_ode() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let fixed = solve(&problem, Rok4a, &options(false, Some(0.01))).unwrap();
    let adaptive = solve(&problem, Rok4a, &options(true, Some(0.1))).unwrap();
    let exact = std::f64::consts::E;
    assert!((fixed.last_state()[0] - exact).abs() < 1.0e-8);
    assert!((adaptive.last_state()[0] - exact).abs() < 1.0e-7);
}

#[test]
fn rok4a_supports_backward_integration() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -2.0 * u[0],
        vec![(-2.0_f64).exp()],
        (1.0, 0.0),
        (),
    );
    let solution = solve(&problem, Rok4a, &options(true, Some(0.01))).unwrap();
    assert!((solution.last_state()[0] - 1.0).abs() < 1.0e-7);
}
