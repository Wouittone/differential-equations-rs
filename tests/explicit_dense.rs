use differential_equations::{OdeProblem, Rk4, SaveMode, SolveOptions, Tsit5, solve};

fn cubic_rate(derivative: &mut [f64], _: &[f64], _: &(), time: f64) {
    derivative[0] = 3.0 * time * time;
}

fn fixed_cubic(span: (f64, f64), initial: f64, step: f64, save_at: Vec<f64>) -> Vec<f64> {
    let problem = OdeProblem::new(cubic_rate, vec![initial], span, ());
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        save: SaveMode::Endpoints,
        save_at,
        ..SolveOptions::default()
    };
    solve(&problem, Rk4, &options).unwrap().values().to_vec()
}

#[test]
fn rk4_save_at_uses_endpoint_hermite_forward_and_backward() {
    let forward = fixed_cubic((0.0, 1.0), 0.0, 1.0, vec![0.25, 0.75]);
    assert!((forward[0] - 0.015625).abs() < 1.0e-14);
    assert!((forward[1] - 0.421875).abs() < 1.0e-14);

    let backward = fixed_cubic((1.0, 0.0), 1.0, 1.0, vec![0.75, 0.25]);
    assert!((backward[0] - 0.421875).abs() < 1.0e-14);
    assert!((backward[1] - 0.015625).abs() < 1.0e-14);
}

#[test]
fn rk4_dense_sampling_preserves_exact_endpoints() {
    let problem = OdeProblem::new(cubic_rate, vec![0.0], (0.0, 1.0), ());
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(1.0),
        save_at: vec![0.0, 1.0],
        ..SolveOptions::default()
    };
    let solution = solve(&problem, Rk4, &options).unwrap();
    assert_eq!(solution.times(), &[0.0, 1.0]);
    assert_eq!(solution.values(), &[0.0, 1.0]);
}

#[test]
fn rejected_explicit_attempts_do_not_emit_dense_samples() {
    let problem = OdeProblem::new(
        |derivative: &mut [f64], state: &[f64], _: &(), _: f64| {
            derivative[0] = state[0];
        },
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions {
        initial_step: Some(1.0),
        absolute_tolerance: 1.0e-12,
        relative_tolerance: 1.0e-12,
        save_at: vec![0.25, 0.5, 0.75],
        ..SolveOptions::default()
    };
    let solution = solve(&problem, Tsit5, &options).unwrap();
    assert!(solution.stats().rejected_steps > 0);
    assert_eq!(solution.times(), &[0.25, 0.5, 0.75]);
    assert_eq!(solution.values().len(), 3);
}
