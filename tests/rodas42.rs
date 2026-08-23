use differential_equations::algorithms::*;
use differential_equations::*;

fn fixed_options(step: f64) -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

type ExponentialProblem = OdeProblem<fn(&mut [f64], &[f64], &(), f64), ()>;

fn exponential_rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
    du[0] = u[0];
}

fn exponential_problem(span: (f64, f64), initial: f64) -> ExponentialProblem {
    OdeProblem::new(exponential_rhs, vec![initial], span, ())
}

#[test]
fn fixed_step_order_and_backward_integration() {
    let coarse = solve(
        &exponential_problem((0.0, 1.0), 1.0),
        Rodas42,
        &fixed_options(0.1),
    )
    .unwrap()
    .last_state()[0];
    let fine = solve(
        &exponential_problem((0.0, 1.0), 1.0),
        Rodas42,
        &fixed_options(0.05),
    )
    .unwrap()
    .last_state()[0];
    let ratio = (coarse - std::f64::consts::E).abs() / (fine - std::f64::consts::E).abs();
    assert!(
        ratio > 10.0,
        "expected fourth-order convergence, ratio={ratio}"
    );

    let backward = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -2.0 * u[0],
        vec![(-2.0_f64).exp()],
        (1.0, 0.0),
        (),
    );
    let endpoint = solve(&backward, Rodas42, &fixed_options(0.02))
        .unwrap()
        .last_state()[0];
    assert!((endpoint - 1.0).abs() < 2.0e-6, "endpoint={endpoint:.17e}");
}

#[test]
fn adaptive_jacobian_callbacks_and_save_at_are_supported() {
    let rhs = |du: &mut [f64], u: &[f64], _: &(), time: f64| {
        du[0] = -1000.0 * (u[0] - time.cos()) - time.sin();
    };
    let problem = OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
        .with_jacobian(|jacobian: &mut [f64], _: &[f64], _: &(), _: f64| jacobian[0] = -1000.0)
        .with_continuous_callback(
            |_, _, time| time - 0.5,
            |state, _, _| {
                state[0] += 0.01;
                CallbackAction::Continue
            },
        );
    let options = SolveOptions {
        absolute_tolerance: 1.0e-8,
        relative_tolerance: 1.0e-8,
        save: SaveMode::Endpoints,
        save_at: vec![0.25, 0.5, 0.75],
        ..SolveOptions::default()
    };
    let solution = solve(&problem, Rodas42, &options).unwrap();
    assert!(solution.last_state()[0].is_finite());
    assert!(solution.stats().callback_invocations > 0);
    for time in options.save_at {
        assert!(solution.times().contains(&time), "missing save_at={time}");
    }
    assert!(solution.stats().jacobian_evaluations > 0);
}
