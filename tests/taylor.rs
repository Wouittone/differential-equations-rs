use differential_equations::solvers::taylor::{
    ExplicitTaylor, ExplicitTaylor2, ExplicitTaylorAdaptiveOrder,
};
use differential_equations::{
    CallbackAction, ConfigurationError, OdeAlgorithm, OdeProblem, SaveMode, SolveError,
    SolveOptions, solve,
};

type Rhs = fn(&mut [f64], &[f64], &(), f64);

fn exponential(derivative: &mut [f64], state: &[f64], _: &(), _: f64) {
    derivative[0] = state[0];
}

fn endpoint<A: OdeAlgorithm>(algorithm: A, step: f64, adaptive: bool) -> f64 {
    let problem = OdeProblem::new(exponential as Rhs, vec![1.0], (0.0, 1.0), ());
    solve(
        &problem,
        algorithm,
        &SolveOptions::default()
            .with_adaptive(adaptive)
            .with_initial_step(Some(step))
            .with_save(SaveMode::Endpoints)
            .with_relative_tolerance(1.0e-8)
            .with_absolute_tolerance(1.0e-10),
    )
    .unwrap()
    .last_state()[0]
}

#[test]
fn fixed_taylor_polynomials_recover_configured_orders() {
    let exact = 1.0_f64.exp();
    let second_coarse = (endpoint(ExplicitTaylor2, 0.1, false) - exact).abs();
    let second_fine = (endpoint(ExplicitTaylor2, 0.05, false) - exact).abs();
    assert!(second_coarse / second_fine > 3.5);

    for order in [4, 6, 8] {
        let coarse = (endpoint(ExplicitTaylor::new(order).unwrap(), 0.4, false) - exact).abs();
        let fine = (endpoint(ExplicitTaylor::new(order).unwrap(), 0.2, false) - exact).abs();
        assert!(
            coarse / fine > 2.0_f64.powi(order as i32) * 0.55,
            "order {order}: coarse={coarse:e}, fine={fine:e}"
        );
    }
}

#[test]
fn fixed_and_adaptive_order_variants_control_error() {
    assert_eq!(ExplicitTaylor::new(8).unwrap().order(), 8);
    let adaptive_order = ExplicitTaylorAdaptiveOrder::new(4, 9).unwrap();
    assert_eq!(adaptive_order.min_order(), 4);
    assert_eq!(adaptive_order.max_order(), 9);
    let exact = 1.0_f64.exp();
    assert!((endpoint(ExplicitTaylor::new(8).unwrap(), 0.2, true) - exact).abs() < 2.0e-7);
    assert!((endpoint(adaptive_order, 0.2, true) - exact).abs() < 2.0e-7);
}

#[test]
fn constructors_preserve_supported_orders_and_reject_invalid_configuration() {
    assert_eq!(ExplicitTaylor::new(1).unwrap().order(), 1);
    assert_eq!(ExplicitTaylor::new(12).unwrap().order(), 12);
    assert_eq!(
        ExplicitTaylor::new(0),
        Err(ConfigurationError::InvalidParameter {
            parameter: "Taylor order",
            reason: "must be between 1 and 12 inclusive",
        })
    );
    assert_eq!(
        ExplicitTaylor::new(13),
        Err(ConfigurationError::InvalidParameter {
            parameter: "Taylor order",
            reason: "must be between 1 and 12 inclusive",
        })
    );

    let full_window = ExplicitTaylorAdaptiveOrder::new(1, 12).unwrap();
    assert_eq!(full_window.min_order(), 1);
    assert_eq!(full_window.max_order(), 12);
    for (minimum, maximum) in [(0, 12), (4, 4), (8, 7), (1, 13)] {
        assert_eq!(
            ExplicitTaylorAdaptiveOrder::new(minimum, maximum),
            Err(ConfigurationError::InvalidBounds {
                context: "adaptive Taylor order window",
                reason: "must satisfy 1 <= min_order < max_order <= 12",
            })
        );
    }
}

#[test]
fn second_order_variant_remains_fixed_step_only() {
    let problem = OdeProblem::new(exponential as Rhs, vec![1.0], (0.0, 1.0), ());
    assert_eq!(
        solve(&problem, ExplicitTaylor2, &SolveOptions::default()),
        Err(SolveError::AdaptiveStepUnsupported)
    );
}

#[test]
fn native_taylor_polynomial_drives_dense_queries_and_roots() {
    let problem = OdeProblem::new(exponential as Rhs, vec![1.0], (0.0, 0.4), ());
    let solution = solve(
        &problem,
        ExplicitTaylor::new(8).unwrap(),
        &SolveOptions::default()
            .with_adaptive(false)
            .with_initial_step(Some(0.4))
            .with_dense_output(true)
            .with_save(SaveMode::Endpoints),
    )
    .unwrap();
    assert_eq!(
        solution.interpolate(0.0).unwrap(),
        solution.state(0).unwrap()
    );
    assert_eq!(solution.interpolate(0.4).unwrap(), solution.last_state());
    assert!((solution.interpolate(0.17).unwrap()[0] - 0.17_f64.exp()).abs() < 2.0e-10);

    let event_problem = OdeProblem::new(exponential as Rhs, vec![1.0], (0.0, 1.0), ())
        .with_continuous_callback(
            |state, _: &(), _: f64| state[0] - 1.5,
            |_: &mut [f64], _: &(), _: f64| CallbackAction::Terminate,
        );
    let event = solve(
        &event_problem,
        ExplicitTaylor::new(8).unwrap(),
        &SolveOptions::default()
            .with_adaptive(false)
            .with_initial_step(Some(0.5))
            .with_dense_output(true)
            .with_event_tolerance(1.0e-12),
    )
    .unwrap();
    let event_time = *event.times().last().unwrap();
    assert!((event_time - 1.5_f64.ln()).abs() < 2.0e-9);
    assert!((event.interpolate(event_time).unwrap()[0] - 1.5).abs() < 1.0e-12);
}
