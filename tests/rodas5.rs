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

#[test]
fn rodas5_fixed_and_adaptive_orders_match_fifth_order_primary() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let fixed = solve(&problem, Rodas5, &fixed_options(0.125)).unwrap();
    assert!((fixed.last_state()[0] - std::f64::consts::E).abs() < 1.0e-8);

    let adaptive = solve(
        &problem,
        Rodas5,
        &SolveOptions {
            absolute_tolerance: 1.0e-9,
            relative_tolerance: 1.0e-9,
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        },
    )
    .unwrap();
    assert!((adaptive.last_state()[0] - std::f64::consts::E).abs() < 1.0e-7);
}

#[test]
fn rodas5_supports_backward_jacobian_callback_and_save_at() {
    let backward_problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -2.0 * u[0],
        vec![(-2.0_f64).exp()],
        (1.0, 0.0),
        (),
    );
    let backward = solve(
        &backward_problem,
        Rodas5,
        &SolveOptions {
            absolute_tolerance: 1.0e-9,
            relative_tolerance: 1.0e-9,
            initial_step: Some(0.01),
            max_step: 0.01,
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        },
    )
    .unwrap();
    assert!((backward.last_state()[0] - 1.0).abs() < 1.0e-7);

    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    )
    .with_jacobian(|jacobian: &mut [f64], _: &[f64], _: &(), _: f64| jacobian[0] = -1.0)
    .with_discrete_callback(
        |_, _, time| time == 0.5,
        |state, _, _| {
            state[0] += 0.25;
            CallbackAction::Continue
        },
    );
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.25),
        save: SaveMode::Endpoints,
        save_at: vec![0.25, 0.5, 0.75],
        ..SolveOptions::default()
    };
    let solution = solve(&problem, Rodas5, &options).unwrap();
    assert_eq!(solution.stats().callback_invocations, 1);
    assert!(solution.stats().jacobian_evaluations > 0);
    for time in options.save_at {
        assert!(solution.times().contains(&time), "missing save_at={time}");
    }
}
