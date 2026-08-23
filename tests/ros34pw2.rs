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

type ExponentialRhs = fn(&mut [f64], &[f64], &(), f64);

fn exponential() -> OdeProblem<ExponentialRhs, ()> {
    OdeProblem::new(
        (|du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0]) as ExponentialRhs,
        vec![1.0],
        (0.0, 1.0),
        (),
    )
}

#[test]
fn ros34pw2_has_pinned_fixed_step_convergence() {
    let coarse = (solve(&exponential(), Ros34Pw2, &fixed_options(0.1))
        .unwrap()
        .last_state()[0]
        - std::f64::consts::E)
        .abs();
    let fine = (solve(&exponential(), Ros34Pw2, &fixed_options(0.05))
        .unwrap()
        .last_state()[0]
        - std::f64::consts::E)
        .abs();
    assert!(coarse / fine > 7.0, "ratio={:.3}", coarse / fine);
}

#[test]
fn ros34pw2_adaptive_jacobian_and_stiff_problem() {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), time: f64) {
        du[0] = -1000.0 * (u[0] - time.cos()) - time.sin();
    }
    let problem = OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
        .with_jacobian(|jacobian: &mut [f64], _: &[f64], _: &(), _: f64| jacobian[0] = -1000.0);
    let options = SolveOptions {
        absolute_tolerance: 1.0e-8,
        relative_tolerance: 1.0e-8,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let solution = solve(&problem, Ros34Pw2, &options).unwrap();
    assert!((solution.last_state()[0] - 1.0_f64.cos()).abs() < 2.0e-6);
    assert!(solution.stats().jacobian_evaluations > 0);
}

#[test]
fn ros34pw2_supports_backward_callbacks_and_save_at() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -2.0 * u[0],
        vec![(-2.0_f64).exp()],
        (1.0, 0.0),
        (),
    )
    .with_discrete_callback(
        |_, _, time| (time - 0.5).abs() < 1.0e-12,
        |state, _, _| {
            state[0] += 0.25;
            CallbackAction::Continue
        },
    );
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.05),
        save: SaveMode::Endpoints,
        save_at: vec![0.75, 0.5, 0.25],
        ..SolveOptions::default()
    };
    let solution = solve(&problem, Ros34Pw2, &options).unwrap();
    assert_eq!(solution.stats().callback_invocations, 1);
    for time in options.save_at {
        assert!(solution.times().contains(&time), "missing save_at={time}");
    }
    assert!(solution.last_state()[0] > 0.9);
}
