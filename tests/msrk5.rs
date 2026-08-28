use differential_equations::solvers::explicit::*;
use differential_equations::*;

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn exponential() -> OdeProblem<TestRhs, ()> {
    fn rhs(du: &mut [f64], state: &[f64], _: &(), _: f64) {
        du[0] = state[0];
    }
    OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
}

fn fixed(step: f64) -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

#[test]
fn msrk5_has_fifth_order_fixed_step_convergence() {
    let coarse = (solve(&exponential(), Msrk5, &fixed(0.1))
        .unwrap()
        .last_state()[0]
        - std::f64::consts::E)
        .abs();
    let fine = (solve(&exponential(), Msrk5, &fixed(0.05))
        .unwrap()
        .last_state()[0]
        - std::f64::consts::E)
        .abs();
    assert!(coarse / fine > 20.0, "expected fifth-order convergence");
}

#[test]
fn msrk5_supports_forward_backward_save_at() {
    let forward = OdeProblem::new(
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = 1.0,
        vec![0.0],
        (0.0, 1.0),
        (),
    );
    let solution = solve(
        &forward,
        Msrk5,
        &SolveOptions {
            adaptive: false,
            initial_step: Some(0.3),
            save_at: vec![0.2, 0.5, 0.8],
            ..SolveOptions::default()
        },
    )
    .unwrap();
    assert_eq!(solution.times(), &[0.2, 0.5, 0.8]);
    for (&time, state) in solution.times().iter().zip(solution.values()) {
        assert!((time - *state).abs() < 2.0e-12);
    }

    let backward = OdeProblem::new(
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = 1.0,
        vec![1.0],
        (1.0, 0.0),
        (),
    );
    let solution = solve(
        &backward,
        Msrk5,
        &SolveOptions {
            adaptive: false,
            initial_step: Some(0.3),
            save_at: vec![0.8, 0.5, 0.2],
            ..SolveOptions::default()
        },
    )
    .unwrap();
    assert_eq!(solution.times(), &[0.8, 0.5, 0.2]);
    for (&time, state) in solution.times().iter().zip(solution.values()) {
        assert!((time - *state).abs() < 2.0e-12);
    }
}

#[test]
fn msrk5_callback_can_terminate_without_post_effect_rhs() {
    let problem = OdeProblem::new(
        |du: &mut [f64], state: &[f64], _: &(), _: f64| {
            du[0] = if state[0] >= 40.0 { f64::NAN } else { 1.0 };
        },
        vec![0.0],
        (0.0, 1.0),
        (),
    )
    .with_continuous_callback(
        |state, _: &(), _| state[0] - 0.5,
        |state, _: &(), _| {
            state[0] = 42.0;
            CallbackAction::Terminate
        },
    );
    let solution = solve(&problem, Msrk5, &fixed(0.25)).unwrap();
    assert_eq!(solution.last_state(), &[42.0]);
    assert_eq!(solution.stats().callback_invocations, 1);
}
