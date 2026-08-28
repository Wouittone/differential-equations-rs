use differential_equations::solvers::explicit::*;
use differential_equations::*;

fn unit_rate(du: &mut [f64], _: &[f64], _: &(), _: f64) {
    du[0] = 1.0;
}

fn fixed_options(step: f64) -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

#[test]
fn alshina6_supports_save_at_and_backward_steps() {
    let forward = OdeProblem::new(unit_rate, vec![0.0], (0.0, 1.0), ());
    let solution = solve(
        &forward,
        Alshina6,
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

    let backward = OdeProblem::new(unit_rate, vec![1.0], (1.0, 0.0), ());
    let solution = solve(
        &backward,
        Alshina6,
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
fn alshina6_continuous_callback_can_terminate() {
    let problem = OdeProblem::new(unit_rate, vec![0.0], (0.0, 1.0), ()).with_continuous_callback(
        |state, _: &(), _| state[0] - 0.5,
        |state, _: &(), _| {
            state[0] = 42.0;
            CallbackAction::Terminate
        },
    );
    let solution = solve(&problem, Alshina6, &fixed_options(0.25)).unwrap();
    assert_eq!(solution.last_state(), &[42.0]);
    assert_eq!(solution.stats().callback_invocations, 1);
}
