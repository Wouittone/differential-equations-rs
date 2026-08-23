use differential_equations::algorithms::*;
use differential_equations::*;

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn exponential() -> OdeProblem<TestRhs, ()> {
    fn rhs(du: &mut [f64], state: &[f64], _: &(), _: f64) {
        du[0] = state[0];
    }
    OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
}

fn linear() -> OdeProblem<TestRhs, ()> {
    fn rhs(du: &mut [f64], _: &[f64], _: &(), _: f64) {
        du[0] = 1.0;
    }
    OdeProblem::new(rhs, vec![0.0], (0.0, 1.0), ())
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
fn sir54_has_fifth_order_fixed_step_convergence() {
    let coarse = (solve(&exponential(), Sir54, &fixed(0.1))
        .unwrap()
        .last_state()[0]
        - std::f64::consts::E)
        .abs();
    let fine = (solve(&exponential(), Sir54, &fixed(0.05))
        .unwrap()
        .last_state()[0]
        - std::f64::consts::E)
        .abs();
    assert!(
        coarse / fine > 15.0,
        "expected fourth-plus-order convergence"
    );
}

#[test]
fn sir54_supports_fixed_steps_and_dense_save_at() {
    let solution = solve(
        &linear(),
        Sir54,
        &SolveOptions {
            adaptive: false,
            initial_step: Some(0.2),
            save: SaveMode::Endpoints,
            save_at: vec![0.2, 0.5, 0.8],
            ..SolveOptions::default()
        },
    )
    .unwrap();
    assert_eq!(solution.times(), &[0.2, 0.5, 0.8]);
    for (&time, state) in solution.times().iter().zip(solution.values()) {
        assert!((*state - time).abs() < 2.0e-12);
    }
}
