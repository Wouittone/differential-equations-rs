use differential_equations::{Bs3, OdeProblem, SaveMode, SolveOptions, solve};

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
    du[0] = -u[0];
}

fn exponential() -> OdeProblem<TestRhs, ()> {
    OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
}

#[test]
fn generated_bs3_tableau_preserves_adaptive_accuracy_and_fsal_work() {
    let options = SolveOptions {
        absolute_tolerance: 1.0e-9,
        relative_tolerance: 1.0e-9,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let solution = solve(&exponential(), Bs3, &options).unwrap();

    assert_eq!(solution.dimension(), 1);
    assert!((solution.last_state()[0] - (-1.0_f64).exp()).abs() < 5.0e-8);
    // The FSAL stage is reused between accepted steps, so the RHS count is
    // strictly below a fresh four-stage evaluation for every accepted step.
    assert!(solution.stats().rhs_evaluations > solution.stats().accepted_steps);
    assert!(solution.stats().rhs_evaluations < 4 * (solution.stats().accepted_steps + 1));
}
