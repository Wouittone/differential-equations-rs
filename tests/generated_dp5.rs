use differential_equations::algorithms::*;
use differential_equations::*;

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
    du[0] = -u[0];
}

fn exponential() -> OdeProblem<TestRhs, ()> {
    OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
}

#[test]
fn generated_dp5_tableau_preserves_tight_endpoint_accuracy() {
    let options = SolveOptions {
        absolute_tolerance: 1.0e-11,
        relative_tolerance: 1.0e-11,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let solution = solve(&exponential(), Dp5, &options).unwrap();

    assert!((solution.last_state()[0] - (-1.0_f64).exp()).abs() < 2.0e-10);
    assert!(solution.stats().rhs_evaluations > solution.stats().accepted_steps);
    assert!(solution.stats().rhs_evaluations < 7 * (solution.stats().accepted_steps + 1));
}
