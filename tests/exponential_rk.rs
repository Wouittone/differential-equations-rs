use differential_equations::solvers::exponential::general::ExponentialAlgorithm;
use differential_equations::solvers::exponential::*;
use differential_equations::{
    OdeProblem, SaveMode, SemilinearOdeProblem, SolveError, SolveOptions, solve,
};

fn fixed(step: f64) -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

fn assert_linear_exact<A: ExponentialAlgorithm>(algorithm: A) {
    let problem = SemilinearOdeProblem::new(
        vec![-2.0],
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = 0.0,
        vec![1.0],
        (0.0, 1.0),
        (),
    )
    .unwrap();
    let solution = solve_exponential(&problem, algorithm, &fixed(0.2)).unwrap();
    assert!((solution.last_state()[0] - (-2.0_f64).exp()).abs() < 3.0e-12);
}

#[test]
fn every_public_constructor_executes_a_genuine_exponential_step() {
    assert_linear_exact(LawsonEuler);
    assert_linear_exact(NorsettEuler);
    assert_linear_exact(ETD1);
    assert_linear_exact(ETDRK2);
    assert_linear_exact(ETDRK3);
    assert_linear_exact(ETDRK4);
    assert_linear_exact(HochOst4);
    assert_linear_exact(Exp4);
    assert_linear_exact(EPIRK4s3A);
    assert_linear_exact(EPIRK4s3B);
    assert_linear_exact(EPIRK5s3);
    assert_linear_exact(EXPRB53s3);
    assert_linear_exact(EPIRK5P1);
    assert_linear_exact(EPIRK5P2);
    assert_linear_exact(ETD2);
    assert_linear_exact(Exprb32);
    assert_linear_exact(Exprb43);
}

#[test]
fn semilinear_vector_operator_is_integrated_exactly() {
    let problem = SemilinearOdeProblem::new(
        vec![0.0, -1.0, 1.0, 0.0],
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du.fill(0.0),
        vec![1.0, 0.0],
        (0.0, std::f64::consts::FRAC_PI_2),
        (),
    )
    .unwrap();
    let solution = solve_exponential(&problem, ETDRK4, &fixed(0.2)).unwrap();
    assert!(solution.last_state()[0].abs() < 2.0e-12);
    assert!((solution.last_state()[1] - 1.0).abs() < 2.0e-12);
    assert!(solution.stats().jacobian_evaluations > 0);
    assert_eq!(solution.stats().rejected_steps, 0);
}

#[test]
fn etd_methods_support_nonautonomous_remainders_and_backward_time() {
    let forward = SemilinearOdeProblem::new(
        vec![-1.0],
        |du: &mut [f64], _: &[f64], _: &(), time: f64| du[0] = time,
        vec![0.0],
        (0.0, 1.0),
        (),
    )
    .unwrap();
    let expected = (-1.0_f64).exp();
    let coarse = solve_exponential(&forward, ETDRK2, &fixed(0.1)).unwrap();
    let fine = solve_exponential(&forward, ETDRK2, &fixed(0.05)).unwrap();
    let coarse_error = (coarse.last_state()[0] - expected).abs();
    let fine_error = (fine.last_state()[0] - expected).abs();
    assert!(coarse_error < 2.0e-12);
    assert!(fine_error < 2.0e-12);

    let backward = SemilinearOdeProblem::new(
        vec![-1.0],
        |du: &mut [f64], _: &[f64], _: &(), time: f64| du[0] = time,
        vec![expected],
        (1.0, 0.0),
        (),
    )
    .unwrap();
    let solution = solve_exponential(&backward, ETDRK3, &fixed(0.025)).unwrap();
    assert!(solution.last_state()[0].abs() < 2.0e-4);
}

#[test]
fn adaptive_exprb_methods_control_error_and_fixed_methods_reject_adaptivity() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    )
    .with_jacobian(|jacobian, _, _, _| jacobian[0] = 1.0);
    let options = SolveOptions::default().with_tolerances(1.0e-10, 1.0e-8);
    for solution in [
        solve(&problem, Exprb32, &options).unwrap(),
        solve(&problem, Exprb43, &options).unwrap(),
    ] {
        assert!((solution.last_state()[0] - std::f64::consts::E).abs() < 2.0e-8);
        assert!(solution.stats().accepted_steps > 0);
    }
    assert_eq!(
        solve(&problem, ETDRK4, &SolveOptions::default()),
        Err(SolveError::AdaptiveStepUnsupported)
    );
    assert_eq!(
        solve(
            &problem,
            ETDRK4,
            &SolveOptions::default().with_adaptive(false)
        ),
        Err(SolveError::InitialStepRequired)
    );
}

#[test]
fn nonlinear_convergence_improves_at_the_declared_rate() {
    fn error<A: ExponentialAlgorithm>(algorithm: A, step: f64) -> f64 {
        let problem = SemilinearOdeProblem::new(
            vec![-1.0],
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0] * u[0],
            vec![0.25],
            (0.0, 1.0),
            (),
        )
        .unwrap();
        let exact = 1.0 / (1.0 + 3.0 * std::f64::consts::E);
        (solve_exponential(&problem, algorithm, &fixed(step))
            .unwrap()
            .last_state()[0]
            - exact)
            .abs()
    }

    assert!(error(ETDRK2, 0.05) < error(ETDRK2, 0.1) / 3.0);
    assert!(error(ETDRK3, 0.05) < error(ETDRK3, 0.1) / 6.0);
    assert!(error(ETDRK4, 0.05) < error(ETDRK4, 0.1) / 10.0);
    assert!(error(HochOst4, 0.05) < error(HochOst4, 0.1) / 10.0);
    assert!(error(ETD2, 0.05) < error(ETD2, 0.1) / 3.0);
    assert!(error(LawsonEuler, 0.05) < error(LawsonEuler, 0.1) / 1.7);
    assert!(error(NorsettEuler, 0.05) < error(NorsettEuler, 0.1) / 1.7);
}

#[test]
fn exponential_rosenbrock_formulas_converge_on_a_nonlinear_problem() {
    fn error<A: ExponentialAlgorithm>(algorithm: A, step: f64) -> f64 {
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0] * (u[0] - 1.0),
            vec![0.25],
            (0.0, 1.0),
            (),
        )
        .with_jacobian(|jacobian, u, _, _| jacobian[0] = 2.0 * u[0] - 1.0);
        let exact = 1.0 / (1.0 + 3.0 * std::f64::consts::E);
        (solve(&problem, algorithm, &fixed(step))
            .unwrap()
            .last_state()[0]
            - exact)
            .abs()
    }
    fn ratio<A: ExponentialAlgorithm + Copy>(algorithm: A) -> f64 {
        error(algorithm, 0.1) / error(algorithm, 0.05)
    }

    assert!(ratio(Exp4) > 8.0);
    assert!(ratio(EPIRK4s3A) > 8.0);
    assert!(ratio(EPIRK4s3B) > 8.0);
    assert!(ratio(EXPRB53s3) > 16.0);
    assert!(ratio(EPIRK5P1) > 16.0);
    assert!(ratio(EPIRK5P2) > 16.0);
    // OrdinaryDiffEq marks EPIRK5s3 broken at the pinned revision, but the
    // published exponential formula must still improve under refinement.
    assert!(ratio(EPIRK5s3) > 2.0);
}

#[test]
fn nonfinite_nonlinear_output_is_reported() {
    assert!(
        SemilinearOdeProblem::new(
            vec![1.0, 0.0],
            |_: &mut [f64], _: &[f64], _: &(), _: f64| {},
            vec![1.0],
            (0.0, 1.0),
            (),
        )
        .is_err()
    );
    let problem = SemilinearOdeProblem::new(
        vec![0.0],
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = f64::NAN,
        vec![1.0],
        (0.0, 1.0),
        (),
    )
    .unwrap();
    assert_eq!(
        solve_exponential(&problem, ETDRK2, &fixed(0.1)),
        Err(SolveError::NonFiniteDerivative)
    );
}
