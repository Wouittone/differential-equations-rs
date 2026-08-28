use differential_equations::solvers::exponential::rkip::solve_rkip;
use differential_equations::solvers::rosenbrock::Rosenbrock23;
use differential_equations::solvers::rosenbrock::amf::{
    AMFOperator, AmfProblem, build_amf_function, solve_amf,
};
use differential_equations::solvers::stabilized::irkc::solve_irkc;
use differential_equations::solvers::{
    exponential::{InteractionPictureAlgorithm, rkip::RKIP},
    rosenbrock::amf::AMF,
    stabilized::IRKC,
};
use differential_equations::{
    CallbackAction, OdeProblem, SaveMode, SemilinearOdeProblem, SolveError, SolveOptions,
    SplitOdeProblem, solve,
};

fn fixed(step: f64) -> SolveOptions {
    SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(step)
        .with_save(SaveMode::Endpoints)
}

#[test]
fn amf_operator_applies_ordered_factor_solves() {
    let j1 = vec![0.0, 1.0, 0.0, 0.0];
    let j2 = vec![0.0, 0.0, 1.0, 0.0];
    let mut operator = AMFOperator::from_split(2, vec![j1, j2]).unwrap();
    operator.factorize(0.2).unwrap();
    let mut rhs = vec![1.0, 2.0];
    operator.solve_ordered(&mut rhs);
    // Solve (I-.2J1)(I-.2J2)x=b explicitly.
    assert!((rhs[0] - 1.4).abs() < 1.0e-14);
    assert!((rhs[1] - 2.28).abs() < 1.0e-14);
    assert_eq!(operator.factor_count(), 2);
}

#[test]
fn structured_amf_runs_rosenbrock_w_with_each_factor() {
    fn rhs(output: &mut [f64], state: &[f64], _: &(), _: f64) {
        output[0] = -3.0 * state[0];
        output[1] = -7.0 * state[1];
    }
    fn jacobian(output: &mut [f64], _: &[f64], _: &(), _: f64) {
        output.copy_from_slice(&[-3.0, 0.0, 0.0, -7.0]);
    }
    fn factors(factors: &mut [Vec<f64>], _: &[f64], _: &(), _: f64) {
        factors[0].copy_from_slice(&[-3.0, 0.0, 0.0, 0.0]);
        factors[1].copy_from_slice(&[0.0, 0.0, 0.0, -7.0]);
    }
    let function = build_amf_function(2, rhs, jacobian, vec![vec![0.0; 4]; 2], factors).unwrap();
    let problem = AmfProblem::new(function, vec![1.0, 1.0], (0.0, 0.5), ()).unwrap();
    let solution = solve_amf(&problem, AMF::new(Rosenbrock23), &fixed(0.01)).unwrap();
    assert!((solution.last_state()[0] - (-1.5_f64).exp()).abs() < 2.0e-4);
    assert!((solution.last_state()[1] - (-3.5_f64).exp()).abs() < 2.0e-4);
    assert!(solution.stats().linear_factorizations >= 2 * solution.stats().accepted_steps);
    assert!(solution.stats().linear_solves >= 4 * solution.stats().accepted_steps);
}

#[test]
fn ordinary_amf_retains_callback_lifecycle() {
    fn rhs(output: &mut [f64], state: &[f64], _: &(), _: f64) {
        output[0] = -state[0];
    }
    let problem = OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
        .with_jacobian(|jacobian, _, _, _| jacobian[0] = -1.0)
        .with_discrete_callback(
            |_, _, time| time >= 0.3,
            |_, _, _| CallbackAction::Terminate,
        );
    let solution = solve(&problem, AMF::new(Rosenbrock23), &fixed(0.1)).unwrap();
    assert!((solution.times().last().unwrap() - 0.3).abs() < 1.0e-12);
    assert_eq!(solution.stats().callback_invocations, 1);
}

#[test]
fn rkip_matches_affine_semilinear_solution_and_recycles_cache() {
    fn nonlinear(output: &mut [f64], _: &[f64], _: &(), _: f64) {
        output[0] = 1.0;
    }
    let problem =
        SemilinearOdeProblem::new(vec![-2.0], nonlinear, vec![1.0], (0.0, 1.0), ()).unwrap();
    let algorithm = RKIP::new(0.1, 0.2, 2).unwrap();
    let first = solve_rkip(&problem, &algorithm, &fixed(0.1)).unwrap();
    let exact = 0.5 + 0.5 * (-2.0_f64).exp();
    assert!((first.last_state()[0] - exact).abs() < 2.0e-10);
    let built = algorithm.cache_stats().exponentials_built;
    let hits = algorithm.cache_stats().cache_hits;
    let second = solve_rkip(&problem, &algorithm, &fixed(0.1)).unwrap();
    assert!((second.last_state()[0] - exact).abs() < 2.0e-10);
    // The floating-point terminal remainder is intentionally single-use, but
    // the geometric-grid exponentials are recycled across the second solve.
    assert!(algorithm.cache_stats().exponentials_built >= built);
    assert!(algorithm.cache_stats().cache_hits > hits);
    assert_eq!(algorithm.cache_stats().cached_step_sizes, 1);
}

#[test]
fn rkip_rejects_invalid_inputs_before_mutating_its_cache() {
    fn zero(output: &mut [f64], _: &[f64], _: &(), _: f64) {
        output.fill(0.0);
    }
    let problem = SemilinearOdeProblem::new(vec![-1.0], zero, vec![1.0], (0.0, 1.0), ()).unwrap();
    let algorithm = RKIP::new(0.1, 0.2, 2).unwrap();
    solve_rkip(&problem, &algorithm, &fixed(0.1)).unwrap();
    let before = algorithm.cache_stats();
    let invalid = SolveOptions::new().with_tolerances(0.0, 1.0e-3);
    assert_eq!(
        solve_rkip(&problem, &algorithm, &invalid),
        Err(SolveError::InvalidTolerance)
    );
    assert_eq!(algorithm.cache_stats(), before);

    assert_eq!(
        InteractionPictureAlgorithm::solve_interaction_picture(&algorithm, &problem, &invalid),
        Err(SolveError::InvalidTolerance)
    );
    assert_eq!(algorithm.cache_stats(), before);
}

#[test]
fn rkip_adaptive_estimator_refines_and_backward_is_inverse_for_linear_case() {
    fn nonlinear(output: &mut [f64], state: &[f64], _: &(), _: f64) {
        output[0] = state[0].sin();
    }
    let problem =
        SemilinearOdeProblem::new(vec![-1.0], nonlinear, vec![0.4], (0.0, 1.0), ()).unwrap();
    let algorithm = RKIP::new(1.0e-4, 0.25, 32).unwrap();
    let solution = solve_rkip(
        &problem,
        &algorithm,
        &SolveOptions::new().with_tolerances(1e-9, 1e-9),
    )
    .unwrap();
    assert!(solution.stats().accepted_steps > 1);

    fn zero(output: &mut [f64], _: &[f64], _: &(), _: f64) {
        output.fill(0.0);
    }
    let forward = SemilinearOdeProblem::new(vec![-2.0], zero, vec![1.0], (0.0, 1.0), ()).unwrap();
    let forward = solve_rkip(&forward, &algorithm, &fixed(0.1)).unwrap();
    let backward = SemilinearOdeProblem::new(
        vec![-2.0],
        zero,
        forward.last_state().to_vec(),
        (1.0, 0.0),
        (),
    )
    .unwrap();
    let backward = solve_rkip(&backward, &algorithm, &fixed(0.1)).unwrap();
    assert!((backward.last_state()[0] - 1.0).abs() < 2.0e-11);
}

#[test]
fn irkc_handles_stiff_split_and_eigenvalue_override() {
    fn explicit(output: &mut [f64], state: &[f64], _: &(), _: f64) {
        output[0] = -100.0 * state[0];
    }
    fn implicit(output: &mut [f64], state: &[f64], _: &(), _: f64) {
        output[0] = -state[0];
    }
    let problem = SplitOdeProblem::new(explicit, implicit, vec![1.0], (0.0, 0.1), ())
        .with_implicit_jacobian(|jacobian, _, _, _| jacobian[0] = -1.0);
    let solution = solve_irkc(
        &problem,
        IRKC::new().with_eigenvalue_estimate(100.0),
        &fixed(0.01),
    )
    .unwrap();
    assert!((solution.last_state()[0] - (-10.1_f64).exp()).abs() < 5.0e-3);
    assert!(solution.stats().nonlinear_iterations > 0);
    assert!(solution.stats().linear_factorizations > 0);
}

#[test]
fn irkc_estimates_eigenvalue_and_reports_configuration_failures() {
    fn explicit(output: &mut [f64], state: &[f64], _: &(), _: f64) {
        output[0] = -20.0 * state[0];
    }
    fn implicit(output: &mut [f64], _: &[f64], _: &(), _: f64) {
        output[0] = 0.0;
    }
    let problem = SplitOdeProblem::new(explicit, implicit, vec![1.0], (0.0, 0.1), ());
    let solution = solve_irkc(&problem, IRKC::new(), &fixed(0.01)).unwrap();
    assert!(solution.stats().rhs_evaluations > 50);
    let overridden = solve_irkc(
        &problem,
        IRKC::new().with_eigenvalue_estimate(20.0),
        &fixed(0.01),
    )
    .unwrap();
    assert!((solution.last_state()[0] - overridden.last_state()[0]).abs() < 1.0e-12);
    assert!(overridden.stats().rhs_evaluations < solution.stats().rhs_evaluations);
    assert_eq!(
        solve_irkc(
            &problem,
            IRKC::new().with_eigenvalue_estimate(f64::NAN),
            &fixed(0.01)
        )
        .unwrap_err(),
        SolveError::InvalidTolerance
    );
}

#[test]
fn irkc_preserves_typed_continuous_callbacks() {
    let problem = SplitOdeProblem::new(
        |output: &mut [f64], _: &[f64], _: &(), _: f64| output[0] = 1.0,
        |output: &mut [f64], _: &[f64], _: &(), _: f64| output[0] = 0.0,
        vec![0.0],
        (0.0, 0.2),
        (),
    )
    .with_continuous_callback(
        |state, _, _| state[0] - 0.1,
        |state, _, _| {
            state[0] = 3.0;
            CallbackAction::Terminate
        },
    );
    let solution = solve_irkc(
        &problem,
        IRKC::new().with_eigenvalue_estimate(1.0),
        &fixed(0.2).with_event_tolerance(1.0e-11),
    )
    .unwrap();
    assert!((solution.times().last().unwrap() - 0.1).abs() < 1.0e-10);
    assert_eq!(solution.last_state(), &[3.0]);
    assert_eq!(solution.stats().callback_invocations, 1);
}

#[test]
fn typed_problem_dense_segments_use_real_derivatives() {
    fn rhs(output: &mut [f64], state: &[f64], _: &(), _: f64) {
        output[0] = -state[0];
    }
    fn jacobian(output: &mut [f64], _: &[f64], _: &(), _: f64) {
        output[0] = -1.0;
    }
    fn factors(factors: &mut [Vec<f64>], _: &[f64], _: &(), _: f64) {
        factors[0][0] = -1.0;
    }
    let function = build_amf_function(1, rhs, jacobian, vec![vec![0.0]], factors).unwrap();
    let amf_problem = AmfProblem::new(function, vec![1.0], (0.0, 1.0), ()).unwrap();
    let dense = fixed(0.05).with_dense_output(true);
    let amf = solve_amf(&amf_problem, AMF::new(Rosenbrock23), &dense).unwrap();
    assert!((amf.interpolate(0.375).unwrap()[0] - (-0.375_f64).exp()).abs() < 5.0e-4);

    fn zero(output: &mut [f64], _: &[f64], _: &(), _: f64) {
        output.fill(0.0);
    }
    let rkip_problem =
        SemilinearOdeProblem::new(vec![-1.0], zero, vec![1.0], (0.0, 1.0), ()).unwrap();
    let rkip_algorithm = RKIP::new(0.05, 0.1, 2).unwrap();
    let rkip = solve_rkip(&rkip_problem, &rkip_algorithm, &dense).unwrap();
    assert!((rkip.interpolate(0.375).unwrap()[0] - (-0.375_f64).exp()).abs() < 2.0e-6);

    let irkc_problem = SplitOdeProblem::new(rhs, zero, vec![1.0], (0.0, 0.1), ());
    let irkc = solve_irkc(
        &irkc_problem,
        IRKC::new().with_eigenvalue_estimate(1.0),
        &fixed(0.001).with_dense_output(true),
    )
    .unwrap();
    assert!((irkc.interpolate(0.0555).unwrap()[0] - (-0.0555_f64).exp()).abs() < 2.0e-4);
}
