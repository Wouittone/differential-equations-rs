use differential_equations::algorithms::linear::*;
use differential_equations::solvers::linear::general::{
    LinearOperatorAlgorithm, solve_lie_group, solve_linear_operator,
};
use differential_equations::{
    CallbackAction, LieGroupProblem, LinearOperatorProblem, OdeProblem, SaveMode, SolveError,
    SolveOptions, solve,
};

fn fixed(step: f64) -> SolveOptions {
    SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(step)
        .with_save(SaveMode::Endpoints)
}

fn rotation_operator(output: &mut [f64], _: &[f64], _: &(), _: f64) {
    output.copy_from_slice(&[0.0, -1.0, 1.0, 0.0]);
}

fn assert_rotation<A: LinearOperatorAlgorithm>(algorithm: A) {
    let problem =
        LinearOperatorProblem::new(rotation_operator, vec![1.0, 0.0], (0.0, 1.0), ()).unwrap();
    let solution = solve_linear_operator(&problem, algorithm, &fixed(0.125)).unwrap();
    let final_state = solution.last_state();
    assert!((final_state[0] - 1.0_f64.cos()).abs() < 2.0e-12);
    assert!((final_state[1] - 1.0_f64.sin()).abs() < 2.0e-12);
    assert!(solution.stats().rhs_evaluations > 0);
}

#[test]
fn every_vector_constructor_executes_a_genuine_exponential_action() {
    assert_rotation(LieEuler);
    assert_rotation(LinearExponential);
    assert_rotation(MagnusMidpoint);
    assert_rotation(MagnusLeapfrog);
    assert_rotation(RKMK2);
    assert_rotation(RKMK4);
    assert_rotation(LieRK4);
    assert_rotation(CG2);
    assert_rotation(CG3);
    assert_rotation(CG4a);
    assert_rotation(MagnusAdapt4);
    assert_rotation(MagnusGauss4);
    assert_rotation(MagnusGL4);
    assert_rotation(MagnusGL6);
    assert_rotation(MagnusNC6);
    assert_rotation(MagnusGL8);
    assert_rotation(MagnusNC8);
}

fn scalar_time_operator(output: &mut [f64], _: &[f64], _: &(), time: f64) {
    output[0] = -(1.0 + time);
}

fn scalar_error<A: LinearOperatorAlgorithm>(algorithm: A, step: f64) -> f64 {
    let problem =
        LinearOperatorProblem::new(scalar_time_operator, vec![1.0], (0.0, 1.0), ()).unwrap();
    let solution = solve_linear_operator(&problem, algorithm, &fixed(step)).unwrap();
    (solution.last_state()[0] - (-1.5_f64).exp()).abs()
}

#[test]
fn midpoint_and_gauss_paths_have_their_expected_time_quadrature_behavior() {
    let euler_coarse = scalar_error(LieEuler, 0.2);
    let euler_fine = scalar_error(LieEuler, 0.1);
    assert!(euler_coarse / euler_fine > 1.8);

    // Midpoint and every Gauss/NC formula integrate this affine scalar
    // generator exactly; ordinary explicit RK substitution would not.
    for error in [
        scalar_error(MagnusMidpoint, 0.2),
        scalar_error(MagnusGauss4, 0.2),
        scalar_error(MagnusGL4, 0.2),
        scalar_error(MagnusGL6, 0.2),
        scalar_error(MagnusNC6, 0.2),
        scalar_error(MagnusGL8, 0.2),
        scalar_error(MagnusNC8, 0.2),
    ] {
        assert!(error < 2.0e-12, "error={error:e}");
    }
}

#[test]
fn forward_and_backward_typed_solves_are_inverse_for_constant_generator() {
    let forward =
        LinearOperatorProblem::new(rotation_operator, vec![1.0, 0.0], (0.0, 1.0), ()).unwrap();
    let forward = solve_linear_operator(&forward, RKMK4, &fixed(0.1)).unwrap();
    let backward = LinearOperatorProblem::new(
        rotation_operator,
        forward.last_state().to_vec(),
        (1.0, 0.0),
        (),
    )
    .unwrap();
    let backward = solve_linear_operator(&backward, RKMK4, &fixed(0.1)).unwrap();
    assert!((backward.last_state()[0] - 1.0).abs() < 2.0e-12);
    assert!(backward.last_state()[1].abs() < 2.0e-12);
}

#[test]
fn adaptive_magnus_controls_error_on_a_noncommuting_time_dependent_system() {
    fn operator(output: &mut [f64], _: &[f64], _: &(), time: f64) {
        output.copy_from_slice(&[0.0, -(1.0 + time * time), 1.0 + 0.5 * time.sin(), 0.0]);
    }
    let problem = LinearOperatorProblem::new(operator, vec![1.0, 0.0], (0.0, 2.0), ()).unwrap();
    let loose = solve_linear_operator(
        &problem,
        MagnusAdapt4,
        &SolveOptions::new()
            .with_initial_step(0.5)
            .with_max_step(0.5)
            .with_tolerances(1.0e-5, 1.0e-5),
    )
    .unwrap();
    let tight = solve_linear_operator(
        &problem,
        MagnusAdapt4,
        &SolveOptions::new()
            .with_initial_step(0.5)
            .with_max_step(0.5)
            .with_tolerances(1.0e-10, 1.0e-10),
    )
    .unwrap();
    assert!(
        tight.stats().accepted_steps > loose.stats().accepted_steps,
        "loose={:?}, tight={:?}",
        loose.stats(),
        tight.stats()
    );
    assert!(tight.stats().rejected_steps > 0);
}

#[test]
fn cayley_conjugation_preserves_trace_and_determinant() {
    fn generator(output: &mut [f64], _: &[f64], _: &(), _: f64) {
        output.copy_from_slice(&[0.0, -1.0, 1.0, 0.0]);
    }
    let problem =
        LieGroupProblem::matrix(generator, vec![2.0, 0.5, 0.5, -1.0], 2, (0.0, 1.0), ()).unwrap();
    let solution = solve_lie_group(&problem, CayleyEuler, &fixed(0.1)).unwrap();
    let state = solution.last_state();
    assert!((state[0] + state[3] - 1.0).abs() < 2.0e-12);
    assert!((state[0] * state[3] - state[1] * state[2] + 2.25).abs() < 2.0e-12);
    assert_eq!(
        solution.stats().linear_factorizations,
        solution.stats().accepted_steps
    );
    assert_eq!(
        solution.stats().linear_solves,
        2 * solution.stats().accepted_steps
    );
}

#[test]
fn fixed_only_methods_reject_adaptive_mode_and_nonfinite_operators_fail() {
    let problem =
        LinearOperatorProblem::new(rotation_operator, vec![1.0, 0.0], (0.0, 1.0), ()).unwrap();
    assert_eq!(
        solve_linear_operator(&problem, MagnusGL4, &SolveOptions::new()).unwrap_err(),
        SolveError::AdaptiveStepUnsupported
    );

    fn invalid(output: &mut [f64], _: &[f64], _: &(), _: f64) {
        output.fill(f64::NAN);
    }
    let invalid = LinearOperatorProblem::new(invalid, vec![1.0], (0.0, 1.0), ()).unwrap();
    assert_eq!(
        solve_linear_operator(&invalid, LieEuler, &fixed(0.1)).unwrap_err(),
        SolveError::NonFiniteDerivative
    );
}

#[test]
fn ordinary_ode_path_uses_analytic_operator_and_shared_callbacks() {
    fn rhs(output: &mut [f64], state: &[f64], _: &(), _: f64) {
        output[0] = -state[0];
    }
    let problem = OdeProblem::new(rhs, vec![1.0], (0.0, 2.0), ())
        .with_jacobian(|jacobian, _, _, _| jacobian[0] = -1.0)
        .with_discrete_callback(
            |_, _, time| time >= 0.5,
            |_, _, _| CallbackAction::Terminate,
        );
    let solution = solve(&problem, MagnusMidpoint, &fixed(0.1)).unwrap();
    assert!((solution.times().last().unwrap() - 0.5).abs() < 1.0e-12);
    assert_eq!(solution.stats().callback_invocations, 1);
    assert!(solution.stats().jacobian_evaluations > 0);
}

#[test]
fn ordinary_path_honors_save_at_and_continuous_callback_lifecycle() {
    fn rhs(output: &mut [f64], state: &[f64], _: &(), _: f64) {
        output[0] = -state[0];
    }
    let saved_problem = OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
        .with_jacobian(|jacobian, _, _, _| jacobian[0] = -1.0);
    let saved = solve(
        &saved_problem,
        MagnusMidpoint,
        &fixed(0.2)
            .with_save_at([0.1, 0.4, 0.9])
            .with_dense_output(true),
    )
    .unwrap();
    assert_eq!(saved.times(), &[0.1, 0.4, 0.9]);
    assert!((saved.interpolate(0.35).unwrap()[0] - (-0.35_f64).exp()).abs() < 2.0e-5);

    let event_problem = OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
        .with_jacobian(|jacobian, _, _, _| jacobian[0] = -1.0)
        .with_continuous_callback(
            |_, _, time| time - 0.35,
            |_, _, _| CallbackAction::Terminate,
        );
    let event = solve(&event_problem, MagnusMidpoint, &fixed(0.2)).unwrap();
    assert!((event.times().last().unwrap() - 0.35).abs() < 1.0e-12);
    assert_eq!(event.stats().callback_invocations, 1);
}

#[test]
fn lie_vector_representation_and_matrix_representation_are_checked() {
    let vector =
        LieGroupProblem::vector(rotation_operator, vec![1.0, 0.0], (0.0, 1.0), ()).unwrap();
    let solution = solve_lie_group(&vector, CG3, &fixed(0.1)).unwrap();
    assert!((solution.last_state()[0] - 1.0_f64.cos()).abs() < 2.0e-12);
    assert_eq!(
        solve_lie_group(&vector, CayleyEuler, &fixed(0.1)).unwrap_err(),
        SolveError::InvalidTableau
    );
}

#[test]
fn typed_linear_and_matrix_group_paths_retain_real_dense_slopes() {
    let vector =
        LinearOperatorProblem::new(rotation_operator, vec![1.0, 0.0], (0.0, 1.0), ()).unwrap();
    let vector = solve_linear_operator(
        &vector,
        MagnusMidpoint,
        &fixed(0.05).with_dense_output(true),
    )
    .unwrap();
    let sample = vector.interpolate(0.375).unwrap();
    assert!((sample[0] - 0.375_f64.cos()).abs() < 2.0e-6);
    assert!((sample[1] - 0.375_f64.sin()).abs() < 2.0e-6);

    let matrix = LieGroupProblem::matrix(
        rotation_operator,
        vec![1.0, 0.0, 0.0, 0.0],
        2,
        (0.0, 1.0),
        (),
    )
    .unwrap();
    let matrix =
        solve_lie_group(&matrix, CayleyEuler, &fixed(0.05).with_dense_output(true)).unwrap();
    let sample = matrix.interpolate(0.375).unwrap();
    let cosine = 0.375_f64.cos();
    let sine = 0.375_f64.sin();
    assert!((sample[0] - cosine * cosine).abs() < 3.0e-4);
    assert!((sample[1] - cosine * sine).abs() < 3.0e-4);
    assert!((sample[3] - sine * sine).abs() < 3.0e-4);
}

fn autonomous_group_error<A: LinearOperatorAlgorithm>(algorithm: A, step: f64) -> f64 {
    fn operator(output: &mut [f64], state: &[f64], _: &(), _: f64) {
        output.copy_from_slice(&[0.0, -1.0, state[0].sin(), 0.0]);
    }
    let problem = LinearOperatorProblem::new(operator, vec![1.0, 0.0], (0.0, 1.0), ()).unwrap();
    let state = solve_linear_operator(&problem, algorithm, &fixed(step))
        .unwrap()
        .last_state()
        .to_vec();
    let exact = autonomous_reference();
    state
        .iter()
        .zip(exact)
        .map(|(actual, exact)| (actual - exact).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn autonomous_reference() -> [f64; 2] {
    fn rhs(state: [f64; 2]) -> [f64; 2] {
        [-state[1], state[0].sin() * state[0]]
    }
    let step = 1.0e-5;
    let mut state = [1.0, 0.0];
    for _ in 0..100_000 {
        let k1 = rhs(state);
        let k2 = rhs([state[0] + step * k1[0] / 2.0, state[1] + step * k1[1] / 2.0]);
        let k3 = rhs([state[0] + step * k2[0] / 2.0, state[1] + step * k2[1] / 2.0]);
        let k4 = rhs([state[0] + step * k3[0], state[1] + step * k3[1]]);
        for component in 0..2 {
            state[component] += step
                * (k1[component] + 2.0 * k2[component] + 2.0 * k3[component] + k4[component])
                / 6.0;
        }
    }
    state
}

#[test]
fn autonomous_lie_compositions_exhibit_expected_refinement() {
    macro_rules! assert_ratio {
        ($algorithm:expr, $threshold:expr) => {{
            let coarse = autonomous_group_error($algorithm, 0.05);
            let fine = autonomous_group_error($algorithm, 0.025);
            assert!(
                coarse / fine > $threshold,
                "{} ratio={}",
                stringify!($algorithm),
                coarse / fine
            );
        }};
    }
    assert_ratio!(LieEuler, 1.8);
    assert_ratio!(RKMK2, 3.5);
    assert_ratio!(CG2, 3.5);
    assert_ratio!(CG3, 6.5);
    assert_ratio!(RKMK4, 6.5);
    assert_ratio!(LieRK4, 12.0);
    assert_ratio!(CG4a, 6.5);
}

fn magnus_time_error<A: LinearOperatorAlgorithm>(algorithm: A, step: f64) -> f64 {
    fn operator(output: &mut [f64], _: &[f64], _: &(), time: f64) {
        let rate = time.sin();
        output.copy_from_slice(&[0.0, -rate, rate, 0.0]);
    }
    let problem = LinearOperatorProblem::new(operator, vec![1.0, 0.0], (0.0, 2.0), ()).unwrap();
    let state = solve_linear_operator(&problem, algorithm, &fixed(step))
        .unwrap()
        .last_state()
        .to_vec();
    let angle = 1.0 - 2.0_f64.cos();
    ((state[0] - angle.cos()).powi(2) + (state[1] - angle.sin()).powi(2)).sqrt()
}

#[test]
fn time_dependent_magnus_quadratures_exhibit_their_pinned_orders() {
    macro_rules! assert_ratio {
        ($algorithm:expr, $threshold:expr) => {{
            let coarse = magnus_time_error($algorithm, 0.5);
            let fine = magnus_time_error($algorithm, 0.25);
            assert!(
                coarse / fine > $threshold,
                "{} ratio={}, coarse={:e}, fine={:e}",
                stringify!($algorithm),
                coarse / fine,
                coarse,
                fine
            );
        }};
    }
    assert_ratio!(MagnusMidpoint, 3.5);
    assert_ratio!(MagnusLeapfrog, 3.5);
    assert_ratio!(MagnusAdapt4, 12.0);
    assert_ratio!(MagnusGauss4, 12.0);
    assert_ratio!(MagnusGL4, 12.0);
    assert_ratio!(MagnusGL6, 40.0);
    assert_ratio!(MagnusNC6, 40.0);
    assert_ratio!(MagnusGL8, 120.0);
    assert_ratio!(MagnusNC8, 120.0);
}
