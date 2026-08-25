use differential_equations::algorithms::multistep::*;
use differential_equations::*;

type SplitRhs = fn(&mut [f64], &[f64], &(), f64);
type TestSplitProblem = SplitOdeProblem<SplitRhs, SplitRhs, ()>;

fn linear_explicit(du: &mut [f64], u: &[f64], _: &(), time: f64) {
    du[0] = 0.5 * u[0] + time.sin();
}

fn autonomous_explicit(du: &mut [f64], u: &[f64], _: &(), _: f64) {
    du[0] = 0.5 * u[0];
}

fn linear_implicit(du: &mut [f64], u: &[f64], _: &(), _: f64) {
    du[0] = -2.0 * u[0];
}

fn fixed(step: f64) -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

fn linear_split(time_span: (f64, f64)) -> TestSplitProblem {
    SplitOdeProblem::new(
        linear_explicit as SplitRhs,
        linear_implicit as SplitRhs,
        vec![1.0],
        time_span,
        (),
    )
    .with_implicit_jacobian(|jacobian: &mut [f64], _: &[f64], _: &(), _: f64| {
        jacobian[0] = -2.0;
    })
}

fn autonomous_split(time_span: (f64, f64)) -> TestSplitProblem {
    SplitOdeProblem::new(
        autonomous_explicit as SplitRhs,
        linear_implicit as SplitRhs,
        vec![1.0],
        time_span,
        (),
    )
}

#[test]
fn configured_and_named_sbdf_aliases_are_identical() {
    let problem = linear_split((0.0, 1.0));
    let configured = solve_split(&problem, SBDF::new(2), &fixed(0.01)).unwrap();
    let named = solve_split(&problem, SBDF2, &fixed(0.01)).unwrap();
    assert_eq!(configured, named);

    let euler = solve_split(&problem, IMEXEuler, &fixed(0.01)).unwrap();
    let order_one = solve_split(&problem, SBDF::new(1), &fixed(0.01)).unwrap();
    assert_eq!(euler, order_one);
}

#[test]
fn imex_ark_staging_is_method_specific() {
    let problem = SplitOdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0] * u[0],
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -u[0],
        vec![0.5],
        (0.0, 0.1),
        (),
    );
    let ordinary = solve_split(&problem, IMEXEuler, &fixed(0.1)).unwrap();
    let ark = solve_split(&problem, IMEXEulerARK, &fixed(0.1)).unwrap();
    assert_ne!(ordinary.last_state(), ark.last_state());
}

#[test]
fn all_split_methods_handle_scalar_vector_and_nonautonomous_parts() {
    let exact =
        (-1.5_f64).exp() + (1.5 * 1.0_f64.sin() - 1.0_f64.cos()) / 3.25 + (-1.5_f64).exp() / 3.25;
    macro_rules! check {
        ($algorithm:expr, $tolerance:expr) => {{
            let solution =
                solve_split(&linear_split((0.0, 1.0)), $algorithm, &fixed(0.0025)).unwrap();
            assert!(solution.last_state()[0].is_finite());
            assert!(
                (solution.last_state()[0] - exact).abs() < $tolerance,
                "endpoint for {}: {}",
                stringify!($algorithm),
                solution.last_state()[0]
            );
            assert!((400..=401).contains(&solution.stats().accepted_steps));
            assert!(solution.stats().rhs_evaluations > 800);
            assert!(solution.stats().nonlinear_iterations >= 400);
            assert!(solution.stats().linear_solves > 0);
        }};
    }
    check!(IMEXEuler, 4.0e-3);
    check!(IMEXEulerARK, 4.0e-3);
    check!(SBDF2, 3.0e-5);
    check!(CNAB2, 3.0e-5);
    check!(CNLF2, 3.0e-5);

    // The pinned mutable SBDF cache advances directly from order one to
    // order three, leaving older startup entries zero-initialized. Preserve
    // those observable pinned endpoints for strict Julia parity.
    let sbdf3 = solve_split(&linear_split((0.0, 1.0)), SBDF3, &fixed(0.0025)).unwrap();
    let sbdf4 = solve_split(&linear_split((0.0, 1.0)), SBDF4, &fixed(0.0025)).unwrap();
    assert!((sbdf3.last_state()[0] - 0.438_414_405_130_520_5).abs() < 2.0e-12);
    assert!((sbdf4.last_state()[0] - 0.485_053_873_520_655_8).abs() < 2.0e-12);

    let vector = SplitOdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), time: f64| {
            du[0] = u[1] + time;
            du[1] = -0.25 * u[0] + time.cos();
        },
        |du: &mut [f64], u: &[f64], _: &(), _: f64| {
            du[0] = -3.0 * u[0];
            du[1] = -2.0 * u[1];
        },
        vec![0.3, -0.7],
        (0.0, 1.0),
        (),
    );
    let solution = solve_split(&vector, SBDF4, &fixed(0.005)).unwrap();
    assert_eq!(solution.dimension(), 2);
    assert!(solution.last_state().iter().all(|value| value.is_finite()));
}

#[test]
fn split_methods_integrate_forward_and_backward() {
    macro_rules! round_trip {
        ($algorithm:expr, $tolerance:expr) => {{
            let forward =
                solve_split(&autonomous_split((0.0, 1.0)), $algorithm, &fixed(0.0025)).unwrap();
            let backward_problem = SplitOdeProblem::new(
                |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = 0.5 * u[0],
                |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -2.0 * u[0],
                forward.last_state().to_vec(),
                (1.0, 0.0),
                (),
            );
            let backward = solve_split(&backward_problem, $algorithm, &fixed(0.0025)).unwrap();
            assert!(
                (backward.last_state()[0] - 1.0).abs() < $tolerance,
                "{} round-trip endpoint: {}",
                stringify!($algorithm),
                backward.last_state()[0]
            );
        }};
    }
    round_trip!(IMEXEuler, 1.1e-2);
    round_trip!(SBDF2, 2.0e-4);
    round_trip!(CNAB2, 8.0e-5);
    round_trip!(CNLF2, 9.0e-3);
}

#[test]
fn second_and_higher_order_imex_methods_converge() {
    let exact = (-1.5_f64).exp();
    macro_rules! ratio {
        ($algorithm:expr) => {{
            let error = |step| {
                (solve_split(&autonomous_split((0.0, 1.0)), $algorithm, &fixed(step))
                    .unwrap()
                    .last_state()[0]
                    - exact)
                    .abs()
            };
            let observed = error(0.02) / error(0.01);
            assert!(
                observed > 3.2,
                "{} ratio: {observed}",
                stringify!($algorithm)
            );
        }};
    }
    ratio!(SBDF2);
    ratio!(CNAB2);
    let cnlf_error = |step| {
        (solve_split(&autonomous_split((0.0, 1.0)), CNLF2, &fixed(step))
            .unwrap()
            .last_state()[0]
            - exact)
            .abs()
    };
    // The pinned implementation starts leapfrog with first-order IMEX Euler;
    // the two-step recurrence itself is second order, but startup dominates
    // this end-to-end refinement sequence.
    assert!(cnlf_error(0.02) / cnlf_error(0.01) > 1.7);

    // Orders three and four retain the pinned zero-cache startup behavior;
    // refinement still reduces their error, though startup dominates the
    // formal high-order stencil on this short interval.
    for algorithm in [SBDF::new(3), SBDF::new(4)] {
        let error = |step| {
            (solve_split(&autonomous_split((0.0, 1.0)), algorithm, &fixed(step))
                .unwrap()
                .last_state()[0]
                - exact)
                .abs()
        };
        assert!(error(0.01) < error(0.02));
    }
}

#[test]
fn split_failures_and_analytic_jacobian_stats_are_reported() {
    let problem = autonomous_split((0.0, 1.0));
    assert_eq!(
        solve_split(&problem, SBDF::new(5), &fixed(0.1)).unwrap_err(),
        SolveError::InvalidMultistepOrder
    );
    assert_eq!(
        solve_split(&problem, SBDF2, &SolveOptions::default()).unwrap_err(),
        SolveError::AdaptiveStepUnsupported
    );
    let no_step = SolveOptions {
        adaptive: false,
        ..SolveOptions::default()
    };
    assert_eq!(
        solve_split(&problem, SBDF2, &no_step).unwrap_err(),
        SolveError::InitialStepRequired
    );

    let singular = SplitOdeProblem::new(
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = 0.0,
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = 10.0 * u[0],
        vec![1.0],
        (0.0, 0.1),
        (),
    )
    .with_implicit_jacobian(|jacobian: &mut [f64], _: &[f64], _: &(), _: f64| {
        jacobian[0] = 10.0;
    });
    assert_eq!(
        solve_split(&singular, IMEXEuler, &fixed(0.1)).unwrap_err(),
        SolveError::SingularLinearSystem
    );

    let analytic = solve_split(&linear_split((0.0, 0.1)), SBDF2, &fixed(0.01)).unwrap();
    assert_eq!(analytic.stats().jacobian_evaluations, 10);
    assert_eq!(analytic.stats().linear_factorizations, 10);
    assert_eq!(analytic.stats().linear_solves, 10);
}

#[test]
fn imex_multistep_dense_output_uses_total_split_derivatives() {
    let problem = autonomous_split((0.0, 1.0));
    let dense = solve_split(&problem, SBDF2, &fixed(0.02).with_dense_output(true)).unwrap();
    let expected = (-1.5_f64 * 0.375).exp();
    let dense_sample = dense.interpolate(0.375).unwrap()[0];
    assert!(
        (dense_sample - expected).abs() < 1.0e-3,
        "dense={dense_sample} expected={expected}"
    );

    let saved = solve_split(&problem, SBDF2, &fixed(0.02).with_save_at([0.375, 1.0])).unwrap();
    assert_eq!(saved.times(), &[0.375, 1.0]);
    assert!((saved.state(0).unwrap()[0] - expected).abs() < 1.0e-3);
}

#[test]
fn vcabm_handles_scalar_vector_nonautonomous_and_reverse_solves() {
    let scalar = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), time: f64| du[0] = u[0] + time,
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions {
        absolute_tolerance: 1.0e-9,
        relative_tolerance: 1.0e-9,
        initial_step: Some(0.001),
        max_step: 0.05,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let forward = solve(&scalar, VCABM, &options).unwrap();
    assert!((forward.last_state()[0] - (2.0 * std::f64::consts::E - 2.0)).abs() < 2.0e-6);
    assert!(forward.stats().accepted_steps > 0);

    let reverse = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), time: f64| du[0] = u[0] + time,
        forward.last_state().to_vec(),
        (1.0, 0.0),
        (),
    );
    let backward = solve(&reverse, VCABM, &options).unwrap();
    assert!((backward.last_state()[0] - 1.0).abs() < 4.0e-6);

    let vector = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), time: f64| {
            du[0] = -0.4 * u[0] + time.sin();
            du[1] = u[0] - 0.2 * u[1] + time.cos();
        },
        vec![0.3, -0.7],
        (0.0, 2.0),
        (),
    );
    let result = solve(&vector, VCABM, &options).unwrap();
    assert_eq!(result.dimension(), 2);
    assert!(result.last_state().iter().all(|value| value.is_finite()));
}

#[test]
fn vcabm_fixed_step_sequence_converges() {
    let problem = || {
        OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
            vec![1.0],
            (0.0, 1.0),
            (),
        )
    };
    let error = |step| {
        (solve(&problem(), VCABM, &fixed(step)).unwrap().last_state()[0] - std::f64::consts::E)
            .abs()
    };
    assert!(error(0.04) / error(0.02) > 3.5);
}
