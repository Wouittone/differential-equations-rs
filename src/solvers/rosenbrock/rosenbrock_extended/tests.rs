use std::cell::Cell;
use std::rc::Rc;

use super::{
    Grk4a, Grk4t, Rodas3, Rodas3P, Rodas3d, Rodas4, Rodas4P, Rodas5P, Rodas5Pr, Rodas6P, Rodas23W,
    Ros2, Ros3, Ros3Pr, Ros3Prl, Ros3Prl2, Ros3p, Ros34Prw, Ros34Pw1a, Ros34Pw1b, Rosenbrock32,
    RosenbrockW6S4OS,
};
use crate::{CallbackAction, OdeProblem, SaveMode, SolveError, SolveOptions, solve};

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

#[test]
fn only_linearly_implicit_workspaces_allocate_linearization_storage() {
    let dimension = 17;
    for (tableau, linear_dimension) in [
        (None, dimension),
        (Some(Rodas5P.tableau().unwrap()), dimension),
        (Some(super::Tsit5DA.tableau().unwrap()), 0),
    ] {
        let workspace = super::workspace::Workspace::new(dimension, tableau, None);
        for buffer in [
            &workspace.perturbed_state,
            &workspace.perturbed_derivative,
            &workspace.time_derivative,
            &workspace.right_hand_side,
        ] {
            assert_eq!(buffer.len(), linear_dimension);
        }
        assert_eq!(workspace.pivots.len(), linear_dimension);
        assert_eq!(
            workspace.jacobian.len(),
            linear_dimension * linear_dimension
        );
        assert_eq!(
            workspace.factorization.len(),
            linear_dimension * linear_dimension
        );
        assert_eq!(workspace.current_derivative.len(), dimension);
        assert_eq!(workspace.stage_state.len(), dimension);
    }
}

fn stiff_problem(span: (f64, f64), initial: f64) -> OdeProblem<TestRhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), time: f64) {
        du[0] = -1000.0 * (u[0] - time.cos()) - time.sin();
    }
    OdeProblem::new(rhs as TestRhs, vec![initial], span, ())
}

fn adaptive_options() -> SolveOptions {
    SolveOptions {
        absolute_tolerance: 1.0e-8,
        relative_tolerance: 1.0e-8,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

#[test]
fn adaptive_methods_solve_a_stiff_nonautonomous_problem() {
    let endpoints = [
        solve(&stiff_problem((0.0, 1.0), 1.0), Ros2, &adaptive_options())
            .unwrap()
            .last_state()[0],
        solve(&stiff_problem((0.0, 1.0), 1.0), Rodas3, &adaptive_options())
            .unwrap()
            .last_state()[0],
        solve(
            &stiff_problem((0.0, 1.0), 1.0),
            Rodas3d,
            &adaptive_options(),
        )
        .unwrap()
        .last_state()[0],
        solve(&stiff_problem((0.0, 1.0), 1.0), Ros3, &adaptive_options())
            .unwrap()
            .last_state()[0],
        solve(&stiff_problem((0.0, 1.0), 1.0), Ros3Pr, &adaptive_options())
            .unwrap()
            .last_state()[0],
        solve(
            &stiff_problem((0.0, 1.0), 1.0),
            Ros3Prl,
            &adaptive_options(),
        )
        .unwrap()
        .last_state()[0],
        solve(
            &stiff_problem((0.0, 1.0), 1.0),
            Ros3Prl2,
            &adaptive_options(),
        )
        .unwrap()
        .last_state()[0],
        solve(&stiff_problem((0.0, 1.0), 1.0), Ros3p, &adaptive_options())
            .unwrap()
            .last_state()[0],
        solve(
            &stiff_problem((0.0, 1.0), 1.0),
            Ros34Prw,
            &adaptive_options(),
        )
        .unwrap()
        .last_state()[0],
        solve(
            &stiff_problem((0.0, 1.0), 1.0),
            Rosenbrock32,
            &adaptive_options(),
        )
        .unwrap()
        .last_state()[0],
        solve(&stiff_problem((0.0, 1.0), 1.0), Rodas4, &adaptive_options())
            .unwrap()
            .last_state()[0],
        solve(
            &stiff_problem((0.0, 1.0), 1.0),
            Rodas4P,
            &adaptive_options(),
        )
        .unwrap()
        .last_state()[0],
        solve(&stiff_problem((0.0, 1.0), 1.0), Grk4a, &adaptive_options())
            .unwrap()
            .last_state()[0],
        solve(&stiff_problem((0.0, 1.0), 1.0), Grk4t, &adaptive_options())
            .unwrap()
            .last_state()[0],
        solve(
            &stiff_problem((0.0, 1.0), 1.0),
            Ros34Pw1b,
            &adaptive_options(),
        )
        .unwrap()
        .last_state()[0],
        solve(
            &stiff_problem((0.0, 1.0), 1.0),
            Rodas5P,
            &adaptive_options(),
        )
        .unwrap()
        .last_state()[0],
        solve(
            &stiff_problem((0.0, 1.0), 1.0),
            Rodas6P,
            &adaptive_options(),
        )
        .unwrap()
        .last_state()[0],
        solve(
            &stiff_problem((0.0, 1.0), 1.0),
            Rodas23W,
            &adaptive_options(),
        )
        .unwrap()
        .last_state()[0],
    ];
    for endpoint in endpoints {
        assert!((endpoint - 1.0_f64.cos()).abs() < 2.0e-6);
    }
}

fn fixed_endpoint<A: crate::OdeAlgorithm>(algorithm: A, step: f64) -> f64 {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    solve(&problem, algorithm, &options).unwrap().last_state()[0]
}

fn convergence_ratio<A: crate::OdeAlgorithm + Copy>(algorithm: A, step: f64) -> f64 {
    let coarse = (fixed_endpoint(algorithm, step) - std::f64::consts::E).abs();
    let fine = (fixed_endpoint(algorithm, step / 2.0) - std::f64::consts::E).abs();
    coarse / fine
}

#[test]
fn ros34pw1a_uses_its_own_tableau() {
    let ros34pw1a = fixed_endpoint(Ros34Pw1a, 0.25);
    let rodas3p = fixed_endpoint(Rodas3P, 0.25);

    assert!(
        (ros34pw1a - rodas3p).abs() > 1.0e-8,
        "ROS34PW1a unexpectedly reproduced the Rodas3P step: {ros34pw1a:.17e}"
    );
    assert!(convergence_ratio(Ros34Pw1a, 0.1) > 7.0);
}

#[test]
fn ros34pw1a_controls_a_zero_embedded_error_with_step_doubling() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions {
        absolute_tolerance: 1.0e-9,
        relative_tolerance: 1.0e-9,
        initial_step: Some(1.0),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };

    let solution = solve(&problem, Ros34Pw1a, &options).unwrap();

    assert!(
        (solution.last_state()[0] - std::f64::consts::E).abs() < 2.0e-7,
        "endpoint={:.17e}",
        solution.last_state()[0]
    );
    assert!(solution.stats().rejected_steps > 0);
}

#[test]
fn methods_have_their_expected_fixed_step_orders() {
    let ratios = [
        convergence_ratio(Ros2, 0.1),
        convergence_ratio(Rosenbrock32, 0.1),
        convergence_ratio(Rodas3, 0.1),
        convergence_ratio(Rodas3d, 0.1),
        convergence_ratio(Ros3Pr, 0.1),
        convergence_ratio(Ros3Prl, 0.1),
        convergence_ratio(Ros3Prl2, 0.1),
        convergence_ratio(Ros3p, 0.1),
        convergence_ratio(Ros34Prw, 0.1),
        convergence_ratio(Rodas4, 0.1),
        convergence_ratio(Rodas4P, 0.1),
        convergence_ratio(Grk4a, 0.1),
        convergence_ratio(Grk4t, 0.1),
        convergence_ratio(Ros34Pw1b, 0.1),
        convergence_ratio(Rodas5P, 0.2),
        convergence_ratio(Rodas6P, 0.2),
        convergence_ratio(RosenbrockW6S4OS, 0.1),
        convergence_ratio(Rodas23W, 0.1),
    ];
    assert!(ratios[0] > 3.0);
    assert!(ratios[1] > 7.0);
    assert!(ratios[2] > 7.0);
    // Rodas3d is fourth order on this linear problem because its damping
    // parameter is a root of the fourth-order linear order condition.
    assert!(ratios[3] > 14.0);
    assert!(ratios[4] > 7.0);
    assert!(ratios[5] > 7.0);
    assert!(ratios[6] > 7.0);
    assert!(ratios[7] > 7.0);
    assert!(ratios[8] > 7.0);
    assert!(ratios[9] > 14.0);
    assert!(ratios[10] > 14.0);
    assert!(ratios[11] > 14.0);
    assert!(ratios[12] > 14.0);
    assert!(ratios[13] > 7.0);
    assert!(ratios[14] > 14.0);
    assert!(ratios[15] > 14.0);
    assert!(ratios[16] > 7.0);
    // Pinned Rodas23W uses a second-order primary solution.
    assert!(ratios[17] > 3.0 && ratios[17] < 5.5);
}

#[test]
fn rodas5pr_matches_rodas5p_on_regular_ode_paths() {
    let fixed_options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.1),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let p = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let rodas5p = solve(&p, Rodas5P, &fixed_options).unwrap();
    let rodas5pr = solve(&p, Rodas5Pr, &fixed_options).unwrap();
    assert!((rodas5p.last_state()[0] - rodas5pr.last_state()[0]).abs() < 1.0e-14);

    let adaptive = adaptive_options();
    let rodas5pr = solve(&stiff_problem((0.0, 1.0), 1.0), Rodas5Pr, &adaptive).unwrap();
    assert!((rodas5pr.last_state()[0] - 1.0_f64.cos()).abs() < 2.0e-6);
    assert!(rodas5pr.stats().rhs_evaluations > 0);
}

#[test]
fn rodas23w_supports_jacobian_backward_callbacks_and_save_at() {
    let jacobian_calls = Rc::new(Cell::new(0));
    let jacobian_calls_for_problem = Rc::clone(&jacobian_calls);
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -2.0 * u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    )
    .with_jacobian(move |jacobian: &mut [f64], _: &[f64], _: &(), _: f64| {
        jacobian_calls_for_problem.set(jacobian_calls_for_problem.get() + 1);
        jacobian[0] = -2.0;
    })
    .with_discrete_callback(
        |_, _, time| time == 0.5,
        |state, _, _| {
            state[0] += 0.25;
            CallbackAction::Continue
        },
    );
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.25),
        save: SaveMode::Endpoints,
        save_at: vec![0.25, 0.5, 0.75],
        ..SolveOptions::default()
    };
    let solution = solve(&problem, Rodas23W, &options).unwrap();
    assert_eq!(solution.stats().callback_invocations, 1);
    assert!(solution.stats().jacobian_evaluations > 0);
    assert!(jacobian_calls.get() > 0);
    for time in options.save_at {
        assert!(solution.times().contains(&time), "missing save_at={time}");
    }

    let backward_problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -2.0 * u[0],
        vec![(-2.0_f64).exp()],
        (1.0, 0.0),
        (),
    );
    let backward_options = SolveOptions {
        initial_step: Some(0.01),
        max_step: 0.01,
        save: SaveMode::Endpoints,
        ..adaptive_options()
    };
    let endpoint = solve(&backward_problem, Rodas23W, &backward_options)
        .unwrap()
        .last_state()[0];
    assert!((endpoint - 1.0).abs() < 1.0e-5, "endpoint={endpoint:.17e}");
}

#[test]
fn w6s4os_is_fixed_step_only_and_supports_backward_integration() {
    let adaptive_error = solve(
        &stiff_problem((0.0, 1.0), 1.0),
        RosenbrockW6S4OS,
        &adaptive_options(),
    )
    .expect_err("RosenbrockW6S4OS must reject adaptive scheduling");
    assert_eq!(adaptive_error, SolveError::AdaptiveStepUnsupported);

    let backward_problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -2.0 * u[0],
        vec![(-2.0_f64).exp()],
        (1.0, 0.0),
        (),
    );
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.05),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let endpoint = solve(&backward_problem, RosenbrockW6S4OS, &options)
        .unwrap()
        .last_state()[0];
    assert!((endpoint - 1.0).abs() < 5.0e-7, "endpoint={endpoint:.17e}");
}

#[test]
fn stiff_methods_preserve_callbacks_and_requested_samples() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    )
    .with_discrete_callback(
        |_, _, time| time == 0.5,
        |state, _, _| {
            state[0] += 0.25;
            CallbackAction::Continue
        },
    );
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.25),
        save: SaveMode::Endpoints,
        save_at: vec![0.25, 0.5, 0.75],
        ..SolveOptions::default()
    };
    let solution = solve(&problem, RosenbrockW6S4OS, &options).unwrap();
    assert_eq!(solution.stats().callback_invocations, 1);
    for &time in &options.save_at {
        assert!(solution.times().contains(&time), "missing save_at={time}");
    }
    assert!(solution.last_state()[0] > 0.0);

    let rodas4p_solution = solve(&problem, Rodas4P, &options).unwrap();
    assert_eq!(rodas4p_solution.stats().callback_invocations, 1);
    assert!(rodas4p_solution.stats().jacobian_evaluations > 0);
    for &time in &options.save_at {
        assert!(
            rodas4p_solution.times().contains(&time),
            "missing Rodas4P save_at={time}"
        );
    }
    assert!(rodas4p_solution.last_state()[0] > 0.0);

    let rodas6p_solution = solve(&problem, Rodas6P, &options).unwrap();
    assert_eq!(rodas6p_solution.stats().callback_invocations, 1);
    assert!(rodas6p_solution.stats().jacobian_evaluations > 0);
    for &time in &options.save_at {
        assert!(
            rodas6p_solution.times().contains(&time),
            "missing Rodas6P save_at={time}"
        );
    }
    assert!(rodas6p_solution.last_state()[0] > 0.0);

    let grk4t_options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.25),
        save: SaveMode::Endpoints,
        save_at: vec![0.25, 0.5, 0.75],
        ..SolveOptions::default()
    };
    let grk4t_solution = solve(&problem, Grk4t, &grk4t_options).unwrap();
    assert_eq!(grk4t_solution.stats().callback_invocations, 1);
    for time in grk4t_options.save_at {
        assert!(
            grk4t_solution.times().contains(&time),
            "missing GRK4T save_at={time}"
        );
    }
    assert!(grk4t_solution.last_state()[0] > 0.0);
}

#[test]
fn ros34pw1b_supports_jacobian_callbacks_and_requested_samples() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    )
    .with_jacobian(|jacobian: &mut [f64], _: &[f64], _: &(), _: f64| {
        jacobian[0] = -1.0;
    })
    .with_discrete_callback(
        |_, _, time| time == 0.5,
        |state, _, _| {
            state[0] += 0.25;
            CallbackAction::Continue
        },
    );
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.25),
        save: SaveMode::Endpoints,
        save_at: vec![0.25, 0.5, 0.75],
        ..SolveOptions::default()
    };
    let solution = solve(&problem, Ros34Pw1b, &options).unwrap();
    assert_eq!(solution.stats().callback_invocations, 1);
    assert!(solution.stats().jacobian_evaluations > 0);
    for time in options.save_at {
        assert!(solution.times().contains(&time), "missing save_at={time}");
    }
    assert!(solution.last_state()[0] > 0.0);
}

#[test]
fn methods_support_backward_integration() {
    let backward_problem = || {
        OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -2.0 * u[0],
            vec![(-2.0_f64).exp()],
            (1.0, 0.0),
            (),
        )
    };
    let ros3p_options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.01),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let ros3p_endpoint = solve(&backward_problem(), Ros3p, &ros3p_options)
        .unwrap()
        .last_state()[0];
    assert!((ros3p_endpoint - 1.0).abs() < 3.0e-6);

    for (name, endpoint) in [
        (
            "ros2",
            solve(&backward_problem(), Ros2, &adaptive_options())
                .unwrap()
                .last_state()[0],
        ),
        (
            "rodas3",
            solve(&backward_problem(), Rodas3, &adaptive_options())
                .unwrap()
                .last_state()[0],
        ),
        (
            "rodas3d",
            solve(&backward_problem(), Rodas3d, &adaptive_options())
                .unwrap()
                .last_state()[0],
        ),
        (
            "rosenbrock32",
            solve(&backward_problem(), Rosenbrock32, &adaptive_options())
                .unwrap()
                .last_state()[0],
        ),
        (
            "rodas4",
            solve(&backward_problem(), Rodas4, &adaptive_options())
                .unwrap()
                .last_state()[0],
        ),
        (
            "rodas4p",
            solve(&backward_problem(), Rodas4P, &adaptive_options())
                .unwrap()
                .last_state()[0],
        ),
        (
            "rodas5p",
            solve(&backward_problem(), Rodas5P, &adaptive_options())
                .unwrap()
                .last_state()[0],
        ),
        (
            "rodas6p",
            solve(&backward_problem(), Rodas6P, &adaptive_options())
                .unwrap()
                .last_state()[0],
        ),
        (
            "grk4a",
            solve(&backward_problem(), Grk4a, &adaptive_options())
                .unwrap()
                .last_state()[0],
        ),
        (
            "ros34pw1b",
            solve(
                &backward_problem(),
                Ros34Pw1b,
                &SolveOptions {
                    initial_step: Some(0.005),
                    max_step: 0.005,
                    ..adaptive_options()
                },
            )
            .unwrap()
            .last_state()[0],
        ),
        (
            "grk4t",
            solve(&backward_problem(), Grk4t, &adaptive_options())
                .unwrap()
                .last_state()[0],
        ),
        (
            "ros34prw",
            solve(&backward_problem(), Ros34Prw, &adaptive_options())
                .unwrap()
                .last_state()[0],
        ),
    ] {
        assert!(
            (endpoint - 1.0).abs() < 3.0e-7,
            "{name}: endpoint={endpoint:.17e}"
        );
    }
}

#[test]
fn ros3pr_supports_fixed_step_backward_integration() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -2.0 * u[0],
        vec![(-2.0_f64).exp()],
        (1.0, 0.0),
        (),
    );
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.01),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let solution = solve(&problem, Ros3Pr, &options).unwrap();
    assert!((solution.last_state()[0] - 1.0).abs() < 3.0e-6);
}

#[test]
fn ros3prl_covers_regular_ode_lifecycle() {
    let jacobian_calls = Rc::new(Cell::new(0));
    let jacobian_calls_for_problem = Rc::clone(&jacobian_calls);
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    )
    .with_jacobian(move |jacobian: &mut [f64], _: &[f64], _: &(), _: f64| {
        jacobian_calls_for_problem.set(jacobian_calls_for_problem.get() + 1);
        jacobian[0] = -1.0;
    })
    .with_discrete_callback(
        |_, _, time| time == 0.5,
        |state, _, _| {
            state[0] += 0.25;
            CallbackAction::Continue
        },
    );
    let fixed_options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.25),
        save: SaveMode::Endpoints,
        save_at: vec![0.25, 0.5, 0.75],
        ..SolveOptions::default()
    };
    let fixed = solve(&problem, Ros3Prl, &fixed_options).unwrap();
    assert_eq!(fixed.stats().callback_invocations, 1);
    assert!(fixed.stats().jacobian_evaluations > 0);
    assert!(jacobian_calls.get() > 0);
    for &time in &fixed_options.save_at {
        assert!(fixed.times().contains(&time), "missing save_at={time}");
    }
    assert!(fixed.last_state()[0] > 0.0);

    let adaptive = solve(
        &stiff_problem((0.0, 1.0), 1.0),
        Ros3Prl,
        &adaptive_options(),
    )
    .unwrap();
    assert!((adaptive.last_state()[0] - 1.0_f64.cos()).abs() < 2.0e-6);

    let backward_problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -2.0 * u[0],
        vec![(-2.0_f64).exp()],
        (1.0, 0.0),
        (),
    );
    let backward_options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.01),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let backward = solve(&backward_problem, Ros3Prl, &backward_options)
        .unwrap()
        .last_state()[0];
    assert!((backward - 1.0).abs() < 3.0e-6);
}

#[test]
fn ros3prl2_covers_regular_ode_lifecycle() {
    let jacobian_calls = Rc::new(Cell::new(0));
    let jacobian_calls_for_problem = Rc::clone(&jacobian_calls);
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    )
    .with_jacobian(move |jacobian: &mut [f64], _: &[f64], _: &(), _: f64| {
        jacobian_calls_for_problem.set(jacobian_calls_for_problem.get() + 1);
        jacobian[0] = 1.0;
    });
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.01),
        save: SaveMode::Endpoints,
        save_at: vec![0.25, 0.5, 0.75],
        ..SolveOptions::default()
    };
    let solution = solve(&problem, Ros3Prl2, &options).unwrap();
    assert!(jacobian_calls.get() > 0);
    for &time in &options.save_at {
        assert!(solution.times().contains(&time), "missing save_at={time}");
    }
    assert!((solution.last_state()[0] - 0.75_f64.exp()).abs() < 3.0e-6);

    let backward_problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -2.0 * u[0],
        vec![(-2.0_f64).exp()],
        (1.0, 0.0),
        (),
    );
    let backward_options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.01),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let backward = solve(&backward_problem, Ros3Prl2, &backward_options)
        .unwrap()
        .last_state()[0];
    assert!((backward - 1.0).abs() < 3.0e-6);
}

#[test]
fn analytic_jacobian_reduces_rhs_work() {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), time: f64) {
        du[0] = -1000.0 * (u[0] - time.cos()) - time.sin();
    }
    type Rhs = fn(&mut [f64], &[f64], &(), f64);
    let numeric = OdeProblem::new(rhs as Rhs, vec![1.0], (0.0, 0.2), ());
    let analytic = OdeProblem::new(rhs as Rhs, vec![1.0], (0.0, 0.2), ())
        .with_jacobian(|jacobian: &mut [f64], _: &[f64], _: &(), _: f64| jacobian[0] = -1000.0);
    let numeric = solve(&numeric, Rodas4, &adaptive_options()).unwrap();
    let analytic = solve(&analytic, Rodas4, &adaptive_options()).unwrap();
    assert!((numeric.last_state()[0] - analytic.last_state()[0]).abs() < 2.0e-10);
    assert!(analytic.stats().rhs_evaluations < numeric.stats().rhs_evaluations);

    let numeric = solve(
        &OdeProblem::new(rhs as Rhs, vec![1.0], (0.0, 0.2), ()),
        Rodas3d,
        &adaptive_options(),
    )
    .unwrap();
    let analytic = solve(
        &OdeProblem::new(rhs as Rhs, vec![1.0], (0.0, 0.2), ())
            .with_jacobian(|jacobian: &mut [f64], _: &[f64], _: &(), _: f64| jacobian[0] = -1000.0),
        Rodas3d,
        &adaptive_options(),
    )
    .unwrap();
    assert!((numeric.last_state()[0] - analytic.last_state()[0]).abs() < 2.0e-10);
    assert!(analytic.stats().rhs_evaluations < numeric.stats().rhs_evaluations);

    let numeric = solve(
        &OdeProblem::new(rhs as Rhs, vec![1.0], (0.0, 0.2), ()),
        Ros34Prw,
        &adaptive_options(),
    )
    .unwrap();
    let analytic = solve(
        &OdeProblem::new(rhs as Rhs, vec![1.0], (0.0, 0.2), ())
            .with_jacobian(|jacobian: &mut [f64], _: &[f64], _: &(), _: f64| jacobian[0] = -1000.0),
        Ros34Prw,
        &adaptive_options(),
    )
    .unwrap();
    assert!((numeric.last_state()[0] - analytic.last_state()[0]).abs() < 2.0e-10);
    assert!(analytic.stats().rhs_evaluations < numeric.stats().rhs_evaluations);

    let numeric = solve(
        &OdeProblem::new(rhs as Rhs, vec![1.0], (0.0, 0.2), ()),
        Grk4a,
        &adaptive_options(),
    )
    .unwrap();
    let analytic = solve(
        &OdeProblem::new(rhs as Rhs, vec![1.0], (0.0, 0.2), ())
            .with_jacobian(|jacobian: &mut [f64], _: &[f64], _: &(), _: f64| jacobian[0] = -1000.0),
        Grk4a,
        &adaptive_options(),
    )
    .unwrap();
    assert!((numeric.last_state()[0] - analytic.last_state()[0]).abs() < 2.0e-10);
    assert!(analytic.stats().rhs_evaluations < numeric.stats().rhs_evaluations);

    let numeric = solve(
        &OdeProblem::new(rhs as Rhs, vec![1.0], (0.0, 0.2), ()),
        Ros3Pr,
        &adaptive_options(),
    )
    .unwrap();
    let analytic = solve(
        &OdeProblem::new(rhs as Rhs, vec![1.0], (0.0, 0.2), ())
            .with_jacobian(|jacobian: &mut [f64], _: &[f64], _: &(), _: f64| jacobian[0] = -1000.0),
        Ros3Pr,
        &adaptive_options(),
    )
    .unwrap();
    assert!((numeric.last_state()[0] - analytic.last_state()[0]).abs() < 2.0e-10);
    assert!(analytic.stats().rhs_evaluations < numeric.stats().rhs_evaluations);

    let numeric = solve(
        &OdeProblem::new(rhs as Rhs, vec![1.0], (0.0, 0.2), ()),
        Rodas3,
        &adaptive_options(),
    )
    .unwrap();
    let analytic = solve(
        &OdeProblem::new(rhs as Rhs, vec![1.0], (0.0, 0.2), ())
            .with_jacobian(|jacobian: &mut [f64], _: &[f64], _: &(), _: f64| jacobian[0] = -1000.0),
        Rodas3,
        &adaptive_options(),
    )
    .unwrap();
    assert!((numeric.last_state()[0] - analytic.last_state()[0]).abs() < 2.0e-10);
    assert!(analytic.stats().rhs_evaluations < numeric.stats().rhs_evaluations);

    let numeric = solve(
        &OdeProblem::new(rhs as Rhs, vec![1.0], (0.0, 0.2), ()),
        Ros2,
        &adaptive_options(),
    )
    .unwrap();
    let analytic = solve(
        &OdeProblem::new(rhs as Rhs, vec![1.0], (0.0, 0.2), ())
            .with_jacobian(|jacobian: &mut [f64], _: &[f64], _: &(), _: f64| jacobian[0] = -1000.0),
        Ros2,
        &adaptive_options(),
    )
    .unwrap();
    assert!((numeric.last_state()[0] - analytic.last_state()[0]).abs() < 2.0e-10);
    assert!(analytic.stats().rhs_evaluations < numeric.stats().rhs_evaluations);
}

#[test]
fn callbacks_invalidate_stiff_step_caches_and_save_at_is_honored() {
    let problem = stiff_problem((0.0, 1.0), 1.0).with_continuous_callback(
        |_, _, time| time - 0.5,
        |state, _, _| {
            state[0] += 0.01;
            CallbackAction::Continue
        },
    );
    let options = SolveOptions {
        save: SaveMode::Endpoints,
        save_at: vec![0.25, 0.5, 0.75],
        ..adaptive_options()
    };
    let solution = solve(&problem, Rodas4, &options).unwrap();
    assert!(solution.stats().callback_invocations > 0);
    assert!(solution.times().contains(&0.25));
    assert!(solution.times().contains(&0.5));
    assert!(solution.times().contains(&0.75));

    let grk4a_solution = solve(&problem, Grk4a, &options).unwrap();
    assert!(grk4a_solution.stats().callback_invocations > 0);
    assert!(grk4a_solution.times().contains(&0.25));
    assert!(grk4a_solution.times().contains(&0.5));
    assert!(grk4a_solution.times().contains(&0.75));

    let rodas3_solution = solve(&problem, Rodas3, &options).unwrap();
    assert!(rodas3_solution.stats().callback_invocations > 0);
    assert!(rodas3_solution.times().contains(&0.25));
    assert!(rodas3_solution.times().contains(&0.5));
    assert!(rodas3_solution.times().contains(&0.75));

    let rodas3d_solution = solve(&problem, Rodas3d, &options).unwrap();
    assert!(rodas3d_solution.stats().callback_invocations > 0);
    assert!(rodas3d_solution.times().contains(&0.25));
    assert!(rodas3d_solution.times().contains(&0.5));
    assert!(rodas3d_solution.times().contains(&0.75));

    let ros2_solution = solve(&problem, Ros2, &options).unwrap();
    assert!(ros2_solution.stats().callback_invocations > 0);
    assert!(ros2_solution.times().contains(&0.25));
    assert!(ros2_solution.times().contains(&0.5));
    assert!(ros2_solution.times().contains(&0.75));

    let ros3pr_solution = solve(&problem, Ros3Pr, &options).unwrap();
    assert!(ros3pr_solution.stats().callback_invocations > 0);
    assert!(ros3pr_solution.times().contains(&0.25));
    assert!(ros3pr_solution.times().contains(&0.5));
    assert!(ros3pr_solution.times().contains(&0.75));
    let ros3p_solution = solve(&problem, Ros3p, &options).unwrap();
    assert!(ros3p_solution.stats().callback_invocations > 0);
    assert!(ros3p_solution.times().contains(&0.25));
    assert!(ros3p_solution.times().contains(&0.5));
    assert!(ros3p_solution.times().contains(&0.75));
    let ros34prw_solution = solve(&problem, Ros34Prw, &options).unwrap();
    assert!(ros34prw_solution.stats().callback_invocations > 0);
    assert!(ros34prw_solution.times().contains(&0.25));
    assert!(ros34prw_solution.times().contains(&0.5));
    assert!(ros34prw_solution.times().contains(&0.75));
}

#[test]
fn reuses_differentiation_after_rejected_rodas_steps() {
    let options = SolveOptions {
        initial_step: Some(1.0),
        ..adaptive_options()
    };

    let solution = solve(&stiff_problem((0.0, 1.0), 1.0), Rodas4, &options).unwrap();
    let stats = solution.stats();

    assert!(stats.rejected_steps > 0);
    assert!(stats.jacobian_evaluations < stats.accepted_steps + stats.rejected_steps);
}

#[test]
fn callback_effect_is_seen_by_the_next_jacobian() {
    let saw_effect_state = Rc::new(Cell::new(false));
    let jacobian_saw_effect = Rc::clone(&saw_effect_state);
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -u[0],
        vec![1.0],
        (0.0, 0.5),
        (),
    )
    .with_jacobian(move |jacobian: &mut [f64], state: &[f64], _: &(), _: f64| {
        if state[0] == 3.0 {
            jacobian_saw_effect.set(true);
        }
        jacobian[0] = -1.0;
    })
    .with_discrete_callback(
        |_, _, time| time == 0.25,
        |state, _, _| {
            state[0] = 3.0;
            CallbackAction::Continue
        },
    );
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.25),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };

    solve(&problem, Rodas4, &options).unwrap();

    assert!(saw_effect_state.get());
}

#[test]
fn terminating_callback_skips_post_effect_rodas_work() {
    let rhs_calls = Rc::new(Cell::new(0));
    let observed_calls = Rc::clone(&rhs_calls);
    let problem = OdeProblem::new(
        move |du: &mut [f64], u: &[f64], _: &(), _: f64| {
            observed_calls.set(observed_calls.get() + 1);
            du[0] = if u[0] == 12_345.0 { f64::NAN } else { -u[0] };
        },
        vec![1.0],
        (0.0, 1.0),
        (),
    )
    .with_discrete_callback(
        |_, _, time| time > 0.0,
        |state, _, _| {
            state[0] = 12_345.0;
            CallbackAction::Terminate
        },
    );
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.25),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };

    let solution = solve(&problem, Rodas4, &options).unwrap();

    assert_eq!(solution.last_state()[0], 12_345.0);
    assert_eq!(rhs_calls.get(), solution.stats().rhs_evaluations);
    assert_eq!(solution.stats().accepted_steps, 1);
    assert_eq!(solution.stats().jacobian_evaluations, 1);
}
