use differential_equations::callbacks::{GeneralDomain, PositiveDomain};
use differential_equations::ndarray::{ArrayViewD, ArrayViewMutD, arr0, array};
use differential_equations::solvers::explicit::split_euler::{SplitEuler, solve_split};
use differential_equations::solvers::explicit::{Euler, Tsit5};
use differential_equations::solvers::multirate::MRIGARKERK22a;
use differential_equations::solvers::multistep::IMEXEuler;
use differential_equations::solvers::stabilized::IRKC;
use differential_equations::{
    CallbackAction, CallbackSave, CallbackSet, ConfigurationError, OdeProblem, SaveMode,
    SolveError, SolveOptions, SplitOdeProblem, solve,
};

fn fixed() -> SolveOptions {
    SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(0.1)
        .with_save(SaveMode::EveryStep)
}

fn zero(du: &mut [f64], _: &[f64], _: &(), _: f64) {
    du.fill(0.0);
}

fn oscillator(du: &mut [f64], u: &[f64], _: &(), _: f64) {
    du[0] = u[1];
    du[1] = -u[0];
}

fn circle(r: &mut [f64], u: &[f64], _: &(), _: f64) {
    r[0] = u.iter().map(|value| value * value).sum::<f64>() - 1.0;
}

fn circle_jacobian(j: &mut [f64], u: &[f64], _: &(), _: f64) {
    for (j, u) in j.iter_mut().zip(u) {
        *j = 2.0 * u;
    }
}

fn circle_policy(analytic: bool) -> GeneralDomain<()> {
    let policy = GeneralDomain::new(1, circle)
        .with_absolute_tolerance(1.0e-3)
        .with_save(CallbackSave::None);
    if analytic {
        policy.with_jacobian(circle_jacobian)
    } else {
        policy
    }
}

#[test]
fn prediction_shrinks_before_attempts_and_projection_preserves_the_constraint() {
    for analytic in [false, true] {
        for span in [(0.0, 1.0), (1.0, 0.0)] {
            let problem = OdeProblem::new(oscillator, [1.0, 0.0], span, ())
                .with_callback_set(circle_policy(analytic).into_callback_set().unwrap());
            let solution = solve(&problem, Euler, &fixed()).unwrap();
            let first_step = (solution.times()[1] - span.0).abs();
            // 0.1 -> 0.05 -> 0.025, then the 0.9 prediction safety factor.
            assert!((first_step - 0.0225).abs() < 1.0e-14);
            assert_eq!(solution.stats().rejected_steps, 0);
            assert_eq!(*solution.times().last().unwrap(), span.1);
            for u in solution.values().chunks_exact(2) {
                assert!((u[0] * u[0] + u[1] * u[1] - 1.0).abs() < 3.0e-15);
            }
            let adaptive = solve(&problem, Tsit5, &fixed().with_adaptive(true)).unwrap();
            for u in adaptive.values().chunks_exact(2) {
                assert!((u[0] * u[0] + u[1] * u[1] - 1.0).abs() < 3.0e-15);
            }
        }
    }
}

#[test]
fn predictor_uses_future_time_parameters_and_signed_residuals() {
    for sign in [-1.0, 1.0] {
        for span in [(0.0_f64, 1.0_f64), (1.0, 0.0)] {
            let policy =
                GeneralDomain::new(1, move |r: &mut [f64], u: &[f64], scale: &f64, t: f64| {
                    r[0] = sign * (scale * (1.0 + t).powi(2) - u[0]);
                })
                .with_jacobian(move |j: &mut [f64], _: &[f64], _: &f64, _| j[0] = -sign)
                .with_absolute_tolerance(2.0e-3)
                .with_save(CallbackSave::None)
                .into_callback_set()
                .unwrap();
            let problem = OdeProblem::new(
                |du: &mut [f64], _: &[f64], scale: &f64, t: f64| {
                    du[0] = 2.0 * scale * (1.0 + t);
                },
                [2.0 * (1.0 + span.0).powi(2)],
                span,
                2.0,
            )
            .with_callback_set(policy);
            let solution = solve(&problem, Euler, &fixed()).unwrap();
            let expected_step = if sign > 0.0 { 0.0225 } else { 0.1 };
            assert!(((solution.times()[1] - span.0).abs() - expected_step).abs() < 1.0e-14);
            for (&time, &state) in solution.times().iter().zip(solution.values()) {
                assert!((state - 2.0 * (1.0 + time).powi(2)).abs() < 3.0e-15);
            }
        }
    }
}

#[test]
fn domain_tolerance_defaults_to_solver_and_can_be_overridden() {
    let run = |tolerance: Option<f64>, reduction| {
        let policy = GeneralDomain::new(1, circle)
            .with_reduction_factor(reduction)
            .with_save(CallbackSave::None);
        let policy = match tolerance {
            Some(tolerance) => policy.with_absolute_tolerance(tolerance),
            None => policy,
        };
        let problem = OdeProblem::new(oscillator, [1.0, 0.0], (0.0, 0.1), ())
            .with_callback_set(policy.into_callback_set().unwrap());
        solve(&problem, Euler, &fixed().with_tolerances(1.0e-3, 1.0e-3)).unwrap()
    };
    assert!((run(None, 0.5).times()[1] - 0.0225).abs() < 1.0e-14);
    assert!((run(Some(0.1), 0.5).times()[1] - 0.1).abs() < 1.0e-14);
    assert!((run(Some(1.0e-3), 0.2).times()[1] - 0.018).abs() < 1.0e-14);
}

#[test]
fn region_projection_handles_inactive_constraints() {
    for analytic in [false, true] {
        let policy = GeneralDomain::new(2, |r: &mut [f64], u: &[f64], _: &(), _| {
            for (r, u) in r.iter_mut().zip(u) {
                *r = (-u).max(0.0);
            }
        });
        let policy = if analytic {
            policy.with_jacobian(|j: &mut [f64], u: &[f64], _: &(), _| {
                j.fill(0.0);
                j[0] = if u[0] < 0.0 { -1.0 } else { 0.0 };
                j[3] = if u[1] < 0.0 { -1.0 } else { 0.0 };
            })
        } else {
            policy
        };
        let problem = OdeProblem::new(zero, [-0.5, 1.0], (0.0, 0.1), ())
            .with_callback_set(policy.into_callback_set().unwrap());
        let solution = solve(&problem, Euler, &fixed()).unwrap();
        assert!(solution.last_state()[0].abs() < 3.0e-15);
        assert_eq!(solution.last_state()[1], 1.0);
    }
}

#[test]
fn region_constraints_can_become_active_or_inactive_between_steps() {
    let displace = CallbackSet::new().with_discrete_callback_saving(
        CallbackSave::None,
        |_, _, _| true,
        |u, _, t| {
            let state = if t < 0.05 {
                [-0.5, 1.0]
            } else if t < 0.15 {
                [1.0, -0.5]
            } else if t < 0.25 {
                [-0.5, -1.0]
            } else {
                [0.5, 1.0]
            };
            u.copy_from_slice(&state);
            CallbackAction::Continue
        },
    );
    let domain = GeneralDomain::new(2, |r: &mut [f64], u: &[f64], _: &(), _| {
        for (r, u) in r.iter_mut().zip(u) {
            *r = (-u).max(0.0);
        }
    })
    .with_save(CallbackSave::None)
    .into_callback_set()
    .unwrap();
    let problem = OdeProblem::new(zero, [1.0, 1.0], (0.0, 0.3), ())
        .with_callback_set(displace.append(domain));
    let solution = solve(&problem, Euler, &fixed()).unwrap();
    assert_eq!(solution.values(), &[0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 1.0]);
}

#[test]
fn the_same_problem_preserves_scalar_vector_and_matrix_shapes() {
    let run = |initial| {
        let problem = OdeProblem::from_array(
            |mut du: ArrayViewMutD<'_, f64>, _: ArrayViewD<'_, f64>, _: &(), _| {
                du.fill(0.0);
            },
            initial,
            (0.0, 0.1),
            (),
        )
        .with_callback_set(circle_policy(false).into_callback_set().unwrap());
        solve(&problem, Euler, &fixed()).unwrap()
    };
    let solutions = [
        run(arr0(2.0).into_dyn()),
        run(array![2.0, 0.0].into_dyn()),
        run(array![[2.0, 0.0], [0.0, 0.0]].into_dyn()),
    ];
    for (solution, shape) in solutions.iter().zip([&[][..], &[2][..], &[2, 2][..]]) {
        assert_eq!(solution.state_shape(), shape);
        assert!((solution.last_state()[0] - 1.0).abs() < 3.0e-15);
        // A forward-difference Jacobian approximates zero partial derivatives
        // with a small bias. The preserved shape and projected norm are exact
        // contracts; direction is only accurate to the differentiation scale.
        assert!(
            solution.last_state()[1..]
                .iter()
                .all(|value| value.abs() < 1.0e-8)
        );
    }
}

#[test]
fn prediction_and_projection_route_through_all_split_drivers() {
    let problem = || {
        SplitOdeProblem::new(oscillator, zero, [1.0, 0.0], (0.0, 0.1), ())
            .with_callback_set(circle_policy(true).into_callback_set().unwrap())
    };
    let solutions = [
        solve_split(&problem(), SplitEuler, &fixed()).unwrap(),
        solve_split(&problem(), MRIGARKERK22a::new(4), &fixed()).unwrap(),
        solve_split(&problem(), IMEXEuler, &fixed()).unwrap(),
        solve_split(&problem(), IRKC::default(), &fixed()).unwrap(),
    ];
    for solution in solutions {
        assert!((solution.times()[1] - 0.0225).abs() < 1.0e-14);
        for u in solution.values().chunks_exact(2) {
            assert!((u[0] * u[0] + u[1] * u[1] - 1.0).abs() < 3.0e-15);
        }
    }
}

#[test]
fn projection_composes_in_order_and_positive_domain_keeps_its_semantics() {
    let displace = || {
        CallbackSet::new().with_discrete_callback(
            |_, _, _| true,
            |u, _, _| {
                u[0] = 2.0;
                CallbackAction::Continue
            },
        )
    };
    let project = || {
        circle_policy(true)
            .with_absolute_tolerance(10.0)
            .into_callback_set()
            .unwrap()
    };
    let run = |callbacks| {
        let problem =
            OdeProblem::new(zero, [1.0, 0.0], (0.0, 0.1), ()).with_callback_set(callbacks);
        solve(&problem, Euler, &fixed()).unwrap()
    };
    assert!((run(displace().append(project())).last_state()[0] - 1.0).abs() < 3.0e-15);
    assert_eq!(run(project().append(displace())).last_state()[0], 2.0);

    let positive = PositiveDomain::new().into_callback_set().unwrap();
    let solution = run(project().append(positive));
    assert_eq!(solution.last_state(), &[1.0, 0.0]);
}

#[test]
fn invalid_settings_and_runtime_failures_are_typed() {
    assert!(GeneralDomain::new(0, circle).into_callback_set().is_err());
    for tolerance in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            GeneralDomain::new(1, circle)
                .with_absolute_tolerance(tolerance)
                .into_callback_set(),
            Err(ConfigurationError::InvalidParameter {
                parameter: "general-domain absolute tolerance",
                ..
            })
        ));
    }
    for factor in [-1.0, 0.0, 1.0, f64::NAN, f64::INFINITY] {
        assert!(
            GeneralDomain::new(1, circle)
                .with_reduction_factor(factor)
                .into_callback_set()
                .is_err()
        );
    }
    for policy in [
        GeneralDomain::new(1, circle).with_projection_absolute_tolerance(0.0),
        GeneralDomain::new(1, circle).with_finite_difference_step(0.0),
        GeneralDomain::new(1, circle).with_max_iterations(0),
    ] {
        assert!(policy.into_callback_set().is_err());
    }

    let nonfinite = GeneralDomain::new(1, |r: &mut [f64], _: &[f64], _: &(), t| {
        r[0] = if t == 0.0 { 0.0 } else { f64::NAN };
    })
    .into_callback_set()
    .unwrap();
    let problem = OdeProblem::new(zero, [1.0], (0.0, 0.1), ()).with_callback_set(nonfinite);
    assert_eq!(
        solve(&problem, Euler, &fixed()),
        Err(SolveError::NonFiniteDomainResidual)
    );

    let singular = GeneralDomain::new(1, |r: &mut [f64], _: &[f64], _: &(), _| r[0] = 1.0)
        .into_callback_set()
        .unwrap();
    let problem = OdeProblem::new(zero, [1.0], (0.0, 0.1), ()).with_callback_set(singular);
    assert_eq!(
        solve(&problem, Euler, &fixed()),
        Err(SolveError::ManifoldProjectionFailed)
    );

    let callbacks = GeneralDomain::new(1, circle)
        .with_absolute_tolerance(0.0)
        .into_callback_set()
        .unwrap();
    let problem = OdeProblem::new(zero, [1.0], (0.0, 0.1), ()).with_callback_set(callbacks);
    assert_eq!(
        solve(&problem, Euler, &fixed()),
        Err(SolveError::StepSizeUnderflow)
    );
}
