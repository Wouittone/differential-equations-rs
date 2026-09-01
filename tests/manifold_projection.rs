use differential_equations::callbacks::ManifoldProjection;
use differential_equations::ndarray::{ArrayViewD, ArrayViewMutD, arr0, array};
use differential_equations::solvers::explicit::Euler;
use differential_equations::solvers::explicit::split_euler::{SplitEuler, solve_split};
use differential_equations::solvers::multirate::MRIGARKERK22a;
use differential_equations::solvers::multistep::IMEXEuler;
use differential_equations::solvers::stabilized::IRKC;
use differential_equations::{
    CallbackAction, CallbackSave, CallbackSet, ConfigurationError, OdeProblem, SaveMode,
    SolveError, SolveOptions, SplitOdeProblem, solve,
};

type SplitRhs = fn(&mut [f64], &[f64], &(), f64);

fn fixed(step: f64) -> SolveOptions {
    SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(step)
        .with_save(SaveMode::EveryStep)
}

fn oscillator(derivative: &mut [f64], state: &[f64], _: &(), _: f64) {
    derivative[0] = state[1];
    derivative[1] = -state[0];
}

fn zero(derivative: &mut [f64], _: &[f64], _: &(), _: f64) {
    derivative.fill(0.0);
}

fn circle_residual(residual: &mut [f64], state: &[f64], _: &(), _: f64) {
    residual[0] = state[0] * state[0] + state[1] * state[1] - 1.0;
}

fn circle_jacobian(jacobian: &mut [f64], state: &[f64], _: &(), _: f64) {
    jacobian[0] = 2.0 * state[0];
    jacobian[1] = 2.0 * state[1];
}

fn circle_callbacks(analytic: bool) -> CallbackSet<()> {
    let projection = ManifoldProjection::new(1, circle_residual)
        .with_absolute_tolerance(1.0e-12)
        .with_save(CallbackSave::None);
    if analytic {
        projection
            .with_jacobian(circle_jacobian)
            .into_callback_set()
            .unwrap()
    } else {
        projection.into_callback_set().unwrap()
    }
}

#[test]
fn rectangular_projection_preserves_a_conservation_law() {
    for analytic in [false, true] {
        let problem = OdeProblem::new(oscillator, [1.0, 0.0], (0.0, 10.0), ())
            .with_callback_set(circle_callbacks(analytic));
        let solution = solve(&problem, Euler, &fixed(0.1)).unwrap();

        for state in solution.values().chunks_exact(2) {
            let radius_squared = state[0] * state[0] + state[1] * state[1];
            assert!((radius_squared - 1.0).abs() < 2.0e-12);
        }
    }
}

#[test]
fn projection_respects_callback_order() {
    let displace = || {
        CallbackSet::new().with_discrete_callback(
            |_, _, _| true,
            |state, _, _| {
                state[0] = 2.0;
                state[1] = 0.0;
                CallbackAction::Continue
            },
        )
    };

    let project_last = displace().append(circle_callbacks(true));
    let problem = OdeProblem::new(zero, [1.0, 0.0], (0.0, 0.1), ()).with_callback_set(project_last);
    let projected = solve(&problem, Euler, &fixed(0.1)).unwrap();
    assert!((projected.last_state()[0] - 1.0).abs() < 1.0e-12);

    let displace_last = circle_callbacks(true).append(displace());
    let problem =
        OdeProblem::new(zero, [1.0, 0.0], (0.0, 0.1), ()).with_callback_set(displace_last);
    let displaced = solve(&problem, Euler, &fixed(0.1)).unwrap();
    assert_eq!(displaced.last_state(), &[2.0, 0.0]);
}

#[test]
fn scalar_vector_and_matrix_states_keep_their_ndarray_shape() {
    let solve_array = |initial, target| {
        let projection =
            ManifoldProjection::new(1, move |residual: &mut [f64], state: &[f64], _: &(), _| {
                residual[0] = state.iter().map(|value| value * value).sum::<f64>() - target;
            })
            .with_save(CallbackSave::None)
            .into_callback_set()
            .unwrap();
        let problem = OdeProblem::from_array(
            |mut derivative: ArrayViewMutD<'_, f64>, _: ArrayViewD<'_, f64>, _: &(), _| {
                derivative.fill(0.0);
            },
            initial,
            (0.0, 0.1),
            (),
        )
        .with_callback_set(projection);
        solve(&problem, Euler, &fixed(0.1)).unwrap()
    };

    let scalar = solve_array(arr0(2.0).into_dyn(), 1.0);
    let vector = solve_array(array![2.0, 0.0].into_dyn(), 1.0);
    let matrix = solve_array(array![[2.0, 0.0], [0.0, 0.0]].into_dyn(), 1.0);
    assert!(scalar.state_shape().is_empty());
    assert_eq!(vector.state_shape(), &[2]);
    assert_eq!(matrix.state_shape(), &[2, 2]);
    for solution in [&scalar, &vector, &matrix] {
        let norm_squared = solution
            .last_state()
            .iter()
            .map(|value| value * value)
            .sum::<f64>();
        assert!((norm_squared - 1.0).abs() < 1.0e-10);
    }
}

fn split_problem() -> SplitOdeProblem<SplitRhs, SplitRhs, ()> {
    SplitOdeProblem::new(
        zero as SplitRhs,
        zero as SplitRhs,
        [2.0, 0.0],
        (0.0, 0.1),
        (),
    )
    .with_callback_set(circle_callbacks(true))
}

#[test]
fn projection_routes_through_every_split_driver_family() {
    let solutions = [
        solve_split(&split_problem(), SplitEuler, &fixed(0.1)).unwrap(),
        solve_split(&split_problem(), MRIGARKERK22a::new(4), &fixed(0.1)).unwrap(),
        solve_split(&split_problem(), IMEXEuler, &fixed(0.1)).unwrap(),
        solve_split(&split_problem(), IRKC::default(), &fixed(0.1)).unwrap(),
    ];
    for solution in solutions {
        assert!((solution.last_state()[0] - 1.0).abs() < 1.0e-12);
        assert!(solution.last_state()[1].abs() < 1.0e-12);
    }
}

#[test]
fn invalid_configuration_and_projection_failures_are_typed() {
    assert!(matches!(
        ManifoldProjection::new(0, circle_residual).into_callback_set(),
        Err(ConfigurationError::EmptyData {
            context: "manifold residual"
        })
    ));
    for tolerance in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            ManifoldProjection::new(1, circle_residual)
                .with_absolute_tolerance(tolerance)
                .into_callback_set(),
            Err(ConfigurationError::InvalidParameter {
                parameter: "manifold projection absolute tolerance",
                ..
            })
        ));
    }
    assert!(matches!(
        ManifoldProjection::new(1, circle_residual)
            .with_max_iterations(0)
            .into_callback_set(),
        Err(ConfigurationError::InvalidParameter {
            parameter: "manifold projection maximum iterations",
            ..
        })
    ));
    for step in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            ManifoldProjection::new(1, circle_residual)
                .with_finite_difference_step(step)
                .into_callback_set(),
            Err(ConfigurationError::InvalidParameter {
                parameter: "manifold projection finite-difference step",
                ..
            })
        ));
    }

    let too_many_constraints =
        ManifoldProjection::new(2, |residual: &mut [f64], state: &[f64], _: &(), _| {
            residual[0] = state[0];
            residual[1] = state[0];
        })
        .into_callback_set()
        .unwrap();
    let problem =
        OdeProblem::new(zero, [1.0], (0.0, 0.1), ()).with_callback_set(too_many_constraints);
    assert_eq!(
        solve(&problem, Euler, &fixed(0.1)),
        Err(SolveError::InvalidManifoldDimension)
    );

    let nonfinite = ManifoldProjection::new(1, |residual: &mut [f64], _: &[f64], _: &(), _| {
        residual[0] = f64::NAN
    })
    .into_callback_set()
    .unwrap();
    let problem = OdeProblem::new(zero, [1.0], (0.0, 0.1), ()).with_callback_set(nonfinite);
    assert_eq!(
        solve(&problem, Euler, &fixed(0.1)),
        Err(SolveError::NonFiniteManifoldProjection)
    );

    let nonfinite_jacobian = ManifoldProjection::new(1, circle_residual)
        .with_jacobian(|jacobian: &mut [f64], _: &[f64], _: &(), _| jacobian.fill(f64::NAN))
        .into_callback_set()
        .unwrap();
    let problem =
        OdeProblem::new(zero, [2.0, 0.0], (0.0, 0.1), ()).with_callback_set(nonfinite_jacobian);
    assert_eq!(
        solve(&problem, Euler, &fixed(0.1)),
        Err(SolveError::NonFiniteManifoldProjection)
    );

    let singular = ManifoldProjection::new(1, |residual: &mut [f64], _: &[f64], _: &(), _| {
        residual[0] = 1.0
    })
    .into_callback_set()
    .unwrap();
    let problem = OdeProblem::new(zero, [1.0], (0.0, 0.1), ()).with_callback_set(singular);
    assert_eq!(
        solve(&problem, Euler, &fixed(0.1)),
        Err(SolveError::ManifoldProjectionFailed)
    );

    let insufficient_iterations = ManifoldProjection::new(1, circle_residual)
        .with_jacobian(circle_jacobian)
        .with_max_iterations(1)
        .into_callback_set()
        .unwrap();
    let problem = OdeProblem::new(zero, [2.0, 0.0], (0.0, 0.1), ())
        .with_callback_set(insufficient_iterations);
    assert_eq!(
        solve(&problem, Euler, &fixed(0.1)),
        Err(SolveError::ManifoldProjectionFailed)
    );
}
