use differential_equations::callbacks::PositiveDomain;
use differential_equations::ndarray::{ArrayViewD, ArrayViewMutD, arr0, array};
use differential_equations::solvers::explicit::split_euler::{SplitEuler, solve_split};
use differential_equations::solvers::explicit::{Euler, Tsit5};
use differential_equations::solvers::multirate::MRIGARKERK22a;
use differential_equations::solvers::multistep::IMEXEuler;
use differential_equations::solvers::stabilized::IRKC;
use differential_equations::{
    CallbackSave, ConfigurationError, OdeProblem, SaveMode, SolveOptions, SplitOdeProblem, solve,
};

type SplitRhs = fn(&mut [f64], &[f64], &(), f64);

fn fixed(step: f64) -> SolveOptions {
    SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(step)
        .with_save(SaveMode::EveryStep)
}

fn decay(derivative: &mut [f64], state: &[f64], _: &(), _: f64) {
    for (derivative, state) in derivative.iter_mut().zip(state) {
        *derivative = -*state;
    }
}

fn zero(derivative: &mut [f64], _: &[f64], _: &(), _: f64) {
    derivative.fill(0.0);
}

fn first_time_after_start(times: &[f64]) -> f64 {
    let start = times[0];
    *times.iter().find(|time| **time != start).unwrap()
}

#[test]
fn prediction_restricts_the_upcoming_step_without_rejecting_an_attempt() {
    let callbacks = PositiveDomain::new()
        .with_save(CallbackSave::None)
        .into_callback_set()
        .unwrap();
    let problem = OdeProblem::new(decay, [1.0], (0.0, 2.0), ()).with_callback_set(callbacks);

    let solution = solve(&problem, Euler, &fixed(2.0)).unwrap();

    assert!((first_time_after_start(solution.times()) - 0.9).abs() < 1.0e-12);
    assert_eq!(solution.stats().rejected_steps, 0);
    assert!(solution.values().iter().all(|value| *value >= 0.0));

    let callbacks = PositiveDomain::new()
        .with_save(CallbackSave::None)
        .into_callback_set()
        .unwrap();
    let adaptive_problem =
        OdeProblem::new(decay, [1.0], (0.0, 2.0), ()).with_callback_set(callbacks);
    let adaptive = solve(
        &adaptive_problem,
        Tsit5,
        &SolveOptions::new()
            .with_initial_step(2.0)
            .with_save(SaveMode::EveryStep),
    )
    .unwrap();
    assert!((first_time_after_start(adaptive.times()) - 0.9).abs() < 1.0e-12);
    assert!(adaptive.values().iter().all(|value| *value >= 0.0));

    let callbacks = PositiveDomain::new()
        .with_save(CallbackSave::None)
        .into_callback_set()
        .unwrap();
    let backward_problem = OdeProblem::new(
        |derivative: &mut [f64], state: &[f64], _: &(), _| derivative[0] = state[0],
        [1.0],
        (2.0, 0.0),
        (),
    )
    .with_callback_set(callbacks);
    let backward = solve(&backward_problem, Euler, &fixed(2.0)).unwrap();
    assert!((first_time_after_start(backward.times()) - 1.1).abs() < 1.0e-12);
    assert!(
        backward
            .times()
            .windows(2)
            .all(|times| times[1] <= times[0])
    );
}

#[test]
fn solve_tolerance_and_policy_override_control_prediction_acceptance() {
    let default_tolerance = PositiveDomain::new()
        .with_save(CallbackSave::None)
        .into_callback_set()
        .unwrap();
    let problem =
        OdeProblem::new(decay, [1.0], (0.0, 2.0), ()).with_callback_set(default_tolerance);
    let solution = solve(&problem, Euler, &fixed(2.0).with_tolerances(2.0, 1.0e-3)).unwrap();
    assert_eq!(solution.times(), &[0.0, 2.0]);
    assert_eq!(solution.last_state(), &[0.0]);

    let override_tolerance = PositiveDomain::new()
        .with_absolute_tolerance(2.0)
        .with_save(CallbackSave::None)
        .into_callback_set()
        .unwrap();
    let problem =
        OdeProblem::new(decay, [1.0], (0.0, 2.0), ()).with_callback_set(override_tolerance);
    let solution = solve(&problem, Euler, &fixed(2.0)).unwrap();
    assert_eq!(solution.times(), &[0.0, 2.0]);
    assert_eq!(solution.last_state(), &[0.0]);
}

#[test]
fn policies_compose_with_the_most_conservative_reduction() {
    let callbacks = PositiveDomain::new()
        .with_reduction_factor(0.8)
        .with_save(CallbackSave::None)
        .into_callback_set()
        .unwrap()
        .append(
            PositiveDomain::new()
                .with_reduction_factor(0.25)
                .with_save(CallbackSave::None)
                .into_callback_set()
                .unwrap(),
        );
    let problem = OdeProblem::new(decay, [1.0], (0.0, 2.0), ()).with_callback_set(callbacks);
    let solution = solve(&problem, Euler, &fixed(2.0)).unwrap();

    assert!((first_time_after_start(solution.times()) - 0.45).abs() < 1.0e-12);
    assert_eq!(solution.stats().rejected_steps, 0);
}

#[test]
fn negative_initial_and_accepted_values_are_clamped_for_every_array_shape() {
    let solve_array = |initial| {
        let callbacks = PositiveDomain::new()
            .with_absolute_tolerance(2.0)
            .with_save(CallbackSave::None)
            .into_callback_set()
            .unwrap();
        let problem = OdeProblem::from_array(
            |mut derivative: ArrayViewMutD<'_, f64>, state: ArrayViewD<'_, f64>, _: &(), _| {
                derivative.zip_mut_with(&state, |derivative, state| *derivative = -*state);
            },
            initial,
            (0.0, 2.0),
            (),
        )
        .with_callback_set(callbacks);
        solve(&problem, Euler, &fixed(2.0)).unwrap()
    };

    let scalar = solve_array(arr0(1.0).into_dyn());
    let vector = solve_array(array![1.0, -1.0].into_dyn());
    let matrix = solve_array(array![[1.0, -1.0], [0.5, 2.0]].into_dyn());
    assert!(scalar.state_shape().is_empty());
    assert_eq!(vector.state_shape(), &[2]);
    assert_eq!(matrix.state_shape(), &[2, 2]);
    for solution in [&scalar, &vector, &matrix] {
        assert!(solution.last_state().iter().all(|value| *value >= 0.0));
    }
}

fn split_problem() -> SplitOdeProblem<SplitRhs, SplitRhs, ()> {
    let callbacks = PositiveDomain::new()
        .with_save(CallbackSave::None)
        .into_callback_set()
        .unwrap();
    SplitOdeProblem::new(decay as SplitRhs, zero as SplitRhs, [1.0], (0.0, 2.0), ())
        .with_callback_set(callbacks)
}

#[test]
fn prediction_routes_through_every_split_driver_family() {
    let split_euler = solve_split(&split_problem(), SplitEuler, &fixed(2.0)).unwrap();
    assert!((first_time_after_start(split_euler.times()) - 0.9).abs() < 1.0e-12);

    let multirate = solve_split(&split_problem(), MRIGARKERK22a::new(4), &fixed(2.0)).unwrap();
    assert!((first_time_after_start(multirate.times()) - 0.9).abs() < 1.0e-12);

    let imex = solve_split(&split_problem(), IMEXEuler, &fixed(2.0)).unwrap();
    assert!((first_time_after_start(imex.times()) - 0.9).abs() < 1.0e-12);

    let irkc = solve_split(&split_problem(), IRKC::default(), &fixed(2.0)).unwrap();
    assert!((first_time_after_start(irkc.times()) - 0.9).abs() < 1.0e-12);
}

#[test]
fn invalid_policy_parameters_are_typed_configuration_errors() {
    for tolerance in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            PositiveDomain::new()
                .with_absolute_tolerance(tolerance)
                .into_callback_set::<()>(),
            Err(ConfigurationError::InvalidParameter {
                parameter: "positive-domain absolute tolerance",
                ..
            })
        ));
    }
    for factor in [0.0, 1.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            PositiveDomain::new()
                .with_reduction_factor(factor)
                .into_callback_set::<()>(),
            Err(ConfigurationError::InvalidParameter {
                parameter: "positive-domain reduction factor",
                ..
            })
        ));
    }
}
