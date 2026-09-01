use differential_equations::callbacks::StepsizeLimiter;
use differential_equations::solvers::explicit::split_euler::{
    SplitEuler, SplitOdeAlgorithm, solve_split,
};
use differential_equations::solvers::explicit::{Rk4, Tsit5};
use differential_equations::solvers::multirate::MRIGARKERK22a;
use differential_equations::solvers::multistep::IMEXEuler;
use differential_equations::solvers::second_order::{
    Dprkn4, Irkn3, NewmarkBeta, Nystrom4, PseudoVerletLeapfrog, SecondOrderOdeAlgorithm,
    SecondOrderOdeProblem, VelocityVerlet, solve_second_order, solve_symplectic,
};
use differential_equations::solvers::stabilized::IRKC;
use differential_equations::{
    ConfigurationError, OdeProblem, SaveMode, SolveError, SolveOptions, SplitOdeProblem, solve,
};

type SplitRhs = fn(&mut [f64], &[f64], &(), f64);
type Acceleration = fn(&mut [f64], &[f64], &[f64], &(), f64);

fn fixed(step: f64) -> SolveOptions {
    SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(step)
        .with_save(SaveMode::EveryStep)
}

fn zero(derivative: &mut [f64], _: &[f64], _: &(), _: f64) {
    derivative.fill(0.0);
}

fn zero_acceleration(acceleration: &mut [f64], _: &[f64], _: &[f64], _: &(), _: f64) {
    acceleration.fill(0.0);
}

fn assert_steps_at_most(times: &[f64], maximum: f64) {
    for window in times.windows(2) {
        let step = (window[1] - window[0]).abs();
        assert!(step <= maximum + 1.0e-12, "step {step} exceeds {maximum}");
    }
}

#[test]
fn limiter_caps_fixed_adaptive_and_backward_steps() {
    let callbacks = StepsizeLimiter::new(|_: &[f64], _: &(), _| 0.1)
        .with_safety_factor(1.0)
        .into_callback_set()
        .unwrap();
    let fixed_problem = OdeProblem::new(zero, [0.0], (0.0, 1.0), ()).with_callback_set(callbacks);
    let fixed_solution = solve(&fixed_problem, Rk4, &fixed(0.4)).unwrap();
    assert_steps_at_most(fixed_solution.times(), 0.1);

    let callbacks = StepsizeLimiter::new(|_: &[f64], _: &(), _| 0.1)
        .with_safety_factor(1.0)
        .into_callback_set()
        .unwrap();
    let adaptive_problem =
        OdeProblem::new(zero, [0.0], (0.0, 1.0), ()).with_callback_set(callbacks);
    let adaptive_solution = solve(
        &adaptive_problem,
        Tsit5,
        &SolveOptions::new()
            .with_initial_step(0.5)
            .with_tolerances(1.0e-2, 1.0e-2)
            .with_save(SaveMode::EveryStep),
    )
    .unwrap();
    assert_steps_at_most(adaptive_solution.times(), 0.1);

    let callbacks = StepsizeLimiter::new(|_: &[f64], _: &(), _| 0.1)
        .with_safety_factor(1.0)
        .into_callback_set()
        .unwrap();
    let backward_problem =
        OdeProblem::new(zero, [0.0], (1.0, 0.0), ()).with_callback_set(callbacks);
    let backward_solution = solve(&backward_problem, Rk4, &fixed(0.4)).unwrap();
    assert_steps_at_most(backward_solution.times(), 0.1);
    assert!(backward_solution.times().windows(2).all(|w| w[1] < w[0]));
}

#[test]
fn max_step_mode_tracks_the_scaled_limit_exactly() {
    let callbacks = StepsizeLimiter::new(|_: &[f64], _: &(), _| 0.2)
        .with_safety_factor(0.5)
        .with_max_step(true)
        .into_callback_set()
        .unwrap();
    let problem = OdeProblem::new(zero, [0.0], (0.0, 0.35), ()).with_callback_set(callbacks);
    let solution = solve(&problem, Rk4, &fixed(0.03)).unwrap();

    for window in solution.times().windows(2).take(3) {
        assert!((window[1] - window[0] - 0.1).abs() < 1.0e-12);
    }
    assert!((solution.times().last().unwrap() - 0.35).abs() < 1.0e-12);
}

fn split_problem() -> SplitOdeProblem<SplitRhs, SplitRhs, ()> {
    let callbacks = StepsizeLimiter::new(|_: &[f64], _: &(), _| 0.1)
        .with_safety_factor(1.0)
        .into_callback_set()
        .unwrap();
    SplitOdeProblem::new(zero as SplitRhs, zero as SplitRhs, [0.0], (0.0, 0.5), ())
        .with_callback_set(callbacks)
}

fn assert_split_limiter<A: SplitOdeAlgorithm + Copy>(algorithm: A) {
    let solution = solve_split(&split_problem(), algorithm, &fixed(0.3)).unwrap();
    assert_steps_at_most(solution.times(), 0.1);
}

fn second_order_problem() -> SecondOrderOdeProblem<Acceleration, ()> {
    let callbacks = StepsizeLimiter::new(|_: &[f64], _: &[f64], _: &(), _| 0.1)
        .with_safety_factor(1.0)
        .into_second_order_callback_set()
        .unwrap();
    SecondOrderOdeProblem::new(
        zero_acceleration as Acceleration,
        [0.0],
        [0.0],
        (0.0, 0.5),
        (),
    )
    .with_callback_set(callbacks)
}

fn assert_second_order_limiter<A: SecondOrderOdeAlgorithm + Copy>(algorithm: A, adaptive: bool) {
    let solution = solve_second_order(
        &second_order_problem(),
        algorithm,
        &fixed(0.3).with_adaptive(adaptive),
    )
    .unwrap();
    assert_steps_at_most(solution.times(), 0.1);
}

#[test]
fn limiter_routes_through_split_and_second_order_driver_families() {
    assert_split_limiter(SplitEuler);
    assert_split_limiter(MRIGARKERK22a::new(4));
    assert_split_limiter(IMEXEuler);
    assert_split_limiter(IRKC::default());

    assert_second_order_limiter(NewmarkBeta::default(), false);
    assert_second_order_limiter(Nystrom4, false);
    assert_second_order_limiter(Dprkn4, true);
    assert_second_order_limiter(Irkn3, false);
    assert_second_order_limiter(VelocityVerlet, false);

    let solution =
        solve_symplectic(&second_order_problem(), PseudoVerletLeapfrog, &fixed(0.3)).unwrap();
    assert_steps_at_most(solution.times(), 0.1);
}

#[test]
fn limiter_preserves_fsal_caches_when_the_step_is_already_safe() {
    let plain = OdeProblem::new(
        |derivative: &mut [f64], state: &[f64], _: &(), _| derivative[0] = -state[0],
        [1.0],
        (0.0, 1.0),
        (),
    );
    let plain_solution = solve(&plain, Tsit5, &fixed(0.1)).unwrap();

    let callbacks = StepsizeLimiter::new(|_: &[f64], _: &(), _| 0.1)
        .with_safety_factor(1.0)
        .into_callback_set()
        .unwrap();
    let limited = OdeProblem::new(
        |derivative: &mut [f64], state: &[f64], _: &(), _| derivative[0] = -state[0],
        [1.0],
        (0.0, 1.0),
        (),
    )
    .with_callback_set(callbacks);
    let limited_solution = solve(&limited, Tsit5, &fixed(0.1)).unwrap();

    assert_eq!(
        limited_solution.stats().rhs_evaluations,
        plain_solution.stats().rhs_evaluations
    );
    assert_eq!(
        limited_solution.stats().callback_invocations,
        limited_solution.stats().accepted_steps + 1
    );
}

#[test]
fn limiter_reports_typed_configuration_and_runtime_errors() {
    for safety_factor in [0.0, -1.0, 1.1, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            StepsizeLimiter::new(|_: &[f64], _: &(), _| 0.1)
                .with_safety_factor(safety_factor)
                .into_callback_set(),
            Err(ConfigurationError::InvalidParameter {
                parameter: "stepsize safety factor",
                ..
            })
        ));
    }

    for limit in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let callbacks = StepsizeLimiter::new(move |_: &[f64], _: &(), _| limit)
            .into_callback_set()
            .unwrap();
        let problem = OdeProblem::new(zero, [0.0], (0.0, 1.0), ()).with_callback_set(callbacks);
        assert_eq!(
            solve(&problem, Rk4, &fixed(0.1)),
            Err(SolveError::InvalidCallbackStepSize)
        );
    }
}
