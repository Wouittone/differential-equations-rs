use std::cell::Cell;

use differential_equations::callbacks::TerminateSteadyState;
use differential_equations::ndarray::{ArrayViewD, ArrayViewMutD, arr0, array};
use differential_equations::solvers::explicit::split_euler::{SplitEuler, solve_split};
use differential_equations::solvers::explicit::{Euler, Tsit5};
use differential_equations::solvers::multirate::MRIGARKERK22a;
use differential_equations::solvers::multistep::IMEXEuler;
use differential_equations::solvers::rosenbrock::Rodas5P;
use differential_equations::solvers::second_order::{
    Dprkn4, Irkn3, NewmarkBeta, Nystrom4, SecondOrderOdeAlgorithm, SecondOrderOdeProblem,
    SecondOrderSolveError, VelocityVerlet, solve_second_order,
};
use differential_equations::solvers::stabilized::IRKC;
use differential_equations::{
    CallbackAction, CallbackSave, CallbackSet, OdeAlgorithm, OdeProblem, SaveMode, SolveError,
    SolveOptions, SplitOdeProblem, solve,
};

fn options() -> SolveOptions {
    SolveOptions::new()
        .with_tolerances(1.0e-10, 1.0e-10)
        .with_initial_step(0.25)
        .with_save(SaveMode::Endpoints)
}

fn assert_decay<A: OdeAlgorithm + Copy>(algorithm: A) {
    for initial in [
        arr0(1.0).into_dyn(),
        array![1.0, 1.0].into_dyn(),
        array![[1.0, 1.0], [1.0, 1.0]].into_dyn(),
    ] {
        let shape = initial.shape().to_vec();
        let problem = OdeProblem::from_array(
            |mut du: ArrayViewMutD<'_, f64>, u: ArrayViewD<'_, f64>, _: &(), _| {
                du.zip_mut_with(&u, |du, u| *du = -*u);
            },
            initial,
            (0.0, 100.0),
            (),
        )
        .with_callback_set(
            TerminateSteadyState::new()
                .with_tolerances(1.0e-6, 0.0)
                .into_callback_set()
                .unwrap(),
        );
        for _ in 0..2 {
            let solution = solve(&problem, algorithm, &options()).unwrap();
            let time = *solution.times().last().unwrap();
            assert!(time > 13.0 && time < 100.0, "termination time {time}");
            assert_eq!(solution.state_shape(), shape);
            assert!(solution.last_state().iter().all(|u| u.abs() <= 1.0e-6));
            assert_eq!(solution.stats().callback_invocations, 1);
            assert_eq!(solution.times().len(), 2);
        }
    }
}

#[test]
fn same_decay_problem_terminates_for_every_shape_and_explicit_or_stiff_solver() {
    assert_decay(Tsit5);
    assert_decay(Rodas5P);
}

#[test]
fn criterion_is_componentwise_maximum_not_sum_or_average() {
    let callbacks = TerminateSteadyState::new()
        .with_component_tolerances([0.1, 0.0], [0.0, 0.1])
        .into_callback_set()
        .unwrap();
    let problem = OdeProblem::new(
        |du: &mut [f64], _: &[f64], _: &(), _| du.copy_from_slice(&[0.09, 0.9]),
        [0.0, 10.0],
        (0.0, 1.0),
        (),
    )
    .with_callback_set(callbacks);
    let solution = solve(&problem, Euler, &options().with_adaptive(false)).unwrap();
    assert_eq!(solution.times(), &[0.0]);
    assert_eq!(solution.stats().rhs_evaluations, 1);
    assert_eq!(solution.stats().accepted_steps, 0);

    let problem = OdeProblem::new(
        |du: &mut [f64], _: &[f64], _: &(), _| du.fill(0.15),
        [1.0],
        (0.0, 1.0),
        (),
    )
    .with_callback_set(
        TerminateSteadyState::new()
            .with_tolerances(0.1, 0.1)
            .into_callback_set()
            .unwrap(),
    );
    let solution = solve(&problem, Euler, &options().with_adaptive(false)).unwrap();
    assert_eq!(*solution.times().last().unwrap(), 1.0);
    assert_eq!(solution.stats().callback_invocations, 0);
}

#[test]
fn minimum_time_uses_absolute_time_in_either_integration_direction() {
    for (span, minimum, expected, invocations) in [
        ((0.0, 1.0), 0.3, 0.5, 1),
        ((1.0, 0.0), 1.1, 0.0, 0),
        ((1.0, 0.0), 0.5, 1.0, 1),
    ] {
        let problem = OdeProblem::new(
            |du: &mut [f64], _: &[f64], _: &(), _| du.fill(0.0),
            [1.0],
            span,
            (),
        )
        .with_callback_set(
            TerminateSteadyState::new()
                .with_min_time(minimum)
                .into_callback_set()
                .unwrap(),
        );
        let solution = solve(&problem, Euler, &options().with_adaptive(false)).unwrap();
        assert_eq!(*solution.times().last().unwrap(), expected);
        assert_eq!(solution.stats().callback_invocations, invocations);
    }
}

#[test]
fn checks_see_initializers_and_earlier_parameter_effects() {
    let callbacks = CallbackSet::new()
        .with_initialize(|u, _: &Cell<bool>, _| u[0] = 2.0)
        .with_discrete_callback(
            |_, _, _| true,
            |u, p, _| {
                assert_eq!(u[0], 2.0);
                p.set(true);
                CallbackAction::Continue
            },
        )
        .append(TerminateSteadyState::new().into_callback_set().unwrap());
    let problem = OdeProblem::new(
        |du: &mut [f64], _: &[f64], p: &Cell<bool>, _| du.fill(if p.get() { 0.0 } else { 1.0 }),
        [0.0],
        (0.0, 1.0),
        Cell::new(false),
    )
    .with_callback_set(callbacks);
    let solution = solve(&problem, Euler, &options().with_adaptive(false)).unwrap();
    assert_eq!(solution.times().last(), Some(&0.0));
    assert_eq!(solution.last_state(), &[2.0]);
    assert_eq!(solution.stats().rhs_evaluations, 1);
    assert_eq!(solution.stats().callback_invocations, 2);
}

fn counted(du: &mut [f64], _: &[f64], calls: &Cell<usize>, _: f64) {
    calls.set(calls.get() + 1);
    du.fill(1.0);
}

#[test]
fn ordinary_statistics_include_checks_without_invalidating_fsal() {
    let problem = OdeProblem::new(counted, [0.0], (0.0, 1.0), Cell::new(0));
    let baseline = solve(&problem, Tsit5, &options().with_adaptive(false)).unwrap();
    let problem = OdeProblem::new(counted, [0.0], (0.0, 1.0), Cell::new(0))
        .with_callback_set(TerminateSteadyState::new().into_callback_set().unwrap());
    let solution = solve(&problem, Tsit5, &options().with_adaptive(false)).unwrap();
    assert_eq!(solution.stats().rhs_evaluations, problem.parameters().get());
    assert_eq!(
        solution.stats().rhs_evaluations,
        baseline.stats().rhs_evaluations + solution.stats().accepted_steps + 1
    );
    assert_eq!(solution.stats().callback_invocations, 0);
}

#[test]
fn split_checks_use_total_derivative_and_count_both_components() {
    for cancel in [false, true] {
        let problem = SplitOdeProblem::new(
            counted,
            move |du: &mut [f64], _: &[f64], calls: &Cell<usize>, _| {
                calls.set(calls.get() + 1);
                du.fill(if cancel { -1.0 } else { 0.0 });
            },
            [0.0],
            (0.0, 1.0),
            Cell::new(0),
        )
        .with_callback_set(TerminateSteadyState::new().into_callback_set().unwrap());
        let solutions = [
            {
                problem.parameters().set(0);
                let s = solve_split(&problem, SplitEuler, &options().with_adaptive(false)).unwrap();
                assert_eq!(s.stats().rhs_evaluations, problem.parameters().get());
                s
            },
            {
                problem.parameters().set(0);
                let s = solve_split(
                    &problem,
                    MRIGARKERK22a::new(4),
                    &options().with_adaptive(false),
                )
                .unwrap();
                assert_eq!(s.stats().rhs_evaluations, problem.parameters().get());
                s
            },
            {
                problem.parameters().set(0);
                let s = solve_split(&problem, IMEXEuler, &options().with_adaptive(false)).unwrap();
                assert_eq!(s.stats().rhs_evaluations, problem.parameters().get());
                s
            },
            {
                problem.parameters().set(0);
                let s = solve_split(&problem, IRKC::default(), &options().with_adaptive(false))
                    .unwrap();
                assert_eq!(s.stats().rhs_evaluations, problem.parameters().get());
                s
            },
        ];
        for solution in solutions {
            assert_eq!(solution.stats().callback_invocations, usize::from(cancel));
            assert_eq!(
                *solution.times().last().unwrap(),
                if cancel { 0.0 } else { 1.0 }
            );
            if cancel {
                assert_eq!(solution.stats().rhs_evaluations, 2);
            }
        }
    }
}

fn assert_partitioned<A: SecondOrderOdeAlgorithm + Copy>(algorithm: A, adaptive: bool) {
    for velocity in [0.0, 1.0] {
        let problem = SecondOrderOdeProblem::new(
            |a: &mut [f64], _: &[f64], _: &[f64], calls: &Cell<usize>, _| {
                calls.set(calls.get() + 1);
                a.fill(0.0);
            },
            [velocity],
            [0.0],
            (0.0, 1.0),
            Cell::new(0),
        )
        .with_callback_set(
            TerminateSteadyState::new()
                .into_second_order_callback_set()
                .unwrap(),
        );
        let solution =
            solve_second_order(&problem, algorithm, &options().with_adaptive(adaptive)).unwrap();
        assert_eq!(solution.stats().rhs_evaluations, problem.parameters().get());
        assert_eq!(
            *solution.times().last().unwrap(),
            if velocity == 0.0 { 0.0 } else { 1.0 }
        );
        assert_eq!(
            solution.stats().callback_invocations,
            usize::from(velocity == 0.0)
        );
    }
}

#[test]
fn every_second_order_driver_requires_small_velocity_as_well_as_acceleration() {
    assert_partitioned(NewmarkBeta::default(), false);
    assert_partitioned(Nystrom4, false);
    assert_partitioned(Dprkn4, true);
    assert_partitioned(Irkn3, false);
    assert_partitioned(VelocityVerlet, false);
}

#[test]
fn invalid_tolerances_and_incomplete_derivatives_are_errors() {
    for tolerance in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(
            TerminateSteadyState::new()
                .with_tolerances(tolerance, 0.0)
                .into_callback_set::<()>()
                .is_err()
        );
        assert!(
            TerminateSteadyState::new()
                .with_tolerances(0.0, tolerance)
                .into_second_order_callback_set::<()>()
                .is_err()
        );
    }
    assert!(
        TerminateSteadyState::new()
            .with_min_time(f64::NAN)
            .into_callback_set::<()>()
            .is_err()
    );
    assert!(
        TerminateSteadyState::new()
            .with_component_tolerances([], [0.0])
            .into_callback_set::<()>()
            .is_err()
    );
    let problem = OdeProblem::new(
        |du: &mut [f64], _: &[f64], _: &(), _| du.fill(0.0),
        [0.0; 2],
        (0.0, 1.0),
        (),
    )
    .with_callback_set(
        TerminateSteadyState::new()
            .with_component_tolerances([0.0; 3], [0.0])
            .with_min_time(2.0)
            .into_callback_set()
            .unwrap(),
    );
    assert_eq!(
        solve(&problem, Euler, &options().with_adaptive(false)),
        Err(SolveError::InvalidSteadyStateDimension)
    );
    let problem = OdeProblem::new(
        |_: &mut [f64], _: &[f64], _: &(), _| {},
        [0.0],
        (0.0, 1.0),
        (),
    )
    .with_callback_set(TerminateSteadyState::new().into_callback_set().unwrap());
    assert_eq!(
        solve(&problem, Euler, &options().with_adaptive(false)),
        Err(SolveError::NonFiniteDerivative)
    );
    let problem = SecondOrderOdeProblem::new(
        |_: &mut [f64], _: &[f64], _: &[f64], _: &(), _| {},
        [0.0],
        [0.0],
        (0.0, 1.0),
        (),
    )
    .with_callback_set(
        TerminateSteadyState::new()
            .into_second_order_callback_set()
            .unwrap(),
    );
    assert_eq!(
        solve_second_order(&problem, Nystrom4, &options().with_adaptive(false)).unwrap_err(),
        SecondOrderSolveError::Solve(SolveError::NonFiniteDerivative)
    );
}

#[test]
fn termination_saving_respects_the_policy_with_requested_output_times() {
    let problem = |save| {
        OdeProblem::new(
            |du: &mut [f64], _: &[f64], _: &(), _| du.fill(0.0),
            [1.0],
            (0.0, 1.0),
            (),
        )
        .with_callback_set(
            TerminateSteadyState::new()
                .with_min_time(0.5)
                .with_save(save)
                .into_callback_set()
                .unwrap(),
        )
    };
    let options = options().with_adaptive(false).with_save_at([0.25]);
    let solution = solve(&problem(CallbackSave::Before), Euler, &options).unwrap();
    assert_eq!(solution.times(), &[0.25, 0.5, 0.5]);
    let solution = solve(&problem(CallbackSave::None), Euler, &options).unwrap();
    assert_eq!(solution.times(), &[0.25, 0.5]);
}
