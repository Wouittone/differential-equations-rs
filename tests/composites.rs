use std::cell::Cell;
use std::rc::Rc;

use differential_equations::solvers::{automatic::*, explicit::*, rosenbrock::*};
use differential_equations::*;

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn problem() -> OdeProblem<TestRhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), time: f64) {
        du[0] = u[0] + time;
    }
    OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
}

fn options() -> SolveOptions {
    SolveOptions::default()
        .with_absolute_tolerance(1.0e-10)
        .with_relative_tolerance(1.0e-10)
        .with_save(SaveMode::Endpoints)
}

fn assert_default_config(config: &AutoSwitchConfig) {
    assert_eq!(config.enter_stiffness(), 0.90);
    assert_eq!(config.exit_stiffness(), 0.75);
    assert_eq!(config.stiff_confirmations(), 10);
    assert_eq!(config.nonstiff_confirmations(), 3);
    assert_eq!(config.rejection_confirmations(), 2);
    assert_eq!(config.minimum_residence_steps(), 3);
    assert_eq!(config.switch_step_factor(), 2.0);
    assert_eq!(config.initial_branch(), AutomaticBranch::NonStiff);
    assert!(config.allow_switch_back());
}

fn assert_nonstiff_solution(solution: &Solution, explicit: &Solution) {
    assert_eq!(solution.last_state(), explicit.last_state());

    let stats = solution.stats();
    assert_eq!(stats.algorithm_switches, 0);
    assert_eq!(stats.nonstiff_accepted_steps, stats.accepted_steps);
    assert_eq!(stats.stiff_accepted_steps, 0);
    assert_eq!(
        stats.final_automatic_branch,
        Some(AutomaticBranch::NonStiff)
    );
}

#[test]
fn every_automatic_facade_exposes_its_components_and_nonstiff_stats() {
    let options = options();

    macro_rules! check_facade {
        ($facade:ident, $explicit:expr) => {{
            let algorithm = $facade::new(Rodas5P);
            assert_eq!(algorithm.stiff_algorithm(), &Rodas5P);
            assert_default_config(algorithm.switch_config());

            let automatic = solve(&problem(), algorithm, &options).unwrap();
            let explicit = solve(&problem(), $explicit, &options).unwrap();
            assert_nonstiff_solution(&automatic, &explicit);
        }};
    }

    check_facade!(AutoTsit5, Tsit5);
    check_facade!(AutoVern6, Vern6);
    check_facade!(AutoVern7, Vern7);
    check_facade!(AutoVern8, Vern8);
    check_facade!(AutoVern9, Vern9);
}

#[test]
fn configured_facade_exposes_the_replaced_switch_policy() {
    let config = AutoSwitchConfig::new()
        .with_stiffness_thresholds(0.25, 0.5)
        .unwrap()
        .with_stiff_confirmations(2)
        .unwrap()
        .with_nonstiff_confirmations(4)
        .unwrap()
        .with_rejection_confirmations(3)
        .unwrap()
        .with_minimum_residence_steps(5)
        .unwrap()
        .with_switch_step_factor(1.5)
        .unwrap()
        .with_initial_branch(AutomaticBranch::Stiff)
        .with_allow_switch_back(false);

    macro_rules! check_facade {
        ($facade:ident) => {{
            let algorithm = $facade::new(Rodas5P).with_switch_config(config).unwrap();
            assert_eq!(algorithm.stiff_algorithm(), &Rodas5P);
            assert_eq!(algorithm.switch_config(), &config);
        }};
    }

    check_facade!(AutoTsit5);
    check_facade!(AutoVern6);
    check_facade!(AutoVern7);
    check_facade!(AutoVern8);
    check_facade!(AutoVern9);
}

#[test]
fn automatic_facades_switch_in_flight_after_an_accepted_nonstiff_step() {
    let config = AutoSwitchConfig::new()
        .with_stiffness_thresholds(1.0e-12, 2.0e-12)
        .unwrap()
        .with_stiff_confirmations(1)
        .unwrap()
        .with_minimum_residence_steps(1)
        .unwrap()
        .with_allow_switch_back(false);
    let stiff_problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -10.0 * u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions::default()
        .with_initial_step(Some(1.0e-3))
        .with_absolute_tolerance(1.0e-9)
        .with_relative_tolerance(1.0e-9)
        .with_save(SaveMode::Endpoints);

    macro_rules! check_switch {
        ($facade:ident) => {{
            let solution = solve(
                &stiff_problem,
                $facade::new(Rodas5P).with_switch_config(config).unwrap(),
                &options,
            )
            .unwrap();
            let stats = solution.stats();

            assert_eq!(stats.algorithm_switches, 1);
            assert!(stats.nonstiff_accepted_steps > 0);
            assert!(stats.stiff_accepted_steps > 0);
            assert_eq!(
                stats.nonstiff_accepted_steps + stats.stiff_accepted_steps,
                stats.accepted_steps
            );
            assert_eq!(stats.final_automatic_branch, Some(AutomaticBranch::Stiff));
            assert!((solution.last_state()[0] - (-10.0_f64).exp()).abs() < 1.0e-7);
        }};
    }

    check_switch!(AutoTsit5);
    check_switch!(AutoVern9);
}

#[test]
fn nonadaptive_stiff_component_is_rejected_before_problem_side_effects() {
    let rhs_calls = Rc::new(Cell::new(0));
    let callback_checks = Rc::new(Cell::new(0));
    let observed_rhs_calls = Rc::clone(&rhs_calls);
    let observed_callback_checks = Rc::clone(&callback_checks);
    let problem = OdeProblem::new(
        move |du: &mut [f64], u: &[f64], _: &(), _: f64| {
            observed_rhs_calls.set(observed_rhs_calls.get() + 1);
            du[0] = u[0];
        },
        vec![1.0],
        (0.0, 1.0),
        (),
    )
    .with_discrete_callback(
        move |_, _, _| {
            observed_callback_checks.set(observed_callback_checks.get() + 1);
            false
        },
        |_, _, _| CallbackAction::Continue,
    );

    let error = solve(
        &problem,
        AutoTsit5::new(RosenbrockW6S4OS),
        &SolveOptions::default(),
    )
    .unwrap_err();

    assert_eq!(
        error,
        SolveError::IncompatibleAutomaticPair {
            reason: AutomaticPairIncompatibility::NonAdaptiveComponent,
        }
    );
    assert_eq!(rhs_calls.get(), 0);
    assert_eq!(callback_checks.get(), 0);
}

#[test]
fn switch_back_requires_a_stiff_branch_diagnostic() {
    let error = solve(
        &problem(),
        AutoTsit5::new(Tsit5DA),
        &SolveOptions::default(),
    )
    .unwrap_err();

    assert_eq!(
        error,
        SolveError::IncompatibleAutomaticPair {
            reason: AutomaticPairIncompatibility::MissingStiffnessDiagnostic,
        }
    );

    let one_way = AutoTsit5::new(Tsit5DA)
        .with_switch_config(AutoSwitchConfig::new().with_allow_switch_back(false))
        .unwrap();
    assert!(solve(&problem(), one_way, &options()).is_ok());
}

#[test]
fn default_facades_delegate_to_their_native_components() {
    let options = options();
    let default = solve(&problem(), DefaultODEAlgorithm::default(), &options).unwrap();
    let explicit = solve(&problem(), Tsit5, &options).unwrap();
    assert_eq!(default.last_state(), explicit.last_state());

    let default_implicit =
        solve(&problem(), DefaultImplicitODEAlgorithm::default(), &options).unwrap();
    let stiff = solve(&problem(), Rodas5P, &options).unwrap();
    assert_eq!(default_implicit.last_state(), stiff.last_state());
}

#[test]
fn explicit_rk_resource_exposes_the_materialized_tableau() {
    assert_eq!(Dp5.tableau().unwrap().name(), "Dp5");
}
