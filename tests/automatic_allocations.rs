use std::alloc::System;
use std::hint::black_box;

use differential_equations::solvers::automatic::{AutoDp5, AutoSwitchConfig, AutomaticBranch};
use differential_equations::solvers::explicit::Dp5;
use differential_equations::solvers::rosenbrock::Rodas5P;
use differential_equations::{OdeProblem, SaveMode, SolveOptions, solve};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[path = "support/allocation.rs"]
mod allocation_support;

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

fn fixed_options(step: f64) -> SolveOptions {
    SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(step)
        .with_max_step(step)
        .with_save(SaveMode::Endpoints)
}

fn never_switch_config() -> AutoSwitchConfig {
    AutoSwitchConfig::new()
        .with_stiffness_thresholds(1.0e100, 1.0e200)
        .expect("the thresholds are ordered")
}

fn switch_after_one_step_config() -> AutoSwitchConfig {
    AutoSwitchConfig::new()
        .with_stiffness_thresholds(1.0e-14, 1.0e-12)
        .expect("the thresholds are ordered")
        .with_stiff_confirmations(1)
        .expect("one confirmation is valid")
        .with_minimum_residence_steps(1)
        .expect("one residence step is valid")
        .with_switch_step_factor(1.0)
        .expect("a unit step factor is valid")
        .with_allow_switch_back(false)
}

fn never_switch_bytes(dimension: usize) -> usize {
    let problem = OdeProblem::new(
        |derivative: &mut [f64], state: &[f64], _: &(), _: f64| {
            for (derivative, state) in derivative.iter_mut().zip(state) {
                *derivative = -*state;
            }
        },
        vec![1.0; dimension],
        (0.0, 0.02),
        (),
    )
    .with_jacobian(|_, _, _, _| panic!("the inactive stiff branch must not be initialized"));
    let options = fixed_options(0.01);
    let algorithm = AutoDp5::new(Rodas5P)
        .with_switch_config(never_switch_config())
        .expect("the automatic configuration is valid");

    allocation_support::minimum_measurement(|| {
        let region = Region::new(GLOBAL);
        let solution = solve(&problem, algorithm.clone(), &options).unwrap();
        let bytes = region.change().bytes_allocated;

        assert_eq!(solution.stats().algorithm_switches, 0);
        assert_eq!(solution.stats().stiff_accepted_steps, 0);
        assert_eq!(solution.stats().jacobian_evaluations, 0);
        assert_eq!(solution.stats().linear_factorizations, 0);
        assert_eq!(
            solution.stats().final_automatic_branch,
            Some(AutomaticBranch::NonStiff)
        );
        black_box(solution);
        bytes
    })
}

fn switched_allocations(step: f64) -> usize {
    let problem = OdeProblem::new(
        |derivative: &mut [f64], state: &[f64], _: &(), _: f64| {
            derivative[0] = -state[0];
        },
        [1.0],
        (0.0, 1.0),
        (),
    );
    let options = fixed_options(step);
    let algorithm = AutoDp5::new(Rodas5P)
        .with_switch_config(switch_after_one_step_config())
        .expect("the automatic configuration is valid");

    allocation_support::minimum_measurement(|| {
        let region = Region::new(GLOBAL);
        let solution = solve(&problem, algorithm.clone(), &options).unwrap();
        let allocations = region.change().allocations;

        assert_eq!(solution.stats().algorithm_switches, 1);
        assert_eq!(solution.stats().nonstiff_accepted_steps, 1);
        assert!(solution.stats().stiff_accepted_steps > 0);
        assert_eq!(
            solution.stats().final_automatic_branch,
            Some(AutomaticBranch::Stiff)
        );
        black_box(solution);
        allocations
    })
}

fn repeated_switch_allocations(step: f64) -> usize {
    let problem = OdeProblem::new(
        |derivative: &mut [f64], state: &[f64], _: &(), time: f64| {
            let rate = if (0.25..0.75).contains(&time) {
                100.0
            } else {
                1.0
            };
            derivative[0] = -rate * (state[0] - time.cos()) - time.sin();
        },
        [1.0],
        (0.0, 1.0),
        (),
    )
    .with_jacobian(|jacobian, _, _, time| {
        jacobian[0] = if (0.25..0.75).contains(&time) {
            -100.0
        } else {
            -1.0
        };
    });
    let options = fixed_options(step).with_time_stops([0.25, 0.75]);
    let config = AutoSwitchConfig::new()
        .with_stiffness_thresholds(0.01, 0.02)
        .expect("the thresholds are ordered")
        .with_stiff_confirmations(1)
        .expect("one confirmation is valid")
        .with_nonstiff_confirmations(1)
        .expect("one confirmation is valid")
        .with_minimum_residence_steps(1)
        .expect("one residence step is valid")
        .with_switch_step_factor(1.0)
        .expect("a unit step factor is valid");
    let algorithm = AutoDp5::new(Rodas5P)
        .with_switch_config(config)
        .expect("the automatic configuration is valid");

    allocation_support::minimum_measurement(|| {
        let region = Region::new(GLOBAL);
        let solution = solve(&problem, algorithm.clone(), &options).unwrap();
        let allocations = region.change().allocations;

        assert_eq!(solution.stats().algorithm_switches, 2);
        assert!(solution.stats().nonstiff_accepted_steps > 0);
        assert!(solution.stats().stiff_accepted_steps > 0);
        assert_eq!(
            solution.stats().final_automatic_branch,
            Some(AutomaticBranch::NonStiff)
        );
        black_box(solution);
        allocations
    })
}

#[test]
fn automatic_branch_caches_are_lazy_and_do_not_allocate_per_step() {
    // Exclude process-lifetime resource parsing from the cache measurements.
    black_box(Dp5.tableau().unwrap());
    black_box(Rodas5P.tableau().unwrap());

    // A dormant Rosenbrock cache contains dense n-by-n work matrices. Linear
    // memory growth therefore demonstrates that a never-selected stiff branch
    // is not constructed. The panic-on-use Jacobian and zero implicit work
    // statistics independently guard against accidental initialization.
    let small_nonstiff = never_switch_bytes(512);
    let large_nonstiff = never_switch_bytes(1024);
    assert!(
        large_nonstiff <= 2 * small_nonstiff + 64 * 1024,
        "inactive stiff cache caused super-linear allocation growth: {small_nonstiff} -> {large_nonstiff} bytes"
    );

    // The first accepted explicit step forces exactly one transition and thus
    // materializes the stiff cache. Once present, taking ten times as many
    // stiff steps must not introduce allocation growth.
    let hundred_steps = switched_allocations(0.01);
    let thousand_steps = switched_allocations(0.001);
    assert!(
        thousand_steps <= hundred_steps,
        "switched automatic solve allocated per step: {hundred_steps} -> {thousand_steps}"
    );

    // Both branch workspaces are reused after a complete explicit -> stiff ->
    // explicit cycle. Taking ten times as many steps through the same two
    // transitions must not add step-proportional allocation.
    let repeated_hundred_steps = repeated_switch_allocations(0.01);
    let repeated_thousand_steps = repeated_switch_allocations(0.001);
    assert!(
        repeated_thousand_steps <= repeated_hundred_steps,
        "re-entered automatic caches allocated per step: {repeated_hundred_steps} -> {repeated_thousand_steps}"
    );
}
