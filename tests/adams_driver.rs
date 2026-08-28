use std::alloc::System;
use std::hint::black_box;

use differential_equations::solvers::multistep::*;
use differential_equations::*;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[path = "support/allocation.rs"]
mod allocation_support;

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn exponential() -> OdeProblem<TestRhs, ()> {
    fn rhs(derivative: &mut [f64], state: &[f64], _: &(), _: f64) {
        derivative[0] = state[0];
    }
    OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
}

fn fixed_adams_allocations(step: f64) -> usize {
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    allocation_support::minimum_measurement(|| {
        let region = Region::new(GLOBAL);
        let solution = solve(&exponential(), Ab5, &options).unwrap();
        black_box(solution.last_state());
        region.change().allocations
    })
}

fn variable_adams_allocations(maximum_step: f64) -> usize {
    let options = SolveOptions {
        absolute_tolerance: 1.0e-9,
        relative_tolerance: 1.0e-9,
        initial_step: Some(maximum_step),
        max_step: maximum_step,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    allocation_support::minimum_measurement(|| {
        let region = Region::new(GLOBAL);
        let solution = solve(&exponential(), Vcab5, &options).unwrap();
        black_box(solution.last_state());
        region.change().allocations
    })
}

#[test]
fn adams_solve_allocations_are_invariant_with_step_count() {
    // Compare two runs that both advance past the multistep startup phase.
    // A single-step solve legitimately skips the fixed-size history setup and
    // therefore is not a valid baseline for per-step allocation growth.
    let hundred_fixed_steps = fixed_adams_allocations(0.01);
    let thousand_fixed_steps = fixed_adams_allocations(0.001);
    assert!(thousand_fixed_steps <= hundred_fixed_steps);
    assert!(
        hundred_fixed_steps <= 25,
        "unexpected fixed Adams allocation count: {hundred_fixed_steps}"
    );

    let hundred_variable_steps = variable_adams_allocations(0.01);
    let many_variable_steps = variable_adams_allocations(0.001);
    assert!(many_variable_steps <= hundred_variable_steps);
    assert!(
        hundred_variable_steps <= 50,
        "unexpected variable Adams allocation count: {hundred_variable_steps}"
    );
}
