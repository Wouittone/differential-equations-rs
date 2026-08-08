use std::alloc::System;
use std::hint::black_box;

use differential_equations::{Ab5, OdeProblem, SaveMode, SolveOptions, Vcab5, solve};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

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
    let region = Region::new(GLOBAL);
    let solution = solve(&exponential(), Ab5, &options).unwrap();
    black_box(solution.last_state());
    region.change().allocations
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
    let region = Region::new(GLOBAL);
    let solution = solve(&exponential(), Vcab5, &options).unwrap();
    black_box(solution.last_state());
    region.change().allocations
}

#[test]
fn adams_solve_allocations_are_invariant_with_step_count() {
    let one_fixed_step = fixed_adams_allocations(1.0);
    let thousand_fixed_steps = fixed_adams_allocations(0.001);
    assert_eq!(thousand_fixed_steps, one_fixed_step);
    assert!(
        one_fixed_step <= 25,
        "unexpected fixed Adams allocation count: {one_fixed_step}"
    );

    let few_variable_steps = variable_adams_allocations(1.0);
    let many_variable_steps = variable_adams_allocations(0.001);
    assert_eq!(many_variable_steps, few_variable_steps);
    assert!(
        few_variable_steps <= 50,
        "unexpected variable Adams allocation count: {few_variable_steps}"
    );
}
