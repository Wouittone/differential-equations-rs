use std::alloc::System;
use std::hint::black_box;

use differential_equations::algorithms::*;
use differential_equations::*;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn rhs(derivative: &mut [f64], state: &[f64], _: &(), _: f64) {
    derivative[0] = state[0];
}

fn allocations_for(step: f64) -> usize {
    let problem = OdeProblem::new(rhs as TestRhs, vec![1.0], (0.0, 1.0), ());
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let region = Region::new(GLOBAL);
    let solution = solve(&problem, Rk4, &options).unwrap();
    black_box(solution.last_state());
    region.change().allocations
}

fn fixed_implicit_allocations_for(step: f64) -> usize {
    let problem = OdeProblem::new(
        |derivative: &mut [f64], state: &[f64], _: &(), _: f64| {
            derivative[0] = -state[0];
        },
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
    let region = Region::new(GLOBAL);
    let solution = solve(&problem, ImplicitEuler, &options).unwrap();
    black_box(solution.last_state());
    region.change().allocations
}

fn adaptive_trbdf2_allocations_for(maximum_step: f64) -> usize {
    let problem = OdeProblem::new(
        |derivative: &mut [f64], state: &[f64], _: &(), _: f64| {
            derivative[0] = -state[0];
        },
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions {
        initial_step: Some(maximum_step),
        max_step: maximum_step,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let region = Region::new(GLOBAL);
    let solution = solve(&problem, Trbdf2, &options).unwrap();
    black_box(solution.last_state());
    region.change().allocations
}

#[test]
fn callback_free_fixed_steps_do_not_allocate_per_step() {
    let one_step = allocations_for(1.0);
    let thousand_steps = allocations_for(0.001);

    assert_eq!(thousand_steps, one_step);
    assert!(
        one_step <= 7,
        "unexpected fixed solve allocation count: {one_step}"
    );

    let one_implicit_step = fixed_implicit_allocations_for(1.0);
    let thousand_implicit_steps = fixed_implicit_allocations_for(0.001);

    assert_eq!(thousand_implicit_steps, one_implicit_step);
    assert!(
        one_implicit_step <= 20,
        "unexpected fixed implicit solve allocation count: {one_implicit_step}"
    );

    let few_trbdf2_steps = adaptive_trbdf2_allocations_for(1.0);
    let many_trbdf2_steps = adaptive_trbdf2_allocations_for(0.001);

    assert_eq!(many_trbdf2_steps, few_trbdf2_steps);
    assert!(
        few_trbdf2_steps <= 25,
        "unexpected adaptive TRBDF2 solve allocation count: {few_trbdf2_steps}"
    );
}
