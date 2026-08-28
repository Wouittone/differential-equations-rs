use std::alloc::System;
use std::hint::black_box;

use differential_equations::solvers::multistep::Trbdf2;
use differential_equations::solvers::{explicit::*, implicit::*};
use differential_equations::*;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn rhs(derivative: &mut [f64], state: &[f64], _: &(), _: f64) {
    derivative[0] = state[0];
}

fn allocations_for(step: f64) -> usize {
    Rk4.tableau().unwrap();
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
    ImplicitEuler.tableau().unwrap();
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

fn minimum_allocations(mut measure: impl FnMut() -> usize) -> usize {
    (0..3)
        .map(|_| measure())
        .min()
        .expect("the allocation sample count is non-zero")
}

#[test]
fn callback_free_fixed_steps_do_not_allocate_per_step() {
    // `StatsAlloc` observes the entire process, including occasional test-harness
    // allocations from other threads. Taking the minimum of repeated samples
    // removes that bounded noise without hiding per-step allocation growth.
    let hundred_steps = minimum_allocations(|| allocations_for(0.01));
    let thousand_steps = minimum_allocations(|| allocations_for(0.001));

    assert!(
        thousand_steps <= hundred_steps,
        "fixed allocations grew with the step count: {hundred_steps} -> {thousand_steps}"
    );
    assert!(
        hundred_steps <= 7,
        "unexpected fixed solve allocation count: {hundred_steps}"
    );

    let hundred_implicit_steps = minimum_allocations(|| fixed_implicit_allocations_for(0.01));
    let thousand_implicit_steps = minimum_allocations(|| fixed_implicit_allocations_for(0.001));

    assert!(
        thousand_implicit_steps <= hundred_implicit_steps,
        "fixed implicit allocations grew with the step count: {hundred_implicit_steps} -> {thousand_implicit_steps}"
    );
    assert!(
        hundred_implicit_steps <= 20,
        "unexpected fixed implicit solve allocation count: {hundred_implicit_steps}"
    );

    let hundred_trbdf2_steps = minimum_allocations(|| adaptive_trbdf2_allocations_for(0.01));
    let many_trbdf2_steps = minimum_allocations(|| adaptive_trbdf2_allocations_for(0.001));

    assert!(
        many_trbdf2_steps <= hundred_trbdf2_steps,
        "adaptive TRBDF2 allocations grew with the step count: {hundred_trbdf2_steps} -> {many_trbdf2_steps}"
    );
    assert!(
        hundred_trbdf2_steps <= 25,
        "unexpected adaptive TRBDF2 solve allocation count: {hundred_trbdf2_steps}"
    );
}
