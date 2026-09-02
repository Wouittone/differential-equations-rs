use std::alloc::System;
use std::hint::black_box;

use differential_equations::callbacks::TerminateSteadyState;
use differential_equations::solvers::explicit::Euler;
use differential_equations::{OdeProblem, SaveMode, SolveOptions, solve};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[path = "support/allocation.rs"]
mod allocation_support;

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

fn allocations_for(step: f64) -> usize {
    Euler.tableau().unwrap();
    let problem = OdeProblem::new(
        |du: &mut [f64], _: &[f64], _: &(), _| du.fill(1.0),
        [0.0],
        (0.0, 1.0),
        (),
    )
    .with_callback_set(TerminateSteadyState::new().into_callback_set().unwrap());
    let options = SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(step)
        .with_save(SaveMode::Endpoints);
    allocation_support::minimum_measurement(|| {
        let region = Region::new(GLOBAL);
        let solution = solve(&problem, Euler, &options).unwrap();
        assert_eq!(solution.stats().callback_invocations, 0);
        black_box(solution.last_state());
        region.change().allocations
    })
}

#[test]
fn derivative_checks_do_not_allocate_per_step() {
    let hundred = allocations_for(0.01);
    let thousand = allocations_for(0.001);
    assert!(
        thousand <= hundred,
        "allocation count grew: {hundred} -> {thousand}"
    );
}
