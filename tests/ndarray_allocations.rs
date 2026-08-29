use std::alloc::System;
use std::hint::black_box;

use differential_equations::ndarray::{ArrayView2, ArrayViewMut2, array};
use differential_equations::solvers::explicit::Rk4;
use differential_equations::{OdeProblem, SaveMode, SolveOptions, solve};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[path = "support/allocation.rs"]
mod allocation_support;

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

fn allocations_for(step: f64) -> usize {
    Rk4.tableau().unwrap();
    let problem = OdeProblem::from_array(
        |mut derivative: ArrayViewMut2<'_, f64>, state: ArrayView2<'_, f64>, _: &(), _: f64| {
            derivative.assign(&state);
        },
        array![[1.0, 2.0], [3.0, 4.0]],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(step)
        .with_save(SaveMode::Endpoints);

    allocation_support::minimum_measurement(|| {
        let region = Region::new(GLOBAL);
        let solution = solve(&problem, Rk4, &options).unwrap();
        black_box(solution.last_state_array());
        region.change().allocations
    })
}

#[test]
fn ndarray_views_do_not_add_per_step_allocations() {
    let hundred_steps = allocations_for(0.01);
    let thousand_steps = allocations_for(0.001);
    assert!(
        thousand_steps <= hundred_steps,
        "ndarray adapter allocations grew with step count: {hundred_steps} -> {thousand_steps}"
    );
}
