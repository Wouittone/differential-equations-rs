use std::alloc::System;
use std::hint::black_box;

use differential_equations::callbacks::GeneralDomain;
use differential_equations::solvers::explicit::Euler;
use differential_equations::{CallbackSave, OdeProblem, SaveMode, SolveOptions, solve};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[path = "support/allocation.rs"]
mod allocation_support;

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

fn allocations_for(step: f64) -> usize {
    Euler.tableau().unwrap();
    let callbacks = GeneralDomain::new(1, |r: &mut [f64], u: &[f64], _: &(), _| {
        r[0] = u[0] * u[0] + u[1] * u[1] - 1.0;
    })
    .with_absolute_tolerance(1.0e-2)
    .with_save(CallbackSave::None)
    .into_callback_set()
    .unwrap();
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _| {
            du[0] = u[1];
            du[1] = -u[0];
        },
        [1.0, 0.0],
        (0.0, 1.0),
        (),
    )
    .with_callback_set(callbacks);
    let options = SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(step)
        .with_save(SaveMode::Endpoints);
    allocation_support::minimum_measurement(|| {
        let region = Region::new(GLOBAL);
        let solution = solve(&problem, Euler, &options).unwrap();
        black_box(solution.last_state());
        region.change().allocations
    })
}

#[test]
fn prediction_and_projection_allocations_do_not_grow_with_step_count() {
    let hundred = allocations_for(0.01);
    let thousand = allocations_for(0.001);
    assert!(
        thousand <= hundred,
        "allocation count grew: {hundred} -> {thousand}"
    );
}
