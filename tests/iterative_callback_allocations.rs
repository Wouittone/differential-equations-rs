use std::alloc::System;
use std::hint::black_box;

use differential_equations::callbacks::IterativeCallback;
use differential_equations::solvers::explicit::Euler;
use differential_equations::{
    CallbackAction, CallbackSave, OdeProblem, SaveMode, SolveOptions, solve,
};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[path = "support/allocation.rs"]
mod allocation_support;

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

fn allocations_for(event_count: usize) -> usize {
    Euler.tableau().unwrap();
    let callbacks = IterativeCallback::new(|_: &[f64], _: &(), time| Some(time + 1.0))
        .with_save(CallbackSave::None)
        .into_callback_set((0.0, event_count as f64), |_, _, _| {
            CallbackAction::ContinueUnmodified
        })
        .unwrap();
    let problem = OdeProblem::new(
        |du: &mut [f64], _: &[f64], _: &(), _| du.fill(0.0),
        [0.0],
        (0.0, event_count as f64),
        (),
    )
    .with_callback_set(callbacks);
    let options = SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(2.0)
        .with_save(SaveMode::Endpoints);
    allocation_support::minimum_measurement(|| {
        let region = Region::new(GLOBAL);
        let solution = solve(&problem, Euler, &options).unwrap();
        assert_eq!(solution.stats().callback_invocations, event_count);
        black_box(solution.last_state());
        region.change().allocations
    })
}

#[test]
fn scheduling_allocations_do_not_grow_with_event_count() {
    let hundred = allocations_for(100);
    let thousand = allocations_for(1000);
    assert!(
        thousand <= hundred,
        "allocation count grew: {hundred} -> {thousand}"
    );
}
