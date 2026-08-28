use std::alloc::System;
use std::hint::black_box;

use differential_equations::solvers::explicit::*;
use differential_equations::*;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

fn allocations_for<A: OdeAlgorithm + Copy>(algorithm: A, step: f64) -> usize {
    let problem = OdeProblem::new(
        |derivative: &mut [f64], state: &[f64], _: &(), _: f64| {
            derivative[0] = state[0];
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
    let solution = solve(&problem, algorithm, &options).unwrap();
    black_box(solution.last_state());
    region.change().allocations
}

#[test]
fn callback_free_low_storage_steps_do_not_allocate_per_step() {
    let hundred_steps = allocations_for(CarpenterKennedy2N54, 0.01);
    let thousand_steps = allocations_for(CarpenterKennedy2N54, 0.001);

    assert!(thousand_steps <= hundred_steps);
    assert!(
        hundred_steps <= 7,
        "unexpected low-storage solve allocation count: {hundred_steps}"
    );

    let hundred_steps = allocations_for(ParsaniKetchesonDeconinck3S32, 0.01);
    let thousand_steps = allocations_for(ParsaniKetchesonDeconinck3S32, 0.001);

    assert!(thousand_steps <= hundred_steps);
    assert!(
        hundred_steps <= 7,
        "unexpected 3S low-storage solve allocation count: {hundred_steps}"
    );

    let hundred_steps = allocations_for(ParsaniKetchesonDeconinck3S173, 0.01);
    let thousand_steps = allocations_for(ParsaniKetchesonDeconinck3S173, 0.001);
    assert!(thousand_steps <= hundred_steps);
    assert!(
        hundred_steps <= 7,
        "unexpected 3S173 low-storage solve allocation count: {hundred_steps}"
    );

    let hundred_steps = allocations_for(ParsaniKetchesonDeconinck3S53, 0.01);
    let thousand_steps = allocations_for(ParsaniKetchesonDeconinck3S53, 0.001);
    assert!(thousand_steps <= hundred_steps);
    assert!(
        hundred_steps <= 7,
        "unexpected 3S53 low-storage solve allocation count: {hundred_steps}"
    );

    let hundred_steps = allocations_for(ParsaniKetchesonDeconinck3S105, 0.01);
    let thousand_steps = allocations_for(ParsaniKetchesonDeconinck3S105, 0.001);
    assert!(thousand_steps <= hundred_steps);
    assert!(
        hundred_steps <= 7,
        "unexpected 3S105 low-storage solve allocation count: {hundred_steps}"
    );

    let hundred_steps = allocations_for(ParsaniKetchesonDeconinck3S82, 0.01);
    let thousand_steps = allocations_for(ParsaniKetchesonDeconinck3S82, 0.001);
    assert!(thousand_steps <= hundred_steps);
    assert!(
        hundred_steps <= 7,
        "unexpected 3S82 low-storage solve allocation count: {hundred_steps}"
    );

    let hundred_steps = allocations_for(ParsaniKetchesonDeconinck3S94, 0.01);
    let thousand_steps = allocations_for(ParsaniKetchesonDeconinck3S94, 0.001);
    assert!(thousand_steps <= hundred_steps);
    assert!(
        hundred_steps <= 7,
        "unexpected 3S94 low-storage solve allocation count: {hundred_steps}"
    );

    let hundred_steps = allocations_for(ParsaniKetchesonDeconinck3S184, 0.01);
    let thousand_steps = allocations_for(ParsaniKetchesonDeconinck3S184, 0.001);
    assert!(thousand_steps <= hundred_steps);
    assert!(
        hundred_steps <= 7,
        "unexpected 3S184 low-storage solve allocation count: {hundred_steps}"
    );

    let hundred_steps = allocations_for(ParsaniKetchesonDeconinck3S205, 0.01);
    let thousand_steps = allocations_for(ParsaniKetchesonDeconinck3S205, 0.001);
    assert!(thousand_steps <= hundred_steps);
    assert!(
        hundred_steps <= 7,
        "unexpected 3S205 low-storage solve allocation count: {hundred_steps}"
    );
}
