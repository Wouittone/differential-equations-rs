use std::alloc::System;
use std::hint::black_box;

use differential_equations::{OdeProblem, SaveMode, SolveOptions, SspRk432, solve};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

fn allocations_for(step: f64) -> usize {
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
    let solution = solve(&problem, SspRk432, &options).unwrap();
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
        "unexpected SSPRK432 allocation count: {one_step}"
    );
}
