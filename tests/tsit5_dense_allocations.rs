use std::alloc::System;
use std::hint::black_box;

use differential_equations::solvers::explicit::*;
use differential_equations::*;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

fn allocations_for(step: f64, retain_dense_output: bool) -> usize {
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
        retain_dense_output,
        ..SolveOptions::default()
    };
    let region = Region::new(GLOBAL);
    let solution = solve(&problem, Tsit5, &options).unwrap();
    black_box(solution.last_state());
    region.change().allocations
}

#[test]
fn accepted_step_segment_allocations_are_gated_by_dense_retention() {
    assert!(allocations_for(0.01, true) > allocations_for(0.01, false));
}
