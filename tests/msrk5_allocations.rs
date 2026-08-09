use std::alloc::System;
use std::hint::black_box;

use differential_equations::{Msrk5, OdeProblem, SaveMode, SolveOptions, solve};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

fn allocations_for(step: f64) -> usize {
    let problem = OdeProblem::new(
        |du: &mut [f64], state: &[f64], _: &(), _: f64| du[0] = state[0],
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
    let solution = solve(&problem, Msrk5, &options).unwrap();
    black_box(solution.last_state());
    region.change().allocations
}

#[test]
fn msrk5_callback_free_allocations_are_step_invariant() {
    let one_step = allocations_for(1.0);
    let many_steps = allocations_for(0.001);
    assert_eq!(many_steps, one_step);
    assert!(
        one_step <= 12,
        "unexpected Msrk5 allocation count: {one_step}"
    );
}
