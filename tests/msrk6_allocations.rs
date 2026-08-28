use std::alloc::System;
use std::hint::black_box;

use differential_equations::solvers::explicit::*;
use differential_equations::*;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

fn allocations_for(step: f64) -> usize {
    Msrk6.tableau().unwrap();
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
    let solution = solve(&problem, Msrk6, &options).unwrap();
    black_box(solution.last_state());
    region.change().allocations
}

#[test]
fn msrk6_callback_free_allocations_are_step_invariant() {
    let hundred_steps = allocations_for(0.01);
    let many_steps = allocations_for(0.001);
    assert!(many_steps <= hundred_steps);
    assert!(
        hundred_steps <= 12,
        "unexpected Msrk6 allocation count: {hundred_steps}"
    );
}
