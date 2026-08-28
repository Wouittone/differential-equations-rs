use std::alloc::System;
use std::hint::black_box;

use differential_equations::solvers::explicit::*;
use differential_equations::*;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

fn rhs(derivative: &mut [f64], state: &[f64], _: &(), _: f64) {
    derivative[0] = state[0];
}

fn allocations_for(step: f64) -> usize {
    let problem = OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ());
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    Vern6.tableau().unwrap();
    let region = Region::new(GLOBAL);
    let solution = solve(&problem, Vern6, &options).unwrap();
    black_box(solution.last_state());
    region.change().allocations
}

#[test]
fn callback_free_vern6_steps_do_not_allocate_per_step() {
    let hundred_steps = allocations_for(0.01);
    let thousand_steps = allocations_for(0.001);

    assert!(thousand_steps <= hundred_steps);
    assert!(
        hundred_steps <= 10,
        "unexpected Vern6 solve allocation count: {hundred_steps}"
    );
}
