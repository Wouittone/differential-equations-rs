use std::alloc::System;
use std::hint::black_box;

use differential_equations::algorithms::*;
use differential_equations::*;
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
    let solution = solve(&problem, Kyk2014DgSsprk3S2, &options).unwrap();
    black_box(solution.last_state());
    region.change().allocations
}

#[test]
fn fixed_steps_do_not_allocate_per_step() {
    let hundred_steps = allocations_for(0.01);
    let thousand_steps = allocations_for(0.001);
    assert!(thousand_steps <= hundred_steps);
    assert!(
        hundred_steps <= 7,
        "unexpected KYK2014 allocation count: {hundred_steps}"
    );
}
