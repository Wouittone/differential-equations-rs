use std::alloc::System;
use std::hint::black_box;

use differential_equations::algorithms::*;
use differential_equations::*;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

fn fixed_allocations(step: f64) -> usize {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -u[0],
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
    let solution = solve(&problem, Rodas23W, &options).unwrap();
    black_box(solution.last_state());
    region.change().allocations
}

#[test]
fn rodas23w_callback_free_steps_do_not_allocate_per_step() {
    let one_step = fixed_allocations(1.0);
    let thousand_steps = fixed_allocations(0.001);
    assert_eq!(thousand_steps, one_step);
    assert!(
        one_step <= 25,
        "unexpected Rodas23W allocation count: {one_step}"
    );
}
