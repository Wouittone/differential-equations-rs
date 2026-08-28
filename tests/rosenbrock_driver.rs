use std::alloc::System;
use std::hint::black_box;

use differential_equations::solvers::rosenbrock::*;
use differential_equations::*;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[path = "support/allocation.rs"]
mod allocation_support;

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

fn fixed_allocations_for<A: OdeAlgorithm>(algorithm: A, step: f64) -> usize {
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
    let solution = solve(&problem, algorithm, &options).unwrap();
    black_box(solution.last_state());
    region.change().allocations
}

fn assert_allocation_invariant<A: OdeAlgorithm + Copy>(algorithm: A) {
    // `StatsAlloc` is process-wide, so take the minimum of repeated samples to
    // exclude bounded test-harness noise while retaining per-step regressions.
    let one_step =
        allocation_support::minimum_measurement(|| fixed_allocations_for(algorithm, 1.0));
    let thousand_steps =
        allocation_support::minimum_measurement(|| fixed_allocations_for(algorithm, 0.001));

    assert_eq!(
        thousand_steps, one_step,
        "Rosenbrock allocations grew with the step count: {one_step} -> {thousand_steps}"
    );
    assert!(
        one_step <= 25,
        "unexpected Rosenbrock solve allocation count: {one_step}"
    );
}

#[test]
fn callback_free_rosenbrock_steps_do_not_allocate_per_step() {
    assert_allocation_invariant(Rosenbrock23);
    assert_allocation_invariant(Ros2);
    assert_allocation_invariant(Rosenbrock32);
    assert_allocation_invariant(Rodas3);
    assert_allocation_invariant(Rodas3d);
    assert_allocation_invariant(Rodas4);
    assert_allocation_invariant(Grk4a);
    assert_allocation_invariant(Ros34Pw1b);
    assert_allocation_invariant(Rodas5P);
    assert_allocation_invariant(Ros34Prw);
    assert_allocation_invariant(Ros3Prl);
    assert_allocation_invariant(RosenbrockW6S4OS);
}
