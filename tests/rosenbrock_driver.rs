use std::alloc::System;
use std::hint::black_box;

use differential_equations::{
    OdeAlgorithm, OdeProblem, Rodas4, Rodas5P, Rosenbrock23, Rosenbrock32, RosenbrockW6S4OS,
    SaveMode, SolveOptions, solve,
};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

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
    let one_step = fixed_allocations_for(algorithm, 1.0);
    let thousand_steps = fixed_allocations_for(algorithm, 0.001);

    assert_eq!(
        thousand_steps, one_step,
        "Rosenbrock allocations grew with the step count"
    );
    assert!(
        one_step <= 25,
        "unexpected Rosenbrock solve allocation count: {one_step}"
    );
}

#[test]
fn callback_free_rosenbrock_steps_do_not_allocate_per_step() {
    assert_allocation_invariant(Rosenbrock23);
    assert_allocation_invariant(Rosenbrock32);
    assert_allocation_invariant(Rodas4);
    assert_allocation_invariant(Rodas5P);
    assert_allocation_invariant(RosenbrockW6S4OS);
}
