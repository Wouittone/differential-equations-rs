use std::alloc::System;
use std::hint::black_box;

use differential_equations::algorithms::*;
use differential_equations::*;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

type ExponentialProblem = OdeProblem<fn(&mut [f64], &[f64], &(), f64), ()>;

fn exponential_rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
    du[0] = u[0];
}

fn exponential_problem(span: (f64, f64), initial: f64) -> ExponentialProblem {
    OdeProblem::new(exponential_rhs, vec![initial], span, ())
}

fn fixed_options(step: f64) -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

fn allocations(step: f64) -> usize {
    let region = Region::new(GLOBAL);
    let solution = solve(
        &exponential_problem((0.0, 1.0), 1.0),
        Ros34Pw3,
        &fixed_options(step),
    )
    .unwrap();
    black_box(solution.last_state());
    region.change().allocations
}

#[test]
fn callback_free_steps_do_not_allocate_per_step() {
    let hundred_steps = allocations(0.01);
    let many_steps = allocations(0.001);
    assert!(
        many_steps <= hundred_steps + 2,
        "step allocations grew unexpectedly: hundred_steps={hundred_steps}, many_steps={many_steps}"
    );
    assert!(
        hundred_steps <= 60,
        "unexpected allocation count: {hundred_steps}"
    );
}
