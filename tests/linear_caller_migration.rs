use std::alloc::System;
use std::hint::black_box;
use std::sync::Mutex;

use differential_equations::solvers::implicit::*;
use differential_equations::*;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[path = "support/allocation.rs"]
mod allocation_support;

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;
static TEST_LOCK: Mutex<()> = Mutex::new(());

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn problem() -> OdeProblem<TestRhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
        du[0] = -10.0 * u[0] + u[1];
        du[1] = -u[1];
    }
    OdeProblem::new(rhs, vec![1.0, 1.0], (0.0, 1.0), ())
}

fn options(step: f64) -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

fn endpoint<A: OdeAlgorithm + Copy>(algorithm: A) -> [f64; 2] {
    let solution = solve(&problem(), algorithm, &options(0.01)).unwrap();
    [solution.last_state()[0], solution.last_state()[1]]
}

fn implicit_methods_retain_compliance_endpoints() {
    assert_eq!(
        endpoint(ImplicitEuler),
        [4.114_352_645_070_348e-2, 3.697_112_123_291_194e-1]
    );
    assert_eq!(
        endpoint(ImplicitMidpoint),
        [4.091_517_292_423_622e-2, 3.678_763_754_762_209e-1]
    );
    assert_eq!(endpoint(Trapezoid), endpoint(ImplicitMidpoint));
}

fn allocations_for<A: OdeAlgorithm + Copy>(algorithm: A, step: f64) -> usize {
    allocation_support::minimum_measurement(|| {
        let region = Region::new(GLOBAL);
        let solution = solve(&problem(), algorithm, &options(step)).unwrap();
        black_box(solution.last_state());
        region.change().allocations
    })
}

#[test]
fn checked_linear_caller_allocations_are_step_invariant() {
    let _guard = TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    implicit_methods_retain_compliance_endpoints();
    fn assert_invariant<A: OdeAlgorithm + Copy>(algorithm: A) {
        let one_step = allocations_for(algorithm, 0.1);
        let thousand_steps = allocations_for(algorithm, 0.01);
        assert_eq!(
            thousand_steps, one_step,
            "implicit allocations grew with the step count"
        );
        assert!(
            one_step <= 30,
            "unexpected implicit solve allocation count: {one_step}"
        );
    }
    assert_invariant(ImplicitEuler);
    assert_invariant(ImplicitMidpoint);
    assert_invariant(Trapezoid);
}
