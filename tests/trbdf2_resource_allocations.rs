use std::alloc::System;
use std::hint::black_box;

use differential_equations::solvers::explicit::Rk4;
use differential_equations::solvers::multistep::Trbdf2;
use differential_equations::{OdeProblem, SaveMode, SolveOptions, solve};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[path = "support/allocation.rs"]
mod allocation_support;

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

#[test]
fn trbdf2_is_lazy_cached_and_does_not_allocate_per_step() {
    let construction = allocation_support::minimum_measurement(|| {
        let region = Region::new(GLOBAL);
        black_box(Trbdf2);
        region.change().allocations
    });
    assert_eq!(construction, 0);
    let first = Region::new(GLOBAL);
    black_box(Trbdf2.tableau().unwrap());
    assert!(first.change().allocations > 0);
    let unrelated = Region::new(GLOBAL);
    black_box(Rk4.tableau().unwrap());
    assert!(unrelated.change().allocations > 0);
    let repeated = allocation_support::minimum_measurement(|| {
        let region = Region::new(GLOBAL);
        for _ in 0..1000 {
            black_box(Trbdf2.tableau().unwrap());
        }
        region.change().allocations
    });
    assert_eq!(repeated, 0);
    let allocations = |step| {
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _| du[0] = -u[0],
            [1.0],
            (0.0, 1.0),
            (),
        );
        let options = SolveOptions::new()
            .with_adaptive(false)
            .with_initial_step(step)
            .with_save(SaveMode::Endpoints);
        allocation_support::minimum_measurement(|| {
            let region = Region::new(GLOBAL);
            black_box(solve(&problem, Trbdf2, &options).unwrap());
            region.change().allocations
        })
    };
    assert_eq!(allocations(0.01), allocations(0.001));
}
