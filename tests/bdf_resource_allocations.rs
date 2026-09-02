use std::alloc::System;
use std::hint::black_box;

use differential_equations::solvers::multistep::{Ab3, Fbdf, Qbdf, Qbdf1, Qndf, Qndf1, Qndf2};
use differential_equations::{OdeProblem, SolveOptions, solve};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[path = "support/allocation.rs"]
mod allocation_support;

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

#[test]
fn bdf_orders_are_independent_lazy_resources_shared_by_all_consumers() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _| du[0] = -u[0],
        [1.0],
        (0.0, 0.1),
        (),
    );
    // Fixed stepping keeps the variable-order kernel at order one. It must
    // not materialize upper-order estimates used only by adaptive stepping.
    solve(
        &problem,
        Qndf,
        &SolveOptions::new()
            .with_adaptive(false)
            .with_initial_step(0.01),
    )
    .unwrap();

    let second = Region::new(GLOBAL);
    black_box(Qndf2.tableau().unwrap());
    assert!(second.change().allocations > 0);
    let fifth = Region::new(GLOBAL);
    black_box(Fbdf.tableau(5).unwrap());
    assert!(fifth.change().allocations > 0);
    let unrelated = Region::new(GLOBAL);
    black_box(Ab3.tableau().unwrap());
    assert!(unrelated.change().allocations > 0);

    let repeated = allocation_support::minimum_measurement(|| {
        let region = Region::new(GLOBAL);
        for _ in 0..1000 {
            black_box(Qndf.tableau(1).unwrap());
            black_box(Qbdf.tableau(1).unwrap());
            black_box(Fbdf.tableau(1).unwrap());
            black_box(Qndf1.tableau().unwrap());
            black_box(Qbdf1.tableau().unwrap());
        }
        region.change().allocations
    });
    assert_eq!(repeated, 0);
}
