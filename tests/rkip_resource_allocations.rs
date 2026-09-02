use std::alloc::System;
use std::hint::black_box;

use differential_equations::solvers::exponential::RKIP;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[path = "support/allocation.rs"]
mod allocation_support;

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

#[test]
fn creating_solvers_does_not_materialize_the_tableau() {
    let first = RKIP::default();
    let second = RKIP::new(0.01, 0.1, 2).unwrap();
    let first_use = Region::new(GLOBAL);
    let tableau = first.tableau().unwrap();
    assert!(first_use.change().allocations > 0);
    assert!(std::ptr::eq(tableau, second.tableau().unwrap()));
    let repeated = allocation_support::minimum_measurement(|| {
        let region = Region::new(GLOBAL);
        for _ in 0..1000 {
            black_box(first.tableau().unwrap());
            black_box(second.tableau().unwrap());
        }
        region.change().allocations
    });
    assert_eq!(repeated, 0);
}
