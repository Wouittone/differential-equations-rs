use std::alloc::System;
use std::hint::black_box;

use differential_equations::solvers::multistep::Abdf2;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

#[test]
fn abdf2_tableau_is_lazy_and_cached() {
    let construction = Region::new(GLOBAL);
    black_box(Abdf2);
    assert_eq!(construction.change().allocations, 0);

    let first = Region::new(GLOBAL);
    black_box(Abdf2.tableau().unwrap());
    assert!(first.change().allocations > 0);

    let cached = Region::new(GLOBAL);
    black_box(Abdf2.tableau().unwrap());
    assert_eq!(cached.change().allocations, 0);
}
