use std::alloc::System;
use std::hint::black_box;

use differential_equations::solvers::second_order::{
    Dprkn4, Dprkn5, Irkn3, Nystrom4, Nystrom4VelocityIndependent,
};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

#[test]
fn second_order_tableaus_initialize_independently_and_cache_without_allocating() {
    let construction = Region::new(GLOBAL);
    black_box((Dprkn4, Dprkn5, Irkn3, Nystrom4, Nystrom4VelocityIndependent));
    assert_eq!(construction.change().allocations, 0);

    let first = Region::new(GLOBAL);
    let dprkn4 = black_box(Dprkn4.tableau().unwrap());
    assert!(first.change().allocations > 0);

    let second = Region::new(GLOBAL);
    let dprkn5 = black_box(Dprkn5.tableau().unwrap());
    assert!(second.change().allocations > 0);
    assert!(!std::ptr::eq(dprkn4, dprkn5));

    let history = Region::new(GLOBAL);
    black_box(Irkn3.tableau().unwrap());
    assert!(history.change().allocations > 0);

    let fixed = Region::new(GLOBAL);
    black_box(Nystrom4.tableau().unwrap());
    assert!(fixed.change().allocations > 0);

    // Inspecting an IRKN history tableau must not eagerly initialize the
    // independent one-step tableau used later by the IRKN solver bootstrap.
    let bootstrap = Region::new(GLOBAL);
    black_box(Nystrom4VelocityIndependent.tableau().unwrap());
    assert!(bootstrap.change().allocations > 0);

    let cached = Region::new(GLOBAL);
    for _ in 0..1_000 {
        black_box(Dprkn4.tableau().unwrap());
        black_box(Dprkn5.tableau().unwrap());
        black_box(Irkn3.tableau().unwrap());
        black_box(Nystrom4.tableau().unwrap());
        black_box(Nystrom4VelocityIndependent.tableau().unwrap());
    }
    assert_eq!(cached.change().allocations, 0);
}
