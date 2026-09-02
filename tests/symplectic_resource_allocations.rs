use std::alloc::System;
use std::hint::black_box;

use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[path = "support/allocation.rs"]
mod allocation_support;

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

mod first {
    use differential_equations::tableau::define_symplectic_from_file;
    define_symplectic_from_file!(pub FileDriftKick, "tests/resources/file_drift_kick.json");
}

mod second {
    use differential_equations::tableau::define_symplectic_from_file;
    define_symplectic_from_file!(pub FileDriftKick, "tests/resources/file_drift_kick.json");
}

#[test]
fn each_tableau_materializes_only_on_its_own_first_use() {
    let first_region = Region::new(GLOBAL);
    let first = first::FileDriftKick::tableau().unwrap();
    let first_allocations = first_region.change().allocations;
    assert!(first_allocations > 0);

    let second_region = Region::new(GLOBAL);
    let second = second::FileDriftKick::tableau().unwrap();
    let second_allocations = second_region.change().allocations;
    assert!(second_allocations > 0);
    assert!(!std::ptr::eq(first, second));

    let repeated = allocation_support::minimum_measurement(|| {
        let region = Region::new(GLOBAL);
        for _ in 0..1000 {
            black_box(first::FileDriftKick::tableau().unwrap());
            black_box(second::FileDriftKick::tableau().unwrap());
        }
        region.change().allocations
    });
    assert_eq!(repeated, 0);
}
