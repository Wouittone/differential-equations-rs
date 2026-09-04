use std::alloc::System;
use std::hint::black_box;

use differential_equations::solvers::multirate::{
    MIS, MRIGARKERK22a, MRIGARKERK22b, MRIGARKERK33a, MRIGARKERK45a, MRIGARKESDIRK34a,
    MRIGARKIRK21a,
};
use differential_equations::tableau::MriTableau;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

#[test]
fn mri_tableaus_are_independently_lazy_and_cached() {
    let construction = Region::new(GLOBAL);
    black_box(MRIGARKERK22a::new(4));
    black_box(MRIGARKERK45a::new(4));
    assert_eq!(construction.change().allocations, 0);

    let mis_first = Region::new(GLOBAL);
    black_box(MIS::new(4).tableau().unwrap());
    assert!(mis_first.change().allocations > 0);
    let mis_cached = Region::new(GLOBAL);
    black_box(MIS::new(4).tableau().unwrap());
    assert_eq!(mis_cached.change().allocations, 0);

    let loaders: [fn() -> &'static MriTableau; 6] = [
        || black_box(MRIGARKERK22a::new(4).tableau().unwrap()),
        || black_box(MRIGARKERK22b::new(4).tableau().unwrap()),
        || black_box(MRIGARKERK33a::new(4).tableau().unwrap()),
        || black_box(MRIGARKERK45a::new(4).tableau().unwrap()),
        || black_box(MRIGARKESDIRK34a::new(4).tableau().unwrap()),
        || black_box(MRIGARKIRK21a::new(4).tableau().unwrap()),
    ];
    for load in loaders {
        let first = Region::new(GLOBAL);
        load();
        assert!(first.change().allocations > 0);
        let cached = Region::new(GLOBAL);
        load();
        assert_eq!(cached.change().allocations, 0);
    }
}
