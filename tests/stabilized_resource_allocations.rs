use std::alloc::System;
use std::hint::black_box;

use diffeq::tableau::{define_rock2_tableau_from_file, load_tableau};
use differential_equations as diffeq;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

define_rock2_tableau_from_file!(
    ROCK2_DEGREE_1,
    "ROCK2",
    1,
    "src/tableau/resources/methods/stabilized/rock2/degree-001.json",
    crate = diffeq
);
define_rock2_tableau_from_file!(
    ROCK2_DEGREE_22,
    "ROCK2",
    22,
    "src/tableau/resources/methods/stabilized/rock2/degree-022.json",
    crate = diffeq
);

#[test]
fn rock2_degree_resources_initialize_independently_and_cache() {
    let first = Region::new(GLOBAL);
    black_box(load_tableau(&ROCK2_DEGREE_1).unwrap());
    assert!(first.change().allocations > 0);

    // Loading degree 1 must not have initialized degree 22.
    let independent = Region::new(GLOBAL);
    black_box(load_tableau(&ROCK2_DEGREE_22).unwrap());
    assert!(independent.change().allocations > 0);

    let cached = Region::new(GLOBAL);
    for _ in 0..1_000 {
        black_box(load_tableau(&ROCK2_DEGREE_1).unwrap());
        black_box(load_tableau(&ROCK2_DEGREE_22).unwrap());
    }
    assert_eq!(cached.change().allocations, 0);
}
