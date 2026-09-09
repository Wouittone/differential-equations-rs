use std::alloc::System;
use std::hint::black_box;

use diffeq::tableau::{
    define_eserk_tableau_from_file, define_rock2_tableau_from_file, define_rock4_tableau_from_file,
    define_serk2_tableau_from_file, load_tableau,
};
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
define_eserk_tableau_from_file!(
    ESERK4_DEGREE_2,
    "ESERK4",
    4,
    2,
    "src/tableau/resources/methods/stabilized/eserk4/degree-0002.json",
    crate = diffeq
);
define_eserk_tableau_from_file!(
    ESERK4_DEGREE_4000,
    "ESERK4",
    4,
    4000,
    "src/tableau/resources/methods/stabilized/eserk4/degree-4000.json",
    crate = diffeq
);
define_eserk_tableau_from_file!(
    ESERK5_DEGREE_1,
    "ESERK5",
    5,
    1,
    "src/tableau/resources/methods/stabilized/eserk5/degree-0001.json",
    crate = diffeq
);
define_eserk_tableau_from_file!(
    ESERK5_DEGREE_2000,
    "ESERK5",
    5,
    2000,
    "src/tableau/resources/methods/stabilized/eserk5/degree-2000.json",
    crate = diffeq
);
define_serk2_tableau_from_file!(
    SERK2_DEGREE_10,
    "SERK2",
    10,
    "src/tableau/resources/methods/stabilized/serk2/degree-010.json",
    crate = diffeq
);
define_serk2_tableau_from_file!(
    SERK2_DEGREE_100,
    "SERK2",
    100,
    "src/tableau/resources/methods/stabilized/serk2/degree-100.json",
    crate = diffeq
);
define_rock4_tableau_from_file!(
    ROCK4_DEGREE_1,
    "ROCK4",
    1,
    "src/tableau/resources/methods/stabilized/rock4/degree-001.json",
    crate = diffeq
);
define_rock4_tableau_from_file!(
    ROCK4_DEGREE_22,
    "ROCK4",
    22,
    "src/tableau/resources/methods/stabilized/rock4/degree-022.json",
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
fn stabilized_degree_resources_initialize_independently_and_cache() {
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

    let references = Region::new(GLOBAL);
    black_box(&ROCK4_DEGREE_1);
    black_box(&ROCK4_DEGREE_22);
    assert_eq!(references.change().allocations, 0);

    let first = Region::new(GLOBAL);
    black_box(load_tableau(&ROCK4_DEGREE_1).unwrap());
    assert!(first.change().allocations > 0);

    // Loading degree 1 must not have initialized degree 22.
    let independent = Region::new(GLOBAL);
    black_box(load_tableau(&ROCK4_DEGREE_22).unwrap());
    assert!(independent.change().allocations > 0);

    let cached = Region::new(GLOBAL);
    for _ in 0..1_000 {
        black_box(load_tableau(&ROCK4_DEGREE_1).unwrap());
        black_box(load_tableau(&ROCK4_DEGREE_22).unwrap());
    }
    assert_eq!(cached.change().allocations, 0);

    let references = Region::new(GLOBAL);
    black_box(&SERK2_DEGREE_10);
    black_box(&SERK2_DEGREE_100);
    assert_eq!(references.change().allocations, 0);

    let first = Region::new(GLOBAL);
    black_box(load_tableau(&SERK2_DEGREE_10).unwrap());
    assert!(first.change().allocations > 0);

    // Loading degree 10 must not initialize degree 100.
    let independent = Region::new(GLOBAL);
    black_box(load_tableau(&SERK2_DEGREE_100).unwrap());
    assert!(independent.change().allocations > 0);

    let cached = Region::new(GLOBAL);
    for _ in 0..1_000 {
        black_box(load_tableau(&SERK2_DEGREE_10).unwrap());
        black_box(load_tableau(&SERK2_DEGREE_100).unwrap());
    }
    assert_eq!(cached.change().allocations, 0);

    let references = Region::new(GLOBAL);
    black_box(&ESERK4_DEGREE_2);
    black_box(&ESERK4_DEGREE_4000);
    black_box(&ESERK5_DEGREE_1);
    black_box(&ESERK5_DEGREE_2000);
    assert_eq!(references.change().allocations, 0);

    let first = Region::new(GLOBAL);
    black_box(load_tableau(&ESERK4_DEGREE_2).unwrap());
    assert!(first.change().allocations > 0);

    let independent = Region::new(GLOBAL);
    black_box(load_tableau(&ESERK4_DEGREE_4000).unwrap());
    assert!(independent.change().allocations > 0);

    let first = Region::new(GLOBAL);
    black_box(load_tableau(&ESERK5_DEGREE_1).unwrap());
    assert!(first.change().allocations > 0);

    let independent = Region::new(GLOBAL);
    black_box(load_tableau(&ESERK5_DEGREE_2000).unwrap());
    assert!(independent.change().allocations > 0);

    let cached = Region::new(GLOBAL);
    for _ in 0..1_000 {
        black_box(load_tableau(&ESERK4_DEGREE_2).unwrap());
        black_box(load_tableau(&ESERK4_DEGREE_4000).unwrap());
        black_box(load_tableau(&ESERK5_DEGREE_1).unwrap());
        black_box(load_tableau(&ESERK5_DEGREE_2000).unwrap());
    }
    assert_eq!(cached.change().allocations, 0);
}
