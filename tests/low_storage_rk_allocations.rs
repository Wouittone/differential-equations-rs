use std::alloc::System;
use std::hint::black_box;

use differential_equations::solvers::explicit::*;
use differential_equations::*;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[path = "support/allocation.rs"]
mod allocation_support;

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

fn allocations_for<A: OdeAlgorithm + Copy>(algorithm: A, step: f64) -> usize {
    let problem = OdeProblem::new(
        |derivative: &mut [f64], state: &[f64], _: &(), _: f64| {
            derivative[0] = state[0];
        },
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    allocation_support::minimum_measurement(|| {
        let region = Region::new(GLOBAL);
        let solution = solve(&problem, algorithm, &options).unwrap();
        black_box(solution.last_state());
        region.change().allocations
    })
}

#[test]
fn low_storage_resources_are_individually_lazy_and_steps_do_not_allocate() {
    let construction = allocation_support::minimum_measurement(|| {
        let region = Region::new(GLOBAL);
        black_box(Ork256);
        black_box(CFRLDDRK64);
        black_box(RDPK3Sp35);
        black_box(SHLDDRK_2N);
        black_box(CKLLSRK43_2);
        region.change().allocations
    });
    assert_eq!(construction, 0);

    let first = Region::new(GLOBAL);
    black_box(Ork256.tableau().unwrap());
    assert!(first.change().allocations > 0);
    let independent = Region::new(GLOBAL);
    black_box(CFRLDDRK64.tableau().unwrap());
    assert!(independent.change().allocations > 0);
    let alternating = Region::new(GLOBAL);
    black_box(SHLDDRK_2N.tableau().unwrap());
    assert!(alternating.change().allocations > 0);
    let three_s = Region::new(GLOBAL);
    black_box(RDPK3Sp35.tableau().unwrap());
    assert!(three_s.change().allocations > 0);
    let pipeline = Region::new(GLOBAL);
    black_box(CKLLSRK43_2.tableau().unwrap());
    assert!(pipeline.change().allocations > 0);
    let repeated = allocation_support::minimum_measurement(|| {
        let region = Region::new(GLOBAL);
        for _ in 0..1000 {
            black_box(Ork256.tableau().unwrap());
            black_box(CFRLDDRK64.tableau().unwrap());
            black_box(SHLDDRK_2N.tableau().unwrap());
            black_box(RDPK3Sp35.tableau().unwrap());
            black_box(CKLLSRK43_2.tableau().unwrap());
        }
        region.change().allocations
    });
    assert_eq!(repeated, 0);

    let hundred_steps = allocations_for(CarpenterKennedy2N54, 0.01);
    let thousand_steps = allocations_for(CarpenterKennedy2N54, 0.001);

    assert!(thousand_steps <= hundred_steps);
    assert!(
        hundred_steps <= 7,
        "unexpected low-storage solve allocation count: {hundred_steps}"
    );

    let hundred_steps = allocations_for(ParsaniKetchesonDeconinck3S32, 0.01);
    let thousand_steps = allocations_for(ParsaniKetchesonDeconinck3S32, 0.001);

    assert!(thousand_steps <= hundred_steps);
    assert!(
        hundred_steps <= 7,
        "unexpected 3S low-storage solve allocation count: {hundred_steps}"
    );

    let hundred_steps = allocations_for(ParsaniKetchesonDeconinck3S173, 0.01);
    let thousand_steps = allocations_for(ParsaniKetchesonDeconinck3S173, 0.001);
    assert!(thousand_steps <= hundred_steps);
    assert!(
        hundred_steps <= 7,
        "unexpected 3S173 low-storage solve allocation count: {hundred_steps}"
    );

    let hundred_steps = allocations_for(ParsaniKetchesonDeconinck3S53, 0.01);
    let thousand_steps = allocations_for(ParsaniKetchesonDeconinck3S53, 0.001);
    assert!(thousand_steps <= hundred_steps);
    assert!(
        hundred_steps <= 7,
        "unexpected 3S53 low-storage solve allocation count: {hundred_steps}"
    );

    let hundred_steps = allocations_for(ParsaniKetchesonDeconinck3S105, 0.01);
    let thousand_steps = allocations_for(ParsaniKetchesonDeconinck3S105, 0.001);
    assert!(thousand_steps <= hundred_steps);
    assert!(
        hundred_steps <= 7,
        "unexpected 3S105 low-storage solve allocation count: {hundred_steps}"
    );

    let hundred_steps = allocations_for(ParsaniKetchesonDeconinck3S82, 0.01);
    let thousand_steps = allocations_for(ParsaniKetchesonDeconinck3S82, 0.001);
    assert!(thousand_steps <= hundred_steps);
    assert!(
        hundred_steps <= 7,
        "unexpected 3S82 low-storage solve allocation count: {hundred_steps}"
    );

    let hundred_steps = allocations_for(ParsaniKetchesonDeconinck3S94, 0.01);
    let thousand_steps = allocations_for(ParsaniKetchesonDeconinck3S94, 0.001);
    assert!(thousand_steps <= hundred_steps);
    assert!(
        hundred_steps <= 7,
        "unexpected 3S94 low-storage solve allocation count: {hundred_steps}"
    );

    let hundred_steps = allocations_for(ParsaniKetchesonDeconinck3S184, 0.01);
    let thousand_steps = allocations_for(ParsaniKetchesonDeconinck3S184, 0.001);
    assert!(thousand_steps <= hundred_steps);
    assert!(
        hundred_steps <= 7,
        "unexpected 3S184 low-storage solve allocation count: {hundred_steps}"
    );

    let hundred_steps = allocations_for(ParsaniKetchesonDeconinck3S205, 0.01);
    let thousand_steps = allocations_for(ParsaniKetchesonDeconinck3S205, 0.001);
    assert!(thousand_steps <= hundred_steps);
    assert!(
        hundred_steps <= 7,
        "unexpected 3S205 low-storage solve allocation count: {hundred_steps}"
    );

    for (label, hundred_steps, thousand_steps, maximum) in [
        (
            "2C",
            allocations_for(CFRLDDRK64, 0.01),
            allocations_for(CFRLDDRK64, 0.001),
            7,
        ),
        (
            "alternating 2N",
            allocations_for(SHLDDRK_2N, 0.01),
            allocations_for(SHLDDRK_2N, 0.001),
            7,
        ),
        (
            "register pipeline",
            allocations_for(CKLLSRK43_2, 0.01),
            allocations_for(CKLLSRK43_2, 0.001),
            8,
        ),
    ] {
        assert!(
            thousand_steps <= hundred_steps,
            "{label} allocations grew with step count: {hundred_steps} -> {thousand_steps}"
        );
        assert!(
            hundred_steps <= maximum,
            "unexpected {label} low-storage solve allocation count: {hundred_steps}"
        );
    }
}
