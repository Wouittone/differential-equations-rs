use std::alloc::System;
use std::hint::black_box;

use differential_equations::solvers::explicit::{Rk4, solve_split};
use differential_equations::solvers::multirate::MRAB;
use differential_equations::solvers::multistep::{Ab3, Ab5, Abm32};
use differential_equations::{SolveOptions, SplitOdeProblem};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[path = "support/allocation.rs"]
mod allocation_support;

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

#[test]
fn only_used_formulas_are_materialized_and_shared_across_solver_families() {
    // One microstep uses AB1 even when a higher nominal order is configured.
    let problem = SplitOdeProblem::new(
        |du: &mut [f64], _: &[f64], _: &(), _| du.fill(1.0),
        |du: &mut [f64], _: &[f64], _: &(), _| du.fill(0.0),
        [0.0],
        (0.0, 0.1),
        (),
    );
    let result = solve_split(
        &problem,
        MRAB::new(5, 1),
        &SolveOptions::new()
            .with_adaptive(false)
            .with_initial_step(0.1),
    )
    .unwrap();
    assert_eq!(result.last_state(), &[0.1]);

    let first_fifth_order = Region::new(GLOBAL);
    black_box(Ab5.tableau().unwrap());
    assert!(first_fifth_order.change().allocations > 0);

    let first_third_order = Region::new(GLOBAL);
    let predictor = Ab3.tableau().unwrap();
    assert!(first_third_order.change().allocations > 0);

    let first_corrector = Region::new(GLOBAL);
    black_box(Abm32.tableau().unwrap());
    assert!(first_corrector.change().allocations > 0);

    // Tableau inspection does not force unrelated startup tableaux to load.
    let first_bootstrap = Region::new(GLOBAL);
    black_box(Rk4.tableau().unwrap());
    assert!(first_bootstrap.change().allocations > 0);

    assert!(std::ptr::eq(predictor, MRAB::new(3, 8).tableau().unwrap()));
    let repeated = allocation_support::minimum_measurement(|| {
        let region = Region::new(GLOBAL);
        for _ in 0..1000 {
            black_box(Ab3.tableau().unwrap());
            black_box(Abm32.predictor_tableau().unwrap());
            black_box(MRAB::new(3, 8).tableau().unwrap());
        }
        region.change().allocations
    });
    assert_eq!(repeated, 0);
}
