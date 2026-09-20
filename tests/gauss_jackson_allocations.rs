use differential_equations::{GaussJackson8, GaussJacksonConfig};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};
use std::{alloc::System, convert::Infallible};
#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

#[test]
fn all_step_paths_reuse_workspace_without_allocations() {
    let construction = Region::new(GLOBAL);
    let mut solver =
        GaussJackson8::new(0., &[1., 2.], &[0., 0.], 0.1, GaussJacksonConfig::default()).unwrap();
    let setup = construction.change();
    assert!(setup.allocations > 0);
    let mut force = |_: f64, q: &[f64], _: &[f64], a: &mut [f64]| {
        for (a, q) in a.iter_mut().zip(q) {
            *a = -q;
        }
        Ok::<_, Infallible>(())
    };
    let startup = Region::new(GLOBAL);
    for _ in 0..8 {
        solver.try_step(&mut force).unwrap();
    }
    let startup_stats = startup.change();
    assert_eq!(startup_stats.allocations, 0);
    assert_eq!(startup_stats.reallocations, 0);
    let steady = Region::new(GLOBAL);
    for _ in 0..40 {
        solver.try_step(&mut force).unwrap();
    }
    let steady_stats = steady.change();
    assert_eq!(steady_stats.allocations, 0);
    assert_eq!(steady_stats.reallocations, 0);
    let mut rejected = |_: f64, _: &[f64], _: &[f64], _: &mut [f64]| Err::<(), _>(42);
    let failure = Region::new(GLOBAL);
    let failed = solver.try_step(&mut rejected).is_err();
    let failure_stats = failure.change();
    assert!(failed);
    assert_eq!(failure_stats.allocations, 0);
    assert_eq!(failure_stats.reallocations, 0);
    let mut q = [0.; 2];
    let mut v = [0.; 2];
    let mut coefficients = [0.; 12];
    let dense = Region::new(GLOBAL);
    solver
        .interpolate_into(solver.time() - 0.05, &mut q, &mut v)
        .unwrap();
    solver.dense_coefficients_into(&mut coefficients).unwrap();
    let dense_stats = dense.change();
    assert_eq!(dense_stats.allocations, 0);
    assert_eq!(dense_stats.reallocations, 0);
    let restart = Region::new(GLOBAL);
    solver.restart(0., &[2., 3.], &[0., 0.], -0.1).unwrap();
    solver.try_step_to(-0.037, &mut force).unwrap();
    let restart_stats = restart.change();
    assert_eq!(restart_stats.allocations, 0);
    assert_eq!(restart_stats.reallocations, 0);
}
