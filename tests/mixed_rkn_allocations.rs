use differential_equations::{solvers::second_order::FineRkn4, stepping::MixedRknStepper};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};
use std::{alloc::System, convert::Infallible};
#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;
#[test]
fn mixed_stm_attempts_reuse_all_storage() {
    let mut s = MixedRknStepper::new(
        FineRkn4.tableau().unwrap(),
        0.,
        &[1.],
        &[0.],
        &[1., 0., 0., 1.],
    )
    .unwrap();
    let mut rhs = |_: f64, q: &[f64], v: &[f64], z: &[f64], a: &mut [f64], dz: &mut [f64]| {
        a[0] = -q[0] - 0.2 * v[0];
        dz[0] = z[2];
        dz[1] = z[3];
        dz[2] = -z[0] - 0.2 * z[2];
        dz[3] = -z[1] - 0.2 * z[3];
        Ok::<_, Infallible>(())
    };
    let region = Region::new(GLOBAL);
    for _ in 0..100 {
        s.attempt(0.1, &mut rhs).unwrap();
        s.reject().unwrap();
        s.attempt(0.01, &mut rhs).unwrap();
        s.accept().unwrap();
    }
    s.reset(0., &[1.], &[0.], &[1., 0., 0., 1.]).unwrap();
    assert_eq!(region.change().allocations, 0);
    assert_eq!(region.change().reallocations, 0);
}
