use differential_equations::{
    solvers::{explicit::Tsit5, second_order::FineRkn4},
    stepping::{AccelerationPolicy, ExplicitRungeKuttaStepper, RknStepper},
};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};
use std::{alloc::System, convert::Infallible, hint::black_box};
#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;
#[test]
fn accepted_rejected_reset_and_zero_attempts_allocate_nothing() {
    let rk = Tsit5.tableau().unwrap();
    let rkn = FineRkn4.tableau().unwrap();
    let mut first = ExplicitRungeKuttaStepper::new(rk, 0., &[1., 0.]).unwrap();
    let mut second =
        RknStepper::new(rkn, AccelerationPolicy::VelocityDependent, 0., &[1.], &[0.]).unwrap();
    let mut f = |_: f64, y: &[f64], d: &mut [f64]| {
        d[0] = y[1];
        d[1] = -y[0];
        Ok::<_, Infallible>(())
    };
    let mut g = |_: f64, q: &[f64], v: &[f64], a: &mut [f64]| {
        a[0] = -q[0] - 0.01 * v[0];
        Ok::<_, Infallible>(())
    };
    let region = Region::new(GLOBAL);
    for _ in 0..100 {
        first.attempt(0.1, &mut f).unwrap();
        first.reject().unwrap();
        first.attempt(0.05, &mut f).unwrap();
        first.accept().unwrap();
        second.attempt(0.1, &mut g).unwrap();
        second.reject().unwrap();
        second.attempt(0.05, &mut g).unwrap();
        second.accept().unwrap();
    }
    first.reset(0., &[1., 0.]).unwrap();
    second.reset(0., &[1.], &[0.]).unwrap();
    black_box(first.state());
    black_box(second.position());
    assert_eq!(region.change().allocations, 0);
    assert_eq!(region.change().reallocations, 0);
}
