use differential_equations::{solvers::rosenbrock::Rodas4, stepping::RosenbrockStepper};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};
use std::{alloc::System, convert::Infallible};
#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;
#[test]
fn rosenbrock_numeric_differentiation_reject_and_accept_allocate_nothing() {
    let mut s = RosenbrockStepper::new(Rodas4.tableau().unwrap(), 0., &[1., 0.]).unwrap();
    let mut rhs = |t: f64, y: &[f64], d: &mut [f64]| {
        d[0] = y[1];
        d[1] = -y[0] + t.sin();
        Ok::<_, Infallible>(())
    };
    let region = Region::new(GLOBAL);
    for _ in 0..100 {
        s.attempt(0.05, &mut rhs, None, None).unwrap();
        s.reject().unwrap();
        s.attempt(0.025, &mut rhs, None, None).unwrap();
        s.accept().unwrap();
    }
    assert_eq!(region.change().allocations, 0);
    assert_eq!(region.change().reallocations, 0);
}
