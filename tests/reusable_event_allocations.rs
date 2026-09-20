use differential_equations::{solvers::explicit::Tsit5, stepping::*};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};
use std::{alloc::System, convert::Infallible};
#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;
#[test]
fn output_free_root_localization_and_caller_recording_allocate_nothing() {
    let mut s = ExplicitRungeKuttaStepper::new(Tsit5.tableau().unwrap(), 0., &[1., 0.]).unwrap();
    let mut c = AdaptiveController::new(ControllerConfig::proportional(5).unwrap(), 0.2).unwrap();
    let mut saved = [0.; 2];
    let mut index = 0;
    let region = Region::new(GLOBAL);
    let result = integrate_rk_until_event(
        &mut s,
        &mut c,
        3.,
        &[0.5, 1.],
        1000,
        RootOptions::default(),
        &mut |_: f64, y: &[f64], d: &mut [f64]| {
            d[0] = y[1];
            d[1] = -y[0];
            Ok::<_, Infallible>(())
        },
        &mut |_: &[f64], _: &[f64], e: &[f64]| Ok(e[0].abs().max(e[1].abs()) / 1e-10),
        &mut |_: f64, y: &[f64]| Ok(y[0]),
        &mut |o: Observation<'_>| {
            if o.requested {
                saved[index] = o.time;
                index += 1;
            }
            Ok(ObserverAction::Continue)
        },
    )
    .unwrap();
    assert_eq!(region.change().allocations, 0);
    assert_eq!(region.change().reallocations, 0);
    assert_eq!(saved, [0.5, 1.]);
    assert!(result.event_value.unwrap().abs() < 1e-9);
}
