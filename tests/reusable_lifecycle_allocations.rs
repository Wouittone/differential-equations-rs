use differential_equations::{
    solvers::explicit::Tsit5,
    stepping::{
        AdaptiveController, ControllerConfig, ExplicitRungeKuttaStepper, ObserverAction,
        integrate_rk,
    },
};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};
use std::{alloc::System, convert::Infallible, hint::black_box};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

fn measure_arc(endpoint: f64) -> (usize, usize) {
    let tableau = Tsit5.tableau().unwrap();
    let mut state = [1.0];
    let mut stepper = ExplicitRungeKuttaStepper::from_buffer(tableau, 0.0, &mut state).unwrap();
    let mut controller =
        AdaptiveController::new(ControllerConfig::proportional(5).unwrap(), 0.1).unwrap();
    let run = |stepper: &mut ExplicitRungeKuttaStepper<'_>,
               controller: &mut AdaptiveController| {
        integrate_rk(
            stepper,
            controller,
            endpoint,
            &[],
            100_000,
            &mut |_: f64, y: &[f64], dy: &mut [f64]| {
                dy[0] = -y[0];
                Ok::<_, Infallible>(())
            },
            &mut |_: &[f64], _: &[f64], error: &[f64]| {
                Ok::<_, Infallible>(error[0].abs() / 1.0e-10)
            },
            &mut |_| Ok::<_, Infallible>(ObserverAction::Continue),
        )
        .unwrap();
        black_box(stepper.state()[0]);
    };
    run(&mut stepper, &mut controller);
    stepper.reset(0.0, &[1.0]).unwrap();
    controller.reset(0.1).unwrap();
    let region = Region::new(GLOBAL);
    run(&mut stepper, &mut controller);
    let change = region.change();
    (change.allocations + change.reallocations, change.bytes_allocated)
}

fn minimum_measurement(endpoint: f64) -> (usize, usize) {
    (0..3)
        .map(|_| measure_arc(endpoint))
        .min_by_key(|&(allocations, bytes)| (allocations, bytes))
        .unwrap()
}

#[test]
fn reusable_downstream_workspace_has_zero_steady_state_allocations() {
    let short = minimum_measurement(1.0);
    let long = minimum_measurement(10.0);
    assert_eq!(short, (0, 0));
    assert_eq!(long, (0, 0));
}
