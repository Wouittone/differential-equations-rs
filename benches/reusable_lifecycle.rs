//! Compare reusable workspace setup with steady-state stepping.
//!
//! Run with `cargo bench --bench reusable_lifecycle`. The setup cases include
//! tableau/workspace construction; the steady-state cases construct once and
//! reuse the same stepper and controller for every measured arc.
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use differential_equations::{
    solvers::explicit::Tsit5,
    stepping::{
        AdaptiveController, ControllerConfig, ExplicitRungeKuttaStepper, ObserverAction,
        integrate_rk,
    },
};
use std::convert::Infallible;

fn rhs(_: f64, state: &[f64], derivative: &mut [f64]) -> Result<(), Infallible> {
    derivative[0] = -state[0];
    Ok(())
}

fn norm(_: &[f64], _: &[f64], error: &[f64]) -> Result<f64, Infallible> {
    Ok(error[0].abs() / 1.0e-10)
}

fn run_arc(
    stepper: &mut ExplicitRungeKuttaStepper<'_>,
    controller: &mut AdaptiveController,
    endpoint: f64,
) {
    integrate_rk(
        stepper,
        controller,
        endpoint,
        &[],
        100_000,
        &mut rhs,
        &mut norm,
        &mut |_| Ok::<_, Infallible>(ObserverAction::Continue),
    )
    .expect("reusable benchmark arc must solve");
}

fn setup(c: &mut Criterion) {
    let mut group = c.benchmark_group("reusable_lifecycle/setup");
    group.bench_function("stepper_and_controller", |b| {
        b.iter(|| {
            let tableau = Tsit5.tableau().expect("Tsit5 tableau");
            let mut state = [black_box(1.0)];
            let stepper =
                ExplicitRungeKuttaStepper::from_buffer(tableau, 0.0, &mut state).unwrap();
            let controller =
                AdaptiveController::new(ControllerConfig::proportional(5).unwrap(), 0.1)
                    .unwrap();
            black_box((stepper, controller));
        });
    });
    group.finish();
}

fn steady_state(c: &mut Criterion) {
    let mut group = c.benchmark_group("reusable_lifecycle/steady_state");
    for (name, endpoint) in [("short_arc", 1.0), ("long_arc", 10.0)] {
        group.bench_function(name, |b| {
            let tableau = Tsit5.tableau().expect("Tsit5 tableau");
            let mut state = [1.0];
            let mut stepper =
                ExplicitRungeKuttaStepper::from_buffer(tableau, 0.0, &mut state).unwrap();
            let mut controller =
                AdaptiveController::new(ControllerConfig::proportional(5).unwrap(), 0.1)
                    .unwrap();
            run_arc(&mut stepper, &mut controller, endpoint);
            b.iter(|| {
                stepper.reset(0.0, &[1.0]).unwrap();
                controller.reset(0.1).unwrap();
                run_arc(&mut stepper, &mut controller, black_box(endpoint));
                black_box(stepper.state()[0])
            });
        });
    }
    group.finish();
}

criterion_group!(benches, setup, steady_state);
criterion_main!(benches);
