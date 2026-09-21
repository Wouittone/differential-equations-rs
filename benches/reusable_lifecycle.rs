//! Compare reusable workspace setup with steady-state stepping across the
//! explicit solver family and a fixed-size orbit workload.
//!
//! Run with `cargo bench --bench reusable_lifecycle`. The setup cases include
//! tableau/workspace construction; the steady-state cases construct once and
//! reuse the same stepper and controller for every measured arc.
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use differential_equations::{
    solvers::explicit::{Tsit5, Vern9},
    stepping::{
        AdaptiveController, ControllerConfig, ExplicitRungeKuttaStepper, ObserverAction,
        integrate_rk,
    },
};
use std::convert::Infallible;

const CONTROLLER_TOLERANCE: f64 = 1.0e-10;
const OUTPUT_POLICY: &str = "endpoint_and_sampled";

fn benchmark_metadata() -> String {
    let compiler = std::env::var("RUSTC").unwrap_or_else(|_| "unknown".to_string());
    let cpu = std::env::var("PROCESSOR_IDENTIFIER")
        .or_else(|_| std::env::var("HOSTTYPE"))
        .unwrap_or_else(|_| "unknown".to_string());
    format!(
        "reusable_lifecycle crate={} version={} target={} compiler={} cpu={} tolerance={} controller=proportional(5) output={}",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        option_env!("TARGET").unwrap_or("unknown"),
        compiler,
        cpu,
        CONTROLLER_TOLERANCE,
        OUTPUT_POLICY,
    )
}

fn scalar_rhs(_: f64, state: &[f64], derivative: &mut [f64]) -> Result<(), Infallible> {
    derivative[0] = -state[0];
    Ok(())
}

fn scalar_norm(_: &[f64], _: &[f64], error: &[f64]) -> Result<f64, Infallible> {
    Ok(error[0].abs() / CONTROLLER_TOLERANCE)
}

fn orbit_velocity_rhs(
    gravitational_parameter: f64,
    drag: f64,
) -> impl FnMut(f64, &[f64], &mut [f64]) -> Result<(), Infallible> {
    move |_: f64, state: &[f64], derivative: &mut [f64]| {
        let (x, y, z, vx, vy, vz) = (state[0], state[1], state[2], state[3], state[4], state[5]);
        let radius = (x * x + y * y + z * z).sqrt().max(1.0e-12);
        let radius_cubed = radius * radius * radius;
        let speed = (vx * vx + vy * vy + vz * vz).sqrt();
        let acceleration_scale = gravitational_parameter / radius_cubed;
        derivative[0] = vx;
        derivative[1] = vy;
        derivative[2] = vz;
        derivative[3] = -acceleration_scale * x - drag * speed * vx;
        derivative[4] = -acceleration_scale * y - drag * speed * vy;
        derivative[5] = -acceleration_scale * z - drag * speed * vz;
        Ok(())
    }
}

fn orbit_norm(_: &[f64], _: &[f64], error: &[f64]) -> Result<f64, Infallible> {
    Ok(error.iter().copied().fold(0.0, f64::max) / CONTROLLER_TOLERANCE)
}

fn run_arc<F, N>(
    stepper: &mut ExplicitRungeKuttaStepper<'_>,
    controller: &mut AdaptiveController,
    endpoint: f64,
    rhs: &mut F,
    norm: &mut N,
) where
    F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), Infallible>,
    N: FnMut(&[f64], &[f64], &[f64]) -> Result<f64, Infallible>,
{
    integrate_rk(
        stepper,
        controller,
        endpoint,
        &[],
        100_000,
        rhs,
        norm,
        &mut |_| Ok::<_, Infallible>(ObserverAction::Continue),
    )
    .expect("reusable benchmark arc must solve");
}

fn setup(c: &mut Criterion) {
    println!("{}", benchmark_metadata());
    let mut group = c.benchmark_group("reusable_lifecycle/setup");
    group.bench_function("tsit5/stepper_and_controller", |b| {
        b.iter(|| {
            let tableau = Tsit5.tableau().expect("Tsit5 tableau");
            let mut state = [black_box(1.0)];
            let stepper = ExplicitRungeKuttaStepper::from_buffer(tableau, 0.0, &mut state).unwrap();
            let controller =
                AdaptiveController::new(ControllerConfig::proportional(5).unwrap(), 0.1).unwrap();
            black_box((stepper, controller));
        });
    });
    group.bench_function("vern9/stepper_and_controller", |b| {
        b.iter(|| {
            let tableau = Vern9.tableau().expect("Vern9 tableau");
            let mut state = [
                black_box(1.0),
                black_box(0.0),
                black_box(0.0),
                black_box(0.0),
                black_box(0.0),
                black_box(0.0),
            ];
            let stepper = ExplicitRungeKuttaStepper::from_buffer(tableau, 0.0, &mut state).unwrap();
            let controller =
                AdaptiveController::new(ControllerConfig::proportional(5).unwrap(), 0.1).unwrap();
            black_box((stepper, controller));
        });
    });
    group.finish();
}

fn steady_state(c: &mut Criterion) {
    println!("{}", benchmark_metadata());
    let mut group = c.benchmark_group("reusable_lifecycle/steady_state");

    for (name, endpoint) in [
        ("tsit5/scalar/short_arc", 1.0),
        ("tsit5/scalar/long_arc", 10.0),
        ("vern9/scalar/short_arc", 1.0),
        ("vern9/scalar/long_arc", 10.0),
    ] {
        group.bench_function(name, |b| {
            let tableau = if name.starts_with("tsit5") {
                Tsit5.tableau().expect("Tsit5 tableau")
            } else {
                Vern9.tableau().expect("Vern9 tableau")
            };
            let mut state = [1.0];
            let mut stepper =
                ExplicitRungeKuttaStepper::from_buffer(tableau, 0.0, &mut state).unwrap();
            let mut controller =
                AdaptiveController::new(ControllerConfig::proportional(5).unwrap(), 0.1).unwrap();
            let mut rhs = scalar_rhs;
            let mut norm = scalar_norm;
            run_arc(&mut stepper, &mut controller, endpoint, &mut rhs, &mut norm);
            b.iter(|| {
                stepper.reset(0.0, &[1.0]).unwrap();
                controller.reset(0.1).unwrap();
                let mut rhs = scalar_rhs;
                let mut norm = scalar_norm;
                run_arc(
                    &mut stepper,
                    &mut controller,
                    black_box(endpoint),
                    &mut rhs,
                    &mut norm,
                );
                black_box(stepper.state()[0])
            });
        });
    }

    for (name, endpoint, initial_state, gravitational_parameter, drag) in [
        (
            "tsit5/orbit/short_arc",
            1.0,
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            0.1,
        ),
        (
            "vern9/orbit/short_arc",
            1.0,
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            0.1,
        ),
    ] {
        group.bench_function(name, |b| {
            let tableau = if name.starts_with("tsit5") {
                Tsit5.tableau().expect("Tsit5 tableau")
            } else {
                Vern9.tableau().expect("Vern9 tableau")
            };
            let mut state = initial_state;
            let mut stepper =
                ExplicitRungeKuttaStepper::from_buffer(tableau, 0.0, &mut state).unwrap();
            let mut controller =
                AdaptiveController::new(ControllerConfig::proportional(5).unwrap(), 0.1).unwrap();
            let mut rhs = orbit_velocity_rhs(gravitational_parameter, drag);
            let mut norm = orbit_norm;
            run_arc(&mut stepper, &mut controller, endpoint, &mut rhs, &mut norm);
            b.iter(|| {
                stepper.reset(0.0, &initial_state).unwrap();
                controller.reset(0.1).unwrap();
                let mut rhs = orbit_velocity_rhs(gravitational_parameter, drag);
                let mut norm = orbit_norm;
                run_arc(
                    &mut stepper,
                    &mut controller,
                    black_box(endpoint),
                    &mut rhs,
                    &mut norm,
                );
                black_box(stepper.state()[0])
            });
        });
    }

    group.finish();
}

fn output_retention(c: &mut Criterion) {
    let mut group = c.benchmark_group("reusable_lifecycle/output");
    let sample_times: Vec<f64> = (0..64).map(|index| index as f64 / 64.0).collect();
    for name in ["tsit5/endpoint_retention", "vern9/endpoint_retention"] {
        group.bench_function(name, |b| {
            let tableau = if name.starts_with("tsit5") {
                Tsit5.tableau().expect("Tsit5 tableau")
            } else {
                Vern9.tableau().expect("Vern9 tableau")
            };
            let mut state = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            let mut stepper =
                ExplicitRungeKuttaStepper::from_buffer(tableau, 0.0, &mut state).unwrap();
            let mut controller =
                AdaptiveController::new(ControllerConfig::proportional(5).unwrap(), 0.1).unwrap();
            b.iter(|| {
                let mut stored = Vec::new();
                let mut rhs = orbit_velocity_rhs(1.0, 0.1);
                let mut norm = orbit_norm;
                stepper.reset(0.0, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
                controller.reset(0.1).unwrap();
                integrate_rk(
                    &mut stepper,
                    &mut controller,
                    1.0,
                    &sample_times,
                    100_000,
                    &mut rhs,
                    &mut norm,
                    &mut |observation| {
                        stored.push(observation.state.to_vec());
                        Ok::<_, Infallible>(ObserverAction::Continue)
                    },
                )
                .expect("output retention benchmark must solve");
                black_box(stored.len())
            });
        });
    }
    group.finish();
}

criterion_group!(benches, setup, steady_state, output_retention);
criterion_main!(benches);
