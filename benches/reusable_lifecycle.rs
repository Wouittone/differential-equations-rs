//! Compare reusable workspace setup with steady-state stepping across the
//! explicit solver family and a fixed-size orbit workload.
//!
//! Run with `cargo bench --bench reusable_lifecycle`. The setup cases include
//! tableau/workspace construction; the steady-state cases construct once and
//! reuse the same stepper and controller for every measured arc. This binary
//! deliberately uses the ordinary system allocator (no allocation-tracking
//! wrapper) so its Criterion timings reflect normal downstream execution.
//! Allocation, step, RHS-call and endpoint-error counters are reported by
//! the separate `reusable_lifecycle_diagnostics` binary instead, which
//! installs `StatsAlloc` as its global allocator; keeping that instrumented
//! allocator out of this binary avoids paying its per-allocation counter
//! overhead on every measured iteration here.
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use differential_equations::{
    solvers::explicit::{Tsit5, Vern9},
    stepping::{
        AdaptiveController, ControllerConfig, ExplicitRungeKuttaStepper, ObserverAction,
        integrate_rk,
    },
    tableau::parse_tableau,
};
use std::convert::Infallible;

#[path = "reusable_lifecycle_support/mod.rs"]
mod support;
use support::{
    ORBIT_INITIAL_STATE, benchmark_metadata, orbit_norm, orbit_velocity_rhs, run_arc, scalar_norm,
    scalar_rhs,
};

/// Raw JSON resources backing the `Tsit5`/`Vern9` tableaus, embedded
/// independently of the crate's own cached `LazyLock` accessors so this
/// binary can call the public, uncached [`parse_tableau`] on every
/// iteration and measure genuine cold parsing rather than a cache lookup.
const TSIT5_TABLEAU_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/tableau/resources/explicit/tsit5.json"
));
const VERN9_TABLEAU_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/tableau/resources/explicit/vern9.json"
));

fn setup(c: &mut Criterion) {
    println!("{}", benchmark_metadata("none"));
    let mut group = c.benchmark_group("reusable_lifecycle/setup");
    // `Tsit5.tableau()`/`Vern9.tableau()` are backed by process-cached
    // `LazyLock`s: Criterion's warm-up always runs the closure before any
    // timed sample, so even the very first timed `b.iter` call only ever
    // observes the cached value, never a genuine cold parse. To measure
    // real parsing cost, these lanes instead call the public, uncached
    // `parse_tableau` directly against the same embedded JSON on every
    // iteration, so every sample re-parses from scratch. The
    // `stepper_and_controller` lanes below resolve the cached tableau once
    // up front so they isolate workspace-construction cost specifically.
    group.bench_function("tsit5/tableau_parse", |b| {
        b.iter(|| {
            black_box(
                parse_tableau(black_box(TSIT5_TABLEAU_JSON), "Tsit5").expect("Tsit5 tableau"),
            );
        });
    });
    group.bench_function("vern9/tableau_parse", |b| {
        b.iter(|| {
            black_box(
                parse_tableau(black_box(VERN9_TABLEAU_JSON), "Vern9").expect("Vern9 tableau"),
            );
        });
    });

    let tsit5_tableau = Tsit5.tableau().expect("Tsit5 tableau");
    group.bench_function("tsit5/stepper_and_controller", |b| {
        b.iter(|| {
            let mut state = ORBIT_INITIAL_STATE.map(black_box);
            let stepper =
                ExplicitRungeKuttaStepper::from_buffer(tsit5_tableau, 0.0, &mut state).unwrap();
            let controller =
                AdaptiveController::new(ControllerConfig::proportional(5).unwrap(), 0.1).unwrap();
            black_box((stepper, controller));
        });
    });
    let vern9_tableau = Vern9.tableau().expect("Vern9 tableau");
    group.bench_function("vern9/stepper_and_controller", |b| {
        b.iter(|| {
            let mut state = ORBIT_INITIAL_STATE.map(black_box);
            let stepper =
                ExplicitRungeKuttaStepper::from_buffer(vern9_tableau, 0.0, &mut state).unwrap();
            let controller =
                AdaptiveController::new(ControllerConfig::proportional(5).unwrap(), 0.1).unwrap();
            black_box((stepper, controller));
        });
    });
    group.finish();
}

fn steady_state(c: &mut Criterion) {
    println!("{}", benchmark_metadata("endpoint_only"));
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

    for (name, endpoint, gravitational_parameter, drag) in [
        ("tsit5/orbit/short_arc", 1.0, 1.0, 0.1),
        ("vern9/orbit/short_arc", 1.0, 1.0, 0.1),
    ] {
        group.bench_function(name, |b| {
            let tableau = if name.starts_with("tsit5") {
                Tsit5.tableau().expect("Tsit5 tableau")
            } else {
                Vern9.tableau().expect("Vern9 tableau")
            };
            let mut state = ORBIT_INITIAL_STATE;
            let mut stepper =
                ExplicitRungeKuttaStepper::from_buffer(tableau, 0.0, &mut state).unwrap();
            let mut controller =
                AdaptiveController::new(ControllerConfig::proportional(5).unwrap(), 0.1).unwrap();
            let mut rhs = orbit_velocity_rhs(gravitational_parameter, drag);
            let mut norm = orbit_norm;
            run_arc(&mut stepper, &mut controller, endpoint, &mut rhs, &mut norm);
            b.iter(|| {
                stepper.reset(0.0, &ORBIT_INITIAL_STATE).unwrap();
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
    println!("{}", benchmark_metadata("endpoint_and_sampled"));
    let mut group = c.benchmark_group("reusable_lifecycle/output");
    let sample_times: Vec<f64> = (0..64).map(|index| index as f64 / 64.0).collect();
    for name in ["tsit5/endpoint_and_sampled", "vern9/endpoint_and_sampled"] {
        group.bench_function(name, |b| {
            let tableau = if name.starts_with("tsit5") {
                Tsit5.tableau().expect("Tsit5 tableau")
            } else {
                Vern9.tableau().expect("Vern9 tableau")
            };
            let mut state = ORBIT_INITIAL_STATE;
            let mut stepper =
                ExplicitRungeKuttaStepper::from_buffer(tableau, 0.0, &mut state).unwrap();
            let mut controller =
                AdaptiveController::new(ControllerConfig::proportional(5).unwrap(), 0.1).unwrap();
            b.iter(|| {
                let mut stored = Vec::new();
                let mut rhs = orbit_velocity_rhs(1.0, 0.1);
                let mut norm = orbit_norm;
                stepper.reset(0.0, &ORBIT_INITIAL_STATE).unwrap();
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
                        // Only retain requested samples and the endpoint; every
                        // other accepted step is intentionally dropped so this
                        // lane measures the advertised sampled-output policy
                        // rather than full-trajectory storage.
                        if observation.requested || observation.endpoint {
                            stored.push(observation.state.to_vec());
                        }
                        Ok::<_, Infallible>(ObserverAction::Continue)
                    },
                )
                .expect("output retention benchmark must solve");
                // Fold every retained component through black_box (not just
                // the count) so the optimizer cannot dead-store-eliminate the
                // `to_vec()` payloads this lane is measuring the cost of.
                let checksum = stored
                    .iter()
                    .flat_map(|state| state.iter().copied())
                    .fold(0.0_f64, |sum, value| sum + black_box(value));
                black_box((stored.len(), checksum))
            });
        });
    }
    group.finish();
}

/// This reusable layer exposes no public dense/polynomial interpolation
/// coefficients (the continuous-extension machinery in
/// `solution::dense::runge_kutta` is crate-private), so a genuine dense-export
/// lane cannot be constructed from this workload alone. What downstream code
/// building its own dense output *would* need as raw material is every
/// accepted step's state, so this lane measures the cost of retaining that —
/// named for what it does rather than for the (unavailable) interpolation it
/// would ultimately feed, and kept separate from the sampled `output` lane
/// above, whose cost scales with the number of requested samples instead of
/// the number of accepted steps.
fn full_trajectory_retention(c: &mut Criterion) {
    println!("{}", benchmark_metadata("every_accepted_step"));
    let mut group = c.benchmark_group("reusable_lifecycle/full_trajectory_retention");
    for name in ["tsit5/every_accepted_step", "vern9/every_accepted_step"] {
        group.bench_function(name, |b| {
            let tableau = if name.starts_with("tsit5") {
                Tsit5.tableau().expect("Tsit5 tableau")
            } else {
                Vern9.tableau().expect("Vern9 tableau")
            };
            let mut state = ORBIT_INITIAL_STATE;
            let mut stepper =
                ExplicitRungeKuttaStepper::from_buffer(tableau, 0.0, &mut state).unwrap();
            let mut controller =
                AdaptiveController::new(ControllerConfig::proportional(5).unwrap(), 0.1).unwrap();
            b.iter(|| {
                let mut stored = Vec::new();
                let mut rhs = orbit_velocity_rhs(1.0, 0.1);
                let mut norm = orbit_norm;
                stepper.reset(0.0, &ORBIT_INITIAL_STATE).unwrap();
                controller.reset(0.1).unwrap();
                integrate_rk(
                    &mut stepper,
                    &mut controller,
                    1.0,
                    &[],
                    100_000,
                    &mut rhs,
                    &mut norm,
                    &mut |observation| {
                        stored.push(observation.state.to_vec());
                        Ok::<_, Infallible>(ObserverAction::Continue)
                    },
                )
                .expect("full trajectory retention benchmark must solve");
                let checksum = stored
                    .iter()
                    .flat_map(|state| state.iter().copied())
                    .fold(0.0_f64, |sum, value| sum + black_box(value));
                black_box((stored.len(), checksum))
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    setup,
    steady_state,
    output_retention,
    full_trajectory_retention
);
criterion_main!(benches);
