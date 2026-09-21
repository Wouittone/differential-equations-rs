//! Compare reusable workspace setup with steady-state stepping across the
//! explicit solver family and a fixed-size orbit workload.
//!
//! Run with `cargo bench --bench reusable_lifecycle`. The setup cases include
//! tableau/workspace construction; the steady-state cases construct once and
//! reuse the same stepper and controller for every measured arc. The
//! diagnostics group is not itself a timing target: its one-time setup pass
//! reports the allocation, step, RHS-call and endpoint-error counters the
//! steady-state loop otherwise hides behind Criterion's statistics.
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use differential_equations::{
    solvers::explicit::{Tsit5, Vern9},
    stepping::{
        AdaptiveController, ControllerConfig, ExplicitRungeKuttaStepper, ObserverAction,
        integrate_rk,
    },
};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};
use std::{alloc::System, convert::Infallible, time::Instant};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

const CONTROLLER_TOLERANCE: f64 = 1.0e-10;

/// Compiler version captured by invoking the toolchain reported at build
/// time (falling back to `rustc` on `PATH`); environment variables such as
/// `RUSTC` are cargo-internal and are not propagated to this process.
fn compiler_version() -> String {
    std::process::Command::new(option_env!("RUSTC").unwrap_or("rustc"))
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|version| version.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Best-effort CPU model/identifier, tried through the mechanism available
/// on each platform, with an explicit `unknown` fallback rather than an
/// empty or misleading value when none of them succeed.
fn cpu_identifier() -> String {
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/cpuinfo") {
            if let Some(model) = contents
                .lines()
                .find(|line| line.starts_with("model name"))
                .and_then(|line| line.split_once(':'))
                .map(|(_, value)| value.trim().to_string())
                .filter(|value| !value.is_empty())
            {
                return model;
            }
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
        {
            if output.status.success() {
                if let Some(model) = String::from_utf8(output.stdout)
                    .ok()
                    .map(|value| value.trim().to_string())
                    .filter(|value| !value.is_empty())
                {
                    return model;
                }
            }
        }
    }
    #[cfg(target_os = "windows")]
    {
        if let Some(identifier) = std::env::var("PROCESSOR_IDENTIFIER")
            .ok()
            .filter(|value| !value.is_empty())
        {
            return identifier;
        }
    }
    "unknown".to_string()
}

/// Metadata printed once per benchmarked configuration. `output_policy`
/// documents that lane's own retention behavior rather than a single
/// process-wide constant, since the suite intentionally exercises several
/// different output policies (endpoint-only, endpoint-and-sampled, and
/// every-accepted-step retention).
fn benchmark_metadata(output_policy: &str) -> String {
    format!(
        "reusable_lifecycle crate={} version={} target={}-{} cpu={} compiler={} tolerance={} controller=proportional(5) output={}",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        std::env::consts::ARCH,
        std::env::consts::OS,
        cpu_identifier(),
        compiler_version(),
        CONTROLLER_TOLERANCE,
        output_policy,
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
    Ok(error
        .iter()
        .copied()
        .fold(0.0_f64, |worst, component| worst.max(component.abs()))
        / CONTROLLER_TOLERANCE)
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

/// Six-state initial condition shared by the orbit-workload lanes so setup
/// costs stay comparable across solver families. With `gravitational_parameter
/// = 1.0` and unit radius, the nonzero tangential velocity component
/// (`vy = 1.0`) makes this a genuine (drag-decaying) circular orbit rather
/// than radial free-fall.
const ORBIT_INITIAL_STATE: [f64; 6] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

/// High-accuracy reference endpoint for the velocity-coupled orbit, computed
/// once with a much tighter tolerance than the measured lanes. Diagnostics
/// compare their endpoint against this to report a meaningful error even
/// though the drag term has no closed-form solution.
fn reference_orbit_endpoint(
    tableau: &differential_equations::tableau::RungeKuttaTableau,
    endpoint: f64,
    gravitational_parameter: f64,
    drag: f64,
) -> [f64; 6] {
    let mut state = ORBIT_INITIAL_STATE;
    let mut stepper = ExplicitRungeKuttaStepper::from_buffer(tableau, 0.0, &mut state).unwrap();
    let mut controller =
        AdaptiveController::new(ControllerConfig::proportional(5).unwrap(), 0.1).unwrap();
    let mut rhs = orbit_velocity_rhs(gravitational_parameter, drag);
    let mut norm = |_: &[f64], _: &[f64], error: &[f64]| -> Result<f64, Infallible> {
        Ok(error
            .iter()
            .copied()
            .fold(0.0_f64, |worst, component| worst.max(component.abs()))
            / 1.0e-13)
    };
    run_arc(&mut stepper, &mut controller, endpoint, &mut rhs, &mut norm);
    state
}

fn setup(c: &mut Criterion) {
    println!("{}", benchmark_metadata("none"));
    let mut group = c.benchmark_group("reusable_lifecycle/setup");
    // `tableau()` is a lazily parsed, process-cached accessor: its true
    // one-time JSON-parsing cost is paid only by the very first call in the
    // whole benchmark binary (this group runs first in `criterion_group!`),
    // and every call after that is a cache lookup. Report both explicitly
    // as separate lanes instead of silently excluding parsing: the
    // `tableau_parse` lanes below capture the cold-parse-then-cached-lookup
    // accessor cost, while `stepper_and_controller` resolves the tableau
    // once up front so it isolates workspace construction cost. Both lanes
    // construct the same six-component state so the reported
    // `stepper_and_controller` timings reflect solver-family setup cost,
    // not differing workspace size.
    group.bench_function("tsit5/tableau_parse", |b| {
        b.iter(|| {
            black_box(Tsit5.tableau().expect("Tsit5 tableau"));
        });
    });
    group.bench_function("vern9/tableau_parse", |b| {
        b.iter(|| {
            black_box(Vern9.tableau().expect("Vern9 tableau"));
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

/// One-time (non-timed) measurement of the counters the issue's acceptance
/// criteria call for: allocation calls/bytes, elapsed time, accepted and
/// rejected steps, RHS calls, and endpoint error against a reference
/// solution. Printed once per configuration before the group's own timed
/// warm loop runs, so it never perturbs Criterion's statistics.
fn diagnostics(c: &mut Criterion) {
    println!("{}", benchmark_metadata("endpoint_only"));
    let mut group = c.benchmark_group("reusable_lifecycle/diagnostics");

    for (name, endpoint) in [("tsit5/scalar", 1.0), ("vern9/scalar", 1.0)] {
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

            stepper.reset(0.0, &[1.0]).unwrap();
            controller.reset(0.1).unwrap();
            stepper.clear_statistics();
            let region = Region::new(GLOBAL);
            let started = Instant::now();
            run_arc(&mut stepper, &mut controller, endpoint, &mut rhs, &mut norm);
            let elapsed = started.elapsed();
            let change = region.change();
            let stats = stepper.statistics();
            let endpoint_error = (stepper.state()[0] - (-endpoint).exp()).abs();
            println!(
                "reusable_lifecycle/diagnostics {name}: allocations={} bytes_allocated={} bytes_deallocated={} elapsed={:?} accepted_steps={} rejected_steps={} rhs_evaluations={} endpoint_error={:e}",
                change.allocations + change.reallocations,
                change.bytes_allocated,
                change.bytes_deallocated,
                elapsed,
                stats.accepted_steps,
                stats.rejected_steps,
                stats.rhs_evaluations,
                endpoint_error,
            );

            b.iter(|| {
                stepper.reset(0.0, &[1.0]).unwrap();
                controller.reset(0.1).unwrap();
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
        ("tsit5/orbit", 1.0, 1.0, 0.1),
        ("vern9/orbit", 1.0, 1.0, 0.1),
    ] {
        group.bench_function(name, |b| {
            let tableau = if name.starts_with("tsit5") {
                Tsit5.tableau().expect("Tsit5 tableau")
            } else {
                Vern9.tableau().expect("Vern9 tableau")
            };
            let reference =
                reference_orbit_endpoint(tableau, endpoint, gravitational_parameter, drag);
            let mut state = ORBIT_INITIAL_STATE;
            let mut stepper =
                ExplicitRungeKuttaStepper::from_buffer(tableau, 0.0, &mut state).unwrap();
            let mut controller =
                AdaptiveController::new(ControllerConfig::proportional(5).unwrap(), 0.1).unwrap();
            let mut rhs = orbit_velocity_rhs(gravitational_parameter, drag);
            let mut norm = orbit_norm;
            run_arc(&mut stepper, &mut controller, endpoint, &mut rhs, &mut norm);

            stepper.reset(0.0, &ORBIT_INITIAL_STATE).unwrap();
            controller.reset(0.1).unwrap();
            stepper.clear_statistics();
            let region = Region::new(GLOBAL);
            let started = Instant::now();
            run_arc(&mut stepper, &mut controller, endpoint, &mut rhs, &mut norm);
            let elapsed = started.elapsed();
            let change = region.change();
            let stats = stepper.statistics();
            let endpoint_error = stepper
                .state()
                .iter()
                .zip(reference.iter())
                .map(|(measured, reference)| (measured - reference).powi(2))
                .sum::<f64>()
                .sqrt();
            println!(
                "reusable_lifecycle/diagnostics {name}: allocations={} bytes_allocated={} bytes_deallocated={} elapsed={:?} accepted_steps={} rejected_steps={} rhs_evaluations={} endpoint_error={:e}",
                change.allocations + change.reallocations,
                change.bytes_allocated,
                change.bytes_deallocated,
                elapsed,
                stats.accepted_steps,
                stats.rejected_steps,
                stats.rhs_evaluations,
                endpoint_error,
            );

            b.iter(|| {
                stepper.reset(0.0, &ORBIT_INITIAL_STATE).unwrap();
                controller.reset(0.1).unwrap();
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
    diagnostics,
    output_retention,
    full_trajectory_retention
);
criterion_main!(benches);
