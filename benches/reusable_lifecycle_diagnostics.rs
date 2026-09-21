//! Report allocation, step, RHS-call and endpoint-error counters for the
//! `reusable_lifecycle` workload, isolated in a separate binary.
//!
//! Run with `cargo bench --bench reusable_lifecycle_diagnostics`. This
//! binary installs `StatsAlloc` as its global allocator so it can report
//! allocation counts; that instrumented allocator adds per-allocation
//! counter overhead to every allocation in the *entire process*, including
//! ones Criterion times. Keeping it confined to this binary (rather than
//! also installed in `reusable_lifecycle`) means the sibling binary's
//! setup/steady-state/output timings are never perturbed by
//! allocation-tracking overhead, and instead reflect normal downstream
//! execution.
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use differential_equations::{
    solvers::explicit::{Tsit5, Vern9},
    stepping::ExplicitRungeKuttaStepper,
};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};
use std::{alloc::System, time::Instant};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

#[path = "reusable_lifecycle_support/mod.rs"]
mod support;
use differential_equations::stepping::{AdaptiveController, ControllerConfig};
use support::{
    ORBIT_INITIAL_STATE, benchmark_metadata, orbit_norm, orbit_velocity_rhs,
    reference_orbit_endpoint, run_arc, scalar_norm, scalar_rhs,
};

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

criterion_group!(benches, diagnostics);
criterion_main!(benches);
