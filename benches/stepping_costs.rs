//! Construction, accepted/rejected attempts, and output costs measured separately.
//! Run timing and allocation-metrics binaries separately; each emits raw CSV.
use differential_equations::solvers::explicit::Tsit5;
use differential_equations::stepping::{
    AdaptiveController, ControllerConfig, ExplicitRungeKuttaStepper, MinimumStepPolicy,
};
use differential_equations::tableau::{LazyDenseStageCoefficients, RungeKuttaCoefficients};
use differential_equations::{OdeProblem, SaveMode, SolveOptions, solve};
#[cfg(feature = "allocation-metrics")]
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};
use std::{convert::Infallible, hint::black_box, time::Instant};
#[cfg(feature = "allocation-metrics")]
#[global_allocator]
static GLOBAL: &StatsAlloc<std::alloc::System> = &INSTRUMENTED_SYSTEM;

fn rhs(_: f64, y: &[f64], out: &mut [f64]) -> Result<(), Infallible> {
    for (d, v) in out.iter_mut().zip(y) {
        *d = -*v;
    }

    Ok(())
}

#[path = "../tableau-core/tests/fixtures/numeris_rkv98.rs"]
mod numeris_rkv98;

fn rkv98() -> differential_equations::tableau::RungeKuttaTableau {
    let rows: Vec<&[f64]> = numeris_rkv98::A[..16]
        .iter()
        .map(|row| &row[..16])
        .collect();
    let dense: Vec<&[f64]> = numeris_rkv98::BI.iter().map(|row| row.as_slice()).collect();
    let sparse: Vec<Vec<(usize, f64)>> = numeris_rkv98::A[16..]
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .filter(|(_, value)| **value != 0.0)
                .map(|(index, &value)| (index, value))
                .collect()
        })
        .collect();
    let lazy: Vec<_> = sparse
        .iter()
        .enumerate()
        .map(|(index, row)| LazyDenseStageCoefficients {
            node: numeris_rkv98::C[16 + index],
            coefficients: row,
        })
        .collect();
    let mut coefficients = RungeKuttaCoefficients::explicit(
        "numeris 0.6.0 RKV98 benchmark",
        9,
        &rows,
        &numeris_rkv98::B[..16],
        &numeris_rkv98::C[..16],
    );
    coefficients.embedded_order = Some(8);
    coefficients.b_hat = Some(&numeris_rkv98::BHAT[..16]);
    coefficients.dense = Some(&dense);
    coefficients.lazy_dense_stages = &lazy;
    coefficients.build().expect("RKV98 fixture must validate")
}
fn measure(name: &str, n: usize, operations: usize, mut run: impl FnMut() -> f64) {
    // Warm-up excluded. This harness reports warm-cache construction separately.
    black_box(run());
    #[cfg(feature = "allocation-metrics")]
    let region = Region::new(GLOBAL);
    let start = Instant::now();
    let checksum = black_box(run());
    let nanoseconds = start.elapsed().as_nanos();
    #[cfg(feature = "allocation-metrics")]
    let (allocations, bytes, retained) = {
        let s = region.change();
        (
            s.allocations + s.reallocations,
            s.bytes_allocated,
            s.bytes_allocated as i128 - s.bytes_deallocated as i128,
        )
    };
    #[cfg(not(feature = "allocation-metrics"))]
    let (allocations, bytes, retained) = (-1_i128, -1_i128, -1_i128);
    println!(
        "{name},{n},{operations},{nanoseconds},{allocations},{bytes},{retained},{checksum:.17e}"
    );
}
fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let repeats = if args.iter().any(|s| s == "--test") {
        2
    } else {
        args.first()
            .map(|s| s.parse::<usize>().expect("positive repetition count"))
            .unwrap_or(10_000)
    };
    assert!(repeats > 0);
    println!(
        "case,dimension,operations,nanoseconds,allocation_calls,allocated_bytes,net_allocated_bytes,checksum"
    );
    for n in [1, 6, 42, 48, 256] {
        let initial = vec![1.0; n];
        let tableau = Tsit5.tableau().unwrap();
        measure("setup_owned", n, repeats, || {
            let mut sum = 0.0;
            for _ in 0..repeats {
                let s = ExplicitRungeKuttaStepper::new(tableau, 0.0, black_box(&initial)).unwrap();
                sum += black_box(s.state()[0]);
            }
            sum
        });
        for reject in [false, true] {
            let mut solver = ExplicitRungeKuttaStepper::new(tableau, 0.0, &initial).unwrap();
            measure(
                if reject {
                    "rejected_attempt"
                } else {
                    "accepted_attempt"
                },
                n,
                repeats,
                || {
                    solver.reset(0.0, &initial).unwrap();
                    let mut checksum = 0.0;
                    for _ in 0..repeats {
                        let view = solver.attempt(0.001, &mut rhs).unwrap();
                        checksum += black_box(view.candidate[0]);
                        if reject {
                            solver.reject().unwrap();
                        } else {
                            solver.accept().unwrap();
                        }
                    }
                    checksum
                },
            );
        }
        let mut solver = ExplicitRungeKuttaStepper::new(tableau, 0.0, &initial).unwrap();
        let mut recording = vec![0.0; n * repeats];
        measure("accepted_preallocated_recording", n, repeats, || {
            solver.reset(0.0, &initial).unwrap();
            for output in recording.chunks_exact_mut(n) {
                solver.attempt(0.001, &mut rhs).unwrap();
                solver.accept().unwrap();
                output.copy_from_slice(solver.state());
            }
            black_box(recording[recording.len() - 1])
        });
        let problem = OdeProblem::new(
            |d: &mut [f64], y: &[f64], _: &(), _: f64| {
                rhs(0.0, y, d).unwrap();
            },
            initial.clone(),
            (0.0, 0.001),
            (),
        );
        let options = SolveOptions::new()
            .with_adaptive(false)
            .with_initial_step(0.001)
            .with_save(SaveMode::Endpoints);
        measure("fresh_one_step_whole_solve", n, repeats, || {
            let mut sum = 0.0;
            for _ in 0..repeats {
                let s = solve(&problem, Tsit5, &options).unwrap();
                sum += black_box(s.last_state()[0]);
            }
            sum
        });
    }
    benchmark_rkv98_controller_and_stages();
}

fn benchmark_rkv98_controller_and_stages() {
    let tableau = rkv98();
    let state = vec![1.0; 16];
    let mut stepper = ExplicitRungeKuttaStepper::new(&tableau, 0.0, &state).unwrap();
    let mut controller = AdaptiveController::new(
        ControllerConfig {
            rejection_exponent: Some(0.14),
            repeated_rejection_maximum: Some(0.5),
            minimum_step_policy: MinimumStepPolicy::ForceAccept,
            ..ControllerConfig::proportional(8).unwrap()
        },
        1.0e-3,
    )
    .unwrap();
    let mut rhs_calls = 0usize;
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    let stage_start = Instant::now();
    for attempt in 0..256 {
        let view = stepper
            .attempt(1.0e-3, &mut |_: f64, y: &[f64], out: &mut [f64]| {
                rhs_calls += 1;
                rhs(0.0, y, out)
            })
            .unwrap();
        black_box(view.candidate[0]);
        let error = if attempt % 17 == 0 { 2.0 } else { 0.2 };
        if error <= 1.0 {
            stepper.accept().unwrap();
            accepted += 1;
        } else {
            stepper.reject().unwrap();
            rejected += 1;
        }
    }
    let stage_nanoseconds = stage_start.elapsed().as_nanos();
    controller.reset(1.0e-3).unwrap();
    let controller_start = Instant::now();
    let mut proposal = 1.0e-3;
    for attempt in 0..256 {
        let error = if attempt % 17 == 0 { 2.0 } else { 0.2 };
        proposal = controller.assess(proposal, error).unwrap().next_step;
    }
    let controller_nanoseconds = controller_start.elapsed().as_nanos();
    let inclusive_nanoseconds = stage_nanoseconds + controller_nanoseconds;
    println!(
        "rkv98_controller_stage_loop,16,256,{inclusive_nanoseconds},{rhs_calls},{accepted},{rejected},{};stage_ns={stage_nanoseconds};controller_ns={controller_nanoseconds};proposal={proposal:.17e}",
        tableau.lazy_dense_stages().len()
    );
}
