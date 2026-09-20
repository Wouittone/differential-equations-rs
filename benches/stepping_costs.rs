//! Construction, accepted/rejected attempts, and output costs measured separately.
//! Run timing and allocation-metrics binaries separately; each emits raw CSV.
use differential_equations::solvers::explicit::Tsit5;
use differential_equations::stepping::ExplicitRungeKuttaStepper;
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
}
