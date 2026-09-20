//! Library-only work/precision sweep. Run exclusively in the final measurement phase:
//! `cargo run --release --example gj_work_precision > gj_work_precision.csv`.
//! Every row includes construction, solve/startup, and endpoint error extraction;
//! CSV formatting and the fixed analytic oracle are outside timing. No warmup is
//! hidden: repeat zero is retained (including any lazy tableau initialization).
//! Compare measured errors, never nominal tolerances. RKN12 is position-only,
//! hence intentionally applies only to Kepler. Blank startup counts mean the
//! public solver does not expose that statistic; blank allocation counts mean
//! allocations are not instrumented. No host or upstream implementation is used.
//! GJ's tolerance column is zero because it has no adaptive endpoint tolerance:
//! startup/corrector tolerances stay at the documented defaults (absolute 1e-14,
//! relative 1e-13) throughout the fixed-step sweep. Adaptive methods have zero
//! in the step column and use identical absolute/relative requested tolerances.
//! Match achieved error thresholds across the complete rows in analysis, keeping
//! failed rows and all sweep points; do not interpret equal nominal settings as
//! equal accuracy. The oracle has ordinary f64 roundoff near machine precision.
use differential_equations::solvers::{
    explicit::Vern9,
    multistep::VCABM,
    second_order::{Dprkn12, SecondOrderOdeProblem, solve_second_order},
};
use differential_equations::*;
use std::{cell::Cell, convert::Infallible, time::Instant};

#[derive(Clone, Copy)]
enum Case {
    Kepler,
    Damped,
}
impl Case {
    fn name(self) -> &'static str {
        match self {
            Self::Kepler => "kepler_e0.6_t30",
            Self::Damped => "damped_oscillator_t20",
        }
    }
    fn end(self) -> f64 {
        match self {
            Self::Kepler => 30.,
            Self::Damped => 20.,
        }
    }
    fn initial(self) -> (Vec<f64>, Vec<f64>) {
        match self {
            Self::Kepler => (vec![0.4, 0.], vec![0., 2.]),
            Self::Damped => (vec![1.], vec![0.]),
        }
    }
    fn acceleration(self, q: &[f64], v: &[f64], a: &mut [f64]) {
        match self {
            Self::Kepler => {
                let r3 = q[0].hypot(q[1]).powi(3);
                for i in 0..2 {
                    a[i] = -q[i] / r3;
                }
            }
            Self::Damped => a[0] = -q[0] - 0.2 * v[0],
        }
    }
    fn oracle(self) -> Vec<f64> {
        let t = self.end();
        match self {
            Self::Kepler => {
                let mean = t.rem_euclid(std::f64::consts::TAU);
                let mut anomaly = mean;
                for _ in 0..20 {
                    anomaly -= (anomaly - 0.6 * anomaly.sin() - mean) / (1. - 0.6 * anomaly.cos());
                }
                let d = 1. - 0.6 * anomaly.cos();
                vec![
                    anomaly.cos() - 0.6,
                    0.8 * anomaly.sin(),
                    -anomaly.sin() / d,
                    0.8 * anomaly.cos() / d,
                ]
            }
            Self::Damped => {
                let w = 0.99_f64.sqrt();
                let decay = (-0.1 * t).exp();
                vec![
                    decay * ((w * t).cos() + 0.1 / w * (w * t).sin()),
                    -decay / w * (w * t).sin(),
                ]
            }
        }
    }
}
fn error(values: &[f64], oracle: &[f64]) -> f64 {
    values
        .iter()
        .zip(oracle)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max)
}
fn options(tol: f64) -> SolveOptions {
    SolveOptions::default()
        .with_absolute_tolerance(tol)
        .with_relative_tolerance(tol)
        .with_initial_step(0.001)
        .with_max_step(0.4)
        .with_save(SaveMode::Endpoints)
}
fn row(
    case: Case,
    method: &str,
    rep: usize,
    h: f64,
    tol: f64,
    calls: usize,
    accepted: usize,
    startup: &str,
    elapsed: u128,
    outcome: Result<f64, String>,
) {
    let (err, status) = match outcome {
        Ok(e) => (format!("{e:.17e}"), "ok".to_owned()),
        Err(e) => (String::new(), e.replace([',', '\n', '\r'], ";")),
    };
    println!(
        "{},{method},{rep},{h:.17e},{tol:.17e},{calls},{accepted},{startup},{elapsed},,{err},{status}",
        case.name()
    );
}
fn first_order<A: OdeAlgorithm>(
    case: Case,
    algorithm: A,
    name: &str,
    tol: f64,
    rep: usize,
    oracle: &[f64],
) {
    let calls = Cell::new(0usize);
    let start = Instant::now();
    let (mut initial, v) = case.initial();
    initial.extend(v);
    let problem = OdeProblem::new(
        |out: &mut [f64], state: &[f64], _: &(), _: f64| {
            calls.set(calls.get() + 1);
            let n = state.len() / 2;
            out[..n].copy_from_slice(&state[n..]);
            case.acceleration(&state[..n], &state[n..], &mut out[n..]);
        },
        initial,
        (0., case.end()),
        (),
    );
    let result = solve(&problem, algorithm, &options(tol));
    let (accepted, outcome) = match result {
        Ok(s) => (s.stats().accepted_steps, Ok(error(s.last_state(), oracle))),
        Err(e) => (0, Err(format!("{e:?}"))),
    };
    row(
        case,
        name,
        rep,
        0.,
        tol,
        calls.get(),
        accepted,
        "",
        start.elapsed().as_nanos(),
        outcome,
    );
}
fn main() {
    println!(
        "case,method,repeat,step,tolerance,force_calls,accepted_steps,startup_steps,setup_and_solve_ns,allocation_count,endpoint_max_abs_error,status"
    );
    for case in [Case::Kepler, Case::Damped] {
        let oracle = case.oracle();
        for rep in 0..5 {
            for h in [0.08, 0.04, 0.02, 0.01, 0.005, 0.0025] {
                let start = Instant::now();
                let (q, v) = case.initial();
                let mut solver =
                    GaussJackson8::new(0., &q, &v, h, GaussJacksonConfig::default()).unwrap();
                let mut force = |_: f64, q: &[f64], v: &[f64], a: &mut [f64]| {
                    case.acceleration(q, v, a);
                    Ok::<_, Infallible>(())
                };
                let mut failure = None;
                while solver.time() < case.end() {
                    if let Err(e) = solver.try_step_to(case.end(), &mut force) {
                        failure = Some(format!("{e:?}"));
                        break;
                    }
                }
                let outcome = if let Some(e) = failure {
                    Err(e)
                } else {
                    let mut endpoint = solver.position().to_vec();
                    endpoint.extend(solver.velocity());
                    Ok(error(&endpoint, &oracle))
                };
                let elapsed = start.elapsed().as_nanos();
                let stats = solver.statistics();
                row(
                    case,
                    "GaussJackson8",
                    rep,
                    h,
                    0.,
                    stats.acceleration_evaluations,
                    stats.accepted_steps,
                    &stats.startup_steps.to_string(),
                    elapsed,
                    outcome,
                );
            }
            for tol in [1e-6, 1e-8, 1e-10, 1e-12, 1e-14] {
                first_order(case, VCABM, "VCABM", tol, rep, &oracle);
                first_order(case, Vern9, "Vern9", tol, rep, &oracle);
                if matches!(case, Case::Kepler) {
                    let calls = Cell::new(0usize);
                    let start = Instant::now();
                    let (q, v) = case.initial();
                    let problem = SecondOrderOdeProblem::new(
                        |a: &mut [f64], v: &[f64], q: &[f64], _: &(), _: f64| {
                            calls.set(calls.get() + 1);
                            case.acceleration(q, v, a);
                        },
                        v,
                        q,
                        (0., case.end()),
                        (),
                    );
                    let result = solve_second_order(&problem, Dprkn12, &options(tol));
                    let (accepted, outcome) = match result {
                        Ok(s) => {
                            let mut endpoint = s.last_position().to_vec();
                            endpoint.extend(s.last_velocity());
                            (s.stats().accepted_steps, Ok(error(&endpoint, &oracle)))
                        }
                        Err(e) => (0, Err(format!("{e:?}"))),
                    };
                    row(
                        case,
                        "Dprkn12",
                        rep,
                        0.,
                        tol,
                        calls.get(),
                        accepted,
                        "",
                        start.elapsed().as_nanos(),
                        outcome,
                    );
                }
            }
        }
    }
}
