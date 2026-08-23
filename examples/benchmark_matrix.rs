#[cfg(feature = "allocation-metrics")]
use std::alloc::System;
use std::hint::black_box;
use std::time::Instant;

use differential_equations::{
    Ab3, Ab4, Ab5, Abm32, Abm43, Abm54, Alshina2, Alshina3, Bs3, Dp5, Euler, Heun, ImplicitEuler,
    ImplicitMidpoint, KenCarp5, Kvaerno5, Midpoint, OdeAlgorithm, OdeProblem, Ralston, Ralston4,
    Rk4, Rkm, Rodas4P, Rodas5P, Rodas5Pr, Rosenbrock23, SaveMode, SolveOptions, SspRk22, SspRk33,
    SspRk43, Trapezoid, Trbdf2, Tsit5, solve,
};
#[cfg(feature = "allocation-metrics")]
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[cfg(feature = "allocation-metrics")]
#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

struct Rates(Vec<f64>);

type DecayRhs = fn(&mut [f64], &[f64], &Rates, f64);

fn decay(derivative: &mut [f64], state: &[f64], rates: &Rates, _: f64) {
    for ((derivative, state), rate) in derivative.iter_mut().zip(state).zip(&rates.0) {
        *derivative = -rate * state;
    }
}

fn problem(dimension: usize, stiffness: f64, end: f64) -> OdeProblem<DecayRhs, Rates> {
    let rates = (0..dimension)
        .map(|index| stiffness * (1.0 + index as f64 / dimension as f64))
        .collect();
    OdeProblem::new(decay, vec![1.0; dimension], (0.0, end), Rates(rates))
}

fn adaptive_options() -> SolveOptions {
    SolveOptions {
        absolute_tolerance: 1.0e-7,
        relative_tolerance: 1.0e-7,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

fn fixed_options() -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(0.01),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

fn benchmark<A: OdeAlgorithm + Copy>(
    name: &str,
    problem: &OdeProblem<DecayRhs, Rates>,
    algorithm: A,
    options: &SolveOptions,
    repetitions: usize,
) {
    black_box(solve(problem, algorithm, options).expect("warm-up solve failed"));

    #[cfg(feature = "allocation-metrics")]
    let region = Region::new(GLOBAL);
    let started = Instant::now();
    let mut checksum = 0.0;
    let mut rhs_evaluations = 0;
    for _ in 0..repetitions {
        let solution = solve(problem, algorithm, options).expect("benchmark solve failed");
        checksum += solution.last_state()[0];
        rhs_evaluations += solution.stats().rhs_evaluations;
        black_box(&solution);
    }
    let elapsed = started.elapsed();
    #[cfg(feature = "allocation-metrics")]
    let allocations = region.change();
    #[cfg(feature = "allocation-metrics")]
    let (bytes_per_solve, allocations_per_solve) = (
        allocations.bytes_allocated as f64 / repetitions as f64,
        allocations.allocations as f64 / repetitions as f64,
    );
    #[cfg(not(feature = "allocation-metrics"))]
    let (bytes_per_solve, allocations_per_solve) = (f64::NAN, f64::NAN);

    println!(
        "rust,{name},{},{:.3},{:.1},{:.1},{:.1},{:.17e}",
        problem.initial_state().len(),
        elapsed.as_nanos() as f64 / repetitions as f64,
        bytes_per_solve,
        allocations_per_solve,
        rhs_evaluations as f64 / repetitions as f64,
        checksum / repetitions as f64,
    );
}

fn main() {
    let mut repetitions = 20;
    let mut selected_algorithm = None;
    let mut positional_repetitions = None;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--repetitions" => {
                repetitions = args
                    .next()
                    .expect("--repetitions requires a value")
                    .parse()
                    .expect("repetitions must be an integer")
            }
            "--algorithm" => {
                selected_algorithm = Some(args.next().expect("--algorithm requires a value"))
            }
            value if !value.starts_with('-') && positional_repetitions.is_none() => {
                positional_repetitions =
                    Some(value.parse().expect("repetitions must be an integer"));
            }
            value => panic!("unknown benchmark argument: {value}"),
        }
    }
    if let Some(value) = positional_repetitions {
        repetitions = value;
    }

    let selected = |name: &str| {
        selected_algorithm
            .as_deref()
            .is_none_or(|filter| filter == name)
    };
    let mut ran = false;
    let nonstiff = problem(128, 0.2, 2.0);
    let stiff = problem(8, 20.0, 1.0);
    let adaptive = adaptive_options();
    let fixed = fixed_options();

    println!(
        "language,algorithm,dimension,nanoseconds_per_solve,bytes_allocated_per_solve,allocations_per_solve,rhs_evaluations_per_solve,checksum"
    );
    macro_rules! run {
        ($name:literal, $problem:expr, $algorithm:expr, $options:expr) => {
            if selected($name) {
                benchmark($name, $problem, $algorithm, $options, repetitions);
                ran = true;
            }
        };
    }
    run!("Tsit5", &nonstiff, Tsit5, &adaptive);
    run!("Midpoint", &nonstiff, Midpoint, &adaptive);
    run!("Heun", &nonstiff, Heun, &adaptive);
    run!("Ralston", &nonstiff, Ralston, &adaptive);
    run!("BS3", &nonstiff, Bs3, &adaptive);
    run!("DP5", &nonstiff, Dp5, &adaptive);
    run!("Euler", &nonstiff, Euler, &fixed);
    run!("RK4", &nonstiff, Rk4, &fixed);
    run!("RKM", &nonstiff, Rkm, &fixed);
    run!("Ralston4", &nonstiff, Ralston4, &fixed);
    run!("Alshina2", &nonstiff, Alshina2, &fixed);
    run!("Alshina3", &nonstiff, Alshina3, &fixed);
    run!("AB3", &nonstiff, Ab3, &fixed);
    run!("AB4", &nonstiff, Ab4, &fixed);
    run!("AB5", &nonstiff, Ab5, &fixed);
    run!("ABM32", &nonstiff, Abm32, &fixed);
    run!("ABM43", &nonstiff, Abm43, &fixed);
    run!("ABM54", &nonstiff, Abm54, &fixed);
    run!("SSPRK22", &nonstiff, SspRk22, &fixed);
    run!("SSPRK33", &nonstiff, SspRk33, &fixed);
    run!("SSPRK43", &nonstiff, SspRk43, &adaptive);
    run!("ImplicitEuler", &stiff, ImplicitEuler, &fixed);
    run!("ImplicitMidpoint", &stiff, ImplicitMidpoint, &fixed);
    run!("Trapezoid", &stiff, Trapezoid, &fixed);
    run!("Rosenbrock23", &stiff, Rosenbrock23, &adaptive);
    run!("TRBDF2", &stiff, Trbdf2, &adaptive);
    run!("Kvaerno5", &stiff, Kvaerno5, &adaptive);
    run!("KenCarp5", &stiff, KenCarp5, &adaptive);
    run!("Rodas4P", &stiff, Rodas4P, &adaptive);
    run!("Rodas5P", &stiff, Rodas5P, &adaptive);
    run!("Rodas5Pr", &stiff, Rodas5Pr, &adaptive);
    if !ran {
        if let Some(algorithm) = selected_algorithm.as_deref() {
            panic!("unknown benchmark algorithm: {algorithm}");
        }
    }
}
