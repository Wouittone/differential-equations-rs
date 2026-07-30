use std::alloc::System;
use std::hint::black_box;
use std::time::Instant;

use differential_equations::{
    Ab3, Ab4, Ab5, Abm32, Abm43, Abm54, Alshina2, Alshina3, Bs3, Dp5, Euler, Heun, ImplicitEuler,
    ImplicitMidpoint, Midpoint, OdeAlgorithm, OdeProblem, Ralston, Ralston4, Rk4, Rkm,
    Rosenbrock23, SaveMode, SolveOptions, SspRk22, SspRk33, SspRk43, Trapezoid, Tsit5, solve,
};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

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
    let allocations = region.change();

    println!(
        "rust,{name},{},{:.3},{:.1},{:.1},{:.1},{:.17e}",
        problem.initial_state().len(),
        elapsed.as_nanos() as f64 / repetitions as f64,
        allocations.bytes_allocated as f64 / repetitions as f64,
        allocations.allocations as f64 / repetitions as f64,
        rhs_evaluations as f64 / repetitions as f64,
        checksum / repetitions as f64,
    );
}

fn main() {
    let repetitions = std::env::args()
        .nth(1)
        .map(|value| value.parse().expect("repetitions must be an integer"))
        .unwrap_or(20);
    let nonstiff = problem(128, 0.2, 2.0);
    let stiff = problem(8, 20.0, 1.0);
    let adaptive = adaptive_options();
    let fixed = fixed_options();

    println!(
        "language,algorithm,dimension,nanoseconds_per_solve,bytes_allocated_per_solve,allocations_per_solve,rhs_evaluations_per_solve,checksum"
    );
    benchmark("Tsit5", &nonstiff, Tsit5, &adaptive, repetitions);
    benchmark("Midpoint", &nonstiff, Midpoint, &adaptive, repetitions);
    benchmark("Heun", &nonstiff, Heun, &adaptive, repetitions);
    benchmark("Ralston", &nonstiff, Ralston, &adaptive, repetitions);
    benchmark("BS3", &nonstiff, Bs3, &adaptive, repetitions);
    benchmark("DP5", &nonstiff, Dp5, &adaptive, repetitions);
    benchmark("Euler", &nonstiff, Euler, &fixed, repetitions);
    benchmark("RK4", &nonstiff, Rk4, &fixed, repetitions);
    benchmark("RKM", &nonstiff, Rkm, &fixed, repetitions);
    benchmark("Ralston4", &nonstiff, Ralston4, &fixed, repetitions);
    benchmark("Alshina2", &nonstiff, Alshina2, &fixed, repetitions);
    benchmark("Alshina3", &nonstiff, Alshina3, &fixed, repetitions);
    benchmark("AB3", &nonstiff, Ab3, &fixed, repetitions);
    benchmark("AB4", &nonstiff, Ab4, &fixed, repetitions);
    benchmark("AB5", &nonstiff, Ab5, &fixed, repetitions);
    benchmark("ABM32", &nonstiff, Abm32, &fixed, repetitions);
    benchmark("ABM43", &nonstiff, Abm43, &fixed, repetitions);
    benchmark("ABM54", &nonstiff, Abm54, &fixed, repetitions);
    benchmark("SSPRK22", &nonstiff, SspRk22, &fixed, repetitions);
    benchmark("SSPRK33", &nonstiff, SspRk33, &fixed, repetitions);
    benchmark("SSPRK43", &nonstiff, SspRk43, &adaptive, repetitions);
    benchmark("ImplicitEuler", &stiff, ImplicitEuler, &fixed, repetitions);
    benchmark(
        "ImplicitMidpoint",
        &stiff,
        ImplicitMidpoint,
        &fixed,
        repetitions,
    );
    benchmark("Trapezoid", &stiff, Trapezoid, &fixed, repetitions);
    benchmark("Rosenbrock23", &stiff, Rosenbrock23, &adaptive, repetitions);
}
