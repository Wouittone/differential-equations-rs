use std::time::Duration;

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use differential_equations::ndarray::{ArrayView2, ArrayViewMut2, array};
use differential_equations::solve_ensemble_parallel;
use differential_equations::solvers::explicit::Tsit5;
use differential_equations::solvers::rosenbrock::Rodas5P;
use differential_equations::{
    OdeProblem, SaveMode, SolveOptions, solve, solve_ensemble_sequential,
};

type EmptyRhs = fn(&mut [f64], &[f64], &(), f64);
type RateRhs = fn(&mut [f64], &[f64], &f64, f64);

fn lorenz(derivative: &mut [f64], state: &[f64], _: &(), _: f64) {
    derivative[0] = 10.0 * (state[1] - state[0]);
    derivative[1] = state[0] * (28.0 - state[2]) - state[1];
    derivative[2] = state[0] * state[1] - (8.0 / 3.0) * state[2];
}

fn stiff_tracking(derivative: &mut [f64], state: &[f64], _: &(), time: f64) {
    derivative[0] = -1_000.0 * (state[0] - time.cos()) - time.sin();
}

fn stiff_tracking_jacobian(jacobian: &mut [f64], _: &[f64], _: &(), _: f64) {
    jacobian[0] = -1_000.0;
}

fn exponential(derivative: &mut [f64], state: &[f64], rate: &f64, _: f64) {
    derivative[0] = *rate * state[0];
}

fn explicit_problem() -> OdeProblem<EmptyRhs, ()> {
    OdeProblem::new(lorenz as EmptyRhs, [1.0, 0.0, 0.0], (0.0, 10.0), ())
}

fn stiff_problem() -> OdeProblem<EmptyRhs, ()> {
    OdeProblem::new(stiff_tracking as EmptyRhs, [1.0], (0.0, 1.0), ())
        .with_jacobian(stiff_tracking_jacobian)
}

fn ensemble_problem(case: (f64, f64)) -> OdeProblem<RateRhs, f64> {
    let (initial, rate) = case;
    OdeProblem::new(exponential as RateRhs, [initial], (0.0, 2.0), rate)
}

fn adaptive_options() -> SolveOptions {
    SolveOptions::new()
        .with_tolerances(1.0e-8, 1.0e-8)
        .with_save(SaveMode::Endpoints)
}

fn fixed_options(step: f64) -> SolveOptions {
    SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(step)
        .with_save(SaveMode::Endpoints)
}

fn solver_throughput(criterion: &mut Criterion) {
    let explicit = explicit_problem();
    let stiff = stiff_problem();
    let matrix = OdeProblem::from_array(
        |mut derivative: ArrayViewMut2<'_, f64>, state: ArrayView2<'_, f64>, _: &(), _: f64| {
            derivative.zip_mut_with(&state, |derivative, state| *derivative = -*state);
        },
        array![[1.0, 2.0], [3.0, 4.0]],
        (0.0, 10.0),
        (),
    );
    let options = adaptive_options();
    let mut group = criterion.benchmark_group("solver");

    group.bench_function("explicit/tsit5_lorenz", |bencher| {
        bencher.iter(|| {
            let solution = solve(black_box(&explicit), Tsit5, black_box(&options))
                .expect("benchmark problem must solve");
            black_box(solution.last_state()[0]);
        });
    });

    group.bench_function("stiff/rodas5p_tracking", |bencher| {
        bencher.iter(|| {
            let solution = solve(black_box(&stiff), Rodas5P, black_box(&options))
                .expect("benchmark problem must solve");
            black_box(solution.last_state()[0]);
        });
    });

    group.bench_function("ndarray/tsit5_matrix", |bencher| {
        bencher.iter(|| {
            let solution = solve(black_box(&matrix), Tsit5, black_box(&options))
                .expect("benchmark problem must solve");
            black_box(solution.last_state_array()[[0, 0]]);
        });
    });

    group.finish();
}

fn dense_output(criterion: &mut Criterion) {
    let problem = explicit_problem();
    let options = fixed_options(0.02).with_dense_output(true);
    let solution = solve(&problem, Tsit5, &options).expect("benchmark problem must solve");
    let query_times: Vec<_> = (1..=64).map(|index| 10.0 * index as f64 / 65.0).collect();

    let mut group = criterion.benchmark_group("dense_output");
    group.throughput(Throughput::Elements(query_times.len() as u64));
    group.bench_function("tsit5_interpolate_64", |bencher| {
        bencher.iter(|| {
            for &time in &query_times {
                let state = solution
                    .interpolate(black_box(time))
                    .expect("query lies inside the solved interval");
                black_box(state);
            }
        });
    });
    group.finish();
}

fn ensembles(criterion: &mut Criterion) {
    let cases: Vec<_> = (0..64)
        .map(|index| (1.0 + index as f64 / 64.0, -0.5 - index as f64 / 256.0))
        .collect();
    let options = fixed_options(0.01);

    let sequential_probe =
        solve_ensemble_sequential(cases.iter().copied(), ensemble_problem, Tsit5, &options);
    assert!(
        sequential_probe
            .iter()
            .all(|outcome| outcome.result.is_ok())
    );

    // Validate the workload and initialize Rayon's pool outside timed iterations.
    let parallel_probe =
        solve_ensemble_parallel(cases.iter().copied(), ensemble_problem, Tsit5, &options);
    assert!(parallel_probe.iter().all(|outcome| outcome.result.is_ok()));

    let mut group = criterion.benchmark_group("ensemble");
    group.throughput(Throughput::Elements(cases.len() as u64));
    group.bench_function("sequential/64_cases", |bencher| {
        bencher.iter(|| {
            let outcomes = solve_ensemble_sequential(
                black_box(cases.iter().copied()),
                ensemble_problem,
                Tsit5,
                black_box(&options),
            );
            black_box(outcomes);
        });
    });
    group.bench_function("parallel/64_cases", |bencher| {
        bencher.iter(|| {
            let outcomes = solve_ensemble_parallel(
                black_box(cases.iter().copied()),
                ensemble_problem,
                Tsit5,
                black_box(&options),
            );
            black_box(outcomes);
        });
    });
    group.finish();
}

criterion_group! {
    name = benchmarks;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5));
    targets = solver_throughput, dense_output, ensembles
}
criterion_main!(benchmarks);
