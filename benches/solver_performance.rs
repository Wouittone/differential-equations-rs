use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use differential_equations::ndarray::{Array2, ArrayView2, ArrayViewMut2, array};
use differential_equations::solve_ensemble_parallel;
use differential_equations::solvers::automatic::{AutoSwitchConfig, AutoTsit5, AutomaticBranch};
use differential_equations::solvers::explicit::{CKLLSRK95_4M, RDPK3SpFSAL510, Tsit5};
use differential_equations::solvers::rosenbrock::{Rodas5P, Tsit5DA};
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

fn moderate_decay(derivative: &mut [f64], state: &[f64], _: &(), _: f64) {
    derivative[0] = -10.0 * state[0];
}

fn moderate_decay_jacobian(jacobian: &mut [f64], _: &[f64], _: &(), _: f64) {
    jacobian[0] = -10.0;
}

fn switching_tracking(derivative: &mut [f64], state: &[f64], _: &(), time: f64) {
    let rate = if (0.25..0.75).contains(&time) {
        100.0
    } else {
        1.0
    };
    derivative[0] = -rate * (state[0] - time.cos()) - time.sin();
}

fn switching_tracking_jacobian(jacobian: &mut [f64], _: &[f64], _: &(), time: f64) {
    jacobian[0] = if (0.25..0.75).contains(&time) {
        -100.0
    } else {
        -1.0
    };
}

fn explicit_problem() -> OdeProblem<EmptyRhs, ()> {
    OdeProblem::new(lorenz as EmptyRhs, [1.0, 0.0, 0.0], (0.0, 10.0), ())
}

fn stiff_problem() -> OdeProblem<EmptyRhs, ()> {
    OdeProblem::new(stiff_tracking as EmptyRhs, [1.0], (0.0, 1.0), ())
        .with_jacobian(stiff_tracking_jacobian)
}

fn automatic_problem() -> OdeProblem<EmptyRhs, ()> {
    OdeProblem::new(moderate_decay as EmptyRhs, [1.0], (0.0, 1.0), ())
        .with_jacobian(moderate_decay_jacobian)
}

fn repeated_switch_problem() -> OdeProblem<EmptyRhs, ()> {
    OdeProblem::new(switching_tracking as EmptyRhs, [1.0], (0.0, 1.0), ())
        .with_jacobian(switching_tracking_jacobian)
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
    RDPK3SpFSAL510
        .tableau()
        .expect("built-in tableau must parse");
    CKLLSRK95_4M.tableau().expect("built-in tableau must parse");
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

    group.bench_function("low_storage/rdpk3spfsal510_lorenz", |bencher| {
        bencher.iter(|| {
            let solution = solve(black_box(&explicit), RDPK3SpFSAL510, black_box(&options))
                .expect("benchmark problem must solve");
            black_box(solution.last_state()[0]);
        });
    });

    group.bench_function("low_storage/ckllsrk95_4m_lorenz", |bencher| {
        bencher.iter(|| {
            let solution = solve(black_box(&explicit), CKLLSRK95_4M, black_box(&options))
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

fn automatic_switching(criterion: &mut Criterion) {
    // Fixed steps keep the compared workloads identical: differences in these
    // cases come from detector dispatch or branch handoff, not from divergent
    // adaptive-controller histories.
    let problem = automatic_problem();
    let repeated_problem = repeated_switch_problem();
    let options = fixed_options(0.01);
    let adaptive = adaptive_options();
    let no_switch = AutoTsit5::new(Rodas5P)
        .with_switch_config(
            AutoSwitchConfig::new()
                .with_stiffness_thresholds(1.0e100, 1.0e200)
                .expect("benchmark thresholds must be valid"),
        )
        .expect("benchmark switching configuration must be valid");
    let one_switch = AutoTsit5::new(Rodas5P)
        .with_switch_config(
            AutoSwitchConfig::new()
                .with_stiffness_thresholds(1.0e-12, 2.0e-12)
                .expect("benchmark thresholds must be valid")
                .with_stiff_confirmations(1)
                .expect("benchmark confirmation count must be valid")
                .with_minimum_residence_steps(1)
                .expect("benchmark residence count must be valid")
                .with_switch_step_factor(1.0)
                .expect("benchmark step factor must be valid")
                .with_allow_switch_back(false),
        )
        .expect("benchmark switching configuration must be valid");
    let two_switch = AutoTsit5::new(Rodas5P)
        .with_switch_config(
            AutoSwitchConfig::new()
                .with_stiffness_thresholds(0.01, 0.02)
                .expect("benchmark thresholds must be valid")
                .with_stiff_confirmations(1)
                .expect("benchmark confirmation count must be valid")
                .with_nonstiff_confirmations(1)
                .expect("benchmark confirmation count must be valid")
                .with_minimum_residence_steps(1)
                .expect("benchmark residence count must be valid")
                .with_switch_step_factor(1.0)
                .expect("benchmark step factor must be valid"),
        )
        .expect("benchmark switching configuration must be valid");
    let repeated_options = fixed_options(0.01).with_time_stops([0.25, 0.75]);

    // Fail before measurement if policy changes invalidate either workload.
    let no_switch_probe =
        solve(&problem, no_switch.clone(), &options).expect("benchmark problem must solve");
    assert_eq!(no_switch_probe.stats().algorithm_switches, 0);
    assert_eq!(
        no_switch_probe.stats().final_automatic_branch,
        Some(AutomaticBranch::NonStiff)
    );
    let one_switch_probe =
        solve(&problem, one_switch.clone(), &options).expect("benchmark problem must solve");
    assert_eq!(one_switch_probe.stats().algorithm_switches, 1);
    assert_eq!(one_switch_probe.stats().nonstiff_accepted_steps, 1);
    assert_eq!(
        one_switch_probe.stats().final_automatic_branch,
        Some(AutomaticBranch::Stiff)
    );
    let adaptive_probe = solve(&problem, no_switch.clone(), &adaptive)
        .expect("adaptive benchmark problem must solve");
    assert_eq!(adaptive_probe.stats().algorithm_switches, 0);
    let two_switch_probe = solve(&repeated_problem, two_switch.clone(), &repeated_options)
        .expect("repeated-switch benchmark problem must solve");
    assert_eq!(two_switch_probe.stats().algorithm_switches, 2);
    assert_eq!(
        two_switch_probe.stats().final_automatic_branch,
        Some(AutomaticBranch::NonStiff)
    );

    let mut group = criterion.benchmark_group("automatic_switching");
    group.bench_function("no_switch/explicit_tsit5", |bencher| {
        bencher.iter(|| {
            let solution = solve(black_box(&problem), Tsit5, black_box(&options))
                .expect("benchmark problem must solve");
            black_box(solution.last_state()[0]);
        });
    });
    group.bench_function("no_switch/auto_tsit5", |bencher| {
        bencher.iter(|| {
            let solution = solve(
                black_box(&problem),
                black_box(no_switch.clone()),
                black_box(&options),
            )
            .expect("benchmark problem must solve");
            black_box(solution.last_state()[0]);
        });
    });
    group.bench_function("one_switch/auto_tsit5_rodas5p", |bencher| {
        bencher.iter(|| {
            let solution = solve(
                black_box(&problem),
                black_box(one_switch.clone()),
                black_box(&options),
            )
            .expect("benchmark problem must solve");
            black_box(solution.last_state()[0]);
        });
    });
    group.bench_function("one_switch/stiff_rodas5p", |bencher| {
        bencher.iter(|| {
            let solution = solve(black_box(&problem), Rodas5P, black_box(&options))
                .expect("benchmark problem must solve");
            black_box(solution.last_state()[0]);
        });
    });
    group.bench_function("adaptive_no_switch/explicit_tsit5", |bencher| {
        bencher.iter(|| {
            let solution = solve(black_box(&problem), Tsit5, black_box(&adaptive))
                .expect("benchmark problem must solve");
            black_box(solution.last_state()[0]);
        });
    });
    group.bench_function("adaptive_no_switch/auto_tsit5", |bencher| {
        bencher.iter(|| {
            let solution = solve(
                black_box(&problem),
                black_box(no_switch.clone()),
                black_box(&adaptive),
            )
            .expect("benchmark problem must solve");
            black_box(solution.last_state()[0]);
        });
    });
    group.bench_function("two_switch/auto_tsit5_rodas5p", |bencher| {
        bencher.iter(|| {
            let solution = solve(
                black_box(&repeated_problem),
                black_box(two_switch.clone()),
                black_box(&repeated_options),
            )
            .expect("benchmark problem must solve");
            black_box(solution.last_state()[0]);
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

fn hybrid_workspace_scaling(criterion: &mut Criterion) {
    // Short solves expose workspace construction costs. Tableau parsing and
    // problem construction are deliberately outside the measured iterations.
    Tsit5DA.tableau().expect("built-in tableau must parse");
    let options = fixed_options(0.01);
    let mut group = criterion.benchmark_group("hybrid_workspace");
    for dimension in [128, 256, 1024] {
        let problem = OdeProblem::from_array(
            |mut du: ArrayViewMut2<'_, f64>, u: ArrayView2<'_, f64>, _: &(), _| {
                du.zip_mut_with(&u, |du, u| *du = -*u);
            },
            Array2::ones((8, dimension / 8)),
            (0.0, 0.02),
            (),
        );
        let probe = solve(&problem, Tsit5DA, &options).expect("benchmark problem must solve");
        assert_eq!(probe.stats().accepted_steps, 2);
        assert_eq!(probe.stats().linear_factorizations, 0);
        group.throughput(Throughput::Elements(dimension as u64));
        group.bench_function(BenchmarkId::new("tsit5da_matrix", dimension), |bencher| {
            bencher.iter(|| {
                let solution = solve(black_box(&problem), Tsit5DA, black_box(&options))
                    .expect("benchmark problem must solve");
                black_box(solution.last_state());
            });
        });
    }
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
    targets = solver_throughput,
        automatic_switching,
        dense_output,
        hybrid_workspace_scaling,
        ensembles
}
criterion_main!(benchmarks);
