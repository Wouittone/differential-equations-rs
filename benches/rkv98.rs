use std::time::Duration;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use differential_equations::{
    stepping::{
        AdaptiveController, ControllerConfig, ExplicitRungeKuttaStepper, Observation,
        ObserverAction, integrate_rk,
    },
    tableau::RungeKuttaCoefficients,
};

#[path = "../tableau-core/tests/fixtures/numeris_rkv98.rs"]
mod numeris_rkv98;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Work {
    attempts: usize,
    rhs_evaluations: usize,
    accepted_steps: usize,
    rejected_steps: usize,
    stage_evaluations: usize,
    dense_stage_evaluations: usize,
}

fn tableau() -> differential_equations::tableau::RungeKuttaTableau {
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
                .map(|(stage, &value)| (stage, value))
                .collect()
        })
        .collect();
    let lazy: Vec<_> = sparse
        .iter()
        .enumerate()
        .map(
            |(index, coefficients)| differential_equations::tableau::LazyDenseStageCoefficients {
                node: numeris_rkv98::C[16 + index],
                coefficients,
            },
        )
        .collect();
    let mut coefficients = RungeKuttaCoefficients::explicit(
        "Numeris RKV98",
        9,
        &rows,
        &numeris_rkv98::B[..16],
        &numeris_rkv98::C[..16],
    );
    coefficients.embedded_order = Some(8);
    coefficients.b_hat = Some(&numeris_rkv98::BHAT[..16]);
    coefficients.dense = Some(&dense);
    coefficients.lazy_dense_stages = &lazy;
    coefficients.build().expect("RKV98 fixture must build")
}

fn rhs(_: f64, state: &[f64], derivative: &mut [f64]) -> Result<(), ()> {
    derivative[0] = -state[0];
    Ok(())
}

fn direct_kernel(tableau: &differential_equations::tableau::RungeKuttaTableau) -> Work {
    let mut stepper = ExplicitRungeKuttaStepper::new(tableau, 0.0, &[1.0]).unwrap();
    for _ in 0..64 {
        let view = stepper.attempt(1.0 / 64.0, &mut rhs).unwrap();
        assert_eq!(view.stage_derivatives.len(), tableau.stages());
        stepper.accept().unwrap();
    }
    let statistics = stepper.statistics();
    Work {
        attempts: statistics.attempts,
        rhs_evaluations: statistics.rhs_evaluations,
        accepted_steps: statistics.accepted_steps,
        rejected_steps: statistics.rejected_steps,
        stage_evaluations: statistics.rhs_evaluations,
        dense_stage_evaluations: 0,
    }
}

fn adaptive_driver(tableau: &differential_equations::tableau::RungeKuttaTableau) -> Work {
    let mut stepper = ExplicitRungeKuttaStepper::new(tableau, 0.0, &[1.0]).unwrap();
    let mut config = ControllerConfig::proportional(8).unwrap();
    config.minimum_factor = 1.0;
    config.maximum_factor = 1.0;
    config.rejection_maximum = 1.0;
    config.rejected_acceptance_maximum = 1.0;
    let mut controller = AdaptiveController::new(config, 1.0 / 64.0).unwrap();
    let mut norm = |_: &[f64], _: &[f64], _: &[f64]| Ok::<_, ()>(0.0);
    integrate_rk(
        &mut stepper,
        &mut controller,
        1.0,
        &[],
        64,
        &mut rhs,
        &mut norm,
        &mut |_: Observation<'_>| Ok::<_, ()>(ObserverAction::Continue),
    )
    .unwrap();
    let statistics = stepper.statistics();
    Work {
        attempts: statistics.attempts,
        rhs_evaluations: statistics.rhs_evaluations,
        accepted_steps: statistics.accepted_steps,
        rejected_steps: statistics.rejected_steps,
        stage_evaluations: statistics.rhs_evaluations,
        dense_stage_evaluations: 0,
    }
}

fn rkv98_stage_kernel_vs_driver(c: &mut Criterion) {
    let tableau = tableau();
    let direct = direct_kernel(&tableau);
    let driver = adaptive_driver(&tableau);
    assert_eq!(direct, driver);
    eprintln!(
        "rkv98 diagnostics: stages={} dense_stages={} attempts={} rhs={} accepted={} rejected={}",
        tableau.stages(),
        numeris_rkv98::A.len() - tableau.stages(),
        direct.attempts,
        direct.rhs_evaluations,
        direct.accepted_steps,
        direct.rejected_steps
    );

    let mut group = c.benchmark_group("rkv98");
    group.bench_function("stage_kernel", |bencher| {
        bencher.iter(|| black_box(direct_kernel(black_box(&tableau))));
    });
    group.bench_function("adaptive_driver", |bencher| {
        bencher.iter(|| black_box(adaptive_driver(black_box(&tableau))));
    });
    group.finish();
}

criterion_group! {
    name = benchmarks;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5));
    targets = rkv98_stage_kernel_vs_driver
}
criterion_main!(benchmarks);
