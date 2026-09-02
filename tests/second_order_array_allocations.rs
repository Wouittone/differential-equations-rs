use std::alloc::System;
use std::hint::black_box;

use differential_equations::ndarray::{Array, ArrayView, ArrayViewMut, Dimension, arr0, array};
use differential_equations::solvers::second_order::SecondOrderOdeProblem;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[path = "support/allocation.rs"]
mod allocation_support;

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

fn assert_evaluation_does_not_allocate<D: Dimension>(initial: Array<f64, D>) {
    let problem = SecondOrderOdeProblem::from_array(
        |mut a: ArrayViewMut<'_, f64, D>,
         v: ArrayView<'_, f64, D>,
         q: ArrayView<'_, f64, D>,
         _: &(),
         _| {
            a.assign(&q);
            a.zip_mut_with(&v, |a, v| *a = -*a - *v);
        },
        Array::zeros(initial.raw_dim()),
        initial,
        (0.0, 1.0),
        (),
    )
    .unwrap();
    let mut output = vec![0.0; problem.initial_position().len()];
    let allocations = allocation_support::minimum_measurement(|| {
        let region = Region::new(GLOBAL);
        for _ in 0..1000 {
            problem
                .evaluate_acceleration(
                    &mut output,
                    problem.initial_velocity(),
                    problem.initial_position(),
                    0.0,
                )
                .unwrap();
            black_box(&output);
        }
        region.change().allocations
    });
    assert_eq!(allocations, 0);
    assert!(output.iter().all(|a| *a == -1.0));
}

#[test]
fn scalar_vector_and_matrix_in_place_adapters_do_not_allocate_per_evaluation() {
    assert_evaluation_does_not_allocate(arr0(1.0));
    assert_evaluation_does_not_allocate(array![1.0, 1.0]);
    assert_evaluation_does_not_allocate(array![[1.0, 1.0], [1.0, 1.0]]);
}
