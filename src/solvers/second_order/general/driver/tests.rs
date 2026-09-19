use super::*;
use crate::InterpolationError;

fn saved_solution(velocities: Vec<f64>, positions: Vec<f64>) -> SecondOrderSolution {
    SecondOrderSolution {
        times: vec![0.0, 1.0],
        velocities,
        positions,
        dimension: 1,
        state_shape: IxDyn(&[1]),
        stats: SolverStats::default(),
        dense_segments: Vec::new(),
    }
}

#[test]
fn saved_interpolation_is_bounded_for_opposite_sign_finite_endpoints() {
    let mut solution = saved_solution(vec![f64::MAX, -f64::MAX], vec![-f64::MAX, f64::MAX]);

    let (velocity, position) = solution.try_interpolate(0.25).unwrap();

    assert!((velocity[0] / f64::MAX - 0.5).abs() <= f64::EPSILON);
    assert!((position[0] / f64::MAX + 0.5).abs() <= f64::EPSILON);

    solution.times = vec![-f64::MAX, f64::MAX];
    assert_eq!(solution.try_interpolate(0.0), Ok((vec![0.0], vec![0.0])));
}

#[test]
fn exact_saved_interpolation_rejects_non_finite_partitions() {
    let solution = saved_solution(vec![1.0, 2.0], vec![f64::INFINITY, 3.0]);

    assert_eq!(
        solution.try_interpolate(0.0),
        Err(InterpolationError::NonFiniteResult {
            context: "saved second-order state",
        })
    );
}

#[test]
fn retained_dense_interpolation_rejects_non_finite_output() {
    let mut solution = saved_solution(vec![1.0, 1.0], vec![1.0, 1.0]);
    solution.times[1] = 2.0;
    solution.dense_segments.push(PartitionedDenseSegment::new(
        0.0,
        2.0,
        &[f64::MAX],
        &[-f64::MAX],
        &[f64::MAX],
        &[f64::MAX],
    ));

    assert_eq!(
        solution.try_interpolate(1.0),
        Err(InterpolationError::NonFiniteResult {
            context: "second-order dense segment",
        })
    );
}
