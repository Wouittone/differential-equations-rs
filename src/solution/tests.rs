use super::{
    DenseSegment, HermiteSegment, InterpolationError, OwnedDenseSegment, Solution, SolverStats,
    TrajectoryRecorder,
};

#[test]
fn exposes_flat_states_as_slices() {
    let solution = Solution::new(
        vec![0.0, 0.5, 1.0],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        2,
        SolverStats::default(),
    );

    assert_eq!(solution.state(0), Some([1.0, 2.0].as_slice()));
    assert_eq!(solution.state(2), Some([5.0, 6.0].as_slice()));
    assert_eq!(solution.state(3), None);
    assert_eq!(solution.last_state(), &[5.0, 6.0]);
    assert_eq!(solution.interpolate(0.25), Some(vec![2.0, 3.0]));
    assert_eq!(solution.interpolate(2.0), None);

    let scalar = Solution::new(vec![0.0], vec![1.0], 1, SolverStats::default());
    assert_eq!(scalar.state(usize::MAX), None);
    assert_eq!(scalar.state_array(usize::MAX), None);
}

#[test]
fn linear_interpolation_handles_extreme_opposite_sign_endpoints() {
    let solution = Solution::new(
        vec![-f64::MAX, f64::MAX],
        vec![-f64::MAX, f64::MAX],
        1,
        SolverStats::default(),
    );

    assert_eq!(solution.try_interpolate(0.0), Ok(vec![0.0]));
}

#[test]
fn public_interpolation_rejects_non_finite_saved_and_dense_results() {
    let saved = Solution::new(vec![0.0], vec![f64::INFINITY], 1, SolverStats::default());
    assert_eq!(
        saved.try_interpolate(0.0),
        Err(InterpolationError::NonFiniteResult {
            context: "saved solution state"
        })
    );

    let linear = Solution::new(
        vec![0.0, 1.0],
        vec![0.0, f64::INFINITY],
        1,
        SolverStats::default(),
    );
    assert_eq!(
        linear.try_interpolate(0.5),
        Err(InterpolationError::NonFiniteResult { context: "linear" })
    );

    let segment = HermiteSegment::new(
        0.0,
        1.0,
        vec![0.0],
        vec![0.0],
        vec![f64::MAX],
        vec![f64::MAX],
    )
    .unwrap();
    let dense = Solution::new_with_dense(
        vec![0.0, 1.0],
        vec![0.0, 0.0],
        1,
        SolverStats::default(),
        vec![OwnedDenseSegment::Hermite(segment)],
    );
    assert_eq!(
        dense.try_interpolate(0.5),
        Err(InterpolationError::NonFiniteResult {
            context: "dense output"
        })
    );
}

#[test]
fn hermite_segment_matches_endpoints_and_midpoint() {
    let segment =
        HermiteSegment::new(0.0, 1.0, vec![0.0], vec![1.0], vec![0.0], vec![2.0]).unwrap();
    let mut output = [0.0];
    segment.interpolate(0.0, &mut output).unwrap();
    assert_eq!(output, [0.0]);
    segment.interpolate(1.0, &mut output).unwrap();
    assert_eq!(output, [1.0]);
    segment.interpolate(0.5, &mut output).unwrap();
    assert!((output[0] - 0.25).abs() < 1.0e-14);
}

#[test]
fn hermite_segment_is_checked_and_exact_at_endpoints() {
    let segment =
        HermiteSegment::new(1.0, 0.0, vec![1.0], vec![0.0], vec![2.0], vec![0.0]).unwrap();
    let mut output = [f64::NAN];
    segment.interpolate(1.0, &mut output).unwrap();
    assert_eq!(output, [1.0]);
    segment.interpolate(0.0, &mut output).unwrap();
    assert_eq!(output, [0.0]);
    assert!(segment.interpolate(1.1, &mut output).is_err());
    assert!(segment.interpolate(0.5, &mut []).is_err());
    assert!(
        HermiteSegment::new(0.0, 1.0, vec![f64::NAN], vec![1.0], vec![0.0], vec![1.0],).is_err()
    );
}

#[test]
fn recorder_uses_accepted_hermite_segment_for_save_at() {
    let options = crate::SolveOptions {
        save_at: vec![0.25, 0.75],
        ..crate::SolveOptions::default()
    };
    let mut recorder = TrajectoryRecorder::new(&[0.0], 0.0, &options);
    let segment =
        HermiteSegment::new(0.0, 1.0, vec![0.0], vec![1.0], vec![0.0], vec![3.0]).unwrap();
    recorder
        .record_step_dense(&[0.0], 0.0, &[1.0], 1.0, true, &segment)
        .unwrap();
    let solution = recorder.finish(SolverStats::default());
    assert_eq!(solution.times(), &[0.25, 0.75]);
    assert!((solution.values()[0] - 0.015625).abs() < 1.0e-14);
    assert!((solution.values()[1] - 0.421875).abs() < 1.0e-14);
}

#[test]
fn indexed_lookup_preserves_duplicate_precedence_in_both_directions() {
    for times in [vec![0.0, 1.0, 1.0, 2.0], vec![2.0, 1.0, 1.0, 0.0]] {
        let solution = super::Solution::from_saved(
            times.clone(),
            vec![0.0, 10.0, 20.0, 30.0],
            &[1],
            super::SolverStats::default(),
        )
        .unwrap();
        assert_eq!(solution.try_interpolate(1.0).unwrap(), [20.0]);
        assert_eq!(
            solution.try_interpolate((times[0] + 1.0) / 2.0).unwrap(),
            [5.0]
        );
        assert_eq!(
            solution.try_interpolate((times[3] + 1.0) / 2.0).unwrap(),
            [25.0]
        );
        assert!(solution.try_interpolate(-1.0).is_err());
        assert!(solution.try_interpolate(3.0).is_err());
    }
}
