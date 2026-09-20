use differential_equations::solvers::explicit::Tsit5;
use differential_equations::{
    DenseSegmentData, InterpolationQuality, OdeProblem, PortableDenseSegment, Solution,
    SolveOptions, solve,
};

fn polynomial(backward: bool) -> PortableDenseSegment {
    PortableDenseSegment::from_data(DenseSegmentData {
        version: 1,
        start_time: if backward { 2.0 } else { 0.0 },
        end_time: if backward { 0.0 } else { 2.0 },
        bound_time: 1.0,
        dimension: 1,
        coefficients: vec![1.0, 2.0, 3.0],
        end_state: vec![6.0],
        bound_state: None,
        quality: InterpolationQuality::MethodSpecific,
    })
    .unwrap()
}
#[test]
fn polynomial_bounds_and_validation() {
    for backward in [false, true] {
        let p = polynomial(backward);
        let mut output = [0.0];
        p.interpolate_into(1.0, &mut output).unwrap();
        assert_eq!(output, [2.75]);
        assert!(
            p.interpolate_into(if backward { 0.5 } else { 1.5 }, &mut output)
                .is_err()
        );
        let mut bad = p.into_data();
        bad.version = 2;
        assert!(PortableDenseSegment::from_data(bad).is_err());
    }
}
#[test]
fn export_import_and_quality_preserve_dense_accuracy() {
    for span in [(0.0_f64, 1.0), (1.0, 0.0)] {
        let problem = OdeProblem::new(
            |dy: &mut [f64], y: &[f64], _: &(), _| dy[0] = y[0],
            [span.0.exp()],
            span,
            (),
        );
        let solution = solve(
            &problem,
            Tsit5,
            &SolveOptions::default()
                .with_tolerances(1e-10, 1e-10)
                .with_dense_output(true),
        )
        .unwrap();
        let imported = Solution::from_data(solution.export_data().unwrap()).unwrap();
        for k in 0..101 {
            let t = k as f64 / 100.0;
            let original = solution.try_interpolate(t).unwrap()[0];
            let portable = imported.try_interpolate(t).unwrap()[0];
            assert!(
                (original - portable).abs() < 2e-14,
                "{t} {original} {portable}"
            );
            assert!((portable - t.exp()).abs() < 1e-8);
        }
    }
    let linear =
        Solution::from_saved(vec![0.0, 1.0], vec![0.0, 1.0], &[1], Default::default()).unwrap();
    assert_eq!(
        linear.interpolation_quality(0.5).unwrap(),
        InterpolationQuality::Linear
    );
    assert!(linear.try_interpolate_method_into(0.5, &mut [0.0]).is_err());
    assert_eq!(
        linear.interpolation_quality(1.0).unwrap(),
        InterpolationQuality::ExactSavedState
    );
}
#[cfg(feature = "serde")]
#[test]
fn serde_is_versioned_and_validated() {
    let p = polynomial(false);
    let json = serde_json::to_string(&p).unwrap();
    let loaded: PortableDenseSegment = serde_json::from_str(&json).unwrap();
    assert_eq!(p, loaded);
    let bad = json.replace("\"version\":1", "\"version\":99");
    assert!(serde_json::from_str::<PortableDenseSegment>(&bad).is_err());
    let bad = json.replace("\"dimension\":1", "\"dimension\":2");
    assert!(serde_json::from_str::<PortableDenseSegment>(&bad).is_err());
}

#[test]
fn malformed_trajectory_segments_rejected() {
    let solution =
        Solution::from_saved(vec![0.0, 2.0], vec![1.0, 6.0], &[1], Default::default()).unwrap();
    let mut data = solution.export_data().unwrap();
    data.segments = vec![polynomial(false), polynomial(false)];
    assert!(Solution::from_data(data).is_err());
    let mut data = solution.export_data().unwrap();
    data.version = 0;
    assert!(Solution::from_data(data).is_err());
}
#[cfg(feature = "serde")]
#[test]
fn serialized_solution_preserves_dense_and_repeated_states() {
    let problem = OdeProblem::new(
        |dy: &mut [f64], y: &[f64], _: &(), _| dy[0] = -y[0],
        [1.0],
        (0.0, 1.0),
        (),
    );
    let solution = solve(
        &problem,
        Tsit5,
        &SolveOptions::default().with_dense_output(true),
    )
    .unwrap();
    let json = serde_json::to_string(&solution.export_data().unwrap()).unwrap();
    let data: differential_equations::SolutionData = serde_json::from_str(&json).unwrap();
    let restored = Solution::from_data(data).unwrap();
    assert_eq!(restored.times(), solution.times());
    assert_eq!(restored.values(), solution.values());
    assert_eq!(restored.stats(), solution.stats());
    assert!(
        (restored.try_interpolate(0.37).unwrap()[0] - solution.try_interpolate(0.37).unwrap()[0])
            .abs()
            < 1e-14
    );
}

#[test]
fn dense_export_can_extend_beyond_requested_saved_times() {
    for (span, saves) in [
        ((0.0, 1.0), vec![0.4, 0.6]),
        ((1.0, 0.0), vec![0.6, 0.4]),
        ((1.0, 0.0), vec![0.5]),
    ] {
        let p = OdeProblem::new(
            |dy: &mut [f64], _: &[f64], _: &(), _| dy[0] = 1.0,
            [span.0],
            span,
            (),
        );
        let s = solve(
            &p,
            Tsit5,
            &SolveOptions::default()
                .with_dense_output(true)
                .with_save_at(saves),
        )
        .unwrap();
        let restored = Solution::from_data(s.export_data().unwrap()).unwrap();
        for t in [0.1, 0.5, 0.9] {
            assert!((restored.try_interpolate(t).unwrap()[0] - t).abs() < 1e-12);
        }
    }
}

#[test]
fn callback_jump_keeps_last_saved_state_precedence_after_export() {
    use differential_equations::{CallbackAction, CallbackSave};
    for span in [(0.0, 1.0), (1.0, 0.0)] {
        let p = OdeProblem::new(
            |dy: &mut [f64], _: &[f64], _: &(), _| dy[0] = 1.0,
            [span.0],
            span,
            (),
        )
        .with_preset_time_callback_saving(
            [0.5],
            CallbackSave::Both,
            |state: &mut [f64], _: &(), _| {
                state[0] += 10.0;
                CallbackAction::Continue
            },
        );
        let s = solve(&p, Tsit5, &SolveOptions::default().with_dense_output(true)).unwrap();
        let imported = Solution::from_data(s.export_data().unwrap()).unwrap();
        assert_eq!(
            imported.try_interpolate(0.5).unwrap(),
            s.try_interpolate(0.5).unwrap()
        );
        assert!((imported.try_interpolate(0.5).unwrap()[0] - 10.5).abs() < 1e-12);
        for t in [0.49, 0.51] {
            assert!(
                (imported.try_interpolate(t).unwrap()[0] - s.try_interpolate(t).unwrap()[0]).abs()
                    < 2e-14
            );
        }
    }
}

#[cfg(feature = "serde")]
#[test]
fn direct_solution_serde_rejects_invalid_versions_and_saved_dimensions() {
    let s = Solution::from_saved(vec![0.0, 1.0], vec![1.0, 2.0], &[1], Default::default()).unwrap();
    let json = serde_json::to_string(&s).unwrap();
    let restored: Solution = serde_json::from_str(&json).unwrap();
    assert_eq!(s, restored);
    let wrong_version = json.replace("\"version\":1", "\"version\":0");
    assert!(serde_json::from_str::<Solution>(&wrong_version).is_err());
    let wrong_shape = json.replace("\"state_shape\":[1]", "\"state_shape\":[2]");
    assert!(serde_json::from_str::<Solution>(&wrong_shape).is_err());
}

#[test]
fn zero_length_clipped_backward_segments_do_not_break_indexing() {
    let mut data = Solution::from_saved(vec![2.0, 0.0], vec![0.0, 0.0], &[1], Default::default())
        .unwrap()
        .export_data()
        .unwrap();
    for (start, end, bound, value) in [
        (2.0, 1.0, 1.0, 10.0),
        (1.0, 0.0, 1.0, 20.0),
        (1.0, 0.0, 0.0, 30.0),
    ] {
        data.segments.push(
            PortableDenseSegment::from_data(DenseSegmentData {
                version: 1,
                start_time: start,
                end_time: end,
                bound_time: bound,
                dimension: 1,
                coefficients: vec![value],
                end_state: vec![value],
                bound_state: None,
                quality: InterpolationQuality::MethodSpecific,
            })
            .unwrap(),
        );
    }
    let solution = Solution::from_data(data).unwrap();
    assert_eq!(solution.try_interpolate(0.5).unwrap(), [30.0]);
    assert_eq!(
        solution.interpolation_quality(0.5).unwrap(),
        InterpolationQuality::MethodSpecific
    );
    assert_eq!(solution.try_interpolate(1.0).unwrap(), [10.0]);
}
#[test]
fn indexed_saved_lookup_matches_linear_reference_for_irregular_repeated_times() {
    for n in [1, 2, 3, 17, 1024] {
        for backward in [false, true] {
            let mut times = Vec::new();
            let mut values = Vec::new();
            let mut t = 0.0;
            for i in 0..n {
                if i % 7 != 0 {
                    t += (i % 5 + 1) as f64 * 0.1;
                }
                times.push(if backward { -t } else { t });
                values.push((i % 13) as f64);
            }
            let solution =
                Solution::from_saved(times.clone(), values.clone(), &[1], Default::default())
                    .unwrap();
            for i in 0..=3 * n {
                let query = if i == 3 * n {
                    times[n - 1]
                } else {
                    times[0] + (times[n - 1] - times[0]) * i as f64 / (3 * n) as f64
                };
                let exact = times.iter().rposition(|&t| t == query);
                let expected = if let Some(j) = exact {
                    values[j]
                } else {
                    let j = (1..n)
                        .find(|&j| {
                            query >= times[j - 1].min(times[j])
                                && query <= times[j - 1].max(times[j])
                        })
                        .unwrap();
                    let theta = (query - times[j - 1]) / (times[j] - times[j - 1]);
                    values[j - 1] * (1.0 - theta) + values[j] * theta
                };
                assert!((solution.try_interpolate(query).unwrap()[0] - expected).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn second_order_export_preserves_partitions_and_reports_linear_velocity() {
    use differential_equations::solvers::second_order::{
        Dprkn6, SecondOrderOdeProblem, SecondOrderSolution, solve_second_order,
    };
    for span in [(0.0_f64, 1.0), (1.0, 0.0)] {
        let p = SecondOrderOdeProblem::new(
            |a: &mut [f64], _: &[f64], q: &[f64], _: &(), _| a[0] = -q[0],
            [span.0.cos()],
            [span.0.sin()],
            span,
            (),
        );
        let s = solve_second_order(
            &p,
            Dprkn6,
            &SolveOptions::default()
                .with_dense_output(true)
                .with_tolerances(1e-10, 1e-10),
        )
        .unwrap();
        let restored = SecondOrderSolution::from_data(s.export_data().unwrap()).unwrap();
        for k in 0..101 {
            let t = k as f64 / 100.0;
            let (a, b) = s.try_interpolate(t).unwrap();
            let (c, d) = restored.try_interpolate(t).unwrap();
            assert!((a[0] - c[0]).abs() < 2e-14);
            assert!((b[0] - d[0]).abs() < 2e-14);
        }
        let t = (s.times()[0] + s.times()[1]) * 0.5;
        assert_eq!(
            s.interpolation_quality(t).unwrap(),
            (
                InterpolationQuality::Linear,
                InterpolationQuality::MethodSpecific
            )
        );
        assert!(
            s.try_interpolate_method_into(t, &mut [0.0], &mut [0.0])
                .is_err()
        );
        #[cfg(feature = "serde")]
        {
            let json = serde_json::to_string(&s).unwrap();
            let restored: SecondOrderSolution = serde_json::from_str(&json).unwrap();
            assert_eq!(restored.stats(), s.stats());
            assert_eq!(restored.times(), s.times());
        }
    }
}
