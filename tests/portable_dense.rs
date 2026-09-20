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
    assert!(
        (restored.try_interpolate(0.37).unwrap()[0] - solution.try_interpolate(0.37).unwrap()[0])
            .abs()
            < 1e-14
    );
}
