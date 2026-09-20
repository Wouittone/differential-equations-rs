use differential_equations::stepping::{StepFailure, initial_step};
#[test]
fn initial_step_direction_scale_and_limit_match_independent_rms_formula() {
    let state = [3., 4.];
    let derivative = [6., 8.];
    let scales = [1., 2.];
    for direction in [-1., 1.] {
        let h = initial_step(&state, &derivative, &scales, direction, 1.).unwrap();
        assert!((h - direction * 0.005).abs() < 1e-16);
        assert_eq!(
            initial_step(&state, &derivative, &scales, direction, 0.001).unwrap(),
            direction * 0.001
        );
    }
    assert_eq!(initial_step(&[0.], &[1.], &[1.], -1., 1.).unwrap(), -1e-6);
    assert_eq!(initial_step(&[1.], &[0.], &[1.], 1., 1.).unwrap(), 1e-6);
}
#[test]
fn initial_step_avoids_spurious_overflow_in_norms_and_scaled_components() {
    let h = initial_step(&[1e200, 2e200], &[2e200, 4e200], &[1., 1.], 1., 1.).unwrap();
    assert!((h - 0.005).abs() < 1e-16);
    let h = initial_step(&[1e200, 2e200], &[2e200, 4e200], &[1e-200, 1e-200], -1., 1.).unwrap();
    assert!((h + 0.005).abs() < 1e-14);
    assert_eq!(
        initial_step(&[f64::MAX], &[1.], &[1.], 1., 0.3).unwrap(),
        0.3
    );
}
#[test]
fn invalid_scales_dimensions_and_direction_are_explicit() {
    assert_eq!(
        initial_step(&[1.], &[1.], &[0.], 1., 1.),
        Err(StepFailure::Direction)
    );
    assert_eq!(
        initial_step(&[1.], &[], &[1.], 1., 1.),
        Err(StepFailure::Dimension)
    );
    assert_eq!(
        initial_step(&[1.], &[1.], &[1.], 0., 1.),
        Err(StepFailure::Direction)
    );
    assert_eq!(
        initial_step(&[1.], &[1.], &[f64::NAN], 1., 1.),
        Err(StepFailure::NonFinite)
    );
}
