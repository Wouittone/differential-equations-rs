/// Default requested absolute tolerance for continuous-event localization.
///
/// Root localization also applies a scale-aware floating-point floor so this
/// tolerance is never smaller than the representable spacing near the event.
pub const DEFAULT_EVENT_TOLERANCE: f64 = 8.0 * f64::EPSILON;

pub(crate) const MAX_EVENT_ROOT_ITERATIONS: usize = 64;

pub(crate) fn effective_event_tolerance(requested: f64, left_time: f64, right_time: f64) -> f64 {
    let time_scale = left_time.abs().max(right_time.abs()).max(1.0);
    requested.max(8.0 * f64::EPSILON * time_scale)
}

pub(crate) fn times_are_numerically_equal(left_time: f64, right_time: f64) -> bool {
    (right_time - left_time).abs() <= effective_event_tolerance(0.0, left_time, right_time)
}

pub(crate) fn event_interval_converged(
    requested: f64,
    step_start: f64,
    step_end: f64,
    left_fraction: f64,
    right_fraction: f64,
) -> bool {
    let left_time = step_start + left_fraction * (step_end - step_start);
    let right_time = step_start + right_fraction * (step_end - step_start);
    (right_time - left_time).abs() <= effective_event_tolerance(requested, left_time, right_time)
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_EVENT_TOLERANCE, effective_event_tolerance, event_interval_converged};

    #[test]
    fn tolerance_has_a_scale_aware_representability_floor() {
        assert_eq!(
            effective_event_tolerance(DEFAULT_EVENT_TOLERANCE, 0.0, 1.0),
            DEFAULT_EVENT_TOLERANCE
        );
        assert!(effective_event_tolerance(f64::EPSILON, 1.0e16, 1.0e16 + 2.0) >= 2.0);
    }

    #[test]
    fn convergence_is_measured_in_time_not_normalized_fraction() {
        assert!(event_interval_converged(
            1.0e-6,
            0.0,
            100.0,
            0.5,
            0.5 + 1.0e-9
        ));
        assert!(!event_interval_converged(
            1.0e-12,
            0.0,
            100.0,
            0.5,
            0.5 + 1.0e-9
        ));
    }
}
