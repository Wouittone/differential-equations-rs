use super::StepInterpolator;
use crate::SolveError;
use crate::callback::{ContinuousCallback, PredictiveDomainPolicy, VectorContinuousCallback};
use crate::event::{MAX_EVENT_ROOT_ITERATIONS, event_interval_converged};

pub(super) struct RootSegment<'a> {
    pub(super) previous_state: &'a [f64],
    pub(super) previous_time: f64,
    pub(super) state: &'a [f64],
    pub(super) time: f64,
}

pub(super) fn locate_root<P>(
    callback: &ContinuousCallback<P>,
    segment: RootSegment<'_>,
    before: f64,
    interpolation: &mut [f64],
    parameters: &P,
    event_tolerance: f64,
    interpolator: &mut Option<&mut StepInterpolator<'_>>,
) -> Result<f64, SolveError> {
    let mut left = 0.0;
    let mut right = 1.0;
    let mut left_value = before;
    for _ in 0..MAX_EVENT_ROOT_ITERATIONS {
        let middle = 0.5 * (left + right);
        if middle == left || middle == right {
            break;
        }
        let middle_time = segment.previous_time + middle * (segment.time - segment.previous_time);
        if let Some(interpolator) = interpolator.as_mut() {
            interpolator(middle_time, interpolation)?;
        } else {
            interpolate(segment.state, segment.previous_state, middle, interpolation);
        }
        let value = (callback.condition)(interpolation, parameters, middle_time);
        if !value.is_finite() {
            return Err(SolveError::NonFiniteCallbackCondition);
        }
        if value == 0.0 {
            return Ok(middle);
        }
        if left_value.signum() == value.signum() {
            left = middle;
            left_value = value;
        } else {
            right = middle;
        }
        if event_interval_converged(
            event_tolerance,
            segment.previous_time,
            segment.time,
            left,
            right,
        ) {
            break;
        }
    }
    // Return the post-crossing side of the final bracket. This prevents a
    // continuing callback from immediately detecting the same root again on
    // the next step when the midpoint lies microscopically before the root.
    Ok(right)
}

pub(super) fn evaluate_vector_condition<P>(
    callback: &VectorContinuousCallback<P>,
    output: &mut [f64],
    state: &[f64],
    parameters: &P,
    time: f64,
) -> Result<(), SolveError> {
    output.fill(f64::NAN);
    (callback.condition)(output, state, parameters, time);
    output
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteCallbackCondition)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn locate_vector_root<P>(
    callback: &VectorContinuousCallback<P>,
    event_index: usize,
    segment: RootSegment<'_>,
    before: f64,
    interpolation: &mut [f64],
    parameters: &P,
    event_tolerance: f64,
    interpolator: &mut Option<&mut StepInterpolator<'_>>,
    condition_values: &mut [f64],
) -> Result<f64, SolveError> {
    let mut left = 0.0;
    let mut right = 1.0;
    let mut left_value = before;
    for _ in 0..MAX_EVENT_ROOT_ITERATIONS {
        let middle = 0.5 * (left + right);
        if middle == left || middle == right {
            break;
        }
        let middle_time = segment.previous_time + middle * (segment.time - segment.previous_time);
        if let Some(interpolator) = interpolator.as_mut() {
            interpolator(middle_time, interpolation)?;
        } else {
            interpolate(segment.state, segment.previous_state, middle, interpolation);
        }
        evaluate_vector_condition(
            callback,
            condition_values,
            interpolation,
            parameters,
            middle_time,
        )?;
        let value = condition_values[event_index];
        if value == 0.0 {
            return Ok(middle);
        }
        if left_value.signum() == value.signum() {
            left = middle;
            left_value = value;
        } else {
            right = middle;
        }
        if event_interval_converged(
            event_tolerance,
            segment.previous_time,
            segment.time,
            left,
            right,
        ) {
            break;
        }
    }
    Ok(right)
}

pub(super) fn interpolate(
    state: &[f64],
    previous_state: &[f64],
    fraction: f64,
    output: &mut [f64],
) {
    for ((output, previous), current) in output.iter_mut().zip(previous_state).zip(state) {
        *output = previous + fraction * (current - previous);
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn predictive_domain_adjusted_step<P>(
    policies: &[PredictiveDomainPolicy<P>],
    parameters: &P,
    state: &[f64],
    derivative: &[f64],
    time: f64,
    proposed_step: f64,
    default_tolerance: f64,
    prediction: &mut [f64],
) -> Result<f64, SolveError> {
    debug_assert_eq!(state.len(), derivative.len());
    debug_assert_eq!(state.len(), prediction.len());
    let mut step = proposed_step;
    let mut modified = false;
    loop {
        for ((predicted, state), derivative) in prediction.iter_mut().zip(state).zip(derivative) {
            *predicted = state + step * derivative;
        }
        let next_time = time + step;
        let mut reduction: Option<f64> = None;
        for policy in policies {
            if !(policy.accepts)(prediction, parameters, next_time, default_tolerance)? {
                reduction = Some(reduction.map_or(policy.reduction_factor, |current| {
                    current.min(policy.reduction_factor)
                }));
            }
        }
        let Some(reduction) = reduction else {
            break;
        };
        let reduced = step * reduction;
        if reduced == step || reduced == 0.0 {
            return Err(SolveError::StepSizeUnderflow);
        }
        step = reduced;
        modified = true;
    }

    if modified {
        step *= 0.9;
        if step == 0.0 {
            return Err(SolveError::StepSizeUnderflow);
        }
    }
    Ok(step)
}

pub(super) fn ensure_finite_callback_state(state: &[f64]) -> Result<(), SolveError> {
    state
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteCallbackState)
}
