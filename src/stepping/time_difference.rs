use super::{StepFailure, finite};
/// Scale- and domain-aware finite differences of the explicit time partial.
///
/// Time differences hold the state fixed; they are not total derivatives along
/// the solution. Probes follow the attempted step direction and never leave its
/// interval, additionally clipped to `bounds`. The default scale is one time
/// unit, independent of an epoch offset. Two one-sided probes provide quadratic
/// exactness; at the time-resolution limit one probe provides a first difference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeDifferencePolicy {
    /// Physical time scale (same units as the independent variable).
    pub scale: f64,
    /// Relative probe spacing; default cube root of machine epsilon.
    pub relative_step: f64,
    /// Maximum fraction of the attempted interval per probe (at most one half).
    pub maximum_fraction: f64,
    /// Optional closed smooth-domain bounds; do not cross discontinuities.
    pub bounds: Option<(f64, f64)>,
}
impl Default for TimeDifferencePolicy {
    fn default() -> Self {
        Self {
            scale: 1.,
            relative_step: f64::EPSILON.cbrt(),
            maximum_fraction: 0.5,
            bounds: None,
        }
    }
}
fn adjacent(value: f64, direction: f64) -> f64 {
    if value == 0. {
        return direction * f64::from_bits(1);
    }
    let bits = value.to_bits();
    f64::from_bits(if (value > 0.) == (direction > 0.) {
        bits + 1
    } else {
        bits - 1
    })
}
impl TimeDifferencePolicy {
    /// Probe times; the optional second point can be absent at floating resolution.
    pub fn probes(self, time: f64, step: f64) -> Result<(f64, Option<f64>), StepFailure> {
        finite(&[
            time,
            step,
            self.scale,
            self.relative_step,
            self.maximum_fraction,
        ])?;
        if step == 0.
            || self.scale <= 0.
            || self.relative_step <= 0.
            || self.maximum_fraction <= 0.
            || self.maximum_fraction > 0.5
        {
            return Err(StepFailure::Direction);
        }
        let direction = step.signum();
        let mut endpoint = super::checked_time(time, step)?;
        if let Some((lo, hi)) = self.bounds {
            finite(&[lo, hi])?;
            if lo > hi || time < lo || time > hi {
                return Err(StepFailure::Direction);
            }
            endpoint = endpoint.clamp(lo, hi);
        }
        let available = (endpoint - time).abs();
        let spacing = (self.relative_step * self.scale.max(step.abs()))
            .min(self.maximum_fraction * available);
        let mut first = time + direction * spacing;
        if first == time {
            first = adjacent(time, direction);
        }
        if (first - time) * direction <= 0. || (endpoint - first) * direction < 0. {
            return Err(StepFailure::TimeResolution);
        }
        let mut second = time + direction * 2. * spacing;
        if (second - first) * direction <= 0. {
            second = adjacent(first, direction);
        }
        let second = if (endpoint - second) * direction >= 0. && second.is_finite() {
            Some(second)
        } else {
            None
        };
        Ok((first, second))
    }
}
pub(crate) fn partial(base: f64, first: f64, second: Option<f64>, d1: f64, d2: Option<f64>) -> f64 {
    let slope1 = (first - base) / d1;
    if let (Some(second), Some(d2)) = (second, d2) {
        let slope2 = (second - base) / d2;
        (d2 * slope1 - d1 * slope2) / (d2 - d1)
    } else {
        slope1
    }
}
