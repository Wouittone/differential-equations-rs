//! Portable, validated continuous output independent of solver workspaces.
use crate::InterpolationError;

/// Quality available for an interpolation query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum InterpolationQuality {
    /// Interpolation between saved states without a continuous extension.
    Linear,
    /// A continuous extension retained by the numerical method.
    MethodSpecific,
    /// An exactly saved state, including the last state at a callback time.
    ExactSavedState,
}

/// Version-one polynomial representation for interchange.
///
/// Rows are degree-major: `coefficients[k * dimension + i]` multiplies
/// `theta.powi(k)` for component `i`, where `theta=(t-start_time)/(end_time-start_time)`.
/// `bound_time` clips the query domain after events without rescaling theta.
/// Endpoint overrides preserve exact saved endpoints independent of polynomial
/// rounding. Values at `bound_time` use `bound_state` when present.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DenseSegmentData {
    /// Schema version, currently one.
    pub version: u32,
    /// Polynomial origin.
    pub start_time: f64,
    /// Polynomial normalization endpoint.
    pub end_time: f64,
    /// Last query time in integration direction.
    pub bound_time: f64,
    /// Number of state components.
    pub dimension: usize,
    /// Flattened degree-major polynomial coefficients, including constant row.
    pub coefficients: Vec<f64>,
    /// Exact state at the polynomial endpoint.
    pub end_state: Vec<f64>,
    /// Optional exact state at the clipped endpoint.
    pub bound_state: Option<Vec<f64>>,
    /// Provenance of the polynomial; polynomial degree is not an accuracy order.
    pub quality: InterpolationQuality,
}

/// Validated immutable polynomial segment, safe to move or share across threads.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(try_from = "DenseSegmentData", into = "DenseSegmentData")
)]
pub struct PortableDenseSegment(DenseSegmentData);
impl TryFrom<DenseSegmentData> for PortableDenseSegment {
    type Error = InterpolationError;
    fn try_from(data: DenseSegmentData) -> Result<Self, Self::Error> {
        let valid_time = data.start_time.is_finite()
            && data.end_time.is_finite()
            && data.bound_time.is_finite()
            && data.start_time != data.end_time
            && (data.end_time - data.start_time).is_finite()
            && data.bound_time >= data.start_time.min(data.end_time)
            && data.bound_time <= data.start_time.max(data.end_time);
        if data.version != 1
            || !valid_time
            || data.dimension == 0
            || data.coefficients.is_empty()
            || data.coefficients.len() % data.dimension != 0
            || data.end_state.len() != data.dimension
            || !data
                .coefficients
                .iter()
                .chain(&data.end_state)
                .all(|x| x.is_finite())
            || data
                .bound_state
                .as_ref()
                .is_some_and(|s| s.len() != data.dimension || s.iter().any(|x| !x.is_finite()))
            || data.quality == InterpolationQuality::ExactSavedState
        {
            return Err(InterpolationError::InvalidSegmentData {
                context: "portable polynomial version, dimensions, times, or coefficients",
            });
        }
        Ok(Self(data))
    }
}
impl From<PortableDenseSegment> for DenseSegmentData {
    fn from(segment: PortableDenseSegment) -> Self {
        segment.0
    }
}
impl PortableDenseSegment {
    /// Validates an interchange representation.
    pub fn from_data(data: DenseSegmentData) -> Result<Self, InterpolationError> {
        data.try_into()
    }
    /// Borrows the versioned representation without allocating.
    pub fn data(&self) -> &DenseSegmentData {
        &self.0
    }
    /// Consumes this segment, returning its interchange representation.
    pub fn into_data(self) -> DenseSegmentData {
        self.0
    }
    /// Closed query domain in integration order.
    pub fn time_bounds(&self) -> (f64, f64) {
        (self.0.start_time, self.0.bound_time)
    }
    /// Number of scalar state components.
    pub fn dimension(&self) -> usize {
        self.0.dimension
    }
    /// Continuous-extension provenance.
    pub fn quality(&self) -> InterpolationQuality {
        self.0.quality
    }
    /// Whether a finite time lies inside this segment's clipped domain.
    pub fn contains(&self, t: f64) -> bool {
        t.is_finite()
            && t >= self.0.start_time.min(self.0.bound_time)
            && t <= self.0.start_time.max(self.0.bound_time)
    }
    /// Evaluates the polynomial without allocating.
    pub fn interpolate_into(
        &self,
        time: f64,
        output: &mut [f64],
    ) -> Result<(), InterpolationError> {
        if output.len() != self.0.dimension {
            return Err(InterpolationError::DimensionMismatch);
        }
        if !time.is_finite() {
            return Err(InterpolationError::NonFiniteTime);
        }
        if !self.contains(time) {
            return Err(InterpolationError::OutsideTimeSpan);
        }
        let d = &self.0;
        if time == d.bound_time {
            if let Some(state) = &d.bound_state {
                output.copy_from_slice(state);
                return Ok(());
            }
        }
        if time == d.end_time {
            output.copy_from_slice(&d.end_state);
            return Ok(());
        }
        let theta = (time - d.start_time) / (d.end_time - d.start_time);
        for (component, out) in output.iter_mut().enumerate() {
            *out = d
                .coefficients
                .chunks_exact(d.dimension)
                .rev()
                .fold(0.0, |value, row| value * theta + row[component]);
        }
        if output.iter().all(|x| x.is_finite()) {
            Ok(())
        } else {
            Err(InterpolationError::NonFiniteResult {
                context: "portable polynomial",
            })
        }
    }
}

/// Versioned trajectory interchange including logical state shape and dense output.
/// Work statistics describe the original integration and are preserved.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SolutionData {
    /// Schema version, currently one.
    pub version: u32,
    /// Monotonic saved times in integration order; duplicates retain callback states.
    pub times: Vec<f64>,
    /// Flattened row-major saved states.
    pub values: Vec<f64>,
    /// Logical state dimensions; empty denotes a scalar.
    pub state_shape: Vec<usize>,
    /// Ordered non-overlapping dense segments.
    pub segments: Vec<PortableDenseSegment>,
    /// Work counters from the original integration.
    #[cfg_attr(feature = "serde", serde(default))]
    pub statistics: crate::SolverStats,
}

#[cfg(test)]
mod tests {

    use crate::solution::{
        CollocationSegment, DenseSegment, HermiteSegment, StiffSegment, TaylorSegment,
    };
    #[test]
    fn all_native_polynomial_exports_preserve_values() {
        for (start, end, bound) in [(0.0, 2.0, 1.5), (2.0, 0.0, 0.5)] {
            let hermite = HermiteSegment::new_bounded(
                start,
                end,
                bound,
                vec![1.0, 2.0],
                vec![3.0, 4.0],
                vec![0.2, 0.3],
                vec![0.4, 0.5],
            )
            .unwrap();
            let stiff = StiffSegment::new(
                start,
                end,
                bound,
                &[1.0, 2.0],
                &[3.0, 4.0],
                &[0.2, 0.3, 0.4, 0.5, 0.1, 0.7],
                3,
            )
            .unwrap();
            let taylor = TaylorSegment::new_bounded(
                start,
                end,
                bound,
                &[1.0, 2.0],
                &[3.0, 4.0],
                &[1.0, 2.0, 0.2, 0.3, 0.4, 0.5],
                2,
            )
            .unwrap();
            for (native, portable) in [
                (&hermite as &dyn DenseSegment, hermite.portable().unwrap()),
                (&stiff as &dyn DenseSegment, stiff.portable().unwrap()),
                (&taylor as &dyn DenseSegment, taylor.portable().unwrap()),
            ] {
                for i in 0..=20 {
                    let t = start + (bound - start) * i as f64 / 20.0;
                    let (mut a, mut b) = ([0.0; 2], [0.0; 2]);
                    native.interpolate(t, &mut a).unwrap();
                    portable.interpolate_into(t, &mut b).unwrap();
                    assert!(
                        a.iter().zip(b).all(|(a, b)| (a - b).abs() < 2e-14),
                        "at {t}: {a:?} {b:?}"
                    );
                }
            }
            for adaptive in [false, true] {
                let native = CollocationSegment::new(
                    start,
                    end,
                    bound,
                    &[1.0],
                    &[2.0],
                    &[3.0],
                    &[0.2, 0.3],
                    &[0.4, 0.5],
                    &[0.6, 0.7],
                    &[1.0, -1.0, 0.0, 1.0],
                    2,
                    adaptive,
                )
                .unwrap();
                let portable = native.portable().unwrap();
                for i in 0..=20 {
                    let t = start + (bound - start) * i as f64 / 20.0;
                    let (mut a, mut b) = ([0.0], [0.0]);
                    native.interpolate(t, &mut a).unwrap();
                    portable
                        .iter()
                        .find(|s| s.contains(t))
                        .unwrap()
                        .interpolate_into(t, &mut b)
                        .unwrap();
                    assert!(
                        (a[0] - b[0]).abs() < 2e-14,
                        "collocation at {t}: {a:?} {b:?}"
                    );
                }
            }
        }
    }
}
