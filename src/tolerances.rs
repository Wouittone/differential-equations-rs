//! Component and block error control for reusable solver attempts.
//!
//! Scaling is `atol[i] + rtol[i] * max(abs(previous[i]), abs(candidate[i]))`.
//! A zero scale contributes zero for zero error and infinity otherwise. Masked
//! components are ignored entirely; an empty contributing set has norm zero.

use std::{fmt, ops::Range};

/// Reduction applied to the dimensionless component errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorNorm {
    /// Root mean square over contributing components.
    Rms,
    /// Largest absolute component.
    Max,
}

/// Invalid tolerance configuration or evaluation input.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToleranceError {
    /// Input vectors do not have the configured dimension.
    DimensionMismatch,
    /// A tolerance is negative or not finite.
    InvalidTolerance,
    /// A block is outside the configured state.
    InvalidBlock,
    /// A contributing state or error component is not finite.
    NonFiniteInput,
    /// A custom norm returned a negative value or NaN.
    InvalidNorm,
}
impl fmt::Display for ToleranceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::DimensionMismatch => "tolerance and state dimensions differ",
            Self::InvalidTolerance => "tolerances must be finite and nonnegative",
            Self::InvalidBlock => "tolerance block lies outside state",
            Self::NonFiniteInput => "contributing error-control inputs must be finite",
            Self::InvalidNorm => "custom error norm must be nonnegative and not NaN",
        })
    }
}
impl std::error::Error for ToleranceError {}

/// Validated owned per-component tolerances and participation mask.
///
/// Setup allocates three vectors. Evaluation allocates nothing. Named physical,
/// STM, or sensitivity blocks can be configured using [`Self::with_block`].
#[derive(Clone, Debug, PartialEq)]
pub struct Tolerances {
    absolute: Vec<f64>,
    relative: Vec<f64>,
    mask: Vec<bool>,
}
impl Tolerances {
    /// Creates scalar tolerances repeated over `dimension` components.
    pub fn scalar(dimension: usize, absolute: f64, relative: f64) -> Result<Self, ToleranceError> {
        validate_tolerance(absolute)?;
        validate_tolerance(relative)?;
        Self::componentwise(vec![absolute; dimension], vec![relative; dimension])
    }
    /// Creates componentwise tolerances of identical lengths.
    pub fn componentwise(absolute: Vec<f64>, relative: Vec<f64>) -> Result<Self, ToleranceError> {
        if absolute.len() != relative.len() {
            return Err(ToleranceError::DimensionMismatch);
        }
        for &value in absolute.iter().chain(&relative) {
            validate_tolerance(value)?;
        }
        let mask = vec![true; absolute.len()];
        Ok(Self {
            absolute,
            relative,
            mask,
        })
    }
    /// Overrides a contiguous block, for example position, velocity, or STM.
    pub fn with_block(
        mut self,
        range: Range<usize>,
        absolute: f64,
        relative: f64,
    ) -> Result<Self, ToleranceError> {
        validate_tolerance(absolute)?;
        validate_tolerance(relative)?;
        if range.start > range.end || range.end > self.dimension() {
            return Err(ToleranceError::InvalidBlock);
        }
        self.absolute[range.clone()].fill(absolute);
        self.relative[range].fill(relative);
        Ok(self)
    }
    /// Sets participation; `false` components neither contribute nor affect RMS count.
    pub fn with_mask(mut self, mask: Vec<bool>) -> Result<Self, ToleranceError> {
        if mask.len() != self.dimension() {
            return Err(ToleranceError::DimensionMismatch);
        }
        self.mask = mask;
        Ok(self)
    }
    /// Number of state components.
    pub fn dimension(&self) -> usize {
        self.absolute.len()
    }
    /// Absolute tolerance for every component.
    pub fn absolute(&self) -> &[f64] {
        &self.absolute
    }
    /// Relative tolerance for every component.
    pub fn relative(&self) -> &[f64] {
        &self.relative
    }
    /// Participation mask.
    pub fn mask(&self) -> &[bool] {
        &self.mask
    }
    /// Evaluates RMS or maximum error using caller-owned candidate/error slices.
    ///
    /// Values at or below one satisfy the tolerance. A stable RMS avoids
    /// intermediate square overflow for otherwise representable norms.
    pub fn error_norm(
        &self,
        previous: &[f64],
        candidate: &[f64],
        error: &[f64],
        norm: ErrorNorm,
    ) -> Result<f64, ToleranceError> {
        self.validate_inputs(previous, candidate, error)?;
        let mut count = 0usize;
        let mut maximum: f64 = 0.0;
        let mut sum = 0.0;
        for i in 0..self.dimension() {
            if !self.mask[i] {
                continue;
            }
            count += 1;
            let value = self.scaled(i, previous[i], candidate[i], error[i]);
            if value.is_infinite() {
                return Ok(f64::INFINITY);
            }
            if value > maximum {
                sum = 1.0 + sum * (maximum / value).powi(2);
                maximum = value;
            } else if maximum != 0.0 {
                sum += (value / maximum).powi(2);
            }
        }
        Ok(match norm {
            ErrorNorm::Max => maximum,
            ErrorNorm::Rms if count != 0 => maximum * (sum / count as f64).sqrt(),
            ErrorNorm::Rms => 0.0,
        })
    }
    /// Calls a custom reduction with an iterator of nonnegative scaled errors.
    /// The iterator contains only contributing components and allocates nothing.
    /// With no participating components the result is zero and the hook is not called.
    pub fn custom_norm<F>(
        &self,
        previous: &[f64],
        candidate: &[f64],
        error: &[f64],
        mut norm: F,
    ) -> Result<f64, ToleranceError>
    where
        F: FnMut(&mut dyn Iterator<Item = f64>) -> f64,
    {
        self.validate_inputs(previous, candidate, error)?;
        if !self.mask.iter().any(|&enabled| enabled) {
            return Ok(0.0);
        }
        let mut values = (0..self.dimension())
            .filter(|&i| self.mask[i])
            .map(|i| self.scaled(i, previous[i], candidate[i], error[i]));
        let value = norm(&mut values);
        if value.is_nan() || value < 0.0 {
            Err(ToleranceError::InvalidNorm)
        } else {
            Ok(value)
        }
    }
    fn scaled(&self, i: usize, previous: f64, candidate: f64, error: f64) -> f64 {
        let scale = self.absolute[i] + self.relative[i] * previous.abs().max(candidate.abs());
        if scale == 0.0 {
            if error == 0.0 { 0.0 } else { f64::INFINITY }
        } else if scale.is_infinite() {
            let magnitude = previous.abs().max(candidate.abs());
            let unit = self.absolute[i].max(self.relative[i]);
            (error.abs() / unit) / (self.absolute[i] / unit + (self.relative[i] / unit) * magnitude)
        } else {
            error.abs() / scale
        }
    }
    fn validate_inputs(
        &self,
        previous: &[f64],
        candidate: &[f64],
        error: &[f64],
    ) -> Result<(), ToleranceError> {
        if [previous.len(), candidate.len(), error.len()]
            .iter()
            .any(|&n| n != self.dimension())
        {
            return Err(ToleranceError::DimensionMismatch);
        }
        for i in 0..self.dimension() {
            if self.mask[i]
                && (!previous[i].is_finite() || !candidate[i].is_finite() || !error[i].is_finite())
            {
                return Err(ToleranceError::NonFiniteInput);
            }
        }
        Ok(())
    }
}
fn validate_tolerance(value: f64) -> Result<(), ToleranceError> {
    if !value.is_finite() || value < 0.0 {
        Err(ToleranceError::InvalidTolerance)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn host_max_and_rms_acceptance_differ_as_expected() {
        let t = Tolerances::scalar(4, 1.0, 0.0).unwrap();
        let zero = [0.0; 4];
        let e = [1.5, 0.0, 0.0, 0.0];
        assert_eq!(t.error_norm(&zero, &zero, &e, ErrorNorm::Max), Ok(1.5));
        assert_eq!(t.error_norm(&zero, &zero, &e, ErrorNorm::Rms), Ok(0.75));
    }
    #[test]
    fn blocks_masks_and_custom_hook() {
        let t = Tolerances::scalar(3, 1.0, 0.0)
            .unwrap()
            .with_block(1..3, 2.0, 0.0)
            .unwrap()
            .with_mask(vec![true, true, false])
            .unwrap();
        let z = [0.0, 0.0, f64::NAN];
        assert_eq!(
            t.error_norm(&z, &z, &[1.0, 2.0, f64::NAN], ErrorNorm::Rms),
            Ok(1.0)
        );
        assert_eq!(
            t.custom_norm(&z, &z, &[1.0, 2.0, 0.0], |v| v.sum()),
            Ok(2.0)
        );
    }
    #[test]
    fn zero_scales_empty_and_large_rms() {
        let t = Tolerances::scalar(1, 0.0, 0.0).unwrap();
        assert_eq!(
            t.error_norm(&[0.0], &[0.0], &[0.0], ErrorNorm::Rms),
            Ok(0.0)
        );
        assert_eq!(
            t.error_norm(&[0.0], &[0.0], &[1.0], ErrorNorm::Rms),
            Ok(f64::INFINITY)
        );
        let t = Tolerances::scalar(2, 1.0, 0.0).unwrap();
        assert_eq!(
            t.error_norm(&[0.0; 2], &[0.0; 2], &[1e300; 2], ErrorNorm::Rms),
            Ok(1e300)
        );
        let t = t.with_mask(vec![false; 2]).unwrap();
        assert_eq!(
            t.custom_norm(&[0.0; 2], &[0.0; 2], &[0.0; 2], |_| panic!(
                "empty norms do not call hooks"
            )),
            Ok(0.0)
        );
        assert_eq!(
            t.error_norm(&[0.0; 2], &[0.0; 2], &[0.0; 2], ErrorNorm::Rms),
            Ok(0.0)
        );
    }
    #[test]
    fn finite_overflowing_scales_preserve_ratios() {
        let t = Tolerances::scalar(1, 1e308, 2.0).unwrap();
        let value = t
            .error_norm(&[1e308], &[1e308], &[1e308], ErrorNorm::Max)
            .unwrap();
        assert!((value - 1.0 / 3.0).abs() < 1e-15);
        let t = Tolerances::scalar(1, 1e308, 1e308).unwrap();
        assert_eq!(
            t.error_norm(&[1.0], &[1.0], &[1e308], ErrorNorm::Max),
            Ok(0.5)
        );
        let value = t
            .error_norm(&[1.1], &[1.1], &[1e308], ErrorNorm::Max)
            .unwrap();
        assert!((value - 1.0 / 2.1).abs() < 1e-15);
    }
    #[test]
    fn rejects_invalid_configuration_and_inputs() {
        for x in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(Tolerances::scalar(0, x, 0.0).is_err());
        }
        let t = Tolerances::scalar(1, 1.0, 1.0).unwrap();
        assert_eq!(
            t.error_norm(&[], &[], &[], ErrorNorm::Max),
            Err(ToleranceError::DimensionMismatch)
        );
        assert_eq!(
            t.error_norm(&[f64::NAN], &[0.0], &[0.0], ErrorNorm::Max),
            Err(ToleranceError::NonFiniteInput)
        );
        assert_eq!(
            t.custom_norm(&[0.0], &[0.0], &[0.0], |_| -1.0),
            Err(ToleranceError::InvalidNorm)
        );
    }
}
