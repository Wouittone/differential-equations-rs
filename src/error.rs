use thiserror::Error;

/// Invalid data supplied while constructing a problem or algorithm.
///
/// Construction errors are separate from [`crate::SolveError`] so callers can
/// distinguish invalid configuration from failures that occur while stepping.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
#[non_exhaustive]
pub enum ConfigurationError {
    /// A required state, matrix collection, or coefficient collection is empty.
    #[error("{context} must be non-empty")]
    EmptyData {
        /// The input whose contents were required.
        context: &'static str,
    },
    /// Computing a required dense dimension overflowed `usize`.
    #[error("{context} dimension overflow")]
    DimensionOverflow {
        /// The object whose dimension could not be represented.
        context: &'static str,
    },
    /// Related state or matrix dimensions do not agree.
    #[error("{context} dimensions do not match")]
    DimensionMismatch {
        /// The object with inconsistent dimensions.
        context: &'static str,
    },
    /// Configuration data contains a NaN or infinity.
    #[error("{context} must contain only finite values")]
    NonFiniteData {
        /// The input containing a non-finite value.
        context: &'static str,
    },
    /// A scalar algorithm parameter lies outside its supported domain.
    #[error("invalid {parameter}: {reason}")]
    InvalidParameter {
        /// The parameter name.
        parameter: &'static str,
        /// Its required domain or relationship.
        reason: &'static str,
    },
    /// A set of related bounds is non-finite, unordered, or otherwise invalid.
    #[error("invalid {context}: {reason}")]
    InvalidBounds {
        /// The bounded object being configured.
        context: &'static str,
        /// The required relationship between its bounds.
        reason: &'static str,
    },
}
