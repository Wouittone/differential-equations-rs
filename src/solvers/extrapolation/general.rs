//! Explicit and linearly implicit extrapolation methods.
//!
//! The base discretizations and extrapolation nodes follow the pinned
//! `OrdinaryDiffEqExtrapolation` implementation: explicit Euler with Romberg
//! nodes, modified midpoint with inverse-square nodes, linearly implicit Euler
//! with inverse-step nodes, and the semi-implicit midpoint variants with
//! inverse-square nodes.  Polynomial extrapolation is evaluated with the
//! Neville recurrence; the Deuflhard and Hairer--Wanner types retain distinct
//! order-window policies.

use crate::integrator::integrate as drive_integration;
use crate::{ConfigurationError, OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions};

mod kernel;

use kernel::ExtrapolationKernel;

/// Subdividing sequences supported by OrdinaryDiffEq's extrapolation methods.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ExtrapolationSequence {
    /// Uses the harmonic subdivision sequence `1, 2, 3, ...`.
    #[default]
    Harmonic,
    /// Uses powers of two, as in Romberg extrapolation.
    Romberg,
    /// Uses the interleaved Bulirsch subdivision sequence.
    Bulirsch,
}

impl ExtrapolationSequence {
    fn term(self, index: usize) -> usize {
        match self {
            Self::Harmonic => index + 1,
            Self::Romberg => 1usize.checked_shl(index as u32).unwrap_or(usize::MAX),
            Self::Bulirsch => {
                if index == 0 {
                    1
                } else if index % 2 == 1 {
                    1usize << index.div_ceil(2)
                } else {
                    3usize << (index / 2 - 1)
                }
            }
        }
    }
}

macro_rules! extrapolation_algorithm {
    ($name:ident, $min:expr, $init:expr, $max:expr, $kind:expr, $policy:expr, $factor:expr) => {
        #[doc = concat!(
                                                    "Configurable `",
                                                    stringify!($name),
                                                    "` extrapolation algorithm built from `",
                                                    stringify!($kind),
                                                    "` base steps."
                                                )]
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub struct $name {
            min_order: usize,
            init_order: usize,
            max_order: usize,
            sequence: ExtrapolationSequence,
        }

        impl Default for $name {
            fn default() -> Self {
                Self {
                    min_order: $min,
                    init_order: $init,
                    max_order: $max,
                    sequence: ExtrapolationSequence::Harmonic,
                }
            }
        }

        impl $name {
            /// Creates an extrapolation algorithm with a bounded order window.
            ///
            /// # Errors
            ///
            /// Returns [`ConfigurationError::InvalidBounds`] when an order is
            /// unsupported or the requested window is not ordered. The
            /// requested values are never silently changed.
            pub fn new(
                min_order: usize,
                init_order: usize,
                max_order: usize,
                sequence: ExtrapolationSequence,
            ) -> Result<Self, ConfigurationError> {
                let strict_window = matches!(
                    $policy,
                    OrderPolicy::HairerWanner | OrderPolicy::Barycentric
                ) || matches!($kind, BaseMethod::LinearlyImplicitEuler);
                let orders_are_supported = min_order >= $min
                    && min_order <= if strict_window { 13 } else { 14 }
                    && init_order <= 14
                    && max_order <= 15;
                let window_is_ordered = if strict_window {
                    min_order < init_order && init_order < max_order
                } else {
                    min_order <= init_order && init_order <= max_order
                };
                if !orders_are_supported || !window_is_ordered {
                    return Err(ConfigurationError::InvalidBounds {
                        context: concat!(stringify!($name), " order window"),
                        reason: if strict_window {
                            "orders must satisfy the method minimum and min < init < max <= 15, with init <= 14"
                        } else {
                            "orders must satisfy the method minimum and min <= init <= max <= 15, with init <= 14"
                        },
                    });
                }
                Ok(Self {
                    min_order,
                    init_order,
                    max_order,
                    sequence,
                })
            }

            /// Returns the lowest order considered by the controller.
            pub fn min_order(&self) -> usize {
                self.min_order
            }
            /// Returns the initial order selected for a new integration.
            pub fn init_order(&self) -> usize {
                self.init_order
            }
            /// Returns the highest order considered by the controller.
            pub fn max_order(&self) -> usize {
                self.max_order
            }
            /// Returns the subdivision sequence used for extrapolation nodes.
            pub fn sequence(&self) -> ExtrapolationSequence {
                self.sequence
            }
        }

        impl OdeAlgorithm for $name {
            fn solve_validated<F, P>(
                &self,
                problem: &OdeProblem<F, P>,
                options: &SolveOptions,
            ) -> Result<Solution, SolveError>
            where
                F: crate::OdeFunction<P>,
            {
                drive_integration(
                    problem,
                    options,
                    ExtrapolationKernel::new(
                        problem.initial_state().len(),
                        $kind,
                        $policy,
                        self.min_order,
                        self.init_order,
                        self.max_order,
                        self.sequence,
                        $factor,
                    ),
                )
            }
        }
    };
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BaseMethod {
    ExplicitEuler,
    ExplicitMidpoint,
    LinearlyImplicitEuler,
    LinearlyImplicitMidpoint,
    SmoothedLinearlyImplicitEuler,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OrderPolicy {
    AitkenNeville,
    Deuflhard,
    HairerWanner,
    Barycentric,
}

extrapolation_algorithm!(
    AitkenNeville,
    1,
    5,
    10,
    BaseMethod::ExplicitEuler,
    OrderPolicy::AitkenNeville,
    1
);
extrapolation_algorithm!(
    ExtrapolationMidpointDeuflhard,
    1,
    5,
    10,
    BaseMethod::ExplicitMidpoint,
    OrderPolicy::Deuflhard,
    2
);
extrapolation_algorithm!(
    ExtrapolationMidpointHairerWanner,
    2,
    5,
    10,
    BaseMethod::ExplicitMidpoint,
    OrderPolicy::HairerWanner,
    2
);
extrapolation_algorithm!(
    ImplicitEulerExtrapolation,
    3,
    5,
    12,
    BaseMethod::LinearlyImplicitEuler,
    OrderPolicy::Deuflhard,
    1
);
extrapolation_algorithm!(
    ImplicitDeuflhardExtrapolation,
    1,
    5,
    10,
    BaseMethod::LinearlyImplicitMidpoint,
    OrderPolicy::Deuflhard,
    4
);
extrapolation_algorithm!(
    ImplicitHairerWannerExtrapolation,
    2,
    5,
    10,
    BaseMethod::LinearlyImplicitMidpoint,
    OrderPolicy::HairerWanner,
    4
);
extrapolation_algorithm!(
    ImplicitEulerBarycentricExtrapolation,
    3,
    5,
    12,
    BaseMethod::SmoothedLinearlyImplicitEuler,
    OrderPolicy::Barycentric,
    2
);

#[cfg(test)]
mod tests;
