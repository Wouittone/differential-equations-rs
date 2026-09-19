use super::kernel::FirkKernel;
use super::tableau::Family;
use crate::integrator::integrate as drive_integration;
use crate::{ConfigurationError, OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions};

/// Third-order, two-stage Radau IIA collocation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RadauIIA3;

/// Fifth-order, three-stage Radau IIA collocation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RadauIIA5;

/// Ninth-order, five-stage Radau IIA collocation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RadauIIA9;

/// Variable-order Radau IIA collocation.
///
/// Supported odd orders range from 3 through 13. The kernel starts at the
/// minimum order and raises or lowers the number of stages after accepted
/// steps according to the embedded step-doubling error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdaptiveRadau {
    min_order: usize,
    max_order: usize,
}

impl Default for AdaptiveRadau {
    fn default() -> Self {
        Self {
            min_order: 5,
            max_order: 13,
        }
    }
}

impl AdaptiveRadau {
    /// Creates a variable-order Radau method over an inclusive odd-order range.
    ///
    /// Returns [`ConfigurationError::InvalidBounds`] unless both bounds are
    /// odd and satisfy `3 <= min_order <= max_order <= 13`.
    pub const fn new(min_order: usize, max_order: usize) -> Result<Self, ConfigurationError> {
        if min_order < 3
            || max_order > 13
            || min_order > max_order
            || min_order % 2 == 0
            || max_order % 2 == 0
        {
            return Err(ConfigurationError::InvalidBounds {
                context: "adaptive Radau order window",
                reason: "bounds must be odd and satisfy 3 <= min_order <= max_order <= 13",
            });
        }
        Ok(Self {
            min_order,
            max_order,
        })
    }

    /// Returns the minimum odd Radau order used by the adaptive controller.
    pub fn min_order(&self) -> usize {
        self.min_order
    }

    /// Returns the maximum odd Radau order used by the adaptive controller.
    pub fn max_order(&self) -> usize {
        self.max_order
    }
}

/// Gauss--Legendre collocation with configurable stage count.
///
/// The default two-stage method is fourth order and symplectic. Adaptive
/// stepping uses Richardson step doubling, matching the pinned upstream
/// algorithm's documented controller foundation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GaussLegendre {
    num_stages: usize,
}

impl Default for GaussLegendre {
    fn default() -> Self {
        Self { num_stages: 2 }
    }
}

impl GaussLegendre {
    /// Creates a Gauss--Legendre collocation method with two to eight stages.
    ///
    /// Returns [`ConfigurationError::InvalidParameter`] when `num_stages` is
    /// outside the supported range.
    pub const fn new(num_stages: usize) -> Result<Self, ConfigurationError> {
        if num_stages < 2 || num_stages > 8 {
            return Err(ConfigurationError::InvalidParameter {
                parameter: "Gauss-Legendre stage count",
                reason: "must be between 2 and 8 inclusive",
            });
        }
        Ok(Self { num_stages })
    }

    /// Returns the configured collocation stage count.
    pub fn num_stages(&self) -> usize {
        self.num_stages
    }
}

macro_rules! impl_fixed_radau {
    ($name:ty, $stages:expr) => {
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
                    FirkKernel::new(
                        problem.initial_state().len(),
                        Family::Radau,
                        $stages,
                        $stages,
                    ),
                )
            }
        }
    };
}

impl_fixed_radau!(RadauIIA3, 2);
impl_fixed_radau!(RadauIIA5, 3);
impl_fixed_radau!(RadauIIA9, 5);

impl OdeAlgorithm for AdaptiveRadau {
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
            FirkKernel::new(
                problem.initial_state().len(),
                Family::Radau,
                self.min_order.div_ceil(2),
                self.max_order.div_ceil(2),
            ),
        )
    }
}

impl OdeAlgorithm for GaussLegendre {
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
            FirkKernel::new(
                problem.initial_state().len(),
                Family::Gauss,
                self.num_stages,
                self.num_stages,
            ),
        )
    }
}
