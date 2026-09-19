//! Public multirate algorithm configurations and their split-ODE dispatch.

use super::super::tableaux::*;
use super::driver::integrate_multirate;
use super::method::Method;
use crate::solvers::explicit::SplitOdeAlgorithm;
use crate::solvers::multistep::adams_bashforth;
use crate::tableau::{
    LinearMultistepTableau, MisTableau, MriTableau, TableauAccessError, TableauError, load_tableau,
};
use crate::{ConfigurationError, Solution, SolveError, SolveOptions, SplitOdeProblem};

pub use super::method::MultirateSequence;

/// Multirate explicit Euler extrapolation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mreef {
    m: usize,
    order: usize,
    sequence: MultirateSequence,
}

impl Mreef {
    /// Configures the microstep count, extrapolation order, and sequence.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigurationError::InvalidParameter`] when `m` is zero or
    /// `order` is outside `2..=10`.
    pub const fn new(
        m: usize,
        order: usize,
        sequence: MultirateSequence,
    ) -> Result<Self, ConfigurationError> {
        if m == 0 {
            return Err(invalid_microstep_count("MREEF microstep count"));
        }
        if order < 2 || order > 10 {
            return Err(ConfigurationError::InvalidParameter {
                parameter: "MREEF order",
                reason: "must be between two and ten",
            });
        }
        Ok(Self::from_valid_config(m, order, sequence))
    }

    const fn from_valid_config(m: usize, order: usize, sequence: MultirateSequence) -> Self {
        Self { m, order, sequence }
    }

    /// Returns the number of fast microsteps per macro step.
    pub const fn microsteps(&self) -> usize {
        self.m
    }

    /// Returns the configured extrapolation order.
    pub const fn order(&self) -> usize {
        self.order
    }

    /// Returns the configured extrapolation sequence.
    pub const fn sequence(&self) -> MultirateSequence {
        self.sequence
    }
}

impl Default for Mreef {
    fn default() -> Self {
        Self::from_valid_config(4, 4, MultirateSequence::Harmonic)
    }
}

/// Exact compatibility spelling from OrdinaryDiffEqMultirate.
pub type MREEF = Mreef;

/// Multirate Adams--Bashforth with frozen slow forcing per macro step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mrab {
    order: usize,
    m: usize,
}

impl Mrab {
    /// Configures the Adams--Bashforth order and fast microstep count.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigurationError::InvalidParameter`] when `order` is
    /// outside `1..=5` or `m` is zero.
    pub const fn new(order: usize, m: usize) -> Result<Self, ConfigurationError> {
        if order < 1 || order > 5 {
            return Err(ConfigurationError::InvalidParameter {
                parameter: "MRAB order",
                reason: "must be between one and five",
            });
        }
        if m == 0 {
            return Err(invalid_microstep_count("MRAB microstep count"));
        }
        Ok(Self::from_valid_config(order, m))
    }

    const fn from_valid_config(order: usize, m: usize) -> Self {
        Self { order, m }
    }

    /// Returns the configured Adams--Bashforth order.
    pub const fn order(&self) -> usize {
        self.order
    }

    /// Returns the nominal-order formula shared with fixed-step Adams solvers.
    /// Startup microsteps load only the lower-order formulas they actually use.
    ///
    /// # Errors
    ///
    /// Preserves resource-validation and family-invariant failures. The
    /// constructor validates the order before a value can reach this method.
    pub fn tableau(&self) -> Result<&'static LinearMultistepTableau, TableauAccessError> {
        adams_bashforth(self.order)
    }

    /// Returns the number of fast microsteps per macro step.
    pub const fn microsteps(&self) -> usize {
        self.m
    }
}

impl Default for Mrab {
    fn default() -> Self {
        Self::from_valid_config(2, 4)
    }
}

/// Exact compatibility spelling from OrdinaryDiffEqMultirate.
pub type MRAB = Mrab;

/// Knoth--Wolke multirate infinitesimal-step method.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mis {
    m: usize,
}

impl Mis {
    /// Configures the number of fast microsteps per macro step.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigurationError::InvalidParameter`] when `m` is zero.
    pub const fn new(m: usize) -> Result<Self, ConfigurationError> {
        if m == 0 {
            return Err(invalid_microstep_count("MIS microstep count"));
        }
        Ok(Self::from_valid_microsteps(m))
    }

    const fn from_valid_microsteps(m: usize) -> Self {
        Self { m }
    }

    /// Returns the number of fast microsteps per macro step.
    pub const fn microsteps(&self) -> usize {
        self.m
    }

    /// Returns the lazily parsed Knoth--Wolke coupling tableau.
    pub fn tableau(&self) -> Result<&'static MisTableau, TableauError> {
        load_tableau(&MIS_TABLEAU)
    }
}

impl Default for Mis {
    fn default() -> Self {
        Self::from_valid_microsteps(4)
    }
}

/// Exact compatibility spelling from OrdinaryDiffEqMultirate.
pub type MIS = Mis;

macro_rules! mri_algorithm {
    ($rust:ident, $exact:ident, $tableau:ident, $docs:literal) => {
        #[doc = $docs]
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub struct $rust {
            m: usize,
        }

        impl $rust {
            /// Configures the number of fast microsteps per macro step.
            ///
            /// # Errors
            ///
            /// Returns [`ConfigurationError::InvalidParameter`] when `m` is
            /// zero.
            pub const fn new(m: usize) -> Result<Self, ConfigurationError> {
                if m == 0 {
                    return Err(invalid_microstep_count(concat!(
                        stringify!($exact),
                        " microstep count"
                    )));
                }
                Ok(Self::from_valid_microsteps(m))
            }

            const fn from_valid_microsteps(m: usize) -> Self {
                Self { m }
            }

            /// Returns the number of fast microsteps per macro step.
            pub const fn microsteps(&self) -> usize {
                self.m
            }

            /// Returns this method's lazily parsed MRI-GARK coupling tableau.
            pub fn tableau(&self) -> Result<&'static MriTableau, TableauError> {
                load_tableau(&$tableau)
            }
        }

        impl Default for $rust {
            fn default() -> Self {
                Self::from_valid_microsteps(4)
            }
        }

        #[allow(non_camel_case_types)]
        #[doc = concat!("Exact OrdinaryDiffEq-compatible spelling alias for [`", stringify!($rust), "`].")]
        pub type $exact = $rust;

        impl SplitOdeAlgorithm for $rust {
            fn solve_validated<FE, FI, P>(
                &self,
                problem: &SplitOdeProblem<FE, FI, P>,
                options: &SolveOptions,
            ) -> Result<Solution, SolveError>
            where
                FE: crate::OdeFunction<P>,
                FI: crate::OdeFunction<P>,
            {
                integrate_multirate(
                    problem,
                    options,
                    Method::Mri {
                        m: self.m,
                        tableau: self.tableau().map_err(SolveError::from)?,
                    },
                )
            }
        }
    };
}

mri_algorithm!(
    MriGarkErk22a,
    MRIGARKERK22a,
    ERK22A_TABLEAU,
    "Explicit second-order MRI-GARK ERK22a."
);
mri_algorithm!(
    MriGarkErk22b,
    MRIGARKERK22b,
    ERK22B_TABLEAU,
    "Explicit second-order MRI-GARK ERK22b."
);
mri_algorithm!(
    MriGarkErk33a,
    MRIGARKERK33a,
    ERK33A_TABLEAU,
    "Explicit third-order MRI-GARK ERK33a."
);
mri_algorithm!(
    MriGarkErk45a,
    MRIGARKERK45a,
    ERK45A_TABLEAU,
    "Explicit fourth-order MRI-GARK ERK45a."
);
mri_algorithm!(
    MriGarkEsdirk34a,
    MRIGARKESDIRK34a,
    ESDIRK34A_TABLEAU,
    "Third-order implicit-slow MRI-GARK ESDIRK34a."
);
mri_algorithm!(
    MriGarkIrk21a,
    MRIGARKIRK21a,
    IRK21A_TABLEAU,
    "Second-order implicit-slow MRI-GARK IRK21a."
);

impl SplitOdeAlgorithm for Mreef {
    fn solve_validated<FE, FI, P>(
        &self,
        problem: &SplitOdeProblem<FE, FI, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        FE: crate::OdeFunction<P>,
        FI: crate::OdeFunction<P>,
    {
        integrate_multirate(
            problem,
            options,
            Method::Mreef {
                m: self.m,
                order: self.order,
                sequence: self.sequence,
            },
        )
    }
}

impl SplitOdeAlgorithm for Mrab {
    fn solve_validated<FE, FI, P>(
        &self,
        problem: &SplitOdeProblem<FE, FI, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        FE: crate::OdeFunction<P>,
        FI: crate::OdeFunction<P>,
    {
        if options.adaptive && self.order == 1 {
            return Err(SolveError::AdaptiveStepUnsupported);
        }
        integrate_multirate(
            problem,
            options,
            Method::Mrab {
                m: self.m,
                order: self.order,
            },
        )
    }
}

impl SplitOdeAlgorithm for Mis {
    fn solve_validated<FE, FI, P>(
        &self,
        problem: &SplitOdeProblem<FE, FI, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        FE: crate::OdeFunction<P>,
        FI: crate::OdeFunction<P>,
    {
        integrate_multirate(
            problem,
            options,
            Method::Mis {
                m: self.m,
                tableau: self.tableau().map_err(SolveError::from)?,
            },
        )
    }
}

const fn invalid_microstep_count(parameter: &'static str) -> ConfigurationError {
    ConfigurationError::InvalidParameter {
        parameter,
        reason: "must be greater than zero",
    }
}
