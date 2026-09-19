use super::super::function::SecondOrderFunction;
use super::contract::{SecondOrderOdeAlgorithm, SecondOrderSolveError};
use super::driver::{
    StructuralParameters, solve_fixed, solve_irkn, solve_newmark, solve_rkn_adaptive,
    solve_rkn_fixed,
};
use super::problem::SecondOrderOdeProblem;
use super::solution::SecondOrderSolution;
use crate::tableau::{
    IrknTableau, LazyRungeKuttaNystromTableau, RungeKuttaNystromKind, RungeKuttaNystromTableau,
    TableauError, define_irkn_tableau_from_file, define_rkn_tableau_from_file, load_tableau,
};
use crate::{ConfigurationError, SolveError, SolveOptions};

/// First-order drift-then-kick symplectic Euler method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SymplecticEuler;

/// Second-order velocity Verlet method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VelocityVerlet;

/// Second-order kick-drift-kick leapfrog method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VerletLeapfrog;

/// Second-order drift-kick-drift leapfrog method.
///
/// This variant evaluates acceleration twice and supports acceleration that
/// depends on velocity, matching OrdinaryDiffEq's implementation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LeapfrogDriftKickDrift;

/// Fourth-order Runge--Kutta--Nystrom method for velocity-dependent acceleration.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Nystrom4;

/// Fourth-order Runge--Kutta--Nystrom method for acceleration independent of velocity.
///
/// The acceleration callback must ignore its velocity argument. This restriction
/// matches the pinned upstream algorithm.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Nystrom4VelocityIndependent;

/// Fifth-order Runge--Kutta--Nystrom method for acceleration independent of velocity.
///
/// The acceleration callback must ignore its velocity argument. This restriction
/// matches the pinned upstream algorithm.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Nystrom5VelocityIndependent;

/// Three-stage RKN method for second-order linear inhomogeneous problems.
///
/// The pinned method is fourth order on that problem class and generally only
/// second order outside it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Rkn4;

/// Classical Newmark--beta structural dynamics method.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NewmarkBeta {
    beta: f64,
    gamma: f64,
}

impl NewmarkBeta {
    /// Creates a Newmark method with parameters in the upstream admissible ranges.
    pub fn new(beta: f64, gamma: f64) -> Result<Self, ConfigurationError> {
        if !beta.is_finite() || !(0.0..=0.5).contains(&beta) {
            return Err(ConfigurationError::InvalidParameter {
                parameter: "Newmark beta",
                reason: "must be finite and in [0, 0.5]",
            });
        }
        if !gamma.is_finite() || !(0.0..=1.0).contains(&gamma) {
            return Err(ConfigurationError::InvalidParameter {
                parameter: "Newmark gamma",
                reason: "must be finite and in [0, 1]",
            });
        }
        Ok(Self { beta, gamma })
    }

    /// Position update coefficient.
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Velocity update coefficient.
    pub fn gamma(&self) -> f64 {
        self.gamma
    }
}

impl Default for NewmarkBeta {
    fn default() -> Self {
        Self {
            beta: 0.25,
            gamma: 0.5,
        }
    }
}

/// Generalized-alpha structural dynamics method of Chung and Hulbert.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeneralizedAlpha {
    alpha_m: f64,
    alpha_f: f64,
    beta: f64,
    gamma: f64,
}

impl GeneralizedAlpha {
    /// Creates a method from all four generalized-alpha parameters.
    pub fn new(
        alpha_m: f64,
        alpha_f: f64,
        beta: f64,
        gamma: f64,
    ) -> Result<Self, ConfigurationError> {
        if ![alpha_m, alpha_f, beta, gamma]
            .iter()
            .all(|value| value.is_finite())
        {
            return Err(ConfigurationError::NonFiniteData {
                context: "generalized-alpha parameters",
            });
        }
        if alpha_m > alpha_f || alpha_f > 0.5 {
            return Err(ConfigurationError::InvalidParameter {
                parameter: "generalized-alpha alpha values",
                reason: "must satisfy alpha_m <= alpha_f <= 0.5",
            });
        }
        let minimum_beta = (0.5 + alpha_f - alpha_m).powi(2) / 4.0;
        if beta < minimum_beta || !(0.0..=1.0).contains(&gamma) {
            return Err(ConfigurationError::InvalidParameter {
                parameter: "generalized-alpha beta or gamma",
                reason: "must satisfy the unconditional-stability bounds",
            });
        }
        Ok(Self {
            alpha_m,
            alpha_f,
            beta,
            gamma,
        })
    }

    /// Uses the recommended spectral-radius-at-infinity parameterization.
    pub fn from_spectral_radius(rho_infinity: f64) -> Result<Self, ConfigurationError> {
        if !rho_infinity.is_finite() || !(0.0..=1.0).contains(&rho_infinity) {
            return Err(ConfigurationError::InvalidParameter {
                parameter: "generalized-alpha spectral radius",
                reason: "must be finite and in [0, 1]",
            });
        }
        let alpha_m = (2.0 * rho_infinity - 1.0) / (rho_infinity + 1.0);
        let alpha_f = rho_infinity / (rho_infinity + 1.0);
        let gamma = 0.5 - alpha_m + alpha_f;
        let beta = (0.5 + alpha_f - alpha_m).powi(2) / 4.0;
        Self::new(alpha_m, alpha_f, beta, gamma)
    }

    /// Uses the Hilber--Hughes--Taylor alpha parameterization.
    pub fn from_hht_alpha(alpha: f64) -> Result<Self, ConfigurationError> {
        if !alpha.is_finite() || !(-1.0 / 3.0..=0.0).contains(&alpha) {
            return Err(ConfigurationError::InvalidParameter {
                parameter: "HHT alpha",
                reason: "must be finite and in [-1/3, 0]",
            });
        }
        Self::new(
            0.0,
            -alpha,
            (1.0 - alpha).powi(2) / 4.0,
            (1.0 - 2.0 * alpha) / 2.0,
        )
    }

    /// Returns `(alpha_m, alpha_f, beta, gamma)`.
    pub fn parameters(&self) -> (f64, f64, f64, f64) {
        (self.alpha_m, self.alpha_f, self.beta, self.gamma)
    }
}

impl Default for GeneralizedAlpha {
    fn default() -> Self {
        Self {
            alpha_m: 0.5,
            alpha_f: 0.5,
            beta: 0.25,
            gamma: 0.5,
        }
    }
}

/// Dormand--Prince fourth-order adaptive Runge--Kutta--Nystrom method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Dprkn4;

/// Dormand--Prince fifth-order adaptive Runge--Kutta--Nystrom method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Dprkn5;

/// Dormand--Prince sixth-order adaptive RKN method with free sixth-order dense output.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Dprkn6;

/// Fine--Montagnier sixth-order adaptive Runge--Kutta--Nystrom method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Dprkn6Fm;

/// Dormand--Prince eighth-order adaptive Runge--Kutta--Nystrom method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Dprkn8;

/// Dormand--Prince twelfth-order adaptive Runge--Kutta--Nystrom method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Dprkn12;

/// Embedded fourth-order Runge--Kutta--Nystrom method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Erkn4;

/// Embedded fifth-order Runge--Kutta--Nystrom method with position-only error control.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Erkn5;

/// Embedded seventh-order Runge--Kutta--Nystrom method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Erkn7;

/// Fine's fourth-order adaptive RKN method for velocity-dependent acceleration.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FineRkn4;

/// Fine's fifth-order adaptive RKN method for velocity-dependent acceleration.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FineRkn5;

/// Third-order fixed-step improved Runge--Kutta--Nystrom two-step method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Irkn3;

/// Fourth-order fixed-step improved Runge--Kutta--Nystrom two-step method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Irkn4;

/// A generic second-order algorithm backed by a validated RKN resource.
///
/// Most users create a named zero-sized algorithm with
/// [`crate::tableau::define_rkn_from_file!`]. This wrapper is public so other
/// compile-time tooling can reuse the built-in fixed and adaptive RKN drivers.
#[derive(Clone, Copy)]
pub struct ResourceRungeKuttaNystrom {
    resource: &'static LazyRungeKuttaNystromTableau,
}

impl ResourceRungeKuttaNystrom {
    /// Wraps one compile-time-validated lazy RKN tableau.
    pub const fn new(resource: &'static LazyRungeKuttaNystromTableau) -> Self {
        Self { resource }
    }

    /// Returns the lazily initialized tableau.
    pub fn tableau(&self) -> Result<&'static RungeKuttaNystromTableau, TableauError> {
        load_tableau(self.resource)
    }
}

/// SciML-compatible constructor spelling for [`Dprkn4`].
pub type DPRKN4 = Dprkn4;
/// SciML-compatible constructor spelling for [`Dprkn5`].
pub type DPRKN5 = Dprkn5;
/// SciML-compatible constructor spelling for [`Dprkn6`].
pub type DPRKN6 = Dprkn6;
/// SciML-compatible constructor spelling for [`Dprkn6Fm`].
pub type DPRKN6FM = Dprkn6Fm;
/// SciML-compatible constructor spelling for [`Dprkn8`].
pub type DPRKN8 = Dprkn8;
/// SciML-compatible constructor spelling for [`Dprkn12`].
pub type DPRKN12 = Dprkn12;
/// SciML-compatible constructor spelling for [`Erkn4`].
pub type ERKN4 = Erkn4;
/// SciML-compatible constructor spelling for [`Erkn5`].
pub type ERKN5 = Erkn5;
/// SciML-compatible constructor spelling for [`Erkn7`].
pub type ERKN7 = Erkn7;
/// SciML-compatible constructor spelling for [`FineRkn4`].
pub type FineRKN4 = FineRkn4;
/// SciML-compatible constructor spelling for [`FineRkn5`].
pub type FineRKN5 = FineRkn5;
/// SciML-compatible constructor spelling for [`Irkn3`].
pub type IRKN3 = Irkn3;
/// SciML-compatible constructor spelling for [`Irkn4`].
pub type IRKN4 = Irkn4;

#[allow(non_upper_case_globals)]
/// Value-form SciML-compatible constructor spelling for [`Dprkn4`].
pub const DPRKN4: Dprkn4 = Dprkn4;
#[allow(non_upper_case_globals)]
/// Value-form SciML-compatible constructor spelling for [`Dprkn5`].
pub const DPRKN5: Dprkn5 = Dprkn5;
#[allow(non_upper_case_globals)]
/// Value-form SciML-compatible constructor spelling for [`Dprkn6`].
pub const DPRKN6: Dprkn6 = Dprkn6;
#[allow(non_upper_case_globals)]
/// Value-form SciML-compatible constructor spelling for [`Dprkn6Fm`].
pub const DPRKN6FM: Dprkn6Fm = Dprkn6Fm;
#[allow(non_upper_case_globals)]
/// Value-form SciML-compatible constructor spelling for [`Dprkn8`].
pub const DPRKN8: Dprkn8 = Dprkn8;
#[allow(non_upper_case_globals)]
/// Value-form SciML-compatible constructor spelling for [`Dprkn12`].
pub const DPRKN12: Dprkn12 = Dprkn12;
#[allow(non_upper_case_globals)]
/// Value-form SciML-compatible constructor spelling for [`Erkn4`].
pub const ERKN4: Erkn4 = Erkn4;
#[allow(non_upper_case_globals)]
/// Value-form SciML-compatible constructor spelling for [`Erkn5`].
pub const ERKN5: Erkn5 = Erkn5;
#[allow(non_upper_case_globals)]
/// Value-form SciML-compatible constructor spelling for [`Erkn7`].
pub const ERKN7: Erkn7 = Erkn7;
#[allow(non_upper_case_globals)]
/// Value-form SciML-compatible constructor spelling for [`FineRkn4`].
pub const FineRKN4: FineRkn4 = FineRkn4;
#[allow(non_upper_case_globals)]
/// Value-form SciML-compatible constructor spelling for [`FineRkn5`].
pub const FineRKN5: FineRkn5 = FineRkn5;
#[allow(non_upper_case_globals)]
/// Value-form SciML-compatible constructor spelling for [`Irkn3`].
pub const IRKN3: Irkn3 = Irkn3;
#[allow(non_upper_case_globals)]
/// Value-form SciML-compatible constructor spelling for [`Irkn4`].
pub const IRKN4: Irkn4 = Irkn4;

define_rkn_tableau_from_file!(
    pub(super) NYSTROM4_TABLEAU,
    "Nystrom4",
    "src/tableau/resources/second_order/nystrom4.json",
    crate = crate
);
define_rkn_tableau_from_file!(
    pub(super) NYSTROM4_VI_TABLEAU,
    "Nystrom4VelocityIndependent",
    "src/tableau/resources/second_order/nystrom4-velocity-independent.json",
    crate = crate
);
define_rkn_tableau_from_file!(
    pub(super) NYSTROM5_VI_TABLEAU,
    "Nystrom5VelocityIndependent",
    "src/tableau/resources/second_order/nystrom5-velocity-independent.json",
    crate = crate
);
define_rkn_tableau_from_file!(
    pub(super) RKN4_TABLEAU,
    "Rkn4",
    "src/tableau/resources/second_order/rkn4.json",
    crate = crate
);
define_rkn_tableau_from_file!(
    pub(super) DPRKN4_ADAPTIVE_TABLEAU,
    "Dprkn4",
    "src/tableau/resources/second_order/dprkn4.json",
    crate = crate
);
define_rkn_tableau_from_file!(
    pub(super) DPRKN5_ADAPTIVE_TABLEAU,
    "Dprkn5",
    "src/tableau/resources/second_order/dprkn5.json",
    crate = crate
);
define_rkn_tableau_from_file!(
    pub(super) DPRKN6_ADAPTIVE_TABLEAU,
    "Dprkn6",
    "src/tableau/resources/second_order/dprkn6.json",
    crate = crate
);
define_rkn_tableau_from_file!(
    pub(super) DPRKN6FM_ADAPTIVE_TABLEAU,
    "Dprkn6Fm",
    "src/tableau/resources/second_order/dprkn6fm.json",
    crate = crate
);
define_rkn_tableau_from_file!(
    pub(super) DPRKN8_ADAPTIVE_TABLEAU,
    "Dprkn8",
    "src/tableau/resources/second_order/dprkn8.json",
    crate = crate
);
define_rkn_tableau_from_file!(
    pub(super) DPRKN12_ADAPTIVE_TABLEAU,
    "Dprkn12",
    "src/tableau/resources/second_order/dprkn12.json",
    crate = crate
);
define_rkn_tableau_from_file!(
    pub(super) ERKN4_ADAPTIVE_TABLEAU,
    "Erkn4",
    "src/tableau/resources/second_order/erkn4.json",
    crate = crate
);
define_rkn_tableau_from_file!(
    pub(super) ERKN5_ADAPTIVE_TABLEAU,
    "Erkn5",
    "src/tableau/resources/second_order/erkn5.json",
    crate = crate
);
define_rkn_tableau_from_file!(
    pub(super) ERKN7_ADAPTIVE_TABLEAU,
    "Erkn7",
    "src/tableau/resources/second_order/erkn7.json",
    crate = crate
);
define_rkn_tableau_from_file!(
    pub(super) FINERKN4_ADAPTIVE_TABLEAU,
    "FineRkn4",
    "src/tableau/resources/second_order/fine-rkn4.json",
    crate = crate
);
define_rkn_tableau_from_file!(
    pub(super) FINERKN5_ADAPTIVE_TABLEAU,
    "FineRkn5",
    "src/tableau/resources/second_order/fine-rkn5.json",
    crate = crate
);
define_irkn_tableau_from_file!(
    pub(super) IRKN3_TABLEAU,
    "Irkn3",
    "src/tableau/resources/second_order/irkn3.json",
    crate = crate
);
define_irkn_tableau_from_file!(
    pub(super) IRKN4_TABLEAU,
    "Irkn4",
    "src/tableau/resources/second_order/irkn4.json",
    crate = crate
);

#[derive(Clone, Copy)]
pub(super) enum Method {
    SymplecticEuler,
    VelocityVerlet,
    VerletLeapfrog,
    LeapfrogDriftKickDrift,
}

macro_rules! impl_algorithm {
    ($algorithm:ty, $method:expr) => {
        impl SecondOrderOdeAlgorithm for $algorithm {
            fn solve_validated<F, P>(
                &self,
                problem: &SecondOrderOdeProblem<F, P>,
                options: &SolveOptions,
            ) -> Result<SecondOrderSolution, SecondOrderSolveError>
            where
                F: SecondOrderFunction<P>,
            {
                solve_fixed(problem, options, $method)
            }
        }
    };
}

impl_algorithm!(SymplecticEuler, Method::SymplecticEuler);
impl_algorithm!(VelocityVerlet, Method::VelocityVerlet);
impl_algorithm!(VerletLeapfrog, Method::VerletLeapfrog);
impl_algorithm!(LeapfrogDriftKickDrift, Method::LeapfrogDriftKickDrift);

macro_rules! impl_rkn_algorithm {
    ($algorithm:ty, $tableau:ident) => {
        impl $algorithm {
            /// Returns this method's lazily initialized, validated tableau.
            pub fn tableau(&self) -> Result<&'static RungeKuttaNystromTableau, TableauError> {
                load_tableau(&$tableau)
            }
        }

        impl SecondOrderOdeAlgorithm for $algorithm {
            fn solve_validated<F, P>(
                &self,
                problem: &SecondOrderOdeProblem<F, P>,
                options: &SolveOptions,
            ) -> Result<SecondOrderSolution, SecondOrderSolveError>
            where
                F: SecondOrderFunction<P>,
            {
                let tableau = self.tableau().map_err(SolveError::from)?;
                if tableau.kind() != RungeKuttaNystromKind::Fixed {
                    return Err(SolveError::InvalidTableau.into());
                }
                solve_rkn_fixed(problem, options, tableau)
            }
        }
    };
}

impl_rkn_algorithm!(Nystrom4, NYSTROM4_TABLEAU);
impl_rkn_algorithm!(Nystrom4VelocityIndependent, NYSTROM4_VI_TABLEAU);
impl_rkn_algorithm!(Nystrom5VelocityIndependent, NYSTROM5_VI_TABLEAU);
impl_rkn_algorithm!(Rkn4, RKN4_TABLEAU);

macro_rules! impl_adaptive_rkn_algorithm {
    ($algorithm:ty, $tableau:ident) => {
        impl $algorithm {
            /// Returns this method's lazily initialized, validated tableau.
            pub fn tableau(&self) -> Result<&'static RungeKuttaNystromTableau, TableauError> {
                load_tableau(&$tableau)
            }
        }

        impl SecondOrderOdeAlgorithm for $algorithm {
            fn solve_validated<F, P>(
                &self,
                problem: &SecondOrderOdeProblem<F, P>,
                options: &SolveOptions,
            ) -> Result<SecondOrderSolution, SecondOrderSolveError>
            where
                F: SecondOrderFunction<P>,
            {
                let tableau = self.tableau().map_err(SolveError::from)?;
                if tableau.kind() != RungeKuttaNystromKind::Adaptive {
                    return Err(SolveError::InvalidTableau.into());
                }
                solve_rkn_adaptive(problem, options, tableau)
            }
        }
    };
}

impl_adaptive_rkn_algorithm!(Dprkn4, DPRKN4_ADAPTIVE_TABLEAU);
impl_adaptive_rkn_algorithm!(Dprkn5, DPRKN5_ADAPTIVE_TABLEAU);
impl_adaptive_rkn_algorithm!(Dprkn6, DPRKN6_ADAPTIVE_TABLEAU);
impl_adaptive_rkn_algorithm!(Dprkn6Fm, DPRKN6FM_ADAPTIVE_TABLEAU);
impl_adaptive_rkn_algorithm!(Dprkn8, DPRKN8_ADAPTIVE_TABLEAU);
impl_adaptive_rkn_algorithm!(Dprkn12, DPRKN12_ADAPTIVE_TABLEAU);
impl_adaptive_rkn_algorithm!(Erkn4, ERKN4_ADAPTIVE_TABLEAU);
impl_adaptive_rkn_algorithm!(Erkn5, ERKN5_ADAPTIVE_TABLEAU);
impl_adaptive_rkn_algorithm!(Erkn7, ERKN7_ADAPTIVE_TABLEAU);
impl_adaptive_rkn_algorithm!(FineRkn4, FINERKN4_ADAPTIVE_TABLEAU);
impl_adaptive_rkn_algorithm!(FineRkn5, FINERKN5_ADAPTIVE_TABLEAU);

impl SecondOrderOdeAlgorithm for ResourceRungeKuttaNystrom {
    fn solve_validated<F, P>(
        &self,
        problem: &SecondOrderOdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<SecondOrderSolution, SecondOrderSolveError>
    where
        F: SecondOrderFunction<P>,
    {
        let tableau = self.tableau().map_err(SolveError::from)?;
        match tableau.kind() {
            RungeKuttaNystromKind::Fixed => solve_rkn_fixed(problem, options, tableau),
            RungeKuttaNystromKind::Adaptive => solve_rkn_adaptive(problem, options, tableau),
        }
    }
}

impl SecondOrderOdeAlgorithm for NewmarkBeta {
    fn solve_validated<F, P>(
        &self,
        problem: &SecondOrderOdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<SecondOrderSolution, SecondOrderSolveError>
    where
        F: SecondOrderFunction<P>,
    {
        solve_newmark(
            problem,
            options,
            StructuralParameters {
                alpha_m: 0.0,
                alpha_f: 0.0,
                beta: self.beta,
                gamma: self.gamma,
            },
        )
    }
}

impl SecondOrderOdeAlgorithm for GeneralizedAlpha {
    fn solve_validated<F, P>(
        &self,
        problem: &SecondOrderOdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<SecondOrderSolution, SecondOrderSolveError>
    where
        F: SecondOrderFunction<P>,
    {
        solve_newmark(
            problem,
            options,
            StructuralParameters {
                alpha_m: self.alpha_m,
                alpha_f: self.alpha_f,
                beta: self.beta,
                gamma: self.gamma,
            },
        )
    }
}

macro_rules! impl_irkn_algorithm {
    ($algorithm:ty, $tableau:ident) => {
        impl $algorithm {
            /// Returns this method's lazily initialized, validated tableau.
            pub fn tableau(&self) -> Result<&'static IrknTableau, TableauError> {
                load_tableau(&$tableau)
            }
        }

        impl SecondOrderOdeAlgorithm for $algorithm {
            fn solve_validated<F, P>(
                &self,
                problem: &SecondOrderOdeProblem<F, P>,
                options: &SolveOptions,
            ) -> Result<SecondOrderSolution, SecondOrderSolveError>
            where
                F: SecondOrderFunction<P>,
            {
                let tableau = self.tableau().map_err(SolveError::from)?;
                let bootstrap = Nystrom4VelocityIndependent
                    .tableau()
                    .map_err(SolveError::from)?;
                solve_irkn(problem, options, tableau, bootstrap)
            }
        }
    };
}

impl_irkn_algorithm!(Irkn3, IRKN3_TABLEAU);
impl_irkn_algorithm!(Irkn4, IRKN4_TABLEAU);
