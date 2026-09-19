//! Additional native Rosenbrock and Rosenbrock--Wanner methods.
//!
//! Coefficients and stage equations are ported from `OrdinaryDiffEqRosenbrock`
//! and `OrdinaryDiffEqRosenbrockTableaus` at commit
//! `211142263781255a9aa2f910f6760b9f18ec29c8`, including each tableau's exact
//! dense-output dispatch. Empty upstream `H` matrices intentionally retain the
//! generic cubic Hermite fallback.

mod kernel;
mod methods;
mod steps;
mod workspace;

#[cfg(test)]
mod tests;

use super::tableaux::ROSENBROCK23_32_TABLEAU;
use crate::tableau::{RosenbrockPairTableau, TableauError, load_tableau};

pub(crate) use kernel::ExtendedRosenbrockKernel;
pub(crate) use methods::ExtendedRosenbrockMethod;

/// The adaptive third-order Rosenbrock 3/2 W-method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Rosenbrock32;

impl Rosenbrock32 {
    /// Returns the shared, lazily parsed Rosenbrock 2/3 pair tableau.
    pub fn tableau(&self) -> Result<&'static RosenbrockPairTableau, TableauError> {
        load_tableau(&ROSENBROCK23_32_TABLEAU)
    }
}

/// The adaptive second-order, two-stage L-stable Rosenbrock method.
///
/// This is the `ROS2` tableau from the pinned `OrdinaryDiffEqRosenbrock`
/// revision. The Rust spelling follows the crate's type-name convention;
/// inventory matching normalizes it back to the upstream `ROS2` name.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Ros2;

/// The four-stage, third-order A-stable Rodas3 method.
///
/// This is the hand-written `Rodas3RodasTableau` from the pinned
/// `OrdinaryDiffEqRosenbrockTableaus` revision. It has no embedded dense
/// interpolant; the shared recorder supplies trajectory samples.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Rodas3;

/// The four-stage, third-order stiffly accurate Rodas3d method.
///
/// This damped method uses the pinned `Rodas3dRodasTableau` coefficients,
/// including its embedded second-order estimator for adaptive stepping.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Rodas3d;
/// The adaptive third-order, three-stage L-stable `ROS3` Rosenbrock method.
///
/// The embedded estimator is second order and strongly A-stable, matching the
/// pinned `OrdinaryDiffEqRosenbrock` tableau.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Ros3;

/// The three-stage, third-order A-stable ROS3PR Rosenbrock method.
///
/// Coefficients are from `ROS3PRRodasTableau` in the pinned
/// `OrdinaryDiffEqRosenbrockTableaus` revision.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Ros3Pr;

/// The four-stage, third-order stiffly accurate low-storage ROS3PRL method.
///
/// Coefficients are from `ROS3PRLRodasTableau` in the pinned
/// `OrdinaryDiffEqRosenbrockTableaus` revision. Its embedded estimator is
/// second order, matching the upstream regular-ODE implementation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Ros3Prl;

/// The four-stage, third-order stiffly accurate low-storage ROS3PRL2 method.
///
/// Coefficients are from `ROS3PRL2RodasTableau` in the pinned
/// `OrdinaryDiffEqRosenbrockTableaus` revision. Unlike `ROS3PRL`, the
/// embedded estimator is consistent on medium-stiff Prothero--Robinson
/// problems as in the upstream method documentation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Ros3Prl2;

/// The adaptive third-order A-stable Rosenbrock method designed for
/// parabolic problems.
///
/// This is the `ROS3P` tableau from the pinned
/// `OrdinaryDiffEqRosenbrockTableaus` revision. The embedded estimator is
/// second order, as in the upstream `ROS3PRodasTableau`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Ros3p;

/// The four-stage, third-order stiffly accurate Rosenbrock-Wanner method.
///
/// This is the `ROS34PRw` tableau from the pinned
/// `OrdinaryDiffEqRosenbrockTableaus` revision. Its embedded estimator is
/// second order and, as in the upstream implementation, its consistency
/// degrades on medium-stiff problems.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Ros34Prw;

/// The four-stage, fourth-order Rosenbrock-W method `ROS34PW3`.
///
/// The public Rust spelling follows the crate's type-name convention. This
/// method is the strongly A-stable (Rinf approximately 0.63) W-method from
/// the pinned OrdinaryDiffEq Rosenbrock tableau.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Ros34Pw3;

/// The four-stage, fourth-order A-stable GRK4A Rosenbrock method.
///
/// This is the `GRK4ARodasTableau` from the pinned
/// `OrdinaryDiffEqRosenbrockTableaus` revision. Its embedded estimator is
/// third order and is used by the shared adaptive controller.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Grk4a;

/// The four-stage, fourth-order efficient GRK4T Rosenbrock method.
///
/// This is the `GRK4TRodasTableau` from the pinned
/// `OrdinaryDiffEqRosenbrockTableaus` revision. Its embedded estimator is
/// third order and is used by the shared adaptive controller.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Grk4t;

/// The four-stage, fourth-order ROK4a Rosenbrock method.
///
/// This is the `ROK4aRodasTableau` from the pinned
/// `OrdinaryDiffEqRosenbrockTableaus` revision. It is an A-stable method with
/// a third-order embedded estimator used by the shared adaptive controller.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Rok4a;

/// The four-stage, third-order Rosenbrock-W method `ROS34PW1b`.
///
/// The upstream method has a fourth-order primary formula and a third-order
/// embedded estimator. It is a W-method, so the shared regular-ODE kernel
/// reuses the Jacobian factorization for all four stages.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Ros34Pw1b;

/// The four-stage, fourth-order stiffly accurate ROS34PW2 Rosenbrock-W method.
///
/// Coefficients are from `ROS34PW2RodasTableau` in the pinned
/// `OrdinaryDiffEqRosenbrockTableaus` revision. The embedded estimator is
/// third order and is used by the shared adaptive controller.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Ros34Pw2;

/// The six-stage, fourth-order L-stable Rodas4 method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Rodas4;

/// The six-stage, fourth-order L-stable Rodas42 method.
///
/// This is the alternative fourth-order Rodas tableau from the pinned
/// `OrdinaryDiffEqRosenbrockTableaus` revision.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Rodas42;

/// The six-stage, fourth-order L-stable Rodas4P method.
///
/// Rodas4P emphasizes stability for parabolic problems. Its coefficients are
/// the `Rodas4PTableau` from the pinned `OrdinaryDiffEqRosenbrockTableaus`
/// revision.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Rodas4P;

/// The six-stage, fourth-order L-stable Rodas4P2 method.
///
/// Rodas4P2 is the improved parabolic-problem variant of Rodas4P from the
/// pinned `OrdinaryDiffEqRosenbrockTableaus` revision.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Rodas4P2;

/// The nine-stage, fourth-order Rosenbrock-W Rodas4PW method.
///
/// Coefficients are from the pinned `Rodas4PWTableau` regular-ODE tableau.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Rodas4PW;

/// The eight-stage, fifth-order L-stable Rodas5P method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Rodas5P;

/// The eight-stage, fifth-order stiffly accurate Rodas5 method.
///
/// This is the original Di Marzo RODAS5(4) tableau from the pinned
/// OrdinaryDiffEqRosenbrockTableaus revision. It is distinct from the newer
/// Rodas5P family despite sharing the stage count and order.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Rodas5;

/// The eight-stage, fifth-order L-stable Rodas5Pe method.
///
/// Rodas5Pe shares Rodas5P's primary tableau and uses the pinned upstream
/// method's modified embedded weights for a more effective stiff error
/// estimate.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Rodas5Pe;
/// The eight-stage, fifth-order Rodas5P variant with residual control.
///
/// `Rodas5Pr` uses the exact `Rodas5PTableau` and performs the additional
/// midpoint residual estimate from the pinned OrdinaryDiffEq
/// `perform_step!` implementation when an adaptive step's embedded estimate
/// is below one. This extra check is useful on problems where the embedded
/// estimate becomes over-optimistic while the method is entering a stiff
/// transient.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Rodas5Pr;

/// The nineteen-stage, sixth-order L-stable Rodas6P Rosenbrock method.
///
/// Rodas6P is the regular-ODE tableau from the pinned
/// `OrdinaryDiffEqRosenbrock` revision. Its embedded estimate is the final
/// stage, as in the upstream `btilde` vector.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Rodas6P;

/// The six-stage, fourth-order Rosenbrock-W method (fixed step only).
///
/// Coefficients are from `RosenbrockW6S4OSRodasTableau` in the pinned
/// `OrdinaryDiffEqRosenbrockTableaus` revision. The upstream algorithm is
/// intentionally fixed-step because it has no embedded error estimator.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RosenbrockW6S4OS;

/// The adaptive Rodas23W Rosenbrock-W method.
///
/// Rodas23W is the five-stage W-method from the pinned
/// `OrdinaryDiffEqRosenbrock` revision. Its tableau metadata advertises order
/// 3 while the pinned regular-ODE convergence fixture measures order 2; the
/// embedded update is second order (`btilde = [0, 0, 0, 1, -1]`).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Rodas23W;

/// The five-stage, third-order parabolic Rodas3P method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Rodas3P;

/// The three-stage, second-order stiffly accurate ROS2PR method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Ros2Pr;

/// The three-stage, second-order Rosenbrock-W ROS2S method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Ros2S;

/// The four-stage, fourth-order ROS34PW1a Rosenbrock-W method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Ros34Pw1a;

/// The four-stage, fourth-order L-stable Ros4LStab method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Ros4LStab;

/// The four-stage, fourth-order A-stable Shampine Rosenbrock method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RosShamp4;

/// The three-stage Scholz4_7 Rosenbrock method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Scholz4_7;

/// The four-stage, fourth-order D-stable Veldd4 Rosenbrock method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Veldd4;

/// The four-stage, fourth-order A-stable Velds4 Rosenbrock-W method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Velds4;

/// The generic hybrid explicit/linear-implicit Rosenbrock method.
///
/// The native regular-ODE instantiation is [`type@Tsit5DA`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HybridExplicitImplicitRK;

/// The Tsit5DA fifth-order hybrid explicit/linear-implicit method.
pub type Tsit5DA = HybridExplicitImplicitRK;

/// Value constructor for the genuine `Tsit5DA` spelling alias.
#[allow(non_upper_case_globals)]
pub const Tsit5DA: HybridExplicitImplicitRK = HybridExplicitImplicitRK;
