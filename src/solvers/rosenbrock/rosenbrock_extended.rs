//! Additional native Rosenbrock and Rosenbrock--Wanner methods.
//!
//! Coefficients and stage equations are ported from `OrdinaryDiffEqRosenbrock`
//! and `OrdinaryDiffEqRosenbrockTableaus` at commit
//! `211142263781255a9aa2f910f6760b9f18ec29c8`, including each tableau's exact
//! dense-output dispatch. Empty upstream `H` matrices intentionally retain the
//! generic cubic Hermite fallback.

use std::marker::PhantomData;

use super::rosenbrock_dense::*;
use crate::callback::CallbackOutcome;
use crate::integrator::{
    ControllerConfig, KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::linear::{factorize, solve_factorized};
use crate::solution::{
    BorrowedHermiteSegment, BorrowedRungeKuttaSegment, BorrowedStiffSegment, HermiteSegment,
    RungeKuttaSegment, StiffSegment, TrajectoryRecorder,
};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

mod coefficient_data {
    use differential_equations_tableau_macros::define_coefficients_from_file;

    define_coefficients_from_file!(
        pub(super),
        "coefficients/rosenbrock/extended.toml",
        crate = crate
    );
}

use coefficient_data::*;

const ROSENBROCK_GAMMA: f64 = 1.0 / (2.0 + std::f64::consts::SQRT_2);
const ROSENBROCK_C32: f64 = 6.0 + std::f64::consts::SQRT_2;
const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 6.0;

/// The adaptive third-order Rosenbrock 3/2 W-method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Rosenbrock32;

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

struct RodasTableau {
    stages: usize,
    gamma: f64,
    a: &'static [f64],
    c_matrix: &'static [f64],
    nodes: &'static [f64],
    time_weights: &'static [f64],
    weights: &'static [f64],
    error_weights: &'static [f64],
}

#[derive(Clone, Copy)]
enum AdaptiveErrorEstimator {
    Embedded,
    RichardsonStepDoubling { method_order: i32 },
}

// ROS2RodasTableau(T, T2) from
// lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl.

// OrdinaryDiffEq uses btilde directly for the embedded error estimate.

const ROS2_TABLEAU: RodasTableau = RodasTableau {
    stages: 2,
    gamma: 1.7071067811865475,
    a: ROS2_A,
    c_matrix: ROS2_C,
    nodes: ROS2_NODES,
    time_weights: ROS2_D,
    weights: ROS2_B,
    error_weights: ROS2_E,
};

// Rodas3RodasTableau(T, T2) from
// lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl.

const RODAS3_TABLEAU: RodasTableau = RodasTableau {
    stages: 4,
    gamma: 0.5,
    a: RODAS3_A,
    c_matrix: RODAS3_C,
    nodes: RODAS3_NODES,
    time_weights: RODAS3_D,
    weights: RODAS3_B,
    error_weights: RODAS3_E,
};

// Rodas3dRodasTableau(T, T2) from
// lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl at
// OrdinaryDiffEq revision 211142263781255a9aa2f910f6760b9f18ec29c8.

const RODAS3D_TABLEAU: RodasTableau = RodasTableau {
    stages: 4,
    gamma: 0.57281606,
    a: RODAS3D_A,
    c_matrix: RODAS3D_C,
    nodes: RODAS3D_NODES,
    time_weights: RODAS3D_D,
    weights: RODAS3D_B,
    error_weights: RODAS3D_E,
};

// ROS3RodasTableau(T, T2) from
// lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl.

const ROS3_TABLEAU: RodasTableau = RodasTableau {
    stages: 3,
    gamma: 0.435866521508459,
    a: ROS3_A,
    c_matrix: ROS3_C,
    nodes: ROS3_NODES,
    time_weights: ROS3_D,
    weights: ROS3_B,
    error_weights: ROS3_E,
};

// ROS3PRRodasTableau(T, T2) from
// lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl.

const ROS3PR_TABLEAU: RodasTableau = RodasTableau {
    stages: 3,
    gamma: 0.788675134594813,
    a: ROS3PR_A,
    c_matrix: ROS3PR_C,
    nodes: ROS3PR_NODES,
    time_weights: ROS3PR_D,
    weights: ROS3PR_B,
    error_weights: ROS3PR_E,
};

// ROS3PRLRodasTableau(T, T2) from
// lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl.

const ROS3PRL_TABLEAU: RodasTableau = RodasTableau {
    stages: 4,
    gamma: 0.435866521508459,
    a: ROS3PRL_A,
    c_matrix: ROS3PRL_C,
    nodes: ROS3PRL_NODES,
    time_weights: ROS3PRL_D,
    weights: ROS3PRL_B,
    error_weights: ROS3PRL_E,
};

// ROS3PRL2RodasTableau(T, T2) from
// lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl at the
// pinned OrdinaryDiffEq.jl revision.

const ROS3PRL2_TABLEAU: RodasTableau = RodasTableau {
    stages: 4,
    gamma: 0.435866521508459,
    a: ROS3PRL2_A,
    c_matrix: ROS3PRL2_C,
    nodes: ROS3PRL2_NODES,
    time_weights: ROS3PRL2_D,
    weights: ROS3PRL2_B,
    error_weights: ROS3PRL2_E,
};

// ROS3PRodasTableau(T, T2) from
// lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl.
// The source computes these values from gamma = 1/2 + sqrt(3)/6. They are
// written as literals here so the solve path remains allocation-free and
// deterministic while retaining the upstream Float64 tableau.

const ROS3P_TABLEAU: RodasTableau = RodasTableau {
    stages: 3,
    gamma: 0.7886751345948129,
    a: ROS3P_A,
    c_matrix: ROS3P_C,
    nodes: ROS3P_NODES,
    time_weights: ROS3P_D,
    weights: ROS3P_B,
    error_weights: ROS3P_E,
};

// ROS34PRwRodasTableau(T, T2) from
// lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl.

const ROS34PRW_TABLEAU: RodasTableau = RodasTableau {
    stages: 4,
    gamma: 0.435866521508459,
    a: ROS34PRW_A,
    c_matrix: ROS34PRW_C,
    nodes: ROS34PRW_NODES,
    time_weights: ROS34PRW_D,
    weights: ROS34PRW_B,
    error_weights: ROS34PRW_E,
};

// ROS34PW3RodasTableau(T, T2) from
// lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_tableaus.jl.

const ROS34PW3_TABLEAU: RodasTableau = RodasTableau {
    stages: 4,
    gamma: 1.0685790213016289,
    a: ROS34PW3_A,
    c_matrix: ROS34PW3_C,
    nodes: ROS34PW3_NODES,
    time_weights: ROS34PW3_D,
    weights: ROS34PW3_B,
    error_weights: ROS34PW3_E,
};

// GRK4ARodasTableau(T, T2) from
// lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl.

const GRK4A_TABLEAU: RodasTableau = RodasTableau {
    stages: 4,
    gamma: 0.395,
    a: GRK4A_A,
    c_matrix: GRK4A_C,
    nodes: GRK4A_NODES,
    time_weights: GRK4A_D,
    weights: GRK4A_B,
    error_weights: GRK4A_E,
};

// GRK4TRodasTableau(T, T2) from
// lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl.

const GRK4T_TABLEAU: RodasTableau = RodasTableau {
    stages: 4,
    gamma: 0.231,
    a: GRK4T_A,
    c_matrix: GRK4T_C,
    nodes: GRK4T_NODES,
    time_weights: GRK4T_D,
    weights: GRK4T_B,
    error_weights: GRK4T_E,
};

// ROK4aRodasTableau(T, T2) from
// lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl at
// 211142263781255a9aa2f910f6760b9f18ec29c8.

const ROK4A_TABLEAU: RodasTableau = RodasTableau {
    stages: 4,
    gamma: 0.572816062482135,
    a: ROK4A_A,
    c_matrix: ROK4A_C,
    nodes: ROK4A_NODES,
    time_weights: ROK4A_D,
    weights: ROK4A_B,
    error_weights: ROK4A_E,
};

// ROS34PW1bRodasTableau(T, T2) from
// lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl at
// 211142263781255a9aa2f910f6760b9f18ec29c8.

const ROS34PW1B_TABLEAU: RodasTableau = RodasTableau {
    stages: 4,
    gamma: 0.435866521508459,
    a: ROS34PW1B_A,
    c_matrix: ROS34PW1B_C,
    nodes: ROS34PW1B_NODES,
    time_weights: ROS34PW1B_D,
    weights: ROS34PW1B_B,
    error_weights: ROS34PW1B_E,
};

// ROS34PW2RodasTableau(T, T2) from
// lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl at
// 211142263781255a9aa2f910f6760b9f18ec29c8.

const ROS34PW2_TABLEAU: RodasTableau = RodasTableau {
    stages: 4,
    gamma: 0.435866521508459,
    a: ROS34PW2_A,
    c_matrix: ROS34PW2_C,
    nodes: ROS34PW2_NODES,
    time_weights: ROS34PW2_D,
    weights: ROS34PW2_B,
    error_weights: ROS34PW2_E,
};

const RODAS4_TABLEAU: RodasTableau = RodasTableau {
    stages: 6,
    gamma: 0.25,
    a: RODAS4_A,
    c_matrix: RODAS4_C,
    nodes: RODAS4_NODES,
    time_weights: RODAS4_D,
    weights: RODAS4_B,
    error_weights: RODAS4_E,
};

// Rodas42Tableau(T, T2) from
// lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl.

const RODAS42_TABLEAU: RodasTableau = RodasTableau {
    stages: 6,
    gamma: 0.25,
    a: RODAS42_A,
    c_matrix: RODAS42_C,
    nodes: RODAS42_NODES,
    time_weights: RODAS42_D,
    weights: RODAS42_B,
    error_weights: RODAS42_E,
};

// Rodas4PTableau(T, T2) from
// lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl.

const RODAS4P_TABLEAU: RodasTableau = RodasTableau {
    stages: 6,
    gamma: 0.25,
    a: RODAS4P_A,
    c_matrix: RODAS4P_C,
    nodes: RODAS4P_NODES,
    time_weights: RODAS4P_D,
    weights: RODAS4P_B,
    error_weights: RODAS4P_E,
};

// Rodas4P2Tableau(T, T2) from
// lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl at
// OrdinaryDiffEq revision 211142263781255a9aa2f910f6760b9f18ec29c8.

const RODAS4P2_TABLEAU: RodasTableau = RodasTableau {
    stages: 6,
    gamma: 0.25,
    a: RODAS4P2_A,
    c_matrix: RODAS4P2_C,
    nodes: RODAS4P2_NODES,
    time_weights: RODAS4P2_D,
    weights: RODAS4P2_B,
    error_weights: RODAS4P2_E,
};

// Rodas4PWTableau(T, T2) from
// lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl at
// OrdinaryDiffEq revision 211142263781255a9aa2f910f6760b9f18ec29c8.

const RODAS4PW_TABLEAU: RodasTableau = RodasTableau {
    stages: 9,
    gamma: 0.25,
    a: RODAS4PW_A,
    c_matrix: RODAS4PW_C,
    nodes: RODAS4PW_NODES,
    time_weights: RODAS4PW_D,
    weights: RODAS4PW_B,
    error_weights: RODAS4PW_E,
};

// Rodas5Tableau(T, T2) from
// lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl at
// 211142263781255a9aa2f910f6760b9f18ec29c8.

const RODAS5_TABLEAU: RodasTableau = RodasTableau {
    stages: 8,
    gamma: 0.19,
    a: RODAS5_A,
    c_matrix: RODAS5_C,
    nodes: RODAS5_NODES,
    time_weights: RODAS5_D,
    weights: RODAS5_B,
    error_weights: RODAS5_E,
};

const RODAS5P_TABLEAU: RodasTableau = RodasTableau {
    stages: 8,
    gamma: 0.21193756319429014,
    a: RODAS5P_A,
    c_matrix: RODAS5P_C,
    nodes: RODAS5P_NODES,
    time_weights: RODAS5P_D,
    weights: RODAS5P_B,
    error_weights: RODAS5P_E,
};

// Rodas5PeTableau(T, T2) from
// lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_tableaus.jl. Rodas5Pe uses
// Rodas5P's primary tableau with a custom embedded estimator.

const RODAS5PE_TABLEAU: RodasTableau = RodasTableau {
    stages: 8,
    gamma: 0.21193756319429014,
    a: RODAS5P_A,
    c_matrix: RODAS5P_C,
    nodes: RODAS5P_NODES,
    time_weights: RODAS5P_D,
    weights: RODAS5P_B,
    error_weights: RODAS5PE_E,
};
// Rodas6PTableau(T, T2) from
// lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_tableaus.jl at
// 211142263781255a9aa2f910f6760b9f18ec29c8.
//
// This is the regular ODE 19-stage sixth-order L-stable tableau. The
// upstream dense-output H matrix is not needed by the shared recorder.

const RODAS6P_TABLEAU: RodasTableau = RodasTableau {
    stages: 19,
    gamma: 0.26,
    a: RODAS6P_A,
    c_matrix: RODAS6P_C,
    nodes: RODAS6P_NODES,
    time_weights: RODAS6P_D,
    weights: RODAS6P_B,
    error_weights: RODAS6P_E,
};

const ROSENBROCK_W6S4OS_TABLEAU: RodasTableau = RodasTableau {
    stages: 6,
    gamma: 0.25,
    a: ROSENBROCK_W6S4OS_A,
    c_matrix: ROSENBROCK_W6S4OS_C,
    nodes: ROSENBROCK_W6S4OS_NODES,
    time_weights: ROSENBROCK_W6S4OS_D,
    weights: ROSENBROCK_W6S4OS_B,
    error_weights: ROSENBROCK_W6S4OS_E,
};

// Rodas23WRodasTableau(T, T2) from
// lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_tableaus.jl at
// 211142263781255a9aa2f910f6760b9f18ec29c8.
//
// The upstream tableau is constructed in Julia using exact rational literals
// for gamma and decimal coefficients for the remaining entries. Keeping the
// same Float64 values here makes the regular-ODE stage path deterministic and
// allocation-free. The upstream H matrix is only used for stiff-aware dense
// interpolation; regular ODE trajectories use the shared recorder instead.

const RODAS23W_TABLEAU: RodasTableau = RodasTableau {
    stages: 5,
    gamma: 1.0 / 3.0,
    a: RODAS23W_A,
    c_matrix: RODAS23W_C,
    nodes: RODAS23W_NODES,
    time_weights: RODAS23W_D,
    weights: RODAS23W_B,
    error_weights: RODAS23W_E,
};

// Rodas3PRodasTableau(T, T2), ROS2PRRodasTableau(T, T2), and the remaining
// tableaus below are from the pinned OrdinaryDiffEq Rosenbrock sources.

const RODAS3P_TABLEAU: RodasTableau = RodasTableau {
    stages: 5,
    gamma: 1.0 / 3.0,
    a: RODAS3P_A,
    c_matrix: RODAS3P_C,
    nodes: RODAS3P_NODES,
    time_weights: RODAS3P_D,
    weights: RODAS3P_B,
    error_weights: RODAS3P_E,
};

const ROS2PR_TABLEAU: RodasTableau = RodasTableau {
    stages: 3,
    gamma: 0.228155493653962,
    a: ROS2PR_A,
    c_matrix: ROS2PR_C,
    nodes: ROS2PR_NODES,
    time_weights: ROS2PR_D,
    weights: ROS2PR_B,
    error_weights: ROS2PR_E,
};

const ROS2S_TABLEAU: RodasTableau = RodasTableau {
    stages: 3,
    gamma: 0.292893218813452,
    a: ROS2S_A,
    c_matrix: ROS2S_C,
    nodes: ROS2S_NODES,
    time_weights: ROS2S_D,
    weights: ROS2S_B,
    error_weights: ROS2S_E,
};

const ROS34PW1A_TABLEAU: RodasTableau = RodasTableau {
    stages: 4,
    gamma: 0.435866521508459,
    a: ROS34PW1A_A,
    c_matrix: ROS34PW1A_C,
    nodes: ROS34PW1A_NODES,
    time_weights: ROS34PW1A_D,
    weights: ROS34PW1A_B,
    error_weights: ROS34PW1A_E,
};

const ROS4LSTAB_TABLEAU: RodasTableau = RodasTableau {
    stages: 4,
    gamma: 0.57282,
    a: ROS4LSTAB_A,
    c_matrix: ROS4LSTAB_C,
    nodes: ROS4LSTAB_NODES,
    time_weights: ROS4LSTAB_D,
    weights: ROS4LSTAB_B,
    error_weights: ROS4LSTAB_E,
};

const ROSSHAMP4_TABLEAU: RodasTableau = RodasTableau {
    stages: 4,
    gamma: 0.5,
    a: ROSSHAMP4_A,
    c_matrix: ROSSHAMP4_C,
    nodes: ROSSHAMP4_NODES,
    time_weights: ROSSHAMP4_D,
    weights: ROSSHAMP4_B,
    error_weights: ROSSHAMP4_E,
};

const SCHOLZ4_7_TABLEAU: RodasTableau = RodasTableau {
    stages: 3,
    gamma: 0.788675134594813,
    a: SCHOLZ4_7_A,
    c_matrix: SCHOLZ4_7_C,
    nodes: SCHOLZ4_7_NODES,
    time_weights: SCHOLZ4_7_D,
    weights: SCHOLZ4_7_B,
    error_weights: SCHOLZ4_7_E,
};

const VELDD4_TABLEAU: RodasTableau = RodasTableau {
    stages: 4,
    gamma: 0.2257081148225682,
    a: VELDD4_A,
    c_matrix: VELDD4_C,
    nodes: VELDD4_NODES,
    time_weights: VELDD4_D,
    weights: VELDD4_B,
    error_weights: VELDD4_E,
};

const VELDS4_TABLEAU: RodasTableau = RodasTableau {
    stages: 4,
    gamma: 0.5,
    a: VELDS4_A,
    c_matrix: VELDS4_C,
    nodes: VELDS4_NODES,
    time_weights: VELDS4_D,
    weights: VELDS4_B,
    error_weights: VELDS4_E,
};

// Tsit5DATableau(T, T2) reduced to the regular ODE path. The hybrid tableau
// has an explicit A matrix and a lower-triangular Gamma matrix; the shared
// Rosenbrock driver uses the same representation for its ODE specialization.

const TSIT5DA_TABLEAU: RodasTableau = RodasTableau {
    stages: 12,
    gamma: 0.15,
    a: TSIT5DA_A,
    c_matrix: TSIT5DA_C,
    nodes: TSIT5DA_NODES,
    time_weights: TSIT5DA_D,
    weights: TSIT5DA_B,
    error_weights: TSIT5DA_E,
};

#[allow(clippy::too_many_arguments)]
trait ExtendedRosenbrockMethod {
    const ERROR_ORDER: usize;
    const ADAPTIVE: bool;
    const DENSE_H: &'static [f64] = &[];
    const DENSE_ORDER: usize = 0;
    const SPECIAL_DENSE: bool = false;

    fn perform_step<F, P>(
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        options: &SolveOptions,
        workspace: &mut Workspace,
        stats: &mut SolverStats,
    ) -> Result<f64, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64);
}

macro_rules! algorithm {
    ($name:ident) => {
        impl OdeAlgorithm for $name {
            fn solve<F, P>(
                &self,
                problem: &OdeProblem<F, P>,
                options: &SolveOptions,
            ) -> Result<Solution, SolveError>
            where
                F: Fn(&mut [f64], &[f64], &P, f64),
            {
                drive_integration(
                    problem,
                    options,
                    ExtendedRosenbrockKernel::<Self>::new(problem.initial_state().len()),
                )
            }
        }
    };
}

algorithm!(Rosenbrock32);
algorithm!(Ros2);
algorithm!(Rodas3);
algorithm!(Rodas3d);
algorithm!(Ros3);
algorithm!(Ros3Pr);
algorithm!(Ros3Prl);
algorithm!(Ros3Prl2);
algorithm!(Ros3p);
algorithm!(Ros34Prw);
algorithm!(Ros34Pw3);
algorithm!(Grk4a);
algorithm!(Grk4t);
algorithm!(Rok4a);
algorithm!(Ros34Pw1b);
algorithm!(Ros34Pw2);
algorithm!(Rodas4);
algorithm!(Rodas42);
algorithm!(Rodas4P);
algorithm!(Rodas4P2);
algorithm!(Rodas4PW);
algorithm!(Rodas5);
algorithm!(Rodas5P);
algorithm!(Rodas5Pe);
algorithm!(Rodas5Pr);
algorithm!(Rodas6P);
algorithm!(RosenbrockW6S4OS);
algorithm!(Rodas23W);
algorithm!(HybridExplicitImplicitRK);
algorithm!(Rodas3P);
algorithm!(Ros2Pr);
algorithm!(Ros2S);
algorithm!(Ros34Pw1a);
algorithm!(Ros4LStab);
algorithm!(RosShamp4);
algorithm!(Scholz4_7);
algorithm!(Veldd4);
algorithm!(Velds4);

impl ExtendedRosenbrockMethod for Rosenbrock32 {
    const ERROR_ORDER: usize = 3;
    const ADAPTIVE: bool = true;
    const SPECIAL_DENSE: bool = true;

    fn perform_step<F, P>(
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        options: &SolveOptions,
        workspace: &mut Workspace,
        stats: &mut SolverStats,
    ) -> Result<f64, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        perform_rosenbrock32(
            problem, candidate, state, time, step, options, workspace, stats,
        )
    }
}

macro_rules! rodas_method {
    ($name:ident, $order:literal, $tableau:ident, dense = $dense:ident, $dense_order:literal) => {
        rodas_method!(@impl $name, $order, $tableau, false, AdaptiveErrorEstimator::Embedded,
            $dense, $dense_order);
    };
    ($name:ident, $order:literal, $tableau:ident) => {
        rodas_method!(@impl $name, $order, $tableau, false,
            AdaptiveErrorEstimator::Embedded, &[], 0);
    };
    ($name:ident, $order:literal, $tableau:ident, $residual_control:expr) => {
        rodas_method!(@impl $name, $order, $tableau, $residual_control,
            AdaptiveErrorEstimator::Embedded, &[], 0);
    };
    (
        $name:ident,
        $order:literal,
        $tableau:ident,
        $residual_control:expr,
        $adaptive_error_estimator:expr
    ) => {
        rodas_method!(@impl $name, $order, $tableau, $residual_control,
            $adaptive_error_estimator, &[], 0);
    };
    (@impl
        $name:ident,
        $order:literal,
        $tableau:ident,
        $residual_control:expr,
        $adaptive_error_estimator:expr,
        $dense:expr,
        $dense_order:literal
    ) => {
        impl ExtendedRosenbrockMethod for $name {
            const ERROR_ORDER: usize = $order;
            const ADAPTIVE: bool = true;
            const DENSE_H: &'static [f64] = $dense;
            const DENSE_ORDER: usize = $dense_order;

            fn perform_step<F, P>(
                problem: &OdeProblem<F, P>,
                state: &[f64],
                time: f64,
                step: f64,
                candidate: &mut [f64],
                options: &SolveOptions,
                workspace: &mut Workspace,
                stats: &mut SolverStats,
            ) -> Result<f64, SolveError>
            where
                F: Fn(&mut [f64], &[f64], &P, f64),
            {
                perform_rodas(
                    problem,
                    candidate,
                    state,
                    time,
                    step,
                    options,
                    &$tableau,
                    $residual_control,
                    $adaptive_error_estimator,
                    workspace,
                    stats,
                )
            }
        }
    };
}

rodas_method!(Ros2, 2, ROS2_TABLEAU);
rodas_method!(Rodas3, 3, RODAS3_TABLEAU);
rodas_method!(Rodas3d, 3, RODAS3D_TABLEAU);
rodas_method!(Ros3, 3, ROS3_TABLEAU);
rodas_method!(Ros3Pr, 3, ROS3PR_TABLEAU);
rodas_method!(Ros3Prl, 3, ROS3PRL_TABLEAU);
rodas_method!(Ros3Prl2, 3, ROS3PRL2_TABLEAU);
rodas_method!(Ros3p, 3, ROS3P_TABLEAU);
rodas_method!(Ros34Prw, 3, ROS34PRW_TABLEAU);
rodas_method!(Ros34Pw3, 4, ROS34PW3_TABLEAU);
rodas_method!(Grk4a, 4, GRK4A_TABLEAU);
rodas_method!(Grk4t, 4, GRK4T_TABLEAU);
rodas_method!(Rok4a, 4, ROK4A_TABLEAU);
rodas_method!(Ros34Pw1b, 3, ROS34PW1B_TABLEAU);
rodas_method!(Ros34Pw2, 3, ROS34PW2_TABLEAU);
rodas_method!(Rodas4, 4, RODAS4_TABLEAU, dense = RODAS4_H, 2);
rodas_method!(Rodas42, 4, RODAS42_TABLEAU, dense = RODAS42_H, 2);
rodas_method!(Rodas4P, 4, RODAS4P_TABLEAU, dense = RODAS4P_H, 2);
rodas_method!(Rodas4P2, 4, RODAS4P2_TABLEAU, dense = RODAS4P2_H, 2);
rodas_method!(Rodas4PW, 4, RODAS4PW_TABLEAU, dense = RODAS4PW_H, 2);
rodas_method!(Rodas5, 5, RODAS5_TABLEAU, dense = RODAS5_H, 3);
rodas_method!(Rodas5P, 5, RODAS5P_TABLEAU, dense = RODAS5P_H, 3);
rodas_method!(Rodas5Pe, 5, RODAS5PE_TABLEAU, dense = RODAS5P_H, 3);
rodas_method!(Rodas6P, 6, RODAS6P_TABLEAU, dense = RODAS6P_H, 4);
rodas_method!(Rodas23W, 3, RODAS23W_TABLEAU, dense = RODAS23W_H, 3);
rodas_method!(Rodas3P, 3, RODAS3P_TABLEAU, dense = RODAS3P_H, 3);
rodas_method!(Ros2Pr, 2, ROS2PR_TABLEAU);
rodas_method!(Ros2S, 2, ROS2S_TABLEAU);
rodas_method!(
    Ros34Pw1a,
    3,
    ROS34PW1A_TABLEAU,
    false,
    AdaptiveErrorEstimator::RichardsonStepDoubling { method_order: 3 }
);
rodas_method!(Ros4LStab, 4, ROS4LSTAB_TABLEAU);
rodas_method!(RosShamp4, 4, ROSSHAMP4_TABLEAU);
rodas_method!(Scholz4_7, 4, SCHOLZ4_7_TABLEAU);
rodas_method!(Veldd4, 4, VELDD4_TABLEAU);
rodas_method!(Velds4, 4, VELDS4_TABLEAU);

impl ExtendedRosenbrockMethod for HybridExplicitImplicitRK {
    const ERROR_ORDER: usize = 5;
    const ADAPTIVE: bool = true;
    const DENSE_H: &'static [f64] = TSIT5DA_H;
    const DENSE_ORDER: usize = 3;

    fn perform_step<F, P>(
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        options: &SolveOptions,
        workspace: &mut Workspace,
        stats: &mut SolverStats,
    ) -> Result<f64, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        perform_tsit5da(
            problem, candidate, state, time, step, options, workspace, stats,
        )
    }
}

impl ExtendedRosenbrockMethod for Rodas5Pr {
    const ERROR_ORDER: usize = 5;
    const ADAPTIVE: bool = true;
    const DENSE_H: &'static [f64] = RODAS5P_H;
    const DENSE_ORDER: usize = 3;

    fn perform_step<F, P>(
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        options: &SolveOptions,
        workspace: &mut Workspace,
        stats: &mut SolverStats,
    ) -> Result<f64, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        perform_rodas(
            problem,
            candidate,
            state,
            time,
            step,
            options,
            &RODAS5P_TABLEAU,
            true,
            AdaptiveErrorEstimator::Embedded,
            workspace,
            stats,
        )
    }
}

impl ExtendedRosenbrockMethod for RosenbrockW6S4OS {
    const ERROR_ORDER: usize = 4;
    const ADAPTIVE: bool = false;

    fn perform_step<F, P>(
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        options: &SolveOptions,
        workspace: &mut Workspace,
        stats: &mut SolverStats,
    ) -> Result<f64, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        perform_rodas(
            problem,
            candidate,
            state,
            time,
            step,
            options,
            &ROSENBROCK_W6S4OS_TABLEAU,
            false,
            AdaptiveErrorEstimator::Embedded,
            workspace,
            stats,
        )
    }
}

struct Workspace {
    current_derivative: Vec<f64>,
    perturbed_state: Vec<f64>,
    perturbed_derivative: Vec<f64>,
    time_derivative: Vec<f64>,
    stage_state: Vec<f64>,
    stage_derivative: Vec<f64>,
    right_hand_side: Vec<f64>,
    error: Vec<f64>,
    stages: Vec<f64>,
    dense_endpoint_state: Vec<f64>,
    dense_endpoint_derivative: Vec<f64>,
    dense_corrections: Vec<f64>,
    jacobian: Vec<f64>,
    factorization: Vec<f64>,
    pivots: Vec<usize>,
    differentiation_valid: bool,
}

impl Workspace {
    fn new(dimension: usize) -> Self {
        Self {
            current_derivative: vec![0.0; dimension],
            perturbed_state: vec![0.0; dimension],
            perturbed_derivative: vec![0.0; dimension],
            time_derivative: vec![0.0; dimension],
            stage_state: vec![0.0; dimension],
            stage_derivative: vec![0.0; dimension],
            right_hand_side: vec![0.0; dimension],
            error: vec![0.0; dimension],
            stages: vec![0.0; 19 * dimension],
            dense_endpoint_state: vec![0.0; dimension],
            dense_endpoint_derivative: vec![0.0; dimension],
            dense_corrections: vec![0.0; 4 * dimension],
            jacobian: vec![0.0; dimension * dimension],
            factorization: vec![0.0; dimension * dimension],
            pivots: vec![0; dimension],
            differentiation_valid: false,
        }
    }
}

struct ExtendedRosenbrockKernel<M> {
    workspace: Workspace,
    dense_endpoint_prepared: bool,
    method: PhantomData<M>,
}

impl<M> ExtendedRosenbrockKernel<M> {
    fn new(dimension: usize) -> Self {
        Self {
            workspace: Workspace::new(dimension),
            dense_endpoint_prepared: false,
            method: PhantomData,
        }
    }

    fn prepare_stiff_corrections(&mut self)
    where
        M: ExtendedRosenbrockMethod,
    {
        let dimension = self.workspace.current_derivative.len();
        let stages = M::DENSE_H.len() / M::DENSE_ORDER;
        for row in 0..M::DENSE_ORDER {
            for component in 0..dimension {
                let mut correction = 0.0;
                for stage in 0..stages {
                    correction += M::DENSE_H[row * stages + stage]
                        * self.workspace.stages[stage * dimension + component];
                }
                self.workspace.dense_corrections[row * dimension + component] = correction;
            }
        }
    }
}

impl<F, P, M> StepKernel<F, P> for ExtendedRosenbrockKernel<M>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
    M: ExtendedRosenbrockMethod,
{
    fn has_custom_dense_output(&self) -> bool {
        true
    }

    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::with_controller(
            M::ADAPTIVE,
            ControllerConfig::proportional(
                M::ERROR_ORDER,
                SAFETY,
                MIN_FACTOR,
                MAX_FACTOR,
                MIN_FACTOR,
            ),
        )
    }

    fn initialize(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        evaluate(
            problem,
            &mut self.workspace.current_derivative,
            state,
            time,
            stats,
        )
    }

    fn estimate_initial_step(
        &mut self,
        _: &OdeProblem<F, P>,
        state: &[f64],
        _: f64,
        _: f64,
        maximum_step: f64,
        _: &mut [f64],
        options: &SolveOptions,
        _: &mut SolverStats,
    ) -> Result<f64, SolveError> {
        Ok(estimate_initial_step(
            state,
            &self.workspace.current_derivative,
            options,
            maximum_step,
        ))
    }

    fn attempt_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        options: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError> {
        self.dense_endpoint_prepared = false;
        Ok(StepEstimate::new(M::perform_step(
            problem,
            state,
            time,
            step,
            candidate,
            options,
            &mut self.workspace,
            stats,
        )?))
    }

    fn apply_step_callbacks(
        &mut self,
        problem: &OdeProblem<F, P>,
        previous_state: &[f64],
        previous_time: f64,
        state: &mut [f64],
        time: &mut f64,
        state_before_effect: &mut [f64],
        event_tolerance: f64,
        stats: &mut SolverStats,
    ) -> Result<CallbackOutcome, SolveError> {
        let attempted_time = *time;
        self.workspace.dense_endpoint_state.copy_from_slice(state);
        if M::SPECIAL_DENSE {
            let dimension = previous_state.len();
            let segment = BorrowedRungeKuttaSegment::new(
                previous_time,
                attempted_time,
                previous_state,
                &self.workspace.dense_endpoint_state,
                &self.workspace.stages[..2 * dimension],
                ROSENBROCK_SPECIAL,
            )
            .map_err(|_| SolveError::NonFiniteDerivative)?;
            let mut interpolate = |sample_time: f64, output: &mut [f64]| {
                crate::solution::DenseSegment::interpolate(&segment, sample_time, output)
                    .map_err(|_| SolveError::NonFiniteDerivative)
            };
            return problem.apply_step_callbacks(
                previous_state,
                previous_time,
                state,
                time,
                state_before_effect,
                event_tolerance,
                Some(&mut interpolate),
            );
        }
        if M::DENSE_ORDER > 0 {
            self.prepare_stiff_corrections();
            let dimension = previous_state.len();
            let segment = BorrowedStiffSegment::new(
                previous_time,
                attempted_time,
                previous_state,
                &self.workspace.dense_endpoint_state,
                &self.workspace.dense_corrections[..M::DENSE_ORDER * dimension],
                M::DENSE_ORDER,
            )
            .map_err(|_| SolveError::NonFiniteDerivative)?;
            let mut interpolate = |sample_time: f64, output: &mut [f64]| {
                crate::solution::DenseSegment::interpolate(&segment, sample_time, output)
                    .map_err(|_| SolveError::NonFiniteDerivative)
            };
            return problem.apply_step_callbacks(
                previous_state,
                previous_time,
                state,
                time,
                state_before_effect,
                event_tolerance,
                Some(&mut interpolate),
            );
        }
        if !problem.has_continuous_callbacks() {
            return problem.apply_step_callbacks(
                previous_state,
                previous_time,
                state,
                time,
                state_before_effect,
                event_tolerance,
                None,
            );
        }
        evaluate(
            problem,
            &mut self.workspace.dense_endpoint_derivative,
            &self.workspace.dense_endpoint_state,
            attempted_time,
            stats,
        )?;
        self.dense_endpoint_prepared = true;
        let segment = BorrowedHermiteSegment::new(
            previous_time,
            attempted_time,
            previous_state,
            &self.workspace.dense_endpoint_state,
            &self.workspace.current_derivative,
            &self.workspace.dense_endpoint_derivative,
        )
        .map_err(|_| SolveError::NonFiniteDerivative)?;
        let mut interpolate = |sample_time: f64, output: &mut [f64]| {
            crate::solution::DenseSegment::interpolate(&segment, sample_time, output)
                .map_err(|_| SolveError::NonFiniteDerivative)
        };
        problem.apply_step_callbacks(
            previous_state,
            previous_time,
            state,
            time,
            state_before_effect,
            event_tolerance,
            Some(&mut interpolate),
        )
    }

    fn record_dense_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        previous_state: &[f64],
        state: &[f64],
        previous_time: f64,
        attempted_time: f64,
        time: f64,
        final_time: bool,
        recorder: &mut TrajectoryRecorder<'_>,
        stats: &mut SolverStats,
    ) -> Result<bool, SolveError> {
        let dimension = previous_state.len();
        if M::SPECIAL_DENSE {
            let segment = BorrowedRungeKuttaSegment::new(
                previous_time,
                attempted_time,
                previous_state,
                &self.workspace.dense_endpoint_state,
                &self.workspace.stages[..2 * dimension],
                ROSENBROCK_SPECIAL,
            )
            .map_err(|_| SolveError::NonFiniteDerivative)?;
            recorder
                .record_step_dense(
                    previous_state,
                    previous_time,
                    state,
                    time,
                    final_time,
                    &segment,
                )
                .map_err(|_| SolveError::NonFiniteDerivative)?;
            if recorder.retains_dense_output() {
                recorder.retain_runge_kutta_segment(
                    RungeKuttaSegment::new(
                        previous_time,
                        attempted_time,
                        time,
                        previous_state,
                        &self.workspace.dense_endpoint_state,
                        &self.workspace.stages[..2 * dimension],
                        ROSENBROCK_SPECIAL,
                    )
                    .map_err(|_| SolveError::NonFiniteDerivative)?,
                );
            }
            return Ok(true);
        }
        if M::DENSE_ORDER > 0 {
            self.prepare_stiff_corrections();
            let corrections = &self.workspace.dense_corrections[..M::DENSE_ORDER * dimension];
            let segment = BorrowedStiffSegment::new(
                previous_time,
                attempted_time,
                previous_state,
                &self.workspace.dense_endpoint_state,
                corrections,
                M::DENSE_ORDER,
            )
            .map_err(|_| SolveError::NonFiniteDerivative)?;
            recorder
                .record_step_dense(
                    previous_state,
                    previous_time,
                    state,
                    time,
                    final_time,
                    &segment,
                )
                .map_err(|_| SolveError::NonFiniteDerivative)?;
            if recorder.retains_dense_output() {
                recorder.retain_stiff_segment(
                    StiffSegment::new(
                        previous_time,
                        attempted_time,
                        time,
                        previous_state,
                        &self.workspace.dense_endpoint_state,
                        corrections,
                        M::DENSE_ORDER,
                    )
                    .map_err(|_| SolveError::NonFiniteDerivative)?,
                );
            }
            return Ok(true);
        }
        if !self.dense_endpoint_prepared {
            evaluate(
                problem,
                &mut self.workspace.dense_endpoint_derivative,
                &self.workspace.dense_endpoint_state,
                attempted_time,
                stats,
            )?;
            self.dense_endpoint_prepared = true;
        }
        let segment = BorrowedHermiteSegment::new(
            previous_time,
            attempted_time,
            previous_state,
            &self.workspace.dense_endpoint_state,
            &self.workspace.current_derivative,
            &self.workspace.dense_endpoint_derivative,
        )
        .map_err(|_| SolveError::NonFiniteDerivative)?;
        recorder
            .record_step_dense(
                previous_state,
                previous_time,
                state,
                time,
                final_time,
                &segment,
            )
            .map_err(|_| SolveError::NonFiniteDerivative)?;
        if recorder.retains_dense_output() {
            recorder.retain_hermite_segment(
                HermiteSegment::new_bounded(
                    previous_time,
                    attempted_time,
                    time,
                    previous_state.to_vec(),
                    self.workspace.dense_endpoint_state.clone(),
                    self.workspace.current_derivative.clone(),
                    self.workspace.dense_endpoint_derivative.clone(),
                )
                .map_err(|_| SolveError::NonFiniteDerivative)?,
            );
        }
        Ok(true)
    }

    fn accept_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        _: &[f64],
        state: &[f64],
        time: f64,
        _: f64,
        callback_applied: bool,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        self.workspace.differentiation_valid = false;
        if self.dense_endpoint_prepared && !callback_applied {
            std::mem::swap(
                &mut self.workspace.current_derivative,
                &mut self.workspace.dense_endpoint_derivative,
            );
            self.dense_endpoint_prepared = false;
            return Ok(());
        }
        self.dense_endpoint_prepared = false;
        evaluate(
            problem,
            &mut self.workspace.current_derivative,
            state,
            time,
            stats,
        )
    }

    fn reject_step(&mut self) {}
}

#[allow(clippy::too_many_arguments)]
fn perform_rosenbrock32<F, P>(
    problem: &OdeProblem<F, P>,
    candidate: &mut [f64],
    state: &[f64],
    time: f64,
    step: f64,
    options: &SolveOptions,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<f64, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let dimension = state.len();
    prepare_factorization(
        problem,
        state,
        time,
        step,
        ROSENBROCK_GAMMA,
        workspace,
        stats,
    )?;
    let gamma_step = ROSENBROCK_GAMMA * step;

    for index in 0..dimension {
        workspace.right_hand_side[index] =
            workspace.current_derivative[index] + gamma_step * workspace.time_derivative[index];
    }
    solve_factorized(
        &workspace.factorization,
        &workspace.pivots,
        &mut workspace.right_hand_side,
        dimension,
    );
    workspace.stages[..dimension].copy_from_slice(&workspace.right_hand_side);
    stats.linear_solves += 1;

    for (index, &value) in state.iter().enumerate() {
        workspace.stage_state[index] = value + 0.5 * step * workspace.stages[index];
    }
    evaluate(
        problem,
        &mut workspace.stage_derivative,
        &workspace.stage_state,
        time + 0.5 * step,
        stats,
    )?;
    for index in 0..dimension {
        workspace.right_hand_side[index] =
            workspace.stage_derivative[index] - workspace.stages[index];
    }
    solve_factorized(
        &workspace.factorization,
        &workspace.pivots,
        &mut workspace.right_hand_side,
        dimension,
    );
    for (index, &value) in state.iter().enumerate() {
        workspace.stages[dimension + index] =
            workspace.right_hand_side[index] + workspace.stages[index];
        workspace.stage_state[index] = value + step * workspace.stages[dimension + index];
    }
    stats.linear_solves += 1;

    evaluate(
        problem,
        &mut workspace.error,
        &workspace.stage_state,
        time + step,
        stats,
    )?;
    for index in 0..dimension {
        workspace.right_hand_side[index] = workspace.error[index]
            - ROSENBROCK_C32
                * (workspace.stages[dimension + index] - workspace.stage_derivative[index])
            - 2.0 * (workspace.stages[index] - workspace.current_derivative[index])
            + step * workspace.time_derivative[index];
    }
    solve_factorized(
        &workspace.factorization,
        &workspace.pivots,
        &mut workspace.right_hand_side,
        dimension,
    );
    workspace.stages[2 * dimension..3 * dimension].copy_from_slice(&workspace.right_hand_side);
    stats.linear_solves += 1;

    for (index, &value) in state.iter().enumerate() {
        candidate[index] = value
            + (step / 6.0)
                * (workspace.stages[index]
                    + 4.0 * workspace.stages[dimension + index]
                    + workspace.stages[2 * dimension + index]);
        workspace.error[index] = (step / 6.0)
            * (workspace.stages[index] - 2.0 * workspace.stages[dimension + index]
                + workspace.stages[2 * dimension + index]);
    }
    Ok(if options.adaptive {
        scaled_error_norm(state, candidate, &workspace.error, options)
    } else {
        0.0
    })
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn perform_rodas<F, P>(
    problem: &OdeProblem<F, P>,
    candidate: &mut [f64],
    state: &[f64],
    time: f64,
    step: f64,
    options: &SolveOptions,
    tableau: &RodasTableau,
    residual_control: bool,
    adaptive_error_estimator: AdaptiveErrorEstimator,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<f64, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let dimension = state.len();
    prepare_factorization(problem, state, time, step, tableau.gamma, workspace, stats)?;
    for stage in 0..tableau.stages {
        workspace.stage_state.copy_from_slice(state);
        for previous in 0..stage {
            let coefficient = tableau.a[stage * tableau.stages + previous];
            if coefficient != 0.0 {
                for component in 0..dimension {
                    workspace.stage_state[component] +=
                        coefficient * workspace.stages[previous * dimension + component];
                }
            }
        }
        if stage == 0 {
            workspace
                .stage_derivative
                .copy_from_slice(&workspace.current_derivative);
        } else {
            evaluate(
                problem,
                &mut workspace.stage_derivative,
                &workspace.stage_state,
                time + tableau.nodes[stage] * step,
                stats,
            )?;
        }
        for component in 0..dimension {
            workspace.right_hand_side[component] = workspace.stage_derivative[component]
                + step * tableau.time_weights[stage] * workspace.time_derivative[component];
        }
        for previous in 0..stage {
            let coefficient = tableau.c_matrix[stage * tableau.stages + previous] / step;
            if coefficient != 0.0 {
                for component in 0..dimension {
                    workspace.right_hand_side[component] +=
                        coefficient * workspace.stages[previous * dimension + component];
                }
            }
        }
        solve_factorized(
            &workspace.factorization,
            &workspace.pivots,
            &mut workspace.right_hand_side,
            dimension,
        );
        for component in 0..dimension {
            workspace.stages[stage * dimension + component] =
                step * tableau.gamma * workspace.right_hand_side[component];
        }
        stats.linear_solves += 1;
    }

    candidate.copy_from_slice(state);
    workspace.error.fill(0.0);
    for stage in 0..tableau.stages {
        for component in 0..dimension {
            let increment = workspace.stages[stage * dimension + component];
            candidate[component] += tableau.weights[stage] * increment;
            workspace.error[component] += tableau.error_weights[stage] * increment;
        }
    }
    let mut error_estimate = if options.adaptive {
        match adaptive_error_estimator {
            AdaptiveErrorEstimator::Embedded => {
                scaled_error_norm(state, candidate, &workspace.error, options)
            }
            AdaptiveErrorEstimator::RichardsonStepDoubling { method_order } => {
                richardson_step_doubling(
                    problem,
                    candidate,
                    state,
                    time,
                    step,
                    options,
                    tableau,
                    method_order,
                    workspace,
                    stats,
                )?
            }
        }
    } else {
        0.0
    };

    // OrdinaryDiffEq's Rodas5Pr performs an additional residual check only
    // when the embedded estimate accepts the step. The three H rows are the
    // pinned Rodas5P dense-output weights; use otherwise-idle stage buffers
    // here so this check remains allocation-free without clobbering the
    // current derivative needed if the integrator rejects and retries.
    if residual_control && options.adaptive && error_estimate < 1.0 {
        let dimension = state.len();
        for component in 0..dimension {
            workspace.error[component] = 0.0;
            workspace.stage_derivative[component] = 0.0;
            workspace.perturbed_derivative[component] = 0.0;
            for stage in 0..tableau.stages {
                let increment = workspace.stages[stage * dimension + component];
                workspace.error[component] += RODAS5P_H[stage] * increment;
                workspace.stage_derivative[component] += RODAS5P_H[8 + stage] * increment;
                workspace.perturbed_derivative[component] += RODAS5P_H[16 + stage] * increment;
            }
            workspace.perturbed_state[component] = 0.5
                * (state[component]
                    + candidate[component]
                    + 0.5
                        * (workspace.error[component]
                            + 0.5
                                * (workspace.stage_derivative[component]
                                    + 0.5 * workspace.perturbed_derivative[component])));
            workspace.right_hand_side[component] = 0.25
                * (workspace.stage_derivative[component]
                    + workspace.perturbed_derivative[component])
                - state[component]
                + candidate[component];
            workspace.right_hand_side[component] /= step;
        }
        evaluate(
            problem,
            &mut workspace.stage_derivative,
            &workspace.perturbed_state,
            time + 0.5 * step,
            stats,
        )?;
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for component in 0..dimension {
            let residual =
                workspace.right_hand_side[component] - workspace.stage_derivative[component];
            let scale = options.absolute_tolerance
                + options.relative_tolerance * workspace.perturbed_state[component].abs();
            numerator += residual * residual;
            denominator += scale * scale;
        }
        if denominator > 0.0 {
            error_estimate = error_estimate.max((numerator / denominator).sqrt());
        }
    }
    Ok(error_estimate)
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn perform_tsit5da<F, P>(
    problem: &OdeProblem<F, P>,
    candidate: &mut [f64],
    state: &[f64],
    time: f64,
    step: f64,
    options: &SolveOptions,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<f64, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let tableau = &TSIT5DA_TABLEAU;
    let dimension = state.len();

    for stage in 0..tableau.stages {
        workspace.stage_state.copy_from_slice(state);
        for previous in 0..stage {
            let coefficient = tableau.a[stage * tableau.stages + previous];
            if coefficient != 0.0 {
                for component in 0..dimension {
                    workspace.stage_state[component] +=
                        coefficient * workspace.stages[previous * dimension + component];
                }
            }
        }

        if stage == 0 {
            workspace
                .stage_derivative
                .copy_from_slice(&workspace.current_derivative);
        } else {
            evaluate(
                problem,
                &mut workspace.stage_derivative,
                &workspace.stage_state,
                time + tableau.nodes[stage] * step,
                stats,
            )?;
        }
        for component in 0..dimension {
            workspace.stages[stage * dimension + component] =
                step * workspace.stage_derivative[component];
        }
    }

    candidate.copy_from_slice(state);
    workspace.error.fill(0.0);
    for stage in 0..tableau.stages {
        for component in 0..dimension {
            let increment = workspace.stages[stage * dimension + component];
            candidate[component] += tableau.weights[stage] * increment;
            workspace.error[component] += tableau.error_weights[stage] * increment;
        }
    }

    Ok(if options.adaptive {
        scaled_error_norm(state, candidate, &workspace.error, options)
    } else {
        0.0
    })
}

/// Estimates local error with two half steps when selected by a method.
///
/// For a method of order `p`, the difference between one full step and two
/// half steps is `(2^p - 1)` times the local error of the refined solution to
/// leading order. The refined solution is retained as the candidate, making
/// this an asymptotically valid fallback rather than a lower-order defect
/// heuristic. This path is intentionally selected per method. `ROS34PW1a`
/// uses it consistently because its published embedded combination has a
/// scalar-linear cancellation blind spot; consistently using Richardson also
/// avoids switching estimators when numerical Jacobians perturb that zero.
#[allow(clippy::too_many_arguments)]
fn richardson_step_doubling<F, P>(
    problem: &OdeProblem<F, P>,
    candidate: &mut [f64],
    state: &[f64],
    time: f64,
    step: f64,
    options: &SolveOptions,
    tableau: &RodasTableau,
    method_order: i32,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<f64, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let dimension = state.len();
    let half_step = step / 2.0;
    let mut midpoint = vec![0.0; dimension];
    let mut refined_candidate = vec![0.0; dimension];
    let mut refinement_workspace = Workspace::new(dimension);
    let mut fixed_options = options.clone();
    fixed_options.adaptive = false;

    evaluate(
        problem,
        &mut refinement_workspace.current_derivative,
        state,
        time,
        stats,
    )?;
    perform_rodas(
        problem,
        &mut midpoint,
        state,
        time,
        half_step,
        &fixed_options,
        tableau,
        false,
        AdaptiveErrorEstimator::Embedded,
        &mut refinement_workspace,
        stats,
    )?;

    refinement_workspace.differentiation_valid = false;
    evaluate(
        problem,
        &mut refinement_workspace.current_derivative,
        &midpoint,
        time + half_step,
        stats,
    )?;
    perform_rodas(
        problem,
        &mut refined_candidate,
        &midpoint,
        time + half_step,
        half_step,
        &fixed_options,
        tableau,
        false,
        AdaptiveErrorEstimator::Embedded,
        &mut refinement_workspace,
        stats,
    )?;

    let richardson_denominator = 2.0_f64.powi(method_order) - 1.0;
    for component in 0..dimension {
        workspace.error[component] =
            (refined_candidate[component] - candidate[component]) / richardson_denominator;
    }
    candidate.copy_from_slice(&refined_candidate);
    Ok(scaled_error_norm(
        state,
        candidate,
        &workspace.error,
        options,
    ))
}

#[allow(clippy::too_many_arguments)]
fn prepare_factorization<F, P>(
    problem: &OdeProblem<F, P>,
    state: &[f64],
    time: f64,
    step: f64,
    gamma: f64,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    if !workspace.differentiation_valid {
        differentiate(problem, state, time, workspace, stats)?;
        workspace.differentiation_valid = true;
    }
    let dimension = state.len();
    for row in 0..dimension {
        for column in 0..dimension {
            workspace.factorization[row * dimension + column] = f64::from(row == column)
                - gamma * step * workspace.jacobian[row * dimension + column];
        }
    }
    factorize(
        &mut workspace.factorization,
        &mut workspace.pivots,
        dimension,
    )?;
    stats.linear_factorizations += 1;
    Ok(())
}

fn differentiate<F, P>(
    problem: &OdeProblem<F, P>,
    state: &[f64],
    time: f64,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let dimension = state.len();
    if problem.evaluate_jacobian(&mut workspace.jacobian, state, time) {
        ensure_finite(&workspace.jacobian)?;
    } else {
        for column in 0..dimension {
            workspace.perturbed_state.copy_from_slice(state);
            let perturbation = f64::EPSILON.sqrt() * state[column].abs().max(1.0);
            workspace.perturbed_state[column] += perturbation;
            evaluate_unchecked(
                problem,
                &mut workspace.perturbed_derivative,
                &workspace.perturbed_state,
                time,
                stats,
            );
            for row in 0..dimension {
                workspace.jacobian[row * dimension + column] =
                    (workspace.perturbed_derivative[row] - workspace.current_derivative[row])
                        / perturbation;
            }
        }
        ensure_finite(&workspace.jacobian)?;
    }
    let time_perturbation = f64::EPSILON.sqrt() * time.abs().max(1.0);
    evaluate_unchecked(
        problem,
        &mut workspace.perturbed_derivative,
        state,
        time + time_perturbation,
        stats,
    );
    for component in 0..dimension {
        workspace.time_derivative[component] = (workspace.perturbed_derivative[component]
            - workspace.current_derivative[component])
            / time_perturbation;
    }
    ensure_finite(&workspace.time_derivative)?;
    stats.jacobian_evaluations += 1;
    Ok(())
}

fn scaled_error_norm(
    state: &[f64],
    candidate: &[f64],
    error: &[f64],
    options: &SolveOptions,
) -> f64 {
    let mut squared_norm = 0.0;
    for ((&value, &candidate), &error) in state.iter().zip(candidate).zip(error) {
        let scale = options.absolute_tolerance
            + options.relative_tolerance * value.abs().max(candidate.abs());
        squared_norm += (error / scale).powi(2);
    }
    (squared_norm / state.len() as f64).sqrt()
}

fn evaluate_unchecked<F, P>(
    problem: &OdeProblem<F, P>,
    derivative: &mut [f64],
    state: &[f64],
    time: f64,
    stats: &mut SolverStats,
) where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    (problem.rhs)(derivative, state, problem.parameters(), time);
    stats.rhs_evaluations += 1;
}

fn evaluate<F, P>(
    problem: &OdeProblem<F, P>,
    derivative: &mut [f64],
    state: &[f64],
    time: f64,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    evaluate_unchecked(problem, derivative, state, time, stats);
    ensure_finite(derivative)
}

fn ensure_finite(values: &[f64]) -> Result<(), SolveError> {
    values
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteDerivative)
}

fn estimate_initial_step(
    state: &[f64],
    derivative: &[f64],
    options: &SolveOptions,
    maximum_step: f64,
) -> f64 {
    let mut state_norm = 0.0;
    let mut derivative_norm = 0.0;
    for (value, derivative) in state.iter().zip(derivative) {
        let scale = options.absolute_tolerance + options.relative_tolerance * value.abs();
        state_norm += (value / scale).powi(2);
        derivative_norm += (derivative / scale).powi(2);
    }
    let dimension = state.len() as f64;
    state_norm = (state_norm / dimension).sqrt();
    derivative_norm = (derivative_norm / dimension).sqrt();
    if state_norm < 1.0e-5 || derivative_norm < 1.0e-5 {
        1.0e-6_f64.min(maximum_step)
    } else {
        (0.01 * state_norm / derivative_norm).min(maximum_step)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use super::{
        Grk4a, Grk4t, Rodas3, Rodas3P, Rodas3d, Rodas4, Rodas4P, Rodas5P, Rodas5Pr, Rodas6P,
        Rodas23W, Ros2, Ros3, Ros3Pr, Ros3Prl, Ros3Prl2, Ros3p, Ros34Prw, Ros34Pw1a, Ros34Pw1b,
        Rosenbrock32, RosenbrockW6S4OS,
    };
    use crate::{CallbackAction, OdeProblem, SaveMode, SolveError, SolveOptions, solve};

    type TestRhs = fn(&mut [f64], &[f64], &(), f64);

    fn stiff_problem(span: (f64, f64), initial: f64) -> OdeProblem<TestRhs, ()> {
        fn rhs(du: &mut [f64], u: &[f64], _: &(), time: f64) {
            du[0] = -1000.0 * (u[0] - time.cos()) - time.sin();
        }
        OdeProblem::new(rhs as TestRhs, vec![initial], span, ())
    }

    fn adaptive_options() -> SolveOptions {
        SolveOptions {
            absolute_tolerance: 1.0e-8,
            relative_tolerance: 1.0e-8,
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        }
    }

    #[test]
    fn adaptive_methods_solve_a_stiff_nonautonomous_problem() {
        let endpoints = [
            solve(&stiff_problem((0.0, 1.0), 1.0), Ros2, &adaptive_options())
                .unwrap()
                .last_state()[0],
            solve(&stiff_problem((0.0, 1.0), 1.0), Rodas3, &adaptive_options())
                .unwrap()
                .last_state()[0],
            solve(
                &stiff_problem((0.0, 1.0), 1.0),
                Rodas3d,
                &adaptive_options(),
            )
            .unwrap()
            .last_state()[0],
            solve(&stiff_problem((0.0, 1.0), 1.0), Ros3, &adaptive_options())
                .unwrap()
                .last_state()[0],
            solve(&stiff_problem((0.0, 1.0), 1.0), Ros3Pr, &adaptive_options())
                .unwrap()
                .last_state()[0],
            solve(
                &stiff_problem((0.0, 1.0), 1.0),
                Ros3Prl,
                &adaptive_options(),
            )
            .unwrap()
            .last_state()[0],
            solve(
                &stiff_problem((0.0, 1.0), 1.0),
                Ros3Prl2,
                &adaptive_options(),
            )
            .unwrap()
            .last_state()[0],
            solve(&stiff_problem((0.0, 1.0), 1.0), Ros3p, &adaptive_options())
                .unwrap()
                .last_state()[0],
            solve(
                &stiff_problem((0.0, 1.0), 1.0),
                Ros34Prw,
                &adaptive_options(),
            )
            .unwrap()
            .last_state()[0],
            solve(
                &stiff_problem((0.0, 1.0), 1.0),
                Rosenbrock32,
                &adaptive_options(),
            )
            .unwrap()
            .last_state()[0],
            solve(&stiff_problem((0.0, 1.0), 1.0), Rodas4, &adaptive_options())
                .unwrap()
                .last_state()[0],
            solve(
                &stiff_problem((0.0, 1.0), 1.0),
                Rodas4P,
                &adaptive_options(),
            )
            .unwrap()
            .last_state()[0],
            solve(&stiff_problem((0.0, 1.0), 1.0), Grk4a, &adaptive_options())
                .unwrap()
                .last_state()[0],
            solve(&stiff_problem((0.0, 1.0), 1.0), Grk4t, &adaptive_options())
                .unwrap()
                .last_state()[0],
            solve(
                &stiff_problem((0.0, 1.0), 1.0),
                Ros34Pw1b,
                &adaptive_options(),
            )
            .unwrap()
            .last_state()[0],
            solve(
                &stiff_problem((0.0, 1.0), 1.0),
                Rodas5P,
                &adaptive_options(),
            )
            .unwrap()
            .last_state()[0],
            solve(
                &stiff_problem((0.0, 1.0), 1.0),
                Rodas6P,
                &adaptive_options(),
            )
            .unwrap()
            .last_state()[0],
            solve(
                &stiff_problem((0.0, 1.0), 1.0),
                Rodas23W,
                &adaptive_options(),
            )
            .unwrap()
            .last_state()[0],
        ];
        for endpoint in endpoints {
            assert!((endpoint - 1.0_f64.cos()).abs() < 2.0e-6);
        }
    }

    fn fixed_endpoint<A: crate::OdeAlgorithm>(algorithm: A, step: f64) -> f64 {
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
            vec![1.0],
            (0.0, 1.0),
            (),
        );
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(step),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };
        solve(&problem, algorithm, &options).unwrap().last_state()[0]
    }

    fn convergence_ratio<A: crate::OdeAlgorithm + Copy>(algorithm: A, step: f64) -> f64 {
        let coarse = (fixed_endpoint(algorithm, step) - std::f64::consts::E).abs();
        let fine = (fixed_endpoint(algorithm, step / 2.0) - std::f64::consts::E).abs();
        coarse / fine
    }

    #[test]
    fn ros34pw1a_uses_its_own_tableau() {
        let ros34pw1a = fixed_endpoint(Ros34Pw1a, 0.25);
        let rodas3p = fixed_endpoint(Rodas3P, 0.25);

        assert!(
            (ros34pw1a - rodas3p).abs() > 1.0e-8,
            "ROS34PW1a unexpectedly reproduced the Rodas3P step: {ros34pw1a:.17e}"
        );
        assert!(convergence_ratio(Ros34Pw1a, 0.1) > 7.0);
    }

    #[test]
    fn ros34pw1a_controls_a_zero_embedded_error_with_step_doubling() {
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
            vec![1.0],
            (0.0, 1.0),
            (),
        );
        let options = SolveOptions {
            absolute_tolerance: 1.0e-9,
            relative_tolerance: 1.0e-9,
            initial_step: Some(1.0),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        let solution = solve(&problem, Ros34Pw1a, &options).unwrap();

        assert!(
            (solution.last_state()[0] - std::f64::consts::E).abs() < 2.0e-7,
            "endpoint={:.17e}",
            solution.last_state()[0]
        );
        assert!(solution.stats().rejected_steps > 0);
    }

    #[test]
    fn methods_have_their_expected_fixed_step_orders() {
        let ratios = [
            convergence_ratio(Ros2, 0.1),
            convergence_ratio(Rosenbrock32, 0.1),
            convergence_ratio(Rodas3, 0.1),
            convergence_ratio(Rodas3d, 0.1),
            convergence_ratio(Ros3Pr, 0.1),
            convergence_ratio(Ros3Prl, 0.1),
            convergence_ratio(Ros3Prl2, 0.1),
            convergence_ratio(Ros3p, 0.1),
            convergence_ratio(Ros34Prw, 0.1),
            convergence_ratio(Rodas4, 0.1),
            convergence_ratio(Rodas4P, 0.1),
            convergence_ratio(Grk4a, 0.1),
            convergence_ratio(Grk4t, 0.1),
            convergence_ratio(Ros34Pw1b, 0.1),
            convergence_ratio(Rodas5P, 0.2),
            convergence_ratio(Rodas6P, 0.2),
            convergence_ratio(RosenbrockW6S4OS, 0.1),
            convergence_ratio(Rodas23W, 0.1),
        ];
        assert!(ratios[0] > 3.0);
        assert!(ratios[1] > 7.0);
        assert!(ratios[2] > 7.0);
        // Rodas3d is fourth order on this linear problem because its damping
        // parameter is a root of the fourth-order linear order condition.
        assert!(ratios[3] > 14.0);
        assert!(ratios[4] > 7.0);
        assert!(ratios[5] > 7.0);
        assert!(ratios[6] > 7.0);
        assert!(ratios[7] > 7.0);
        assert!(ratios[8] > 7.0);
        assert!(ratios[9] > 14.0);
        assert!(ratios[10] > 14.0);
        assert!(ratios[11] > 14.0);
        assert!(ratios[12] > 14.0);
        assert!(ratios[13] > 7.0);
        assert!(ratios[14] > 14.0);
        assert!(ratios[15] > 14.0);
        assert!(ratios[16] > 7.0);
        // Pinned Rodas23W uses a second-order primary solution.
        assert!(ratios[17] > 3.0 && ratios[17] < 5.5);
    }

    #[test]
    fn rodas5pr_matches_rodas5p_on_regular_ode_paths() {
        let fixed_options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.1),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };
        let p = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
            vec![1.0],
            (0.0, 1.0),
            (),
        );
        let rodas5p = solve(&p, Rodas5P, &fixed_options).unwrap();
        let rodas5pr = solve(&p, Rodas5Pr, &fixed_options).unwrap();
        assert!((rodas5p.last_state()[0] - rodas5pr.last_state()[0]).abs() < 1.0e-14);

        let adaptive = adaptive_options();
        let rodas5pr = solve(&stiff_problem((0.0, 1.0), 1.0), Rodas5Pr, &adaptive).unwrap();
        assert!((rodas5pr.last_state()[0] - 1.0_f64.cos()).abs() < 2.0e-6);
        assert!(rodas5pr.stats().rhs_evaluations > 0);
    }

    #[test]
    fn rodas23w_supports_jacobian_backward_callbacks_and_save_at() {
        let jacobian_calls = Rc::new(Cell::new(0));
        let jacobian_calls_for_problem = Rc::clone(&jacobian_calls);
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -2.0 * u[0],
            vec![1.0],
            (0.0, 1.0),
            (),
        )
        .with_jacobian(move |jacobian: &mut [f64], _: &[f64], _: &(), _: f64| {
            jacobian_calls_for_problem.set(jacobian_calls_for_problem.get() + 1);
            jacobian[0] = -2.0;
        })
        .with_discrete_callback(
            |_, _, time| time == 0.5,
            |state, _, _| {
                state[0] += 0.25;
                CallbackAction::Continue
            },
        );
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.25),
            save: SaveMode::Endpoints,
            save_at: vec![0.25, 0.5, 0.75],
            ..SolveOptions::default()
        };
        let solution = solve(&problem, Rodas23W, &options).unwrap();
        assert_eq!(solution.stats().callback_invocations, 1);
        assert!(solution.stats().jacobian_evaluations > 0);
        assert!(jacobian_calls.get() > 0);
        for time in options.save_at {
            assert!(solution.times().contains(&time), "missing save_at={time}");
        }

        let backward_problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -2.0 * u[0],
            vec![(-2.0_f64).exp()],
            (1.0, 0.0),
            (),
        );
        let backward_options = SolveOptions {
            initial_step: Some(0.01),
            max_step: 0.01,
            save: SaveMode::Endpoints,
            ..adaptive_options()
        };
        let endpoint = solve(&backward_problem, Rodas23W, &backward_options)
            .unwrap()
            .last_state()[0];
        assert!((endpoint - 1.0).abs() < 1.0e-5, "endpoint={endpoint:.17e}");
    }

    #[test]
    fn w6s4os_is_fixed_step_only_and_supports_backward_integration() {
        let adaptive_error = solve(
            &stiff_problem((0.0, 1.0), 1.0),
            RosenbrockW6S4OS,
            &adaptive_options(),
        )
        .expect_err("RosenbrockW6S4OS must reject adaptive scheduling");
        assert_eq!(adaptive_error, SolveError::AdaptiveStepUnsupported);

        let backward_problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -2.0 * u[0],
            vec![(-2.0_f64).exp()],
            (1.0, 0.0),
            (),
        );
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.05),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };
        let endpoint = solve(&backward_problem, RosenbrockW6S4OS, &options)
            .unwrap()
            .last_state()[0];
        assert!((endpoint - 1.0).abs() < 5.0e-7, "endpoint={endpoint:.17e}");
    }

    #[test]
    fn stiff_methods_preserve_callbacks_and_requested_samples() {
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -u[0],
            vec![1.0],
            (0.0, 1.0),
            (),
        )
        .with_discrete_callback(
            |_, _, time| time == 0.5,
            |state, _, _| {
                state[0] += 0.25;
                CallbackAction::Continue
            },
        );
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.25),
            save: SaveMode::Endpoints,
            save_at: vec![0.25, 0.5, 0.75],
            ..SolveOptions::default()
        };
        let solution = solve(&problem, RosenbrockW6S4OS, &options).unwrap();
        assert_eq!(solution.stats().callback_invocations, 1);
        for &time in &options.save_at {
            assert!(solution.times().contains(&time), "missing save_at={time}");
        }
        assert!(solution.last_state()[0] > 0.0);

        let rodas4p_solution = solve(&problem, Rodas4P, &options).unwrap();
        assert_eq!(rodas4p_solution.stats().callback_invocations, 1);
        assert!(rodas4p_solution.stats().jacobian_evaluations > 0);
        for &time in &options.save_at {
            assert!(
                rodas4p_solution.times().contains(&time),
                "missing Rodas4P save_at={time}"
            );
        }
        assert!(rodas4p_solution.last_state()[0] > 0.0);

        let rodas6p_solution = solve(&problem, Rodas6P, &options).unwrap();
        assert_eq!(rodas6p_solution.stats().callback_invocations, 1);
        assert!(rodas6p_solution.stats().jacobian_evaluations > 0);
        for &time in &options.save_at {
            assert!(
                rodas6p_solution.times().contains(&time),
                "missing Rodas6P save_at={time}"
            );
        }
        assert!(rodas6p_solution.last_state()[0] > 0.0);

        let grk4t_options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.25),
            save: SaveMode::Endpoints,
            save_at: vec![0.25, 0.5, 0.75],
            ..SolveOptions::default()
        };
        let grk4t_solution = solve(&problem, Grk4t, &grk4t_options).unwrap();
        assert_eq!(grk4t_solution.stats().callback_invocations, 1);
        for time in grk4t_options.save_at {
            assert!(
                grk4t_solution.times().contains(&time),
                "missing GRK4T save_at={time}"
            );
        }
        assert!(grk4t_solution.last_state()[0] > 0.0);
    }

    #[test]
    fn ros34pw1b_supports_jacobian_callbacks_and_requested_samples() {
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -u[0],
            vec![1.0],
            (0.0, 1.0),
            (),
        )
        .with_jacobian(|jacobian: &mut [f64], _: &[f64], _: &(), _: f64| {
            jacobian[0] = -1.0;
        })
        .with_discrete_callback(
            |_, _, time| time == 0.5,
            |state, _, _| {
                state[0] += 0.25;
                CallbackAction::Continue
            },
        );
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.25),
            save: SaveMode::Endpoints,
            save_at: vec![0.25, 0.5, 0.75],
            ..SolveOptions::default()
        };
        let solution = solve(&problem, Ros34Pw1b, &options).unwrap();
        assert_eq!(solution.stats().callback_invocations, 1);
        assert!(solution.stats().jacobian_evaluations > 0);
        for time in options.save_at {
            assert!(solution.times().contains(&time), "missing save_at={time}");
        }
        assert!(solution.last_state()[0] > 0.0);
    }

    #[test]
    fn methods_support_backward_integration() {
        let backward_problem = || {
            OdeProblem::new(
                |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -2.0 * u[0],
                vec![(-2.0_f64).exp()],
                (1.0, 0.0),
                (),
            )
        };
        let ros3p_options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.01),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };
        let ros3p_endpoint = solve(&backward_problem(), Ros3p, &ros3p_options)
            .unwrap()
            .last_state()[0];
        assert!((ros3p_endpoint - 1.0).abs() < 3.0e-6);

        for (name, endpoint) in [
            (
                "ros2",
                solve(&backward_problem(), Ros2, &adaptive_options())
                    .unwrap()
                    .last_state()[0],
            ),
            (
                "rodas3",
                solve(&backward_problem(), Rodas3, &adaptive_options())
                    .unwrap()
                    .last_state()[0],
            ),
            (
                "rodas3d",
                solve(&backward_problem(), Rodas3d, &adaptive_options())
                    .unwrap()
                    .last_state()[0],
            ),
            (
                "rosenbrock32",
                solve(&backward_problem(), Rosenbrock32, &adaptive_options())
                    .unwrap()
                    .last_state()[0],
            ),
            (
                "rodas4",
                solve(&backward_problem(), Rodas4, &adaptive_options())
                    .unwrap()
                    .last_state()[0],
            ),
            (
                "rodas4p",
                solve(&backward_problem(), Rodas4P, &adaptive_options())
                    .unwrap()
                    .last_state()[0],
            ),
            (
                "rodas5p",
                solve(&backward_problem(), Rodas5P, &adaptive_options())
                    .unwrap()
                    .last_state()[0],
            ),
            (
                "rodas6p",
                solve(&backward_problem(), Rodas6P, &adaptive_options())
                    .unwrap()
                    .last_state()[0],
            ),
            (
                "grk4a",
                solve(&backward_problem(), Grk4a, &adaptive_options())
                    .unwrap()
                    .last_state()[0],
            ),
            (
                "ros34pw1b",
                solve(
                    &backward_problem(),
                    Ros34Pw1b,
                    &SolveOptions {
                        initial_step: Some(0.005),
                        max_step: 0.005,
                        ..adaptive_options()
                    },
                )
                .unwrap()
                .last_state()[0],
            ),
            (
                "grk4t",
                solve(&backward_problem(), Grk4t, &adaptive_options())
                    .unwrap()
                    .last_state()[0],
            ),
            (
                "ros34prw",
                solve(&backward_problem(), Ros34Prw, &adaptive_options())
                    .unwrap()
                    .last_state()[0],
            ),
        ] {
            assert!(
                (endpoint - 1.0).abs() < 3.0e-7,
                "{name}: endpoint={endpoint:.17e}"
            );
        }
    }

    #[test]
    fn ros3pr_supports_fixed_step_backward_integration() {
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -2.0 * u[0],
            vec![(-2.0_f64).exp()],
            (1.0, 0.0),
            (),
        );
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.01),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };
        let solution = solve(&problem, Ros3Pr, &options).unwrap();
        assert!((solution.last_state()[0] - 1.0).abs() < 3.0e-6);
    }

    #[test]
    fn ros3prl_covers_regular_ode_lifecycle() {
        let jacobian_calls = Rc::new(Cell::new(0));
        let jacobian_calls_for_problem = Rc::clone(&jacobian_calls);
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -u[0],
            vec![1.0],
            (0.0, 1.0),
            (),
        )
        .with_jacobian(move |jacobian: &mut [f64], _: &[f64], _: &(), _: f64| {
            jacobian_calls_for_problem.set(jacobian_calls_for_problem.get() + 1);
            jacobian[0] = -1.0;
        })
        .with_discrete_callback(
            |_, _, time| time == 0.5,
            |state, _, _| {
                state[0] += 0.25;
                CallbackAction::Continue
            },
        );
        let fixed_options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.25),
            save: SaveMode::Endpoints,
            save_at: vec![0.25, 0.5, 0.75],
            ..SolveOptions::default()
        };
        let fixed = solve(&problem, Ros3Prl, &fixed_options).unwrap();
        assert_eq!(fixed.stats().callback_invocations, 1);
        assert!(fixed.stats().jacobian_evaluations > 0);
        assert!(jacobian_calls.get() > 0);
        for &time in &fixed_options.save_at {
            assert!(fixed.times().contains(&time), "missing save_at={time}");
        }
        assert!(fixed.last_state()[0] > 0.0);

        let adaptive = solve(
            &stiff_problem((0.0, 1.0), 1.0),
            Ros3Prl,
            &adaptive_options(),
        )
        .unwrap();
        assert!((adaptive.last_state()[0] - 1.0_f64.cos()).abs() < 2.0e-6);

        let backward_problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -2.0 * u[0],
            vec![(-2.0_f64).exp()],
            (1.0, 0.0),
            (),
        );
        let backward_options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.01),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };
        let backward = solve(&backward_problem, Ros3Prl, &backward_options)
            .unwrap()
            .last_state()[0];
        assert!((backward - 1.0).abs() < 3.0e-6);
    }

    #[test]
    fn ros3prl2_covers_regular_ode_lifecycle() {
        let jacobian_calls = Rc::new(Cell::new(0));
        let jacobian_calls_for_problem = Rc::clone(&jacobian_calls);
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
            vec![1.0],
            (0.0, 1.0),
            (),
        )
        .with_jacobian(move |jacobian: &mut [f64], _: &[f64], _: &(), _: f64| {
            jacobian_calls_for_problem.set(jacobian_calls_for_problem.get() + 1);
            jacobian[0] = 1.0;
        });
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.01),
            save: SaveMode::Endpoints,
            save_at: vec![0.25, 0.5, 0.75],
            ..SolveOptions::default()
        };
        let solution = solve(&problem, Ros3Prl2, &options).unwrap();
        assert!(jacobian_calls.get() > 0);
        for &time in &options.save_at {
            assert!(solution.times().contains(&time), "missing save_at={time}");
        }
        assert!((solution.last_state()[0] - 0.75_f64.exp()).abs() < 3.0e-6);

        let backward_problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -2.0 * u[0],
            vec![(-2.0_f64).exp()],
            (1.0, 0.0),
            (),
        );
        let backward_options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.01),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };
        let backward = solve(&backward_problem, Ros3Prl2, &backward_options)
            .unwrap()
            .last_state()[0];
        assert!((backward - 1.0).abs() < 3.0e-6);
    }

    #[test]
    fn analytic_jacobian_reduces_rhs_work() {
        fn rhs(du: &mut [f64], u: &[f64], _: &(), time: f64) {
            du[0] = -1000.0 * (u[0] - time.cos()) - time.sin();
        }
        type Rhs = fn(&mut [f64], &[f64], &(), f64);
        let numeric = OdeProblem::new(rhs as Rhs, vec![1.0], (0.0, 0.2), ());
        let analytic = OdeProblem::new(rhs as Rhs, vec![1.0], (0.0, 0.2), ())
            .with_jacobian(|jacobian: &mut [f64], _: &[f64], _: &(), _: f64| jacobian[0] = -1000.0);
        let numeric = solve(&numeric, Rodas4, &adaptive_options()).unwrap();
        let analytic = solve(&analytic, Rodas4, &adaptive_options()).unwrap();
        assert!((numeric.last_state()[0] - analytic.last_state()[0]).abs() < 2.0e-10);
        assert!(analytic.stats().rhs_evaluations < numeric.stats().rhs_evaluations);

        let numeric = solve(
            &OdeProblem::new(rhs as Rhs, vec![1.0], (0.0, 0.2), ()),
            Rodas3d,
            &adaptive_options(),
        )
        .unwrap();
        let analytic = solve(
            &OdeProblem::new(rhs as Rhs, vec![1.0], (0.0, 0.2), ()).with_jacobian(
                |jacobian: &mut [f64], _: &[f64], _: &(), _: f64| jacobian[0] = -1000.0,
            ),
            Rodas3d,
            &adaptive_options(),
        )
        .unwrap();
        assert!((numeric.last_state()[0] - analytic.last_state()[0]).abs() < 2.0e-10);
        assert!(analytic.stats().rhs_evaluations < numeric.stats().rhs_evaluations);

        let numeric = solve(
            &OdeProblem::new(rhs as Rhs, vec![1.0], (0.0, 0.2), ()),
            Ros34Prw,
            &adaptive_options(),
        )
        .unwrap();
        let analytic = solve(
            &OdeProblem::new(rhs as Rhs, vec![1.0], (0.0, 0.2), ()).with_jacobian(
                |jacobian: &mut [f64], _: &[f64], _: &(), _: f64| jacobian[0] = -1000.0,
            ),
            Ros34Prw,
            &adaptive_options(),
        )
        .unwrap();
        assert!((numeric.last_state()[0] - analytic.last_state()[0]).abs() < 2.0e-10);
        assert!(analytic.stats().rhs_evaluations < numeric.stats().rhs_evaluations);

        let numeric = solve(
            &OdeProblem::new(rhs as Rhs, vec![1.0], (0.0, 0.2), ()),
            Grk4a,
            &adaptive_options(),
        )
        .unwrap();
        let analytic = solve(
            &OdeProblem::new(rhs as Rhs, vec![1.0], (0.0, 0.2), ()).with_jacobian(
                |jacobian: &mut [f64], _: &[f64], _: &(), _: f64| jacobian[0] = -1000.0,
            ),
            Grk4a,
            &adaptive_options(),
        )
        .unwrap();
        assert!((numeric.last_state()[0] - analytic.last_state()[0]).abs() < 2.0e-10);
        assert!(analytic.stats().rhs_evaluations < numeric.stats().rhs_evaluations);

        let numeric = solve(
            &OdeProblem::new(rhs as Rhs, vec![1.0], (0.0, 0.2), ()),
            Ros3Pr,
            &adaptive_options(),
        )
        .unwrap();
        let analytic = solve(
            &OdeProblem::new(rhs as Rhs, vec![1.0], (0.0, 0.2), ()).with_jacobian(
                |jacobian: &mut [f64], _: &[f64], _: &(), _: f64| jacobian[0] = -1000.0,
            ),
            Ros3Pr,
            &adaptive_options(),
        )
        .unwrap();
        assert!((numeric.last_state()[0] - analytic.last_state()[0]).abs() < 2.0e-10);
        assert!(analytic.stats().rhs_evaluations < numeric.stats().rhs_evaluations);

        let numeric = solve(
            &OdeProblem::new(rhs as Rhs, vec![1.0], (0.0, 0.2), ()),
            Rodas3,
            &adaptive_options(),
        )
        .unwrap();
        let analytic = solve(
            &OdeProblem::new(rhs as Rhs, vec![1.0], (0.0, 0.2), ()).with_jacobian(
                |jacobian: &mut [f64], _: &[f64], _: &(), _: f64| jacobian[0] = -1000.0,
            ),
            Rodas3,
            &adaptive_options(),
        )
        .unwrap();
        assert!((numeric.last_state()[0] - analytic.last_state()[0]).abs() < 2.0e-10);
        assert!(analytic.stats().rhs_evaluations < numeric.stats().rhs_evaluations);

        let numeric = solve(
            &OdeProblem::new(rhs as Rhs, vec![1.0], (0.0, 0.2), ()),
            Ros2,
            &adaptive_options(),
        )
        .unwrap();
        let analytic = solve(
            &OdeProblem::new(rhs as Rhs, vec![1.0], (0.0, 0.2), ()).with_jacobian(
                |jacobian: &mut [f64], _: &[f64], _: &(), _: f64| jacobian[0] = -1000.0,
            ),
            Ros2,
            &adaptive_options(),
        )
        .unwrap();
        assert!((numeric.last_state()[0] - analytic.last_state()[0]).abs() < 2.0e-10);
        assert!(analytic.stats().rhs_evaluations < numeric.stats().rhs_evaluations);
    }

    #[test]
    fn callbacks_invalidate_stiff_step_caches_and_save_at_is_honored() {
        let problem = stiff_problem((0.0, 1.0), 1.0).with_continuous_callback(
            |_, _, time| time - 0.5,
            |state, _, _| {
                state[0] += 0.01;
                CallbackAction::Continue
            },
        );
        let options = SolveOptions {
            save: SaveMode::Endpoints,
            save_at: vec![0.25, 0.5, 0.75],
            ..adaptive_options()
        };
        let solution = solve(&problem, Rodas4, &options).unwrap();
        assert!(solution.stats().callback_invocations > 0);
        assert!(solution.times().contains(&0.25));
        assert!(solution.times().contains(&0.5));
        assert!(solution.times().contains(&0.75));

        let grk4a_solution = solve(&problem, Grk4a, &options).unwrap();
        assert!(grk4a_solution.stats().callback_invocations > 0);
        assert!(grk4a_solution.times().contains(&0.25));
        assert!(grk4a_solution.times().contains(&0.5));
        assert!(grk4a_solution.times().contains(&0.75));

        let rodas3_solution = solve(&problem, Rodas3, &options).unwrap();
        assert!(rodas3_solution.stats().callback_invocations > 0);
        assert!(rodas3_solution.times().contains(&0.25));
        assert!(rodas3_solution.times().contains(&0.5));
        assert!(rodas3_solution.times().contains(&0.75));

        let rodas3d_solution = solve(&problem, Rodas3d, &options).unwrap();
        assert!(rodas3d_solution.stats().callback_invocations > 0);
        assert!(rodas3d_solution.times().contains(&0.25));
        assert!(rodas3d_solution.times().contains(&0.5));
        assert!(rodas3d_solution.times().contains(&0.75));

        let ros2_solution = solve(&problem, Ros2, &options).unwrap();
        assert!(ros2_solution.stats().callback_invocations > 0);
        assert!(ros2_solution.times().contains(&0.25));
        assert!(ros2_solution.times().contains(&0.5));
        assert!(ros2_solution.times().contains(&0.75));

        let ros3pr_solution = solve(&problem, Ros3Pr, &options).unwrap();
        assert!(ros3pr_solution.stats().callback_invocations > 0);
        assert!(ros3pr_solution.times().contains(&0.25));
        assert!(ros3pr_solution.times().contains(&0.5));
        assert!(ros3pr_solution.times().contains(&0.75));
        let ros3p_solution = solve(&problem, Ros3p, &options).unwrap();
        assert!(ros3p_solution.stats().callback_invocations > 0);
        assert!(ros3p_solution.times().contains(&0.25));
        assert!(ros3p_solution.times().contains(&0.5));
        assert!(ros3p_solution.times().contains(&0.75));
        let ros34prw_solution = solve(&problem, Ros34Prw, &options).unwrap();
        assert!(ros34prw_solution.stats().callback_invocations > 0);
        assert!(ros34prw_solution.times().contains(&0.25));
        assert!(ros34prw_solution.times().contains(&0.5));
        assert!(ros34prw_solution.times().contains(&0.75));
    }

    #[test]
    fn reuses_differentiation_after_rejected_rodas_steps() {
        let options = SolveOptions {
            initial_step: Some(1.0),
            ..adaptive_options()
        };

        let solution = solve(&stiff_problem((0.0, 1.0), 1.0), Rodas4, &options).unwrap();
        let stats = solution.stats();

        assert!(stats.rejected_steps > 0);
        assert!(stats.jacobian_evaluations < stats.accepted_steps + stats.rejected_steps);
    }

    #[test]
    fn callback_effect_is_seen_by_the_next_jacobian() {
        let saw_effect_state = Rc::new(Cell::new(false));
        let jacobian_saw_effect = Rc::clone(&saw_effect_state);
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -u[0],
            vec![1.0],
            (0.0, 0.5),
            (),
        )
        .with_jacobian(move |jacobian: &mut [f64], state: &[f64], _: &(), _: f64| {
            if state[0] == 3.0 {
                jacobian_saw_effect.set(true);
            }
            jacobian[0] = -1.0;
        })
        .with_discrete_callback(
            |_, _, time| time == 0.25,
            |state, _, _| {
                state[0] = 3.0;
                CallbackAction::Continue
            },
        );
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.25),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        solve(&problem, Rodas4, &options).unwrap();

        assert!(saw_effect_state.get());
    }

    #[test]
    fn terminating_callback_skips_post_effect_rodas_work() {
        let rhs_calls = Rc::new(Cell::new(0));
        let observed_calls = Rc::clone(&rhs_calls);
        let problem = OdeProblem::new(
            move |du: &mut [f64], u: &[f64], _: &(), _: f64| {
                observed_calls.set(observed_calls.get() + 1);
                du[0] = if u[0] == 12_345.0 { f64::NAN } else { -u[0] };
            },
            vec![1.0],
            (0.0, 1.0),
            (),
        )
        .with_discrete_callback(
            |_, _, time| time > 0.0,
            |state, _, _| {
                state[0] = 12_345.0;
                CallbackAction::Terminate
            },
        );
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.25),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        let solution = solve(&problem, Rodas4, &options).unwrap();

        assert_eq!(solution.last_state()[0], 12_345.0);
        assert_eq!(rhs_calls.get(), solution.stats().rhs_evaluations);
        assert_eq!(solution.stats().accepted_steps, 1);
        assert_eq!(solution.stats().jacobian_evaluations, 1);
    }
}
