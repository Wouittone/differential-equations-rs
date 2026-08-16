//! Additional native Rosenbrock and Rosenbrock--Wanner methods.
//!
//! Coefficients and stage equations are ported from `OrdinaryDiffEqRosenbrock`
//! and `OrdinaryDiffEqRosenbrockTableaus` at commit
//! `211142263781255a9aa2f910f6760b9f18ec29c8`. The method-specific stiff dense
//! interpolants are not included; trajectory sampling uses the crate's shared
//! recorder.

use std::marker::PhantomData;

use crate::integrator::{
    ControllerConfig, KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::linear::{factorize, solve_factorized};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

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

// ROS2RodasTableau(T, T2) from
// lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl.
const ROS2_A: &[f64] = &[0.0, 0.0, 0.585786437626905, 0.0];
const ROS2_C: &[f64] = &[0.0, 0.0, -1.17157287525381, 0.0];
const ROS2_NODES: &[f64] = &[0.0, 1.0];
const ROS2_D: &[f64] = &[1.7071067811865475, -1.7071067811865475];
const ROS2_B: &[f64] = &[0.8786796564403574, 0.2928932188134525];
// OrdinaryDiffEq uses btilde directly for the embedded error estimate.
const ROS2_E: &[f64] = &[0.2928932188134525, 0.2928932188134525];
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
const RODAS3_A: &[f64] = &[
    0.0, 0.0, 0.0, 0.0, // stage 1
    0.0, 0.0, 0.0, 0.0, // stage 2
    2.0, 0.0, 0.0, 0.0, // stage 3
    2.0, 0.0, 1.0, 0.0, // stage 4
];
const RODAS3_C: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0, // stage 1
    4.0,
    0.0,
    0.0,
    0.0, // stage 2
    1.0,
    -1.0,
    0.0,
    0.0, // stage 3
    1.0,
    -1.0,
    -8.0 / 3.0,
    0.0, // stage 4
];
const RODAS3_NODES: &[f64] = &[0.0, 0.0, 1.0, 1.0];
const RODAS3_D: &[f64] = &[0.5, 1.5, 0.0, 0.0];
const RODAS3_B: &[f64] = &[2.0, 0.0, 1.0, 1.0];
const RODAS3_E: &[f64] = &[0.0, 0.0, 0.0, 1.0];
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
#[allow(clippy::excessive_precision)]
const RODAS3D_A: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0, // stage 1
    2.1736562342774159,
    0.0,
    0.0,
    0.0, // stage 2
    1.745761108723104,
    0.0,
    0.0,
    0.0, // stage 3
    1.745761108723104,
    0.0,
    1.0,
    0.0, // stage 4
];
#[allow(clippy::excessive_precision)]
const RODAS3D_C: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0, // stage 1
    -13.387001858207178,
    0.0,
    0.0,
    0.0, // stage 2
    0.30442314006596932,
    0.30745278826153299,
    0.0,
    0.0, // stage 3
    0.57287646414081528,
    0.34771098605699358,
    -2.7425340696473901,
    0.0, // stage 4
];
const RODAS3D_NODES: &[f64] = &[0.0, 1.2451051999132263, 1.0, 1.0];
const RODAS3D_D: &[f64] = &[0.57281606, -3.819703409768521, 0.0, 0.0];
const RODAS3D_B: &[f64] = &[1.745761108723104, 0.0, 1.0, 1.0];
const RODAS3D_E: &[f64] = &[0.0, 0.0, 0.0, 1.0];
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
const ROS3_A: &[f64] = &[
    0.0, 0.0, 0.0, // stage 1
    1.0, 0.0, 0.0, // stage 2
    1.0, 1.0, 0.0, // stage 3
];
const ROS3_C: &[f64] = &[
    0.0,
    0.0,
    0.0, // stage 1
    -1.0156171083877703,
    0.0,
    0.0, // stage 2
    4.07599564525377,
    9.20767942983308,
    0.0, // stage 3
];
const ROS3_NODES: &[f64] = &[0.0, 0.435866521508459, 0.435866521508459];
const ROS3_D: &[f64] = &[0.435866521508459, 0.24291996454816805, 2.185138002766406];
const ROS3_B: &[f64] = &[1.0000000000000002, 6.1697947043828245, -0.42772256543218573];
const ROS3_E: &[f64] = &[0.49999999999999983, -2.907955871680547, 0.22354069897811568];
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
const ROS3PR_A: &[f64] = &[
    0.0,
    0.0,
    0.0, // stage 1
    3.0000000000000018,
    0.0,
    0.0, // stage 2
    3.80384757729337,
    1.2679491924311226,
    0.0, // stage 3
];
const ROS3PR_C: &[f64] = &[
    0.0,
    0.0,
    0.0, // stage 1
    -3.80384757729337,
    0.0,
    0.0, // stage 2
    -5.673079295488928,
    -1.7384634363911504,
    0.0, // stage 3
];
const ROS3PR_NODES: &[f64] = &[0.0, 2.36602540378444, 1.0];
const ROS3PR_D: &[f64] = &[0.788675134594813, -1.577350269189627, -0.577350269189621];
const ROS3PR_B: &[f64] = &[4.5358983848622305, 1.2679491924311173, 1.0];
const ROS3PR_E: &[f64] = &[-0.4598572229918937, -0.22992861149594657, 0.0];
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
const ROS3PRL_A: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0, // stage 1
    1.147140180139521,
    0.0,
    0.0,
    0.0, // stage 2
    2.4630707730300534,
    1.147140180139521,
    0.0,
    0.0, // stage 3
    2.4630707730300534,
    1.147140180139521,
    0.0,
    0.0, // stage 4
];
const ROS3PRL_C: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0, // stage 1
    -2.631861185781065,
    0.0,
    0.0,
    0.0, // stage 2
    -2.038451402734394,
    1.8551577240019121,
    0.0,
    0.0, // stage 3
    -1.8050630466729911,
    3.411439279441918,
    -1.7057196397209593,
    0.0, // stage 4
];
const ROS3PRL_NODES: &[f64] = &[0.0, 0.5, 1.0, 1.0];
const ROS3PRL_D: &[f64] = &[
    0.435866521508459,
    -0.064133478491541,
    -0.0032561147686690495,
    0.0,
];
const ROS3PRL_B: &[f64] = &[2.4630707730300534, 1.1471401801395211, 0.0, 1.0];
const ROS3PRL_E: &[f64] = &[
    0.14188781262114447,
    0.9677841576948438,
    -0.06855321332716582,
    0.26131506383377634,
];
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

// ROS3PRodasTableau(T, T2) from
// lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl.
// The source computes these values from gamma = 1/2 + sqrt(3)/6. They are
// written as literals here so the solve path remains allocation-free and
// deterministic while retaining the upstream Float64 tableau.
const ROS3P_A: &[f64] = &[
    0.0,
    0.0,
    0.0, // stage 1
    1.2679491924311228,
    0.0,
    0.0, // stage 2
    1.2679491924311228,
    0.0,
    0.0, // stage 3
];
const ROS3P_C: &[f64] = &[
    0.0,
    0.0,
    0.0, // stage 1
    -1.6076951545867364,
    0.0,
    0.0, // stage 2
    -3.4641016151377553,
    -1.7320508075688774,
    0.0, // stage 3
];
const ROS3P_NODES: &[f64] = &[0.0, 1.0, 1.0];
const ROS3P_D: &[f64] = &[0.7886751345948129, -0.2113248654051871, -1.077350269189626];
const ROS3P_B: &[f64] = &[2.0, 0.5773502691896257, 0.42264973081037427];
const ROS3P_E: &[f64] = &[
    -0.1132486540518709,
    -0.42264973081037427,
    5.551115123125783e-17,
];
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
const ROS34PRW_A: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0, // stage 1
    2.0,
    0.0,
    0.0,
    0.0, // stage 2
    1.9166355646921893,
    -0.7305046154473316,
    0.0,
    0.0, // stage 3
    3.7075384385487764,
    1.984721005641544,
    -0.7228174329072325,
    0.0, // stage 4
];
const ROS34PRW_C: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0, // stage 1
    -4.588560720558084,
    0.0,
    0.0,
    0.0, // stage 2
    -1.4496008611374558,
    2.6585485498967283,
    0.0,
    0.0, // stage 3
    -0.8142320398640468,
    2.1949369533270104,
    -0.9042300763629808,
    0.0, // stage 4
];
const ROS34PRW_NODES: &[f64] = &[
    0.0,
    0.871733043016918,
    1.1537997822626886,
    0.9999999999999999,
];
const ROS34PRW_D: &[f64] = &[
    0.435866521508459,
    -0.435866521508459,
    -0.34459816128502135,
    5.551115123125783e-17,
];
const ROS34PRW_B: &[f64] = &[
    3.7075384385487764,
    1.9847210056415439,
    -0.7228174329072324,
    1.0,
];
const ROS34PRW_E: &[f64] = &[
    -0.08016142700721947,
    0.15059517863671545,
    -0.29187352202361583,
    0.26131506383377556,
];
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
const ROS34PW3_A: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0,
    2.3541034887609085,
    0.0,
    0.0,
    0.0,
    2.1274518517432335,
    0.7018666706430658,
    0.0,
    0.0,
    1.6573541366907125,
    0.37998365119129385,
    0.7677537933767512,
    0.0,
];
const ROS34PW3_C: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0,
    -2.20302237067446,
    0.0,
    0.0,
    0.0,
    -2.750060114993467,
    -0.8408569631081119,
    0.0,
    0.0,
    -3.0077871973155896,
    -0.7070774625618249,
    -1.1874362749274354,
    0.0,
];
const ROS34PW3_NODES: &[f64] = &[
    0.0,
    2.5155456020628817,
    1.2577728010314408,
    0.6288864005157204,
];
const ROS34PW3_D: &[f64] = &[
    1.0685790213016289,
    -1.4469665807612528,
    -0.7714762485313431,
    -0.29371172261080924,
];
const ROS34PW3_B: &[f64] = &[
    2.5468758076906703,
    0.5527809704437551,
    0.920521963049926,
    0.7201674983256354,
];
const ROS34PW3_E: &[f64] = &[
    0.20632710213677674,
    0.0026042321416049896,
    0.006723394920071124,
    0.7201674983256354,
];
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
const GRK4A_A: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0, // stage 1
    1.108860759493671,
    0.0,
    0.0,
    0.0, // stage 2
    2.37708526198336,
    0.1850114988899692,
    0.0,
    0.0, // stage 3
    2.37708526198336,
    0.1850114988899692,
    0.0,
    0.0, // stage 4
];
const GRK4A_C: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0, // stage 1
    -4.920188402397641,
    0.0,
    0.0,
    0.0, // stage 2
    1.055588686048583,
    3.351817267668938,
    0.0,
    0.0, // stage 3
    3.846869007049313,
    3.42710924126818,
    -2.162408848753263,
    0.0, // stage 4
];
const GRK4A_NODES: &[f64] = &[0.0, 0.438, 0.87, 0.87];
const GRK4A_D: &[f64] = &[
    0.395,
    -0.372672395484092,
    0.06629196544571492,
    0.4340946962568634,
];
const GRK4A_B: &[f64] = &[
    1.84568324040584,
    0.1369796894360503,
    0.7129097783291559,
    0.6329113924050632,
];
const GRK4A_E: &[f64] = &[
    0.04831870177201765,
    -0.6471108651049505,
    0.218687666050024,
    -0.6329113924050632,
];
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
const GRK4T_A: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0, // stage 1
    2.0,
    0.0,
    0.0,
    0.0, // stage 2
    4.524708207373116,
    4.163528788597648,
    0.0,
    0.0, // stage 3
    4.524708207373116,
    4.163528788597648,
    0.0,
    0.0, // stage 4
];
const GRK4T_C: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0, // stage 1
    -5.071675338776316,
    0.0,
    0.0,
    0.0, // stage 2
    6.020152728650786,
    0.1597506846727117,
    0.0,
    0.0, // stage 3
    -1.856343618686113,
    -8.505380858179826,
    -2.084075136023187,
    0.0, // stage 4
];
const GRK4T_NODES: &[f64] = &[0.0, 0.462, 0.8802083333333334, 0.8802083333333334];
const GRK4T_D: &[f64] = &[
    0.231,
    -0.03962966775244303,
    0.5507789395789127,
    -0.05535098457052764,
];
const GRK4T_B: &[f64] = &[
    3.957503746640777,
    4.624892388363313,
    0.6174772638750108,
    1.282612945269037,
];
const GRK4T_E: &[f64] = &[
    2.302155402932996,
    3.073634485392623,
    -0.8732808018045032,
    -1.282612945269037,
];
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

// ROS34PW1bRodasTableau(T, T2) from
// lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl at
// 211142263781255a9aa2f910f6760b9f18ec29c8.
const ROS34PW1B_A: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0, // stage 1
    5.0905205106702045,
    0.0,
    0.0,
    0.0, // stage 2
    5.0905205106702045,
    0.0,
    0.0,
    0.0, // stage 3
    4.976281110107875,
    0.027726816471584953,
    0.22942803602790418,
    0.0, // stage 4
];
const ROS34PW1B_C: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0, // stage 1
    -11.679081231228288,
    0.0,
    0.0,
    0.0, // stage 2
    -16.40573264673668,
    -0.27726816471584953,
    0.0,
    0.0, // stage 3
    -8.38103960500476,
    -0.8483284091993433,
    0.28700986043310556,
    0.0, // stage 4
];
const ROS34PW1B_NODES: &[f64] = &[0.0, 2.218787467653286, 2.218787467653286, 1.553923375357884];
const ROS34PW1B_D: &[f64] = &[
    0.435866521508459,
    -1.7829209461448272,
    -2.4654190049693425,
    -0.8055299979063697,
];
const ROS34PW1B_B: &[f64] = &[
    5.2258276123309395,
    -0.5569711481541647,
    0.35797946935364533,
    1.7233739852106407,
];
const ROS34PW1B_E: &[f64] = &[-5.168452127840395, -1.2635194260384186, 0.0, 0.0];
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
const ROS34PW2_A: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0, // stage 1
    2.0,
    0.0,
    0.0,
    0.0, // stage 2
    1.4192173174557647,
    -0.2592322116729697,
    0.0,
    0.0, // stage 3
    4.18476048231916,
    -0.28519201735549593,
    2.294280360279042,
    0.0, // stage 4
];
const ROS34PW2_C: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0, // stage 1
    -4.588560720558084,
    0.0,
    0.0,
    0.0, // stage 2
    -4.18476048231916,
    0.28519201735549593,
    0.0,
    0.0, // stage 3
    -6.368179200128359,
    -6.795620944466837,
    2.8700986043310563,
    0.0, // stage 4
];
const ROS34PW2_NODES: &[f64] = &[0.0, 0.871733043016918, 0.7315799577888524, 1.0];
const ROS34PW2_D: &[f64] = &[
    0.435866521508459,
    -0.435866521508459,
    -0.4133333762338865,
    -5.551115123125783e-17,
];
const ROS34PW2_B: &[f64] = &[
    4.1847604823191595,
    -0.28519201735549565,
    2.2942803602790414,
    1.0,
];
const ROS34PW2_E: &[f64] = &[
    0.2777499476479681,
    -1.4032398951759992,
    1.7726301276675507,
    0.5,
];
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

const RODAS4_A: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 1
    1.544,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 2
    0.9466785280815826,
    0.2557011698983284,
    0.0,
    0.0,
    0.0,
    0.0, // stage 3
    3.314825187068521,
    2.896124015972201,
    0.9986419139977817,
    0.0,
    0.0,
    0.0, // stage 4
    1.221224509226641,
    6.019134481288629,
    12.53708332932087,
    -0.687886036105895,
    0.0,
    0.0, // stage 5
    1.221224509226641,
    6.019134481288629,
    12.53708332932087,
    -0.687886036105895,
    1.0,
    0.0, // stage 6
];
const RODAS4_C: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 1
    -5.6688,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 2
    -2.430093356833875,
    -0.2063599157091915,
    0.0,
    0.0,
    0.0,
    0.0, // stage 3
    -0.1073529058151375,
    -9.594562251023355,
    -20.47028614809616,
    0.0,
    0.0,
    0.0, // stage 4
    7.496443313967647,
    -10.24680431464352,
    -33.99990352819905,
    11.7089089320616,
    0.0,
    0.0, // stage 5
    8.083246795921522,
    -7.981132988064893,
    -31.52159432874371,
    16.31930543123136,
    -6.058818238834054,
    0.0, // stage 6
];
const RODAS4_NODES: &[f64] = &[0.0, 0.386, 0.21, 0.63, 1.0, 1.0];
const RODAS4_D: &[f64] = &[0.25, -0.1043, 0.1035, -0.0362, 0.0, 0.0];
const RODAS4_B: &[f64] = &[
    1.221224509226641,
    6.019134481288629,
    12.53708332932087,
    -0.687886036105895,
    1.0,
    1.0,
];
const RODAS4_E: &[f64] = &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
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
const RODAS42_A: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 1
    1.4028884,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 2
    0.6581212688557198,
    -1.320936088384301,
    0.0,
    0.0,
    0.0,
    0.0, // stage 3
    7.131197445744498,
    16.02964143958207,
    -5.561572550509766,
    0.0,
    0.0,
    0.0,
    // stage 4
    22.73885722420363,
    67.38147284535289,
    -31.2187749303856,
    0.7285641833203814,
    0.0,
    0.0, // stage 5
    22.73885722420363,
    67.38147284535289,
    -31.2187749303856,
    0.7285641833203814,
    1.0,
    0.0, // stage 6
];
const RODAS42_C: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 1
    -5.1043536,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 2
    -2.899967805418783,
    4.040399359702244,
    0.0,
    0.0,
    0.0,
    0.0, // stage 3
    -32.64449927841361,
    -99.35311008728094,
    49.99119122405989,
    0.0,
    0.0,
    0.0, // stage 4
    -76.46023087151691,
    -278.5942120829058,
    153.9294840910643,
    10.97101866258358,
    0.0,
    0.0, // stage 5
    -76.29701586804983,
    -294.2795630511232,
    162.0029695867566,
    23.6516690309527,
    -7.652977706771382,
    0.0, // stage 6
];
const RODAS42_NODES: &[f64] = &[0.0, 0.3507221, 0.2557041, 0.681779, 1.0, 1.0];
const RODAS42_D: &[f64] = &[0.25, -0.0690221, -0.0009672, -0.087979, 0.0, 0.0];
const RODAS42_B: &[f64] = &[
    22.73885722420363,
    67.38147284535289,
    -31.2187749303856,
    0.7285641833203814,
    1.0,
    1.0,
];
const RODAS42_E: &[f64] = &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
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
const RODAS4P_A: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 1
    3.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 2
    1.831036793486759,
    0.4955183967433795,
    0.0,
    0.0,
    0.0,
    0.0, // stage 3
    2.304376582692669,
    -0.05249275245743001,
    -1.176798761832782,
    0.0,
    0.0,
    0.0, // stage 4
    -7.170454962423024,
    -4.741636671481785,
    -16.31002631330971,
    -1.062004044111401,
    0.0,
    0.0, // stage 5
    -7.170454962423024,
    -4.741636671481785,
    -16.31002631330971,
    -1.062004044111401,
    1.0,
    0.0, // stage 6
];
const RODAS4P_C: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 1
    -12.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 2
    -8.791795173947035,
    -2.207865586973518,
    0.0,
    0.0,
    0.0,
    0.0, // stage 3
    10.81793056857153,
    6.780270611428266,
    19.5348594464241,
    0.0,
    0.0,
    0.0, // stage 4
    34.19095006749676,
    15.49671153725963,
    54.7476087596413,
    14.16005392148534,
    0.0,
    0.0, // stage 5
    34.62605830930532,
    15.30084976114473,
    56.99955578662667,
    18.40807009793095,
    -5.714285714285717,
    0.0, // stage 6
];
const RODAS4P_NODES: &[f64] = &[0.0, 0.75, 0.21, 0.63, 1.0, 1.0];
const RODAS4P_D: &[f64] = &[0.25, -0.5, -0.023504, -0.0362, 0.0, 0.0];
const RODAS4P_B: &[f64] = &[
    -7.170454962423024,
    -4.741636671481785,
    -16.31002631330971,
    -1.062004044111401,
    1.0,
    1.0,
];
const RODAS4P_E: &[f64] = &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
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

// Rodas4PWTableau(T, T2) from
// lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl at
// OrdinaryDiffEq revision 211142263781255a9aa2f910f6760b9f18ec29c8.
#[allow(clippy::excessive_precision)]
const RODAS4PW_A: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    2.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    2.9351406859394085,
    -0.2900839547917462,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    2.5023808526102953,
    0.16405393216724992,
    0.5533771741907447,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    3.750686547696225,
    -3.4281864369523043,
    3.040310330969146,
    3.9186650053435343,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    6.144641638053066,
    -3.5875118429024027,
    4.896418320248426,
    3.22902579478038,
    1.8769572180937995,
    0.0,
    0.0,
    0.0,
    0.0,
    7.024950636618132,
    -0.16004404386219917,
    3.011738084182477,
    2.6255276779916845,
    1.064210639984151,
    1.458733196183423,
    0.0,
    0.0,
    0.0,
    2.435229134779613,
    -3.645692955953369,
    1.8428193142644484,
    -1.3604738908042335,
    1.9007297861385437,
    1.2049388559324536,
    0.5873423923113256,
    0.0,
    0.0,
    2.435229134779612,
    -3.6456929559533693,
    1.8428193142644478,
    -1.3604738908042329,
    1.9007297861385437,
    1.2049388559324536,
    0.5873423923113257,
    1.0,
    0.0,
];
#[allow(clippy::excessive_precision)]
const RODAS4PW_C: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -8.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.11431136138808462,
    5.111960520884256,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -8.68739530736641,
    -0.21553233254388848,
    -2.2134894813624935,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -10.812900826181544,
    5.419387230163944,
    -6.845923178683723,
    -5.6022546699245925,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -13.17876026787501,
    8.4031432262977,
    -14.402912022742473,
    -1.8949265796948946,
    -7.507992684163551,
    0.0,
    0.0,
    0.0,
    0.0,
    2.4245226070350387,
    3.916211538629165,
    6.334875580590554,
    -18.204882737666242,
    11.2386753540205,
    4.394215434356857,
    0.0,
    0.0,
    0.0,
    12.966547752453305,
    1.6658702157873542,
    -1.952036642801561,
    44.8607385053866,
    -16.33815656155744,
    -7.339351477557218,
    6.415927010710538,
    0.0,
    0.0,
    -3.4230613926569706,
    10.549037870633642,
    -22.013331787653808,
    59.20345912605831,
    -38.48170615459573,
    -17.83635248514032,
    5.0140912391709005,
    -5.714232814492047,
    0.0,
];
const RODAS4PW_NODES: &[f64] = &[
    0.0,
    0.5,
    0.806306160182747,
    0.5500769630661511,
    0.645123695869199,
    0.7460176197877518,
    0.38581216095480647,
    1.0,
    1.0,
];
const RODAS4PW_D: &[f64] = &[
    0.25,
    -0.25,
    -0.062353072468327386,
    -0.24498696841659628,
    -0.3146820705324951,
    -0.16763676372651826,
    0.10469903956242915,
    0.0,
    0.0,
];
const RODAS4PW_B: &[f64] = &[
    2.435229134779612,
    -3.6456929559533693,
    1.8428193142644478,
    -1.3604738908042329,
    1.9007297861385437,
    1.2049388559324536,
    0.5873423923113257,
    1.0,
    1.0,
];
const RODAS4PW_E: &[f64] = &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
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
const RODAS5_A: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 1
    2.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 2
    3.040894194418781,
    1.041747909077569,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 3
    2.576417536461461,
    1.62208306077664,
    -0.9089668560264532,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 4
    2.760842080225597,
    1.446624659844071,
    -0.3036980084553738,
    0.2877498600325443,
    0.0,
    0.0,
    0.0,
    0.0, // stage 5
    -14.09640773051259,
    6.925207756232704,
    -41.47510893210728,
    2.343771018586405,
    24.13215229196062,
    0.0,
    0.0,
    0.0, // stage 6
    -14.09640773051259,
    6.925207756232704,
    -41.47510893210728,
    2.343771018586405,
    24.13215229196062,
    1.0,
    0.0,
    0.0, // stage 7
    -14.09640773051259,
    6.925207756232704,
    -41.47510893210728,
    2.343771018586405,
    24.13215229196062,
    1.0,
    1.0,
    0.0, // stage 8
];
const RODAS5_C: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 1
    -10.31323885133993,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 2
    -21.04823117650003,
    -7.234992135176716,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 3
    32.22751541853323,
    -4.943732386540191,
    19.44922031041879,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 4
    -20.69865579590063,
    -8.816374604402768,
    1.260436877740897,
    -0.7495647613787146,
    0.0,
    0.0,
    0.0,
    0.0, // stage 5
    -46.22004352711257,
    -17.49534862857472,
    -289.6389582892057,
    93.60855400400906,
    318.3822534212147,
    0.0,
    0.0,
    0.0, // stage 6
    34.20013733472935,
    -14.1553540271769,
    57.823356409884,
    25.83362985412365,
    1.408950972071624,
    -6.551835421242162,
    0.0,
    0.0, // stage 7
    42.57076742291101,
    -13.80770672017997,
    93.98938432427124,
    18.77919633714503,
    -31.5835918722337,
    -6.685968952921985,
    -5.810979938412932,
    0.0, // stage 8
];
const RODAS5_NODES: &[f64] = &[
    0.0,
    0.38,
    0.3878509998321533,
    0.483971893787384,
    0.457047700881958,
    1.0,
    1.0,
    1.0,
];
const RODAS5_D: &[f64] = &[
    0.19,
    -0.18230792253337146,
    -0.3192318321868749,
    0.3449828624725343,
    -0.37741756439208984,
    0.0,
    0.0,
    0.0,
];
const RODAS5_B: &[f64] = &[
    -14.09640773051259,
    6.925207756232704,
    -41.47510893210728,
    2.343771018586405,
    24.13215229196062,
    1.0,
    1.0,
    1.0,
];
const RODAS5_E: &[f64] = &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
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

const RODAS5P_A: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 1
    3.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 2
    2.849394379747939,
    0.45842242204463923,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 3
    -6.954028509809101,
    2.489845061869568,
    -10.358996098473584,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 4
    2.8029986275628964,
    0.5072464736228206,
    -0.3988312541770524,
    -0.04721187230404641,
    0.0,
    0.0,
    0.0,
    0.0, // stage 5
    -7.502846399306121,
    2.561846144803919,
    -11.627539656261098,
    -0.18268767659942256,
    0.030198172008377946,
    0.0,
    0.0,
    0.0, // stage 6
    -7.502846399306121,
    2.561846144803919,
    -11.627539656261098,
    -0.18268767659942256,
    0.030198172008377946,
    1.0,
    0.0,
    0.0, // stage 7
    -7.502846399306121,
    2.561846144803919,
    -11.627539656261098,
    -0.18268767659942256,
    0.030198172008377946,
    1.0,
    1.0,
    0.0, // stage 8
];
const RODAS5P_C: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 1
    -14.155112264123755,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 2
    -17.97296035885952,
    -2.859693295451294,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 3
    147.12150275711716,
    -1.41221402718213,
    71.68940251302358,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 4
    165.43517024871676,
    -0.4592823456491126,
    42.90938336958603,
    -5.961986721573306,
    0.0,
    0.0,
    0.0,
    0.0, // stage 5
    24.854864614690072,
    -3.0009227002832186,
    47.4931110020768,
    5.5814197821558125,
    -0.6610691825249471,
    0.0,
    0.0,
    0.0, // stage 6
    30.91273214028599,
    -3.1208243349937974,
    77.79954646070892,
    34.28646028294783,
    -19.097331116725623,
    -28.087943162872662,
    0.0,
    0.0, // stage 7
    37.80277123390563,
    -3.2571969029072276,
    112.26918849496327,
    66.9347231244047,
    -40.06618937091002,
    -54.66780262877968,
    -9.48861652309627,
    0.0, // stage 8
];
const RODAS5P_NODES: &[f64] = &[
    0.0,
    0.6358126895828704,
    0.4095798393397535,
    0.9769306725060716,
    0.4288403609558664,
    1.0,
    1.0,
    1.0,
];
const RODAS5P_D: &[f64] = &[
    0.21193756319429014,
    -0.42387512638858027,
    -0.3384627126235924,
    1.8046452872882734,
    2.325825639765069,
    0.0,
    0.0,
    0.0,
];
const RODAS5P_B: &[f64] = &[
    -7.502846399306121,
    2.561846144803919,
    -11.627539656261098,
    -0.18268767659942256,
    0.030198172008377946,
    1.0,
    1.0,
    1.0,
];
const RODAS5P_E: &[f64] = &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
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
const RODAS5PE_E: &[f64] = &[
    0.2606326497975715,
    -0.005158627295444251,
    1.3038988631109731,
    1.235000722062074,
    -0.7931985603795049,
    -1.005448461135913,
    -0.18044626132120234,
    0.17051519239113755,
];
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
// RODAS5PH from lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_tableaus.jl at
// 211142263781255a9aa2f910f6760b9f18ec29c8. Rodas5Pr uses this same H matrix
// for its residual-control midpoint estimate (the regular ODE path does not
// otherwise need stiff-aware dense interpolation).
const RODAS5P_H: &[f64] = &[
    25.948786856663858,
    -2.5579724845846235,
    10.433815404888879,
    -2.3679251022685204,
    0.524948541321073,
    1.1241088310450404,
    0.4272876194431874,
    -0.17202221070155493,
    -9.91568850695171,
    -0.9689944594115154,
    3.0438037242978453,
    -24.495224566215796,
    20.176138334709044,
    15.98066361424651,
    -6.789040303419874,
    -6.710236069923372,
    11.419903575922262,
    2.8879645146136994,
    72.92137995996029,
    80.12511834622643,
    -52.072871366152654,
    -59.78993625266729,
    -0.15582684282751913,
    4.883087185713722,
];

// Rodas6PTableau(T, T2) from
// lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_tableaus.jl at
// 211142263781255a9aa2f910f6760b9f18ec29c8.
//
// This is the regular ODE 19-stage sixth-order L-stable tableau. The
// upstream dense-output H matrix is not needed by the shared recorder.
const RODAS6P_A: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 1
    1.7111784962693573,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 2
    3.338661438538325,
    1.7785154948506772,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 3
    2.936071270275081,
    0.9182685464146361,
    0.3700626437020361,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 4
    4.659498341685848,
    1.750740798902701,
    0.5870646872926452,
    0.8880273208834594,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 5
    4.0197306615530755,
    2.839611966871549,
    -0.5985886977898102,
    0.08804800108767567,
    1.5622259206803966,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 6
    1.988416726724047,
    -0.379547946940864,
    0.9004347186464728,
    1.4277449221484224,
    -0.7433508015345144,
    -0.042432590368607255,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 7
    1.8376133238441654,
    1.9114959548124457,
    -0.6715227349230231,
    0.2358079620635186,
    3.6095202089874117,
    0.8151701113738031,
    0.9206065341545108,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 8
    -0.766306772356088,
    3.209956697664864,
    -3.3123779344961592,
    -3.0203200762095332,
    4.800864725315542,
    1.1604579105760842,
    0.4424812765132964,
    0.3706918590956091,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 9
    6.232416226700401,
    2.6089061288608786,
    -0.6004565639275875,
    -3.3845987889094653,
    0.42397260663019737,
    0.35421155529651493,
    0.30716464971632756,
    1.5008969261275715,
    0.5102657561692372,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 10
    0.023392109748070492,
    1.4081998520657641,
    -0.7199787823918794,
    0.7361286083371824,
    2.4632772861278043,
    0.46923886035475726,
    0.1205787235019629,
    -0.8578747086506138,
    -0.2588726092696778,
    -0.4397748045492015,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 11
    2.953951852943472,
    -0.5094757863221286,
    0.3109577019600045,
    -3.5298051247141733,
    -3.545755924579993,
    -0.33681829638738314,
    -0.5663219967973026,
    1.1332773651373889,
    0.15030559921640937,
    0.25755454716019555,
    0.29836356640198125,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 12
    3.613614004197333,
    0.6635854700997046,
    0.021719370087612728,
    -1.4950066071478674,
    0.7257768429136315,
    -0.05542296424699332,
    0.6617050893162496,
    1.5916006835996634,
    0.004468857383033254,
    0.3492741589610665,
    -0.20270398239783438,
    0.6407744206145284,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 13
    0.6650675322630164,
    3.8649437891996143,
    -3.5568168140908862,
    0.30445082364848014,
    6.687033712252074,
    1.7577448564663951,
    0.7252352806302017,
    0.8340620415656512,
    0.288756122559755,
    -0.014344518613253377,
    -0.9202387269679146,
    0.1235675186947092,
    0.5210532009614854,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 14
    0.6650675322630163,
    3.864943789199614,
    -3.5568168140908876,
    0.30445082364847914,
    6.687033712252074,
    1.7577448564663951,
    0.7252352806302018,
    0.834062041565651,
    0.2887561225597551,
    -0.014344518613253487,
    -0.9202387269679145,
    0.12356751869470915,
    0.5210532009614851,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 15
    0.6650675322630177,
    3.864943789199614,
    -3.5568168140908876,
    0.30445082364847964,
    6.687033712252074,
    1.7577448564663947,
    0.7252352806302018,
    0.8340620415656512,
    0.2887561225597553,
    -0.014344518613253388,
    -0.9202387269679146,
    0.12356751869470915,
    0.5210532009614847,
    0.9999999999999998,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 16
    19.50552823691581,
    19.347153108620457,
    8.914549824296701,
    19.841538938728135,
    -30.357562810836335,
    -5.8051482228142195,
    -20.70248562796752,
    -21.376641384544037,
    -8.3987892732033,
    -7.736419267379574,
    2.0685219772918724,
    -8.760228330317224,
    5.418493374208782,
    1.6147496952752922,
    -2.4936078709750045,
    -2.2780749437287913,
    0.0,
    0.0,
    0.0, // stage 17
    56.666079713903386,
    5.165818087889462,
    22.35860178936739,
    10.763846891315566,
    -12.19955923582867,
    -0.6557677952689249,
    6.699011373303813,
    8.198746354754231,
    1.677616608414471,
    3.1277285768268968,
    1.4179582645760416,
    -2.2744851263878836,
    -1.7540829181840385,
    -2.0011264405011304,
    -2.473136836317977,
    -1.8837562152980634,
    0.10307912091592847,
    0.0,
    0.0, // stage 18
    0.4516803503254423,
    -1.1534041395134922,
    -4.845280220499032,
    -23.586737710513358,
    1.6878417137075876,
    0.7868554635343545,
    3.0270371561482805,
    6.0990245036017745,
    2.600755391135945,
    1.3246029905944345,
    2.0716827798306645,
    1.0885240892776982,
    -2.406764303616696,
    -1.0934035335348031,
    -0.08817028752603433,
    0.1628623060793163,
    -0.0494871328758008,
    -0.053891050773493175,
    0.0, // stage 19
];
const RODAS6P_C: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 1
    -6.581455754882143,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 2
    -17.99898897860265,
    -8.573983492685619,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 3
    -9.381383431453385,
    -3.147640353879416,
    -1.3459246069197102,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 4
    -2.6265331637613007,
    -4.114341661049238,
    2.3552716210903446,
    0.7916860595752533,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 5
    13.234071865054425,
    -6.5531726714288245,
    10.73126008968739,
    7.881893740344428,
    -12.771533510641573,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 6
    2.830906994202388,
    0.2604988641272497,
    1.2537810312593667,
    -3.3671244579321455,
    -10.786563365589606,
    -1.9308385166591397,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 7
    -11.060311196714387,
    -1.3456656966931244,
    -0.7657970115506183,
    6.107723730659436,
    2.2037867523584938,
    -0.07238767937020778,
    -0.8050462039096485,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 8
    -18.12844382677043,
    -8.753725825758918,
    2.21059342699439,
    11.608007179779365,
    -0.05812583279366939,
    -0.5568300956262869,
    0.22469855334210373,
    -3.2370311176417705,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 9
    -27.976343920035056,
    -8.591555353546772,
    0.7281452536154736,
    17.638493457476986,
    -5.306189757628467,
    -3.0476569146401444,
    -5.904770682441327,
    -11.929084037829442,
    -5.050568446376497,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 10
    16.354749252471326,
    1.4669223994142209,
    5.928484441955681,
    10.74513480723443,
    10.673355125609953,
    3.688805562318594,
    9.180717730517506,
    10.247712646451996,
    1.465303310058304,
    2.6508985881732774,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 11
    6.914935441832399,
    5.38984958631352,
    5.862037438875566,
    5.348681436005972,
    -7.013382408252529,
    -1.0246660674824237,
    -2.9837100715597376,
    -5.836566084094612,
    -1.6109549842142277,
    -1.1760399923017764,
    1.9280128334739643,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 12
    4.198711896534994,
    0.6181084056782703,
    0.8167077246607498,
    -11.236495255410707,
    -4.824409089261172,
    0.7728367826492113,
    1.3033341851336357,
    3.1220057171705977,
    1.917167519177393,
    0.740911936448596,
    2.3884839541537675,
    -1.0646261824518854,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 13
    8.439457273665866,
    -12.395217711948733,
    8.717809686910918,
    -22.620864451522902,
    -20.113605406532766,
    -4.689776805197894,
    1.6982447341017708,
    4.543406791431926,
    1.613366829028557,
    1.9707273458768597,
    5.732578765805261,
    1.7624664316708778,
    -5.245856774591262,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 14
    1.71450490512272,
    -13.840042084553074,
    6.437347401872298,
    -39.22048912909508,
    -25.603335547270504,
    -8.064795628637777,
    -0.0416146802576695,
    1.2346482358729156,
    2.7009872115209252,
    0.896525973043981,
    11.303609670096813,
    1.7547024586563815,
    -10.02276713006834,
    -6.93945857648056,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 15
    19.61213176916848,
    -14.311603723508286,
    12.96694128305561,
    -28.107332490598257,
    -27.192311861144052,
    -4.26577843330566,
    4.2210733410202135,
    10.487402162301366,
    0.8300940888935481,
    2.8025411207314455,
    8.878715452726594,
    1.4612348690788786,
    -9.853595041510669,
    -7.6808377648250294,
    -6.870056791600298,
    0.0,
    0.0,
    0.0,
    0.0, // stage 16
    -371.4374434114548,
    -183.45541467767708,
    -245.2624549214238,
    -955.0201586246473,
    19.423673137681643,
    -0.9607917923114558,
    84.24290713671253,
    150.79467360315542,
    63.95986343894305,
    44.75849836276605,
    140.71019894715903,
    48.45923399488863,
    -146.47504553991118,
    -81.54609927566207,
    -25.07839993898753,
    3.829875020371404,
    0.0,
    0.0,
    0.0, // stage 17
    -164.17333724090065,
    -123.20834760012879,
    41.95443520088893,
    -239.77049586199965,
    -114.21790985713952,
    -12.341946126859007,
    15.750890701780254,
    39.790586318490696,
    -15.10728752295998,
    -12.824535901748304,
    16.73312629333741,
    -11.045750202895642,
    -14.639022723428816,
    -4.819493713790297,
    0.12987727446051117,
    0.04274137296376776,
    1.204312390836273,
    0.0,
    0.0, // stage 18
    461.9368267656361,
    52.53840694165619,
    144.8183760448168,
    455.9093893534564,
    35.81740627464267,
    40.776479432911195,
    17.76376684531324,
    -7.606209560029927,
    -15.381956921574087,
    3.9317973949968428,
    -41.37771137941743,
    -44.920577584346,
    53.37495929469229,
    10.211466474320808,
    -15.84128439059478,
    -19.507483543094224,
    1.884309895179932,
    5.745356704710484,
    0.0, // stage 19
];
const RODAS6P_NODES: &[f64] = &[
    0.0,
    0.4449064090300329,
    0.5391930604628539,
    0.3920739557917205,
    0.5393851240464334,
    0.7496615946466092,
    0.09171052879621677,
    0.716762001806476,
    0.9201684737037024,
    0.7017495611178288,
    0.5587152179138446,
    0.10896187906446,
    0.5073827520419607,
    0.9999999999999999,
    0.9999999999999999,
    1.0000000000000002,
    0.19999999999999996,
    0.4999999999999998,
    0.8,
];
const RODAS6P_D: &[f64] = &[
    0.26,
    -0.18490640903003291,
    -0.5445316852875675,
    -0.03230297796648507,
    -0.05985832397786847,
    0.08292573124960323,
    0.4158601113780379,
    -0.4887636036121086,
    -0.5305551731438798,
    0.12166683722729399,
    -0.14899579330238244,
    0.20995126195089908,
    -0.06287825975966793,
    -1.1102230246251565e-16,
    1.1102230246251565e-16,
    2.220446049250313e-16,
    8.520155173756681,
    -7.34858003171262,
    1.5593201340906078,
];
const RODAS6P_B: &[f64] = &[
    0.6650675322630177,
    3.864943789199614,
    -3.5568168140908876,
    0.30445082364847964,
    6.687033712252074,
    1.7577448564663947,
    0.7252352806302018,
    0.8340620415656512,
    0.2887561225597553,
    -0.014344518613253388,
    -0.9202387269679146,
    0.12356751869470915,
    0.5210532009614847,
    0.9999999999999998,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
];
const RODAS6P_E: &[f64] = &[
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
];
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

const ROSENBROCK_W6S4OS_A: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.5812383407115008,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.903962441371467,
    1.861519155534501,
    0.0,
    0.0,
    0.0,
    0.0,
    2.076579719675,
    0.1884255381414796,
    1.870158967491032,
    0.0,
    0.0,
    0.0,
    4.435550638484312,
    5.457181798610189,
    4.61635078806893,
    3.118111952402361,
    0.0,
    0.0,
    10.79170169848326,
    -10.05691522584131,
    14.99564485428419,
    5.274339954390943,
    1.42973087126119,
    0.0,
];
const ROSENBROCK_W6S4OS_C: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -2.661294105131369,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -3.128450202373838,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -6.920335474535658,
    -1.202675288266817,
    -9.73356181141362,
    0.0,
    0.0,
    0.0,
    -28.09530629102695,
    20.37126295479377,
    -41.04375275302869,
    -19.66373175620895,
    0.0,
    0.0,
    9.7998186780974,
    11.93579288660318,
    3.673874929013201,
    14.8078285410955,
    0.831858399869068,
    0.0,
];
const ROSENBROCK_W6S4OS_NODES: &[f64] = &[
    0.0,
    0.1453095851778752,
    0.3817422770256738,
    0.6367813704374599,
    0.7560744496323561,
    0.927104723987567,
];
const ROSENBROCK_W6S4OS_D: &[f64] = &[
    0.25,
    0.0836691184292894,
    0.0544718623516351,
    -0.3402289722355864,
    0.0337651588339529,
    -0.090307426761854,
];
const ROSENBROCK_W6S4OS_B: &[f64] = &[
    6.456217074653235,
    -4.853141317768053,
    9.76531833406926,
    2.081084177278723,
    0.6603936866352417,
    0.6,
];
const ROSENBROCK_W6S4OS_E: &[f64] = &[0.0; 6];
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
const RODAS23W_A: &[f64] = &[
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 1
    4.0 / 3.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 2
    0.0,
    0.0,
    0.0,
    0.0,
    0.0, // stage 3
    2.90625,
    3.375,
    0.40625,
    0.0,
    0.0, // stage 4
    2.90625,
    3.375,
    0.40625,
    0.0,
    0.0, // stage 5
];
const RODAS23W_C: &[f64] = &[
    0.0, 0.0, 0.0, 0.0, 0.0, // stage 1
    -4.0, 0.0, 0.0, 0.0, 0.0, // stage 2
    8.25, 6.75, 0.0, 0.0, 0.0, // stage 3
    1.21875, -5.0625, -1.96875, 0.0, 0.0, // stage 4
    4.03125, -15.1875, -4.03125, 6.0, 0.0, // stage 5
];
const RODAS23W_NODES: &[f64] = &[0.0, 4.0 / 9.0, 0.0, 1.0, 1.0];
const RODAS23W_D: &[f64] = &[1.0 / 3.0, -1.0 / 9.0, 1.0, 0.0, 0.0];
const RODAS23W_B: &[f64] = &[2.90625, 3.375, 0.40625, 1.0, 0.0];
const RODAS23W_E: &[f64] = &[0.0, 0.0, 0.0, 1.0, -1.0];
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

#[allow(clippy::too_many_arguments)]
trait ExtendedRosenbrockMethod {
    const ERROR_ORDER: usize;
    const ADAPTIVE: bool;

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
algorithm!(Ros3p);
algorithm!(Ros34Prw);
algorithm!(Ros34Pw3);
algorithm!(Grk4a);
algorithm!(Grk4t);
algorithm!(Ros34Pw1b);
algorithm!(Ros34Pw2);
algorithm!(Rodas4);
algorithm!(Rodas42);
algorithm!(Rodas4P);
algorithm!(Rodas4PW);
algorithm!(Rodas5);
algorithm!(Rodas5P);
algorithm!(Rodas5Pe);
algorithm!(Rodas5Pr);
algorithm!(Rodas6P);
algorithm!(RosenbrockW6S4OS);
algorithm!(Rodas23W);

impl ExtendedRosenbrockMethod for Rosenbrock32 {
    const ERROR_ORDER: usize = 3;
    const ADAPTIVE: bool = true;

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
    ($name:ident, $order:literal, $tableau:ident) => {
        impl ExtendedRosenbrockMethod for $name {
            const ERROR_ORDER: usize = $order;
            const ADAPTIVE: bool = true;

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
                    problem, candidate, state, time, step, options, &$tableau, false, workspace,
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
rodas_method!(Ros3p, 3, ROS3P_TABLEAU);
rodas_method!(Ros34Prw, 3, ROS34PRW_TABLEAU);
rodas_method!(Ros34Pw3, 4, ROS34PW3_TABLEAU);
rodas_method!(Grk4a, 4, GRK4A_TABLEAU);
rodas_method!(Grk4t, 4, GRK4T_TABLEAU);
rodas_method!(Ros34Pw1b, 3, ROS34PW1B_TABLEAU);
rodas_method!(Ros34Pw2, 3, ROS34PW2_TABLEAU);
rodas_method!(Rodas4, 4, RODAS4_TABLEAU);
rodas_method!(Rodas42, 4, RODAS42_TABLEAU);
rodas_method!(Rodas4P, 4, RODAS4P_TABLEAU);
rodas_method!(Rodas4PW, 4, RODAS4PW_TABLEAU);
rodas_method!(Rodas5, 5, RODAS5_TABLEAU);
rodas_method!(Rodas5P, 5, RODAS5P_TABLEAU);
rodas_method!(Rodas5Pe, 5, RODAS5PE_TABLEAU);
rodas_method!(Rodas6P, 6, RODAS6P_TABLEAU);
rodas_method!(Rodas23W, 3, RODAS23W_TABLEAU);

impl ExtendedRosenbrockMethod for Rodas5Pr {
    const ERROR_ORDER: usize = 5;
    const ADAPTIVE: bool = true;

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
            jacobian: vec![0.0; dimension * dimension],
            factorization: vec![0.0; dimension * dimension],
            pivots: vec![0; dimension],
            differentiation_valid: false,
        }
    }
}

struct ExtendedRosenbrockKernel<M> {
    workspace: Workspace,
    method: PhantomData<M>,
}

impl<M> ExtendedRosenbrockKernel<M> {
    fn new(dimension: usize) -> Self {
        Self {
            workspace: Workspace::new(dimension),
            method: PhantomData,
        }
    }
}

impl<F, P, M> StepKernel<F, P> for ExtendedRosenbrockKernel<M>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
    M: ExtendedRosenbrockMethod,
{
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

    fn accept_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        _: &[f64],
        state: &[f64],
        time: f64,
        _: f64,
        _: bool,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        self.workspace.differentiation_valid = false;
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
        scaled_error_norm(state, candidate, &workspace.error, options)
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
        Grk4a, Grk4t, Rodas3, Rodas3d, Rodas4, Rodas4P, Rodas5P, Rodas5Pr, Rodas6P, Rodas23W, Ros2,
        Ros3, Ros3Pr, Ros3Prl, Ros3p, Ros34Prw, Ros34Pw1b, Rosenbrock32, RosenbrockW6S4OS,
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
    fn methods_have_their_expected_fixed_step_orders() {
        let ratios = [
            convergence_ratio(Ros2, 0.1),
            convergence_ratio(Rosenbrock32, 0.1),
            convergence_ratio(Rodas3, 0.1),
            convergence_ratio(Rodas3d, 0.1),
            convergence_ratio(Ros3Pr, 0.1),
            convergence_ratio(Ros3Prl, 0.1),
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
        assert!(ratios[8] > 14.0);
        assert!(ratios[9] > 14.0);
        assert!(ratios[10] > 14.0);
        assert!(ratios[11] > 7.0);
        assert!(ratios[12] > 7.0);
        assert!(ratios[13] > 14.0);
        assert!(ratios[14] > 14.0);
        assert!(ratios[15] > 7.0);
        // Pinned Rodas23W uses a second-order primary solution.
        assert!(ratios[16] > 3.0 && ratios[16] < 5.5);
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
