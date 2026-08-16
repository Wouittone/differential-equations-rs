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

/// The eight-stage, fifth-order L-stable Rodas5P method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Rodas5P;

/// The six-stage, fourth-order Rosenbrock-W method (fixed step only).
///
/// Coefficients are from `RosenbrockW6S4OSRodasTableau` in the pinned
/// `OrdinaryDiffEqRosenbrockTableaus` revision. The upstream algorithm is
/// intentionally fixed-step because it has no embedded error estimator.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RosenbrockW6S4OS;

/// The adaptive third-order, second-order embedded Rodas23W Rosenbrock-W
/// method.
///
/// Rodas23W is the five-stage W-method from the pinned
/// `OrdinaryDiffEqRosenbrock` revision. The primary update is third order and
/// the embedded update is second order (`btilde = [0, 0, 0, 1, -1]`).
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
algorithm!(Ros3);
algorithm!(Ros3Pr);
algorithm!(Ros3p);
algorithm!(Ros34Prw);
algorithm!(Ros34Pw3);
algorithm!(Grk4a);
algorithm!(Grk4t);
algorithm!(Ros34Pw1b);
algorithm!(Ros34Pw2);
algorithm!(Rodas4);
algorithm!(Rodas5P);
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
                    problem, candidate, state, time, step, options, &$tableau, workspace, stats,
                )
            }
        }
    };
}

rodas_method!(Ros2, 2, ROS2_TABLEAU);
rodas_method!(Rodas3, 3, RODAS3_TABLEAU);
rodas_method!(Ros3, 3, ROS3_TABLEAU);
rodas_method!(Ros3Pr, 3, ROS3PR_TABLEAU);
rodas_method!(Ros3p, 3, ROS3P_TABLEAU);
rodas_method!(Ros34Prw, 3, ROS34PRW_TABLEAU);
rodas_method!(Ros34Pw3, 4, ROS34PW3_TABLEAU);
rodas_method!(Grk4a, 4, GRK4A_TABLEAU);
rodas_method!(Grk4t, 4, GRK4T_TABLEAU);
rodas_method!(Ros34Pw1b, 3, ROS34PW1B_TABLEAU);
rodas_method!(Ros34Pw2, 3, ROS34PW2_TABLEAU);
rodas_method!(Rodas4, 4, RODAS4_TABLEAU);
rodas_method!(Rodas5P, 5, RODAS5P_TABLEAU);
rodas_method!(Rodas23W, 3, RODAS23W_TABLEAU);

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
            stages: vec![0.0; 8 * dimension],
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
    Ok(if options.adaptive {
        scaled_error_norm(state, candidate, &workspace.error, options)
    } else {
        0.0
    })
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
        Grk4a, Grk4t, Rodas3, Rodas4, Rodas5P, Rodas23W, Ros2, Ros3, Ros3Pr, Ros3p, Ros34Prw,
        Ros34Pw1b, Rosenbrock32, RosenbrockW6S4OS,
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
            solve(&stiff_problem((0.0, 1.0), 1.0), Ros3, &adaptive_options())
                .unwrap()
                .last_state()[0],
            solve(&stiff_problem((0.0, 1.0), 1.0), Ros3Pr, &adaptive_options())
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
            convergence_ratio(Ros3Pr, 0.1),
            convergence_ratio(Ros3p, 0.1),
            convergence_ratio(Ros34Prw, 0.1),
            convergence_ratio(Rodas4, 0.1),
            convergence_ratio(Grk4a, 0.1),
            convergence_ratio(Grk4t, 0.1),
            convergence_ratio(Ros34Pw1b, 0.1),
            convergence_ratio(Rodas5P, 0.2),
            convergence_ratio(RosenbrockW6S4OS, 0.1),
            convergence_ratio(Rodas23W, 0.1),
        ];
        assert!(ratios[0] > 3.0);
        assert!(ratios[1] > 7.0);
        assert!(ratios[2] > 7.0);
        assert!(ratios[3] > 7.0);
        // ROS3P is a third-order method; the adjacent fourth-order methods
        // retain the stricter ratio checks below.
        assert!(ratios[4] > 7.0);
        assert!(ratios[5] > 7.0);
        assert!(ratios[6] > 14.0);
        // GRK4A is fourth order; Rodas5P is the fifth-order method in this
        // table and keeps the stronger ratio check below.
        assert!(ratios[7] > 14.0);
        assert!(ratios[8] > 7.0);
        // ROS34PW1b is third order in its primary fixed-step update.
        assert!(ratios[9] > 7.0);
        assert!(ratios[10] > 14.0);
        assert!(ratios[11] > 7.0);
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
    fn w6s4os_preserves_callbacks_and_requested_samples() {
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
        for time in options.save_at {
            assert!(solution.times().contains(&time), "missing save_at={time}");
        }
        assert!(solution.last_state()[0] > 0.0);

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
                "rodas5p",
                solve(&backward_problem(), Rodas5P, &adaptive_options())
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
