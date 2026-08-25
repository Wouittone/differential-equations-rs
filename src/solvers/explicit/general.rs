use super::coefficient_data::*;
use super::coefficient_data::{
    BS3_A_ROWS, BS3_B as GENERATED_BS3_B, BS3_E as GENERATED_BS3_E, BS3_STAGE_TIMES, BS5_DENSE,
    BS5_EXTRA_STAGES, DP5_A_ROWS, DP5_B as GENERATED_DP5_B, DP5_E as GENERATED_DP5_E,
    DP5_STAGE_TIMES,
};
use crate::callback::CallbackOutcome;
use crate::integrator::{
    KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::solution::{
    BorrowedHermiteSegment, BorrowedRungeKuttaSegment, HermiteSegment, RungeKuttaSegment,
    TrajectoryRecorder, interpolate_runge_kutta,
};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};
use std::marker::PhantomData;

// Compatibility reexports for the historical `explicit::general` façade.
pub use super::prk::{KuttaPRK2p5, KuttaPrk2p5Tableau};
pub use super::qprk::{QPRK98, Qprk98Tableau};
pub use super::split_euler::SplitEuler;
pub use super::tsit5::Tsit5;

/// Coefficients and method properties for an explicit Runge–Kutta method.
///
/// `COEFFICIENTS[i]` is the strictly lower-triangular row for stage `i`, so it must contain exactly `i` entries.
/// All other coefficient arrays must contain one entry per stage.
/// [`ExplicitRungeKutta`] validates these invariants before solving.
pub trait ButcherTableau {
    const NODES: &'static [f64];
    const COEFFICIENTS: &'static [&'static [f64]];
    const WEIGHTS: &'static [f64];
    const ERROR_WEIGHTS: Option<&'static [f64]>;
    /// A second embedded error estimator, combined with the first by taking
    /// the larger scaled norm. Most methods use only [`Self::ERROR_WEIGHTS`].
    const SECOND_ERROR_WEIGHTS: Option<&'static [f64]> = None;
    const ORDER: usize;
    const FSAL: bool;
    /// Optional method-specific continuous-extension coefficients.
    ///
    /// Each row corresponds to one RK stage and stores `r0, r1, ...` for the
    /// stage weight `theta * (r0 + r1*theta + ...)`.
    const DENSE_COEFFICIENTS: Option<&'static [&'static [f64]]> = None;
    /// Optional stages evaluated lazily only when the continuous extension is
    /// requested by saving, root localization, or retained dense output.
    #[doc(hidden)]
    const LAZY_DENSE_STAGES: &'static [LazyDenseStage] = &[];
}

/// One sparse explicit stage used only by a method-specific continuous
/// extension.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LazyDenseStage {
    node: f64,
    coefficients: &'static [(usize, f64)],
}

impl LazyDenseStage {
    /// Creates a lazy dense stage at `node` from zero-based prior-stage
    /// coefficient pairs.
    #[doc(hidden)]
    pub const fn new(node: f64, coefficients: &'static [(usize, f64)]) -> Self {
        Self { node, coefficients }
    }
}

/// The centralized explicit Runge–Kutta solver for a [`ButcherTableau`].
///
/// Named algorithms such as [`Rk4`] are lightweight facades over
/// this type. It can also be instantiated with a user-defined tableau marker.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExplicitRungeKutta<T> {
    marker: PhantomData<fn() -> T>,
}

/// Upstream-compatible name for a user-supplied explicit Runge--Kutta
/// tableau marker.
pub type ExplicitRK<T> = ExplicitRungeKutta<T>;

impl<T> ExplicitRungeKutta<T> {
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

impl<T> Default for ExplicitRungeKutta<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> OdeAlgorithm for ExplicitRungeKutta<T>
where
    T: ButcherTableau,
{
    fn solve<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        integrate::<F, P, T>(problem, options)
    }
}

const RKM_A: &[&[f64]] = &[EMPTY, RKM_A2, RKM_A3, RKM_A4, RKM_A5, RKM_A6];

// Tsitouras' Runge--Kutta--Oliver six-stage fifth-order method. The pinned
// OrdinaryDiffEq implementation starts with a stage at c=2/3 (rather than
// evaluating f at the left endpoint). The shared explicit driver reserves an
// unweighted c=0 stage for dense-output Hermite segments, then stores the
// upstream six stages at indices 1..=6. This preserves the published tableau
// exactly while keeping the driver's endpoint and save-at semantics sound.

const RKO65_A: &[&[f64]] = &[
    EMPTY, RKO65_A1, RKO65_A2, RKO65_A3, RKO65_A4, RKO65_A5, RKO65_A6,
];

// Misha Stepanov's eight-stage fifth-order method with a final FSAL stage.
// Coefficients are copied from OrdinaryDiffEqLowOrderRK's
// `MSRK5ConstantCache` at the pinned upstream revision.

const MSRK5_A: &[&[f64]] = &[
    EMPTY,
    MSRK5_A2,
    MSRK5_A3,
    MSRK5_A4,
    MSRK5_A5,
    MSRK5_A6,
    MSRK5_A7,
    MSRK5_A8,
    MSRK5_FSAL_ROW,
];

// Misha Stepanov's embedded (4,5) seven-stage FSAL pair. The tableau and
// embedded estimator are copied from OrdinaryDiffEqLowOrderRK's
// `Stepanov5ConstantCache` at the pinned upstream revision. The final stage
// evaluates the endpoint derivative and therefore repeats the primary update
// weights, preserving the FSAL lifecycle used by the upstream implementation.

const STEPANOV5_A: &[&[f64]] = &[
    EMPTY,
    STEPANOV5_A2,
    STEPANOV5_A3,
    STEPANOV5_A4,
    STEPANOV5_A5,
    STEPANOV5_A6,
    STEPANOV5_FSAL_ROW,
];

// Kovalnogov--Simos--Tsitouras' seven-stage embedded (4,5) pair designed for
// SIR-type epidemic models. Coefficients are copied from
// `SIR54ConstantCache` in OrdinaryDiffEqLowOrderRK at the pinned revision.
// The eighth row evaluates the accepted endpoint derivative and repeats the
// primary weights, enabling the shared driver's FSAL lifecycle.

const SIR54_A: &[&[f64]] = &[
    EMPTY,
    SIR54_A2,
    SIR54_A3,
    SIR54_A4,
    SIR54_A5,
    SIR54_A6,
    SIR54_A7,
    SIR54_FSAL_ROW,
];

// Misha Stepanov's eight-stage sixth-order fixed-step method. The tableau is
// copied from OrdinaryDiffEqLowOrderRK's `MSRK6ConstantCache` at the pinned
// upstream revision. OrdinaryDiffEq evaluates one additional endpoint
// derivative for its default FSAL lifecycle, represented here by the final
// row equal to the update weights and a zero final weight.

const MSRK6_A: &[&[f64]] = &[
    EMPTY,
    MSRK6_A2,
    MSRK6_A3,
    MSRK6_A4,
    MSRK6_A5,
    MSRK6_A6,
    MSRK6_A7,
    MSRK6_A8,
    MSRK6_FSAL_ROW,
];

const RALSTON4_A: &[&[f64]] = &[EMPTY, RALSTON4_A2, RALSTON4_A3, RALSTON4_A4];

const ALSHINA3_A: &[&[f64]] = &[EMPTY, HALF_STAGE_ROW, ALSHINA3_A3];

// Alshina's optimal sixth-order, seven-stage fixed-step scheme. The
// coefficients are copied from OrdinaryDiffEqLowOrderRK's
// `Alshina6ConstantCache` at the pinned upstream revision. The final update
// uses only stages 1, 5, 6, and 7 (the omitted b2-b4 entries are zero).

const ALSHINA6_A: &[&[f64]] = &[
    EMPTY,
    ALSHINA6_A2,
    ALSHINA6_A3,
    ALSHINA6_A4,
    ALSHINA6_A5,
    ALSHINA6_A6,
    ALSHINA6_A7,
];

const BS3_A: &[&[f64]] = BS3_A_ROWS;
const BS3_B: &[f64] = &GENERATED_BS3_B;
const BS3_E: &[f64] = &GENERATED_BS3_E;
const BS3_C: &[f64] = &BS3_STAGE_TIMES;

const DP5_A: &[&[f64]] = DP5_A_ROWS;
const DP5_B: &[f64] = &GENERATED_DP5_B;
const DP5_E: &[f64] = &GENERATED_DP5_E;
const DP5_C: &[f64] = &DP5_STAGE_TIMES;
// Dormand--Prince's fourth-order continuous extension, expanded from the
// four dense combinations used by OrdinaryDiffEq's DP5 cache into weights
// over the seven step stages already retained by the shared RK workspace.
const DP5_D1: f64 = -12_715_105_075.0 / 11_282_082_432.0;
const DP5_D3: f64 = 87_487_479_700.0 / 32_700_410_799.0;
const DP5_D4: f64 = -10_690_763_975.0 / 1_880_347_072.0;
const DP5_D5: f64 = 701_980_252_875.0 / 199_316_789_632.0;
const DP5_D6: f64 = -1_453_857_185.0 / 822_651_844.0;
const DP5_D7: f64 = 69_997_945.0 / 29_380_423.0;
const DP5_R1: &[f64] = &[
    1.0,
    3.0 * (35.0 / 384.0) - 2.0 + DP5_D1,
    -2.0 * (35.0 / 384.0) + 1.0 - 2.0 * DP5_D1,
    DP5_D1,
];

const DP5_R3: &[f64] = &[
    0.0,
    3.0 * (500.0 / 1_113.0) + DP5_D3,
    -2.0 * (500.0 / 1_113.0) - 2.0 * DP5_D3,
    DP5_D3,
];
const DP5_R4: &[f64] = &[
    0.0,
    3.0 * (125.0 / 192.0) + DP5_D4,
    -2.0 * (125.0 / 192.0) - 2.0 * DP5_D4,
    DP5_D4,
];
const DP5_R5: &[f64] = &[
    0.0,
    3.0 * (-2_187.0 / 6_784.0) + DP5_D5,
    -2.0 * (-2_187.0 / 6_784.0) - 2.0 * DP5_D5,
    DP5_D5,
];
const DP5_R6: &[f64] = &[
    0.0,
    3.0 * (11.0 / 84.0) + DP5_D6,
    -2.0 * (11.0 / 84.0) - 2.0 * DP5_D6,
    DP5_D6,
];
const DP5_R7: &[f64] = &[0.0, -1.0 + DP5_D7, 1.0 - 2.0 * DP5_D7, DP5_D7];
const DP5_DENSE: &[&[f64]] = &[DP5_R1, DP5_R2, DP5_R3, DP5_R4, DP5_R5, DP5_R6, DP5_R7];

// Owren-Zennaro 3/2 pair.

const OWREN_ZEN3_A: &[&[f64]] = &[EMPTY, OWREN_ZEN3_A2, OWREN_ZEN3_A3, OWREN_ZEN3_A4];

const OWREN_ZEN3_DENSE: &[&[f64]] = &[OWREN_ZEN3_R1, OWREN_ZEN3_R2, OWREN_ZEN3_R3, OWREN_ZEN3_R4];

// Owren-Zennaro 4/3 pair.

const OWREN_ZEN4_A: &[&[f64]] = &[
    EMPTY,
    OWREN_ZEN4_A2,
    OWREN_ZEN4_A3,
    OWREN_ZEN4_A4,
    OWREN_ZEN4_A5,
    OWREN_ZEN4_A6,
];

const OWREN_ZEN4_DENSE: &[&[f64]] = &[
    OWREN_ZEN4_R1,
    OWREN_ZEN4_R2,
    OWREN_ZEN4_R3,
    OWREN_ZEN4_R4,
    OWREN_ZEN4_R5,
    OWREN_ZEN4_R6,
];

// Owren-Zennaro 5/4 pair.

const OWREN_ZEN5_A: &[&[f64]] = &[
    EMPTY,
    OWREN_ZEN5_A2,
    OWREN_ZEN5_A3,
    OWREN_ZEN5_A4,
    OWREN_ZEN5_A5,
    OWREN_ZEN5_A6,
    OWREN_ZEN5_A7,
    OWREN_ZEN5_A8,
];

const OWREN_ZEN5_DENSE: &[&[f64]] = &[
    OWREN_ZEN5_R1,
    OWREN_ZEN5_R2,
    OWREN_ZEN5_R3,
    OWREN_ZEN5_R4,
    OWREN_ZEN5_R5,
    OWREN_ZEN5_R6,
    OWREN_ZEN5_R7,
    OWREN_ZEN5_R8,
];

// Bogacki-Shampine 5/4 pair. Its controller uses the maximum of two embedded
// estimators, represented by ERROR_WEIGHTS and SECOND_ERROR_WEIGHTS.

const BS5_A: &[&[f64]] = &[
    EMPTY, BS5_A2, BS5_A3, BS5_A4, BS5_A5, BS5_A6, BS5_A7, BS5_A8,
];

const SSPRK22_A: &[&[f64]] = &[EMPTY, ENDPOINT_STAGE_ROW];
const SSPRK22_B: &[f64] = EXPLICIT_TRAPEZOID_WEIGHTS;
const SSPRK22_C: &[f64] = EXPLICIT_ENDPOINT_NODES;

const SSPRK22_DENSE: &[&[f64]] = &[SSPRK22_DENSE_1, SSPRK22_DENSE_2];

const SSPRK33_A: &[&[f64]] = &[EMPTY, SSPRK33_A2, SSPRK33_A3];

const SSPRK33_DENSE: &[&[f64]] = &[SSPRK33_DENSE_1, SSPRK33_DENSE_2, SSPRK33_DENSE_3];

const SSPRK43_A: &[&[f64]] = &[EMPTY, SSPRK43_A2, SSPRK43_A3, SSPRK43_A4];

const SSPRK43_DENSE: &[&[f64]] = &[
    SSPRK43_DENSE_1,
    SSPRK43_DENSE_2,
    SSPRK43_DENSE_3,
    SSPRK43_DENSE_4,
];

// Four-stage, third-order pseudo-symplectic Runge–Kutta method. These
// coefficients are copied from OrdinaryDiffEqLowOrderRK's
// `PSRK3p5q4ConstantCache` at the pinned upstream revision.

const PSRK3P5Q4_A: &[&[f64]] = &[EMPTY, PSRK3P5Q4_A2, PSRK3P5Q4_A3, PSRK3P5Q4_A4];

// Five-stage, third-order pseudo-symplectic Runge--Kutta method. Coefficients
// are copied from OrdinaryDiffEqLowOrderRK's `PSRK3p6q5ConstantCache` at the
// pinned upstream revision.

const PSRK3P6Q5_A: &[&[f64]] = &[
    EMPTY,
    PSRK3P6Q5_A2,
    PSRK3P6Q5_A3,
    PSRK3P6Q5_A4,
    PSRK3P6Q5_A5,
];

// Six-stage, fourth-order pseudo-symplectic Runge--Kutta method. Coefficients
// are copied from OrdinaryDiffEqLowOrderRK's `PSRK4p7q6ConstantCache` at the
// pinned upstream revision.

const PSRK4P7Q6_A: &[&[f64]] = &[
    EMPTY,
    PSRK4P7Q6_A2,
    PSRK4P7Q6_A3,
    PSRK4P7Q6_A4,
    PSRK4P7Q6_A5,
    PSRK4P7Q6_A6,
];

macro_rules! algorithm {
    (
        $name:ident,
        $documentation:literal,
        nodes = $nodes:ident,
        coefficients = $coefficients:ident,
        weights = $weights:ident,
        error_weights = $error_weights:expr,
        dense_coefficients = $dense_coefficients:ident,
        order = $order:literal,
        fsal = $fsal:literal
    ) => {
        algorithm!(@impl $name, $documentation, $nodes, $coefficients, $weights,
            $error_weights, Some($dense_coefficients), $order, $fsal);
    };
    (
        $name:ident,
        $documentation:literal,
        nodes = $nodes:ident,
        coefficients = $coefficients:ident,
        weights = $weights:ident,
        error_weights = $error_weights:expr,
        order = $order:literal,
        fsal = $fsal:literal
    ) => {
        algorithm!(@impl $name, $documentation, $nodes, $coefficients, $weights,
            $error_weights, None, $order, $fsal);
    };
    (@impl
        $name:ident,
        $documentation:literal,
        $nodes:ident,
        $coefficients:ident,
        $weights:ident,
        $error_weights:expr,
        $dense_coefficients:expr,
        $order:literal,
        $fsal:literal
    ) => {
        #[doc = $documentation]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        pub struct $name;

        impl ButcherTableau for $name {
            const NODES: &'static [f64] = $nodes;
            const COEFFICIENTS: &'static [&'static [f64]] = $coefficients;
            const WEIGHTS: &'static [f64] = $weights;
            const ERROR_WEIGHTS: Option<&'static [f64]> = $error_weights;
            const DENSE_COEFFICIENTS: Option<&'static [&'static [f64]]> = $dense_coefficients;
            const ORDER: usize = $order;
            const FSAL: bool = $fsal;
        }

        impl OdeAlgorithm for $name {
            fn solve<F, P>(
                &self,
                problem: &OdeProblem<F, P>,
                options: &SolveOptions,
            ) -> Result<Solution, SolveError>
            where
                F: Fn(&mut [f64], &[f64], &P, f64),
            {
                ExplicitRungeKutta::<Self>::new().solve(problem, options)
            }
        }
    };
}

crate::define_explicit_rk_from_file!(pub Euler, "tableaux/explicit/euler.toml", crate = crate);
crate::define_explicit_rk_from_file!(pub Midpoint, "tableaux/explicit/midpoint.toml", crate = crate);
crate::define_explicit_rk_from_file!(pub Heun, "tableaux/explicit/heun.toml", crate = crate);
crate::define_explicit_rk_from_file!(pub Ralston, "tableaux/explicit/ralston.toml", crate = crate);
crate::define_explicit_rk_from_file!(pub Rk4, "tableaux/explicit/rk4.toml", crate = crate);
algorithm!(
    Rkm,
    "The fixed-step six-stage, fourth-order Mead–Renaut Runge–Kutta method.",
    nodes = RKM_C,
    coefficients = RKM_A,
    weights = RKM_B,
    error_weights = None,
    order = 4,
    fsal = false
);
algorithm!(
    Rko65,
    "Tsitouras' six-stage, fifth-order Runge--Kutta--Oliver method.",
    nodes = RKO65_C,
    coefficients = RKO65_A,
    weights = RKO65_B,
    error_weights = None,
    order = 5,
    fsal = false
);
algorithm!(
    Msrk5,
    "The fixed-step eight-stage, fifth-order Misha Stepanov Runge–Kutta method.",
    nodes = MSRK5_C,
    coefficients = MSRK5_A,
    weights = MSRK5_B,
    error_weights = None,
    order = 5,
    fsal = true
);
algorithm!(
    Msrk6,
    "The fixed-step eight-stage, sixth-order Misha Stepanov Runge–Kutta method.",
    nodes = MSRK6_C,
    coefficients = MSRK6_A,
    weights = MSRK6_B,
    error_weights = None,
    order = 6,
    fsal = true
);
algorithm!(
    Stepanov5,
    "Misha Stepanov's adaptive embedded (4,5) seven-stage FSAL Runge–Kutta method.",
    nodes = STEPANOV5_C,
    coefficients = STEPANOV5_A,
    weights = STEPANOV5_B,
    error_weights = Some(STEPANOV5_B_TILDE),
    order = 5,
    fsal = true
);
algorithm!(
    Sir54,
    "The adaptive embedded (4,5) seven-stage FSAL Runge–Kutta method for SIR-type epidemic models.",
    nodes = SIR54_C,
    coefficients = SIR54_A,
    weights = SIR54_B,
    error_weights = Some(SIR54_ERROR),
    order = 5,
    fsal = true
);
algorithm!(
    Ralston4,
    "Ralston's fixed-step four-stage, fourth-order Runge–Kutta method.",
    nodes = RALSTON4_C,
    coefficients = RALSTON4_A,
    weights = RALSTON4_B,
    error_weights = None,
    order = 4,
    fsal = false
);
crate::define_explicit_rk_from_file!(pub Alshina2, "tableaux/explicit/alshina2.toml", crate = crate);
algorithm!(
    Alshina3,
    "The adaptive optimal three-stage, third-order Alshina method.",
    nodes = ALSHINA3_C,
    coefficients = ALSHINA3_A,
    weights = ALSHINA3_B,
    error_weights = Some(ALSHINA3_E),
    order = 3,
    fsal = false
);
algorithm!(
    Alshina6,
    "The fixed-step optimal seven-stage, sixth-order Alshina method.",
    nodes = ALSHINA6_C,
    coefficients = ALSHINA6_A,
    weights = ALSHINA6_B,
    error_weights = None,
    order = 6,
    fsal = false
);
algorithm!(
    Bs3,
    "The adaptive Bogacki–Shampine 3/2 method.",
    nodes = BS3_C,
    coefficients = BS3_A,
    weights = BS3_B,
    error_weights = Some(BS3_E),
    order = 3,
    fsal = true
);
algorithm!(
    Dp5,
    "The adaptive Dormand–Prince 5/4 method.",
    nodes = DP5_C,
    coefficients = DP5_A,
    weights = DP5_B,
    error_weights = Some(DP5_E),
    dense_coefficients = DP5_DENSE,
    order = 5,
    fsal = true
);
algorithm!(
    OwrenZen3,
    "The adaptive Owren-Zennaro 3/2 method with a free third-order interpolant upstream.",
    nodes = OWREN_ZEN3_C,
    coefficients = OWREN_ZEN3_A,
    weights = OWREN_ZEN3_B,
    error_weights = Some(OWREN_ZEN3_E),
    dense_coefficients = OWREN_ZEN3_DENSE,
    order = 3,
    fsal = true
);
algorithm!(
    OwrenZen4,
    "The adaptive Owren-Zennaro 4/3 method with a free fourth-order interpolant upstream.",
    nodes = OWREN_ZEN4_C,
    coefficients = OWREN_ZEN4_A,
    weights = OWREN_ZEN4_B,
    error_weights = Some(OWREN_ZEN4_E),
    dense_coefficients = OWREN_ZEN4_DENSE,
    order = 4,
    fsal = true
);
algorithm!(
    OwrenZen5,
    "The adaptive Owren-Zennaro 5/4 method with a free fifth-order interpolant upstream.",
    nodes = OWREN_ZEN5_C,
    coefficients = OWREN_ZEN5_A,
    weights = OWREN_ZEN5_B,
    error_weights = Some(OWREN_ZEN5_E),
    dense_coefficients = OWREN_ZEN5_DENSE,
    order = 5,
    fsal = true
);

/// The adaptive Bogacki-Shampine 5/4 method.
///
/// OrdinaryDiffEq uses the maximum of two embedded estimators for this method;
/// both are retained here. Its optional extra stages are used only by the
/// upstream dense interpolant and are therefore outside the shared step kernel.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Bs5;

impl ButcherTableau for Bs5 {
    const NODES: &'static [f64] = BS5_C;
    const COEFFICIENTS: &'static [&'static [f64]] = BS5_A;
    const WEIGHTS: &'static [f64] = BS5_B;
    const ERROR_WEIGHTS: Option<&'static [f64]> = Some(BS5_E1);
    const SECOND_ERROR_WEIGHTS: Option<&'static [f64]> = Some(BS5_E2);
    const DENSE_COEFFICIENTS: Option<&'static [&'static [f64]]> = Some(BS5_DENSE);
    const LAZY_DENSE_STAGES: &'static [LazyDenseStage] = BS5_EXTRA_STAGES;
    const ORDER: usize = 5;
    const FSAL: bool = true;
}

impl OdeAlgorithm for Bs5 {
    fn solve<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        ExplicitRungeKutta::<Self>::new().solve(problem, options)
    }
}
algorithm!(
    SspRk22,
    "The fixed-step two-stage, second-order SSP Runge–Kutta method.",
    nodes = SSPRK22_C,
    coefficients = SSPRK22_A,
    weights = SSPRK22_B,
    error_weights = None,
    dense_coefficients = SSPRK22_DENSE,
    order = 2,
    fsal = false
);
algorithm!(
    SspRk33,
    "The fixed-step three-stage, third-order SSP Runge–Kutta method.",
    nodes = SSPRK33_C,
    coefficients = SSPRK33_A,
    weights = SSPRK33_B,
    error_weights = None,
    dense_coefficients = SSPRK33_DENSE,
    order = 3,
    fsal = false
);
algorithm!(
    SspRk43,
    "The adaptive four-stage, third-order SSP Runge–Kutta method.",
    nodes = SSPRK43_C,
    coefficients = SSPRK43_A,
    weights = SSPRK43_B,
    error_weights = Some(SSPRK43_E),
    dense_coefficients = SSPRK43_DENSE,
    order = 3,
    fsal = false
);
algorithm!(
    Psrk3p5q4,
    "The fixed-step four-stage, third-order pseudo-symplectic Runge–Kutta method.",
    nodes = PSRK3P5Q4_C,
    coefficients = PSRK3P5Q4_A,
    weights = PSRK3P5Q4_B,
    error_weights = None,
    order = 3,
    fsal = false
);
algorithm!(
    Psrk3p6q5,
    "The fixed-step five-stage, third-order pseudo-symplectic Runge–Kutta method.",
    nodes = PSRK3P6Q5_C,
    coefficients = PSRK3P6Q5_A,
    weights = PSRK3P6Q5_B,
    error_weights = None,
    order = 3,
    fsal = false
);
algorithm!(
    Psrk4p7q6,
    "The fixed-step six-stage, fourth-order pseudo-symplectic Runge–Kutta method.",
    nodes = PSRK4P7Q6_C,
    coefficients = PSRK4P7Q6_A,
    weights = PSRK4P7Q6_B,
    error_weights = None,
    order = 4,
    fsal = false
);

struct Workspace {
    // Flat stage-major storage: every stage is one contiguous component array.
    // The other work vectors remain separate arrays rather than per-component
    // structs, keeping the hot saxpy-style loops friendly to SIMD.
    stages: Vec<f64>,
    dimension: usize,
    temporary: Vec<f64>,
}

impl Workspace {
    fn new(stage_count: usize, dimension: usize) -> Self {
        Self {
            stages: vec![0.0; stage_count * dimension],
            dimension,
            temporary: vec![0.0; dimension],
        }
    }

    fn stage(&self, index: usize) -> &[f64] {
        let start = index * self.dimension;
        &self.stages[start..start + self.dimension]
    }

    fn swap_stages(&mut self, left: usize, right: usize) {
        let left_start = left * self.dimension;
        let right_start = right * self.dimension;
        for offset in 0..self.dimension {
            self.stages.swap(left_start + offset, right_start + offset);
        }
    }
}

fn validate_tableau<T: ButcherTableau>() -> Result<(), SolveError> {
    let stage_count = T::WEIGHTS.len();
    let dense_stage_count = stage_count + T::LAZY_DENSE_STAGES.len();
    let structurally_valid = stage_count > 0
        && T::ORDER > 0
        && T::NODES.first() == Some(&0.0)
        && T::NODES.len() == stage_count
        && T::COEFFICIENTS.len() == stage_count
        && T::COEFFICIENTS
            .iter()
            .enumerate()
            .all(|(stage, row)| row.len() == stage)
        && T::ERROR_WEIGHTS.is_none_or(|weights| weights.len() == stage_count);
    let error_estimators_valid = T::SECOND_ERROR_WEIGHTS
        .is_none_or(|weights| T::ERROR_WEIGHTS.is_some() && weights.len() == stage_count);
    let lazy_dense_stages_valid = (T::LAZY_DENSE_STAGES.is_empty()
        || T::DENSE_COEFFICIENTS.is_some())
        && T::LAZY_DENSE_STAGES
            .iter()
            .enumerate()
            .all(|(offset, stage)| {
                stage.node.is_finite()
                    && !stage.coefficients.is_empty()
                    && stage.coefficients.iter().all(|&(index, coefficient)| {
                        index < stage_count + offset && coefficient.is_finite()
                    })
            });
    let coefficients_finite = T::NODES.iter().all(|value| value.is_finite())
        && T::WEIGHTS.iter().all(|value| value.is_finite())
        && T::COEFFICIENTS
            .iter()
            .flat_map(|row| row.iter())
            .all(|value| value.is_finite())
        && T::ERROR_WEIGHTS.is_none_or(|weights| weights.iter().all(|value| value.is_finite()));
    let second_error_estimator_finite =
        T::SECOND_ERROR_WEIGHTS.is_none_or(|weights| weights.iter().all(|value| value.is_finite()));
    let dense_coefficients_valid = T::DENSE_COEFFICIENTS.is_none_or(|rows| {
        rows.len() == dense_stage_count
            && rows.iter().enumerate().all(|(stage, row)| {
                let endpoint_weight = T::WEIGHTS.get(stage).copied().unwrap_or(0.0);
                let coefficient_scale = row.iter().map(|value| value.abs()).sum::<f64>();
                !row.is_empty()
                    && row.iter().all(|coefficient| coefficient.is_finite())
                    && (row.iter().sum::<f64>() - endpoint_weight).abs()
                        <= 1.0e-12 * (1.0 + endpoint_weight.abs())
                            + 64.0 * f64::EPSILON * coefficient_scale
            })
    });
    let fsal_valid = !T::FSAL
        || (stage_count > 0
            && T::NODES.last() == Some(&1.0)
            && T::WEIGHTS.last() == Some(&0.0)
            && T::COEFFICIENTS
                .last()
                .is_some_and(|last_row| *last_row == &T::WEIGHTS[..stage_count - 1]));

    (structurally_valid
        && error_estimators_valid
        && lazy_dense_stages_valid
        && coefficients_finite
        && second_error_estimator_finite
        && dense_coefficients_valid
        && fsal_valid)
        .then_some(())
        .ok_or(SolveError::InvalidTableau)
}

fn integrate<F, P, T>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
) -> Result<Solution, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
    T: ButcherTableau,
{
    validate_tableau::<T>()?;
    drive_integration(
        problem,
        options,
        ExplicitKernel::<T>::new(problem.initial_state().len()),
    )
}

struct ExplicitKernel<T> {
    workspace: Workspace,
    stage_zero_is_current: bool,
    dense_endpoint_state: Vec<f64>,
    dense_endpoint_prepared: bool,
    dense_stages_prepared: bool,
    marker: PhantomData<fn() -> T>,
}

impl<T> ExplicitKernel<T> {
    fn new(dimension: usize) -> Self
    where
        T: ButcherTableau,
    {
        Self {
            workspace: Workspace::new(T::WEIGHTS.len() + T::LAZY_DENSE_STAGES.len(), dimension),
            stage_zero_is_current: false,
            dense_endpoint_state: vec![0.0; dimension],
            dense_endpoint_prepared: false,
            dense_stages_prepared: false,
            marker: PhantomData,
        }
    }
}

impl<F, P, T> StepKernel<F, P> for ExplicitKernel<T>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
    T: ButcherTableau,
{
    fn has_custom_dense_output(&self) -> bool {
        true
    }

    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::new(T::ERROR_WEIGHTS.is_some(), T::ORDER)
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
            &mut self.workspace.stages[..self.workspace.dimension],
            state,
            time,
            stats,
        );
        ensure_finite(&self.workspace.stages[..self.workspace.dimension])?;
        self.stage_zero_is_current = true;
        Ok(())
    }

    fn estimate_initial_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        direction: f64,
        maximum_step: f64,
        candidate: &mut [f64],
        options: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<f64, SolveError> {
        estimate_initial_step(
            problem,
            options,
            (state, candidate),
            (time, direction, maximum_step),
            T::ORDER,
            &mut self.workspace,
            stats,
        )
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
        if !self.stage_zero_is_current {
            evaluate(
                problem,
                &mut self.workspace.stages[..self.workspace.dimension],
                state,
                time,
                stats,
            );
            ensure_finite(&self.workspace.stages[..self.workspace.dimension])?;
        }
        perform_step::<F, P, T>(
            problem,
            state,
            time,
            step,
            candidate,
            &mut self.workspace,
            stats,
        );
        self.dense_stages_prepared = false;
        ensure_finite(candidate)?;
        let error = if options.adaptive {
            let error_weights = T::ERROR_WEIGHTS.ok_or(SolveError::AdaptiveStepUnsupported)?;
            let primary_error = error_norm(
                &self.workspace.stages,
                self.workspace.dimension,
                (state, candidate),
                step,
                options,
                error_weights,
                &mut self.workspace.temporary,
            );
            T::SECOND_ERROR_WEIGHTS.map_or(primary_error, |weights| {
                primary_error.max(error_norm(
                    &self.workspace.stages,
                    self.workspace.dimension,
                    (state, candidate),
                    step,
                    options,
                    weights,
                    &mut self.workspace.temporary,
                ))
            })
        } else {
            0.0
        };
        Ok(StepEstimate::new(error))
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
        let Some(coefficients) = T::DENSE_COEFFICIENTS else {
            if !problem.has_continuous_callbacks() {
                self.dense_endpoint_prepared = false;
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
            self.dense_endpoint_state.copy_from_slice(state);
            evaluate(problem, &mut self.workspace.temporary, state, *time, stats);
            ensure_finite(&self.workspace.temporary)?;
            self.dense_endpoint_prepared = true;
            let attempted_time = *time;
            let segment = BorrowedHermiteSegment::new(
                previous_time,
                attempted_time,
                previous_state,
                &self.dense_endpoint_state,
                self.workspace.stage(0),
                &self.workspace.temporary,
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
        };
        self.dense_endpoint_prepared = false;
        let attempted_time = *time;
        if !T::LAZY_DENSE_STAGES.is_empty() && problem.has_continuous_callbacks() {
            perform_lazy_dense_stages::<F, P, T>(
                problem,
                previous_state,
                previous_time,
                attempted_time - previous_time,
                &mut self.workspace,
                stats,
            )?;
            self.dense_stages_prepared = true;
        }
        let stages = &self.workspace.stages;
        let mut interpolate = |sample_time: f64, output: &mut [f64]| {
            interpolate_runge_kutta(
                previous_time,
                attempted_time,
                previous_state,
                stages,
                coefficients,
                sample_time,
                output,
            )
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
        if let Some(coefficients) = T::DENSE_COEFFICIENTS {
            if !self.dense_stages_prepared && !T::LAZY_DENSE_STAGES.is_empty() {
                perform_lazy_dense_stages::<F, P, T>(
                    problem,
                    previous_state,
                    previous_time,
                    attempted_time - previous_time,
                    &mut self.workspace,
                    stats,
                )?;
            }
            let segment = BorrowedRungeKuttaSegment::new(
                previous_time,
                attempted_time,
                previous_state,
                state,
                &self.workspace.stages,
                coefficients,
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
                let segment = RungeKuttaSegment::new(
                    previous_time,
                    attempted_time,
                    time,
                    previous_state,
                    state,
                    &self.workspace.stages,
                    coefficients,
                )
                .map_err(|_| SolveError::NonFiniteDerivative)?;
                recorder.retain_runge_kutta_segment(segment);
            }
            self.dense_stages_prepared = false;
        } else {
            if !recorder.needs_dense_sampling() && !recorder.retains_dense_output() {
                self.dense_endpoint_prepared = false;
                return Ok(false);
            }
            if !self.dense_endpoint_prepared {
                self.dense_endpoint_state.copy_from_slice(state);
                evaluate(
                    problem,
                    &mut self.workspace.temporary,
                    state,
                    attempted_time,
                    stats,
                );
                ensure_finite(&self.workspace.temporary)?;
            }
            let segment = BorrowedHermiteSegment::new(
                previous_time,
                attempted_time,
                previous_state,
                &self.dense_endpoint_state,
                self.workspace.stage(0),
                &self.workspace.temporary,
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
                let segment = HermiteSegment::new_bounded(
                    previous_time,
                    attempted_time,
                    time,
                    previous_state.to_vec(),
                    self.dense_endpoint_state.clone(),
                    self.workspace.stage(0).to_vec(),
                    self.workspace.temporary.clone(),
                )
                .map_err(|_| SolveError::NonFiniteDerivative)?;
                recorder.retain_hermite_segment(segment);
            }
            self.dense_endpoint_prepared = false;
        }
        Ok(true)
    }

    fn accept_step(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: &[f64],
        _: f64,
        _: f64,
        callback_applied: bool,
        _: &mut SolverStats,
    ) -> Result<(), SolveError> {
        if T::FSAL && !callback_applied {
            self.workspace.swap_stages(0, T::WEIGHTS.len() - 1);
            self.stage_zero_is_current = true;
        } else {
            self.stage_zero_is_current = false;
        }
        Ok(())
    }

    fn reject_step(&mut self) {
        self.stage_zero_is_current = true;
        self.dense_stages_prepared = false;
    }
}

fn evaluate<F, P>(
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

fn ensure_finite(values: &[f64]) -> Result<(), SolveError> {
    values
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteDerivative)
}

fn estimate_initial_step<F, P>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
    states: (&[f64], &mut [f64]),
    integration: (f64, f64, f64),
    order: usize,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<f64, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let (state, scratch) = states;
    let (time, direction, maximum_step) = integration;
    let dimension = state.len() as f64;
    let mut state_norm = 0.0;
    let mut derivative_norm = 0.0;
    for (value, derivative) in state.iter().zip(workspace.stage(0)) {
        let scale = options.absolute_tolerance + options.relative_tolerance * value.abs();
        state_norm += (value / scale).powi(2);
        derivative_norm += (derivative / scale).powi(2);
    }
    state_norm = (state_norm / dimension).sqrt();
    derivative_norm = (derivative_norm / dimension).sqrt();
    let trial_step = if state_norm < 1.0e-5 || derivative_norm < 1.0e-5 {
        1.0e-6
    } else {
        0.01 * state_norm / derivative_norm
    }
    .min(maximum_step);

    for ((trial, value), derivative) in workspace
        .temporary
        .iter_mut()
        .zip(state)
        .zip(&workspace.stages[..workspace.dimension])
    {
        *trial = value + direction * trial_step * derivative;
    }
    evaluate(
        problem,
        scratch,
        &workspace.temporary,
        time + direction * trial_step,
        stats,
    );
    ensure_finite(scratch)?;

    let mut curvature_norm = 0.0;
    for ((next, initial), value) in scratch
        .iter()
        .zip(&workspace.stages[..workspace.dimension])
        .zip(state)
    {
        let scale = options.absolute_tolerance + options.relative_tolerance * value.abs();
        curvature_norm += ((next - initial) / scale).powi(2);
    }
    curvature_norm = (curvature_norm / dimension).sqrt() / trial_step;
    let largest = derivative_norm.max(curvature_norm);
    let accuracy_step = if largest <= 1.0e-15 {
        (trial_step * 1.0e-3).max(1.0e-6)
    } else {
        (0.01 / largest).powf(1.0 / order as f64)
    };
    Ok((100.0 * trial_step).min(accuracy_step).min(maximum_step))
}

fn perform_step<F, P, T>(
    problem: &OdeProblem<F, P>,
    state: &[f64],
    time: f64,
    step: f64,
    candidate: &mut [f64],
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) where
    F: Fn(&mut [f64], &[f64], &P, f64),
    T: ButcherTableau,
{
    let stage_count = T::WEIGHTS.len();
    for stage_index in 1..stage_count {
        combine(
            &mut workspace.temporary,
            state,
            step,
            &workspace.stages,
            workspace.dimension,
            stage_index,
            T::COEFFICIENTS[stage_index],
        );
        let start = stage_index * workspace.dimension;
        evaluate(
            problem,
            &mut workspace.stages[start..start + workspace.dimension],
            &workspace.temporary,
            time + T::NODES[stage_index] * step,
            stats,
        );
    }
    combine(
        candidate,
        state,
        step,
        &workspace.stages,
        workspace.dimension,
        stage_count,
        T::WEIGHTS,
    );
}

fn perform_lazy_dense_stages<F, P, T>(
    problem: &OdeProblem<F, P>,
    state: &[f64],
    time: f64,
    step: f64,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
    T: ButcherTableau,
{
    let base_stage_count = T::WEIGHTS.len();
    for (offset, stage) in T::LAZY_DENSE_STAGES.iter().enumerate() {
        workspace.temporary.copy_from_slice(state);
        for &(source, coefficient) in stage.coefficients {
            let source_start = source * workspace.dimension;
            for (value, derivative) in workspace
                .temporary
                .iter_mut()
                .zip(&workspace.stages[source_start..source_start + workspace.dimension])
            {
                *value += step * coefficient * derivative;
            }
        }
        let target = base_stage_count + offset;
        let target_start = target * workspace.dimension;
        evaluate(
            problem,
            &mut workspace.stages[target_start..target_start + workspace.dimension],
            &workspace.temporary,
            time + stage.node * step,
            stats,
        );
        ensure_finite(&workspace.stages[target_start..target_start + workspace.dimension])?;
    }
    Ok(())
}

fn combine(
    output: &mut [f64],
    state: &[f64],
    step: f64,
    stages: &[f64],
    dimension: usize,
    stage_count: usize,
    weights: &[f64],
) {
    output.fill(0.0);
    for (stage_index, weight) in weights.iter().take(stage_count).enumerate() {
        let start = stage_index * dimension;
        let stage = &stages[start..start + dimension];
        for (increment, stage_value) in output.iter_mut().zip(stage) {
            *increment += weight * stage_value;
        }
    }
    for (output_value, state_value) in output.iter_mut().zip(state) {
        *output_value = state_value + step * *output_value;
    }
}

fn error_norm(
    stages: &[f64],
    dimension: usize,
    states: (&[f64], &[f64]),
    step: f64,
    options: &SolveOptions,
    error_weights: &[f64],
    error_buffer: &mut [f64],
) -> f64 {
    let (state, candidate) = states;
    error_buffer.fill(0.0);
    for (stage_index, weight) in error_weights.iter().enumerate() {
        let start = stage_index * dimension;
        let stage = &stages[start..start + dimension];
        for (error, stage_value) in error_buffer.iter_mut().zip(stage) {
            *error += weight * stage_value;
        }
    }
    let mut squared_norm = 0.0;
    for ((error, state), candidate) in error_buffer.iter().zip(state).zip(candidate) {
        let error = step * error;
        let scale = options.absolute_tolerance
            + options.relative_tolerance * state.abs().max(candidate.abs());
        squared_norm += (error / scale).powi(2);
    }
    (squared_norm / state.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use std::f64::consts::E;

    use super::{
        Alshina2, Alshina3, Alshina6, Bs3, Dp5, Euler, Heun, Midpoint, Ralston, Ralston4, Rk4, Rkm,
        SspRk22, SspRk33, SspRk43,
    };
    use super::{Bs5, ButcherTableau, ExplicitRungeKutta, OwrenZen3, OwrenZen4, OwrenZen5};
    use crate::{OdeProblem, SaveMode, SolveError, SolveOptions, solve};

    type TestRhs = fn(&mut [f64], &[f64], &(), f64);

    struct CustomEuler;

    impl ButcherTableau for CustomEuler {
        const NODES: &'static [f64] = &[0.0];
        const COEFFICIENTS: &'static [&'static [f64]] = &[&[]];
        const WEIGHTS: &'static [f64] = &[1.0];
        const ERROR_WEIGHTS: Option<&'static [f64]> = None;
        const ORDER: usize = 1;
        const FSAL: bool = false;
    }

    struct MalformedTableau;

    impl ButcherTableau for MalformedTableau {
        const NODES: &'static [f64] = &[0.0, 1.0];
        const COEFFICIENTS: &'static [&'static [f64]] = &[&[]];
        const WEIGHTS: &'static [f64] = &[1.0];
        const ERROR_WEIGHTS: Option<&'static [f64]> = None;
        const ORDER: usize = 1;
        const FSAL: bool = false;
    }

    struct EmptyFsalTableau;

    impl ButcherTableau for EmptyFsalTableau {
        const NODES: &'static [f64] = &[];
        const COEFFICIENTS: &'static [&'static [f64]] = &[];
        const WEIGHTS: &'static [f64] = &[];
        const ERROR_WEIGHTS: Option<&'static [f64]> = None;
        const ORDER: usize = 1;
        const FSAL: bool = true;
    }

    struct SingleStageAdaptiveTableau;

    impl ButcherTableau for SingleStageAdaptiveTableau {
        const NODES: &'static [f64] = &[0.0];
        const COEFFICIENTS: &'static [&'static [f64]] = &[&[]];
        const WEIGHTS: &'static [f64] = &[1.0];
        const ERROR_WEIGHTS: Option<&'static [f64]> = Some(&[0.0]);
        const ORDER: usize = 1;
        const FSAL: bool = false;
    }

    struct DualEstimatorHeun;

    impl ButcherTableau for DualEstimatorHeun {
        const NODES: &'static [f64] = &[0.0, 1.0];
        const COEFFICIENTS: &'static [&'static [f64]] = &[&[], &[1.0]];
        const WEIGHTS: &'static [f64] = &[0.5, 0.5];
        const ERROR_WEIGHTS: Option<&'static [f64]> = Some(&[0.0, 0.0]);
        const SECOND_ERROR_WEIGHTS: Option<&'static [f64]> = Some(&[-0.5, 0.5]);
        const ORDER: usize = 2;
        const FSAL: bool = false;
    }

    struct MalformedSecondEstimator;

    impl ButcherTableau for MalformedSecondEstimator {
        const NODES: &'static [f64] = &[0.0];
        const COEFFICIENTS: &'static [&'static [f64]] = &[&[]];
        const WEIGHTS: &'static [f64] = &[1.0];
        const ERROR_WEIGHTS: Option<&'static [f64]> = Some(&[0.0]);
        const SECOND_ERROR_WEIGHTS: Option<&'static [f64]> = Some(&[]);
        const ORDER: usize = 1;
        const FSAL: bool = false;
    }

    struct MalformedDenseTableau;

    impl ButcherTableau for MalformedDenseTableau {
        const NODES: &'static [f64] = &[0.0];
        const COEFFICIENTS: &'static [&'static [f64]] = &[&[]];
        const WEIGHTS: &'static [f64] = &[1.0];
        const ERROR_WEIGHTS: Option<&'static [f64]> = None;
        const ORDER: usize = 1;
        const FSAL: bool = false;
        const DENSE_COEFFICIENTS: Option<&'static [&'static [f64]]> = Some(&[&[]]);
    }

    fn exponential() -> OdeProblem<TestRhs, ()> {
        fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
            du[0] = u[0];
        }

        OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
    }

    fn adaptive_options() -> SolveOptions {
        SolveOptions {
            absolute_tolerance: 1.0e-9,
            relative_tolerance: 1.0e-9,
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        }
    }

    #[test]
    fn adaptive_embedded_methods_solve_exponential_growth() {
        for endpoint in [
            solve(&exponential(), Midpoint, &adaptive_options())
                .unwrap()
                .last_state()[0],
            solve(&exponential(), Heun, &adaptive_options())
                .unwrap()
                .last_state()[0],
            solve(&exponential(), Ralston, &adaptive_options())
                .unwrap()
                .last_state()[0],
            solve(&exponential(), Bs3, &adaptive_options())
                .unwrap()
                .last_state()[0],
            solve(&exponential(), Dp5, &adaptive_options())
                .unwrap()
                .last_state()[0],
        ] {
            assert!((endpoint - E).abs() < 2.0e-7);
        }
    }

    fn fixed_endpoint<T: crate::OdeAlgorithm>(algorithm: T, step: f64) -> f64 {
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(step),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };
        solve(&exponential(), algorithm, &options)
            .unwrap()
            .last_state()[0]
    }

    fn convergence_ratio<T: crate::OdeAlgorithm + Copy>(algorithm: T, step: f64) -> f64 {
        let coarse = (fixed_endpoint(algorithm, step) - E).abs();
        let fine = (fixed_endpoint(algorithm, step / 2.0) - E).abs();
        coarse / fine
    }

    #[test]
    fn owren_zen_and_bs5_have_their_expected_orders() {
        let ratios = [
            convergence_ratio(OwrenZen3, 0.1),
            convergence_ratio(OwrenZen4, 0.1),
            convergence_ratio(OwrenZen5, 0.1),
            convergence_ratio(Bs5, 0.1),
        ];
        assert!(ratios[0] > 7.0);
        assert!(ratios[1] > 14.0);
        assert!(ratios[2] > 25.0);
        assert!(ratios[3] > 25.0);
    }

    #[test]
    fn owren_zen_and_bs5_adaptive_solvers_reach_tight_tolerance() {
        for endpoint in [
            solve(&exponential(), OwrenZen3, &adaptive_options())
                .unwrap()
                .last_state()[0],
            solve(&exponential(), OwrenZen4, &adaptive_options())
                .unwrap()
                .last_state()[0],
            solve(&exponential(), OwrenZen5, &adaptive_options())
                .unwrap()
                .last_state()[0],
            solve(&exponential(), Bs5, &adaptive_options())
                .unwrap()
                .last_state()[0],
        ] {
            assert!((endpoint - E).abs() < 2.0e-7);
        }
    }

    #[test]
    fn bs5_retains_both_upstream_error_estimators() {
        assert!(Bs5::ERROR_WEIGHTS.is_some());
        assert!(Bs5::SECOND_ERROR_WEIGHTS.is_some());
        assert_ne!(Bs5::ERROR_WEIGHTS, Bs5::SECOND_ERROR_WEIGHTS);
    }

    #[test]
    fn fixed_methods_have_expected_convergence() {
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.001),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };
        let euler_error =
            (solve(&exponential(), Euler, &options).unwrap().last_state()[0] - E).abs();
        let rk4_error = (solve(&exponential(), Rk4, &options).unwrap().last_state()[0] - E).abs();
        let rkm_error = (solve(&exponential(), Rkm, &options).unwrap().last_state()[0] - E).abs();
        let ralston4_error = (solve(&exponential(), Ralston4, &options)
            .unwrap()
            .last_state()[0]
            - E)
            .abs();
        let alshina2_error = (solve(&exponential(), Alshina2, &options)
            .unwrap()
            .last_state()[0]
            - E)
            .abs();
        let alshina3_error = (solve(&exponential(), Alshina3, &options)
            .unwrap()
            .last_state()[0]
            - E)
            .abs();
        let alshina6_error = (solve(&exponential(), Alshina6, &options)
            .unwrap()
            .last_state()[0]
            - E)
            .abs();

        assert!(euler_error < 0.002);
        assert!(rk4_error < 1.0e-12);
        assert!(rkm_error < 1.0e-12);
        assert!(ralston4_error < 1.0e-12);
        assert!(alshina2_error < 1.0e-6);
        assert!(alshina3_error < 1.0e-9);
        assert!(alshina6_error < 1.0e-12);
        assert!(convergence_ratio(Alshina6, 0.1) > 40.0);
    }

    #[test]
    fn fixed_only_methods_reject_adaptive_configuration() {
        assert_eq!(
            solve(&exponential(), Euler, &SolveOptions::default()),
            Err(SolveError::AdaptiveStepUnsupported)
        );
        assert_eq!(
            solve(&exponential(), Rk4, &SolveOptions::default()),
            Err(SolveError::AdaptiveStepUnsupported)
        );
        assert_eq!(
            solve(&exponential(), Alshina6, &SolveOptions::default()),
            Err(SolveError::AdaptiveStepUnsupported)
        );
    }

    #[test]
    fn named_solver_is_a_facade_over_the_generic_kernel() {
        let problem = exponential();
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.01),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        let named = solve(&problem, Rk4, &options).unwrap();
        let generic = solve(&problem, ExplicitRungeKutta::<Rk4>::new(), &options).unwrap();

        assert_eq!(named, generic);
    }

    #[test]
    fn supports_custom_tableaus_and_rejects_malformed_ones() {
        let problem = exponential();
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.01),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        let custom = solve(&problem, ExplicitRungeKutta::<CustomEuler>::new(), &options).unwrap();
        let named = solve(&problem, Euler, &options).unwrap();

        assert_eq!(custom, named);
        assert_eq!(
            solve(
                &problem,
                ExplicitRungeKutta::<MalformedTableau>::new(),
                &options,
            ),
            Err(SolveError::InvalidTableau)
        );
        assert_eq!(
            solve(
                &problem,
                ExplicitRungeKutta::<EmptyFsalTableau>::new(),
                &options,
            ),
            Err(SolveError::InvalidTableau)
        );
        assert_eq!(
            solve(
                &problem,
                ExplicitRungeKutta::<MalformedSecondEstimator>::new(),
                &adaptive_options(),
            ),
            Err(SolveError::InvalidTableau)
        );
        assert_eq!(
            solve(
                &problem,
                ExplicitRungeKutta::<MalformedDenseTableau>::new(),
                &options,
            ),
            Err(SolveError::InvalidTableau)
        );
        assert!(
            solve(
                &problem,
                ExplicitRungeKutta::<SingleStageAdaptiveTableau>::new(),
                &adaptive_options(),
            )
            .is_ok()
        );
    }

    #[test]
    fn combines_two_error_estimators_by_their_maximum_norm() {
        let options = SolveOptions {
            absolute_tolerance: 1.0e-9,
            relative_tolerance: 1.0e-9,
            initial_step: Some(1.0),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        let solution = solve(
            &exponential(),
            ExplicitRungeKutta::<DualEstimatorHeun>::new(),
            &options,
        )
        .unwrap();

        assert!(solution.stats().rejected_steps > 0);
    }

    #[test]
    fn reports_non_finite_stage_derivatives() {
        let problem = OdeProblem::new(
            |du: &mut [f64], _: &[f64], _: &(), time: f64| {
                du[0] = if time == 0.0 { 1.0 } else { f64::NAN };
            },
            vec![1.0],
            (0.0, 1.0),
            (),
        );
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(1.0),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        assert_eq!(
            solve(&problem, Rk4, &options),
            Err(SolveError::NonFiniteDerivative)
        );
    }

    #[test]
    fn ssp_methods_solve_exponential_growth() {
        let fixed = SolveOptions {
            adaptive: false,
            initial_step: Some(0.001),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        let endpoints = [
            solve(&exponential(), SspRk22, &fixed).unwrap().last_state()[0],
            solve(&exponential(), SspRk33, &fixed).unwrap().last_state()[0],
            solve(&exponential(), SspRk43, &adaptive_options())
                .unwrap()
                .last_state()[0],
        ];

        assert!((endpoints[0] - E).abs() < 1.0e-6);
        assert!((endpoints[1] - E).abs() < 1.0e-9);
        assert!((endpoints[2] - E).abs() < 2.0e-7);
    }
}
