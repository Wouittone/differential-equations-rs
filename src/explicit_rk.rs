use crate::generated_coefficients::{
    BS3_A_ROWS, BS3_B as GENERATED_BS3_B, BS3_E as GENERATED_BS3_E, BS3_STAGE_TIMES, DP5_A_ROWS,
    DP5_B as GENERATED_DP5_B, DP5_E as GENERATED_DP5_E, DP5_STAGE_TIMES, RK4_A_ROWS,
    RK4_B as GENERATED_RK4_B, RK4_STAGE_TIMES,
};
use crate::integrator::{
    KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::solution::{BorrowedHermiteSegment, TrajectoryRecorder};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};
use std::marker::PhantomData;

/// Coefficients and method properties for an explicit Runge–Kutta method.
///
/// `COEFFICIENTS[i]` is the strictly lower-triangular row for stage `i`, so it
/// must contain exactly `i` entries. All other coefficient arrays must contain
/// one entry per stage. [`ExplicitRungeKutta`] validates these invariants before
/// solving.
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
}

/// The centralized explicit Runge–Kutta solver for a [`ButcherTableau`].
///
/// Named algorithms such as [`Rk4`](crate::Rk4) are lightweight facades over
/// this type. It can also be instantiated with a user-defined tableau marker.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExplicitRungeKutta<T> {
    marker: PhantomData<fn() -> T>,
}

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

const EMPTY: &[f64] = &[];
const EULER_A: &[&[f64]] = &[EMPTY];
const EULER_B: &[f64] = &[1.0];
const EULER_C: &[f64] = &[0.0];

const MIDPOINT_A2: &[f64] = &[0.5];
const MIDPOINT_A: &[&[f64]] = &[EMPTY, MIDPOINT_A2];
const MIDPOINT_B: &[f64] = &[0.0, 1.0];
const MIDPOINT_E: &[f64] = &[-1.0, 1.0];
const MIDPOINT_C: &[f64] = &[0.0, 0.5];

const HEUN_A2: &[f64] = &[1.0];
const HEUN_A: &[&[f64]] = &[EMPTY, HEUN_A2];
const HEUN_B: &[f64] = &[0.5, 0.5];
const HEUN_E: &[f64] = &[-0.5, 0.5];
const HEUN_C: &[f64] = &[0.0, 1.0];

const RALSTON_A2: &[f64] = &[2.0 / 3.0];
const RALSTON_A: &[&[f64]] = &[EMPTY, RALSTON_A2];
const RALSTON_B: &[f64] = &[0.25, 0.75];
const RALSTON_E: &[f64] = &[-0.75, 0.75];
const RALSTON_C: &[f64] = &[0.0, 2.0 / 3.0];

const RK4_A: &[&[f64]] = RK4_A_ROWS;
const RK4_B: &[f64] = &GENERATED_RK4_B;
const RK4_C: &[f64] = &RK4_STAGE_TIMES;

const RKM_A2: &[f64] = &[0.167_266_187_050_662];
const RKM_A3: &[f64] = &[0.0, 0.484_574_582_244_783];
const RKM_A4: &[f64] = &[0.0, 0.0, 0.536_909_403_373_491];
const RKM_A5: &[f64] = &[0.0, 0.0, 0.0, 0.082_069_535_961_948];
const RKM_A6: &[f64] = &[0.0, 0.0, 0.0, 0.0, 0.853_923_000_035_347];
const RKM_A: &[&[f64]] = &[EMPTY, RKM_A2, RKM_A3, RKM_A4, RKM_A5, RKM_A6];
const RKM_B: &[f64] = &[
    -0.028_289_441_132_839,
    0.463_968_918_564_71,
    -0.434_414_348_751_899,
    0.693_796_229_087_598,
    0.0,
    0.304_938_642_232_43,
];
const RKM_C: &[f64] = &[
    0.0,
    0.167_266_187_050_662,
    0.484_574_582_244_783,
    0.536_909_403_373_491,
    0.082_069_535_961_948,
    0.853_923_000_035_347,
];

// Misha Stepanov's eight-stage fifth-order method with a final FSAL stage.
// Coefficients are copied from OrdinaryDiffEqLowOrderRK's
// `MSRK5ConstantCache` at the pinned upstream revision.
const MSRK5_A2: &[f64] = &[4.0 / 45.0];
const MSRK5_A3: &[f64] = &[1.0 / 30.0, 1.0 / 10.0];
const MSRK5_A4: &[f64] = &[1.0 / 20.0, 0.0, 3.0 / 20.0];
const MSRK5_A5: &[f64] = &[1.0 / 2.0, 0.0, -15.0 / 8.0, 15.0 / 8.0];
const MSRK5_A6: &[f64] = &[-11.0 / 135.0, 0.0, 23.0 / 45.0, -2.0 / 27.0, 8.0 / 45.0];
const MSRK5_A7: &[f64] = &[
    5.0 / 108.0,
    0.0,
    35.0 / 72.0,
    -59.0 / 216.0,
    -25.0 / 27.0,
    3.0 / 2.0,
];
const MSRK5_A8: &[f64] = &[
    31.0 / 128.0,
    0.0,
    -7563.0 / 4480.0,
    233.0 / 112.0,
    3461.0 / 2240.0,
    -765.0 / 448.0,
    153.0 / 320.0,
];
const MSRK5_B: &[f64] = &[
    29.0 / 456.0,
    0.0,
    0.0,
    11.0 / 38.0,
    2.0 / 27.0,
    11.0 / 40.0,
    4.0 / 19.0,
    224.0 / 2565.0,
    0.0,
];
const MSRK5_FSAL_ROW: &[f64] = &[
    29.0 / 456.0,
    0.0,
    0.0,
    11.0 / 38.0,
    2.0 / 27.0,
    11.0 / 40.0,
    4.0 / 19.0,
    224.0 / 2565.0,
];
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
const MSRK5_C: &[f64] = &[
    0.0,
    4.0 / 45.0,
    2.0 / 15.0,
    1.0 / 5.0,
    1.0 / 2.0,
    8.0 / 15.0,
    5.0 / 6.0,
    19.0 / 20.0,
    1.0,
];

// Misha Stepanov's eight-stage sixth-order fixed-step method. The tableau is
// copied from OrdinaryDiffEqLowOrderRK's `MSRK6ConstantCache` at the pinned
// upstream revision. Unlike MSRK5, MSRK6 is not FSAL in the upstream
// algorithm, so the final stage is the eighth derivative used in the update.
const MSRK6_A2: &[f64] = &[1.0 / 14.0];
const MSRK6_A3: &[f64] = &[0.0, 1.0 / 7.0];
const MSRK6_A4: &[f64] = &[3.0 / 56.0, 0.0, 9.0 / 56.0];
const MSRK6_A5: &[f64] = &[29.0 / 72.0, 0.0, -35.0 / 24.0, 14.0 / 9.0];
const MSRK6_A6: &[f64] = &[-17.0 / 56.0, 0.0, 93.0 / 56.0, -8.0 / 7.0, 3.0 / 7.0];
const MSRK6_A7: &[f64] = &[
    199.0 / 1372.0,
    0.0,
    -195.0 / 196.0,
    1259.0 / 784.0,
    -3855.0 / 5488.0,
    45.0 / 56.0,
];
const MSRK6_A8: &[f64] = &[
    4903.0 / 25596.0,
    0.0,
    4487.0 / 2844.0,
    -255101.0 / 102384.0,
    33847.0 / 11376.0,
    -94325.0 / 51192.0,
    3773.0 / 6399.0,
];
const MSRK6_A: &[&[f64]] = &[
    EMPTY, MSRK6_A2, MSRK6_A3, MSRK6_A4, MSRK6_A5, MSRK6_A6, MSRK6_A7, MSRK6_A8,
];
const MSRK6_B: &[f64] = &[
    16.0 / 243.0,
    0.0,
    0.0,
    16807.0 / 53460.0,
    53.0 / 300.0,
    2401.0 / 12150.0,
    2401.0 / 12150.0,
    79.0 / 1650.0,
];
const MSRK6_C: &[f64] = &[
    0.0,
    1.0 / 14.0,
    1.0 / 7.0,
    3.0 / 14.0,
    1.0 / 2.0,
    9.0 / 14.0,
    6.0 / 7.0,
    1.0,
];

const RALSTON4_A2: &[f64] = &[0.4];
const RALSTON4_A3: &[f64] = &[0.296_977_609_247_753_57, 0.158_759_644_971_035_84];
const RALSTON4_A4: &[f64] = &[
    0.218_100_388_225_920_04,
    -3.050_965_148_692_930_6,
    3.832_864_760_467_010_5,
];
const RALSTON4_A: &[&[f64]] = &[EMPTY, RALSTON4_A2, RALSTON4_A3, RALSTON4_A4];
const RALSTON4_B: &[f64] = &[
    0.174_760_282_262_690_4,
    -0.551_480_662_878_733,
    1.205_535_599_396_523_5,
    0.171_184_781_219_519_02,
];
const RALSTON4_C: &[f64] = &[0.0, 0.4, 0.455_737_254_218_789_4, 1.0];

const ALSHINA2_E: &[f64] = &[1.0, 0.0];
const ALSHINA3_A3: &[f64] = &[0.0, 0.75];
const ALSHINA3_A: &[&[f64]] = &[EMPTY, MIDPOINT_A2, ALSHINA3_A3];
const ALSHINA3_B: &[f64] = &[2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0];
const ALSHINA3_E: &[f64] = &[0.0, 4.0 / 9.0, 0.0];
const ALSHINA3_C: &[f64] = &[0.0, 0.5, 0.75];

// Alshina's optimal sixth-order, seven-stage fixed-step scheme. The
// coefficients are copied from OrdinaryDiffEqLowOrderRK's
// `Alshina6ConstantCache` at the pinned upstream revision. The final update
// uses only stages 1, 5, 6, and 7 (the omitted b2-b4 entries are zero).
const ALSHINA6_A2: &[f64] = &[0.571_428_571_428_571_4];
const ALSHINA6_A3: &[f64] = &[1.026_785_714_285_714_2, -0.312_5];
const ALSHINA6_A4: &[f64] = &[
    0.934_920_634_920_634_9,
    0.277_777_777_777_777_8,
    -0.355_555_555_555_555_57,
];
const ALSHINA6_A5: &[f64] = &[
    0.180_025_671_442_084_34,
    0.147_379_406_839_616_14,
    0.016_070_163_293_314_288,
    -0.067_082_039_324_993_7,
];
const ALSHINA6_A6: &[f64] = &[
    -0.079_797_658_566_031_38,
    0.025_290_591_998_992_584,
    -0.351_632_620_226_681_3,
    0.320_729_490_168_751_6,
    0.809_016_994_374_947_5,
];
const ALSHINA6_A7: &[f64] = &[
    0.498_859_935_619_735_2,
    -0.863_349_994_193_042_9,
    1.677_812_284_666_834_9,
    -1.268_237_254_218_789_4,
    -0.427_050_983_124_842_35,
    1.381_966_011_250_105,
];
const ALSHINA6_A: &[&[f64]] = &[
    EMPTY,
    ALSHINA6_A2,
    ALSHINA6_A3,
    ALSHINA6_A4,
    ALSHINA6_A5,
    ALSHINA6_A6,
    ALSHINA6_A7,
];
const ALSHINA6_B: &[f64] = &[
    1.0 / 12.0,
    0.0,
    0.0,
    0.0,
    5.0 / 12.0,
    5.0 / 12.0,
    1.0 / 12.0,
];
const ALSHINA6_C: &[f64] = &[
    0.0,
    0.571_428_571_428_571_4,
    0.714_285_714_285_714_3,
    0.857_142_857_142_857_1,
    0.276_393_202_250_021,
    0.723_606_797_749_978_9,
    1.0,
];

const BS3_A: &[&[f64]] = BS3_A_ROWS;
const BS3_B: &[f64] = &GENERATED_BS3_B;
const BS3_E: &[f64] = &GENERATED_BS3_E;
const BS3_C: &[f64] = &BS3_STAGE_TIMES;

const DP5_A: &[&[f64]] = DP5_A_ROWS;
const DP5_B: &[f64] = &GENERATED_DP5_B;
const DP5_E: &[f64] = &GENERATED_DP5_E;
const DP5_C: &[f64] = &DP5_STAGE_TIMES;

// Owren-Zennaro 3/2 pair.
const OWREN_ZEN3_A2: &[f64] = &[12.0 / 23.0];
const OWREN_ZEN3_A3: &[f64] = &[-68.0 / 375.0, 368.0 / 375.0];
const OWREN_ZEN3_A4: &[f64] = &[31.0 / 144.0, 529.0 / 1_152.0, 125.0 / 384.0];
const OWREN_ZEN3_A: &[&[f64]] = &[EMPTY, OWREN_ZEN3_A2, OWREN_ZEN3_A3, OWREN_ZEN3_A4];
const OWREN_ZEN3_B: &[f64] = &[31.0 / 144.0, 529.0 / 1_152.0, 125.0 / 384.0, 0.0];
const OWREN_ZEN3_E: &[f64] = &[-25.0 / 144.0, 575.0 / 1_152.0, -125.0 / 384.0, 0.0];
const OWREN_ZEN3_C: &[f64] = &[0.0, 12.0 / 23.0, 4.0 / 5.0, 1.0];

// Owren-Zennaro 4/3 pair.
const OWREN_ZEN4_A2: &[f64] = &[1.0 / 6.0];
const OWREN_ZEN4_A3: &[f64] = &[44.0 / 1_369.0, 363.0 / 1_369.0];
const OWREN_ZEN4_A4: &[f64] = &[3_388.0 / 4_913.0, -8_349.0 / 4_913.0, 8_140.0 / 4_913.0];
const OWREN_ZEN4_A5: &[f64] = &[
    -36_764.0 / 408_375.0,
    767.0 / 1_125.0,
    -32_708.0 / 136_125.0,
    210_392.0 / 408_375.0,
];
const OWREN_ZEN4_A6: &[f64] = &[
    1_697.0 / 18_876.0,
    0.0,
    50_653.0 / 116_160.0,
    299_693.0 / 1_626_240.0,
    3_375.0 / 11_648.0,
];
const OWREN_ZEN4_A: &[&[f64]] = &[
    EMPTY,
    OWREN_ZEN4_A2,
    OWREN_ZEN4_A3,
    OWREN_ZEN4_A4,
    OWREN_ZEN4_A5,
    OWREN_ZEN4_A6,
];
const OWREN_ZEN4_B: &[f64] = &[
    1_697.0 / 18_876.0,
    0.0,
    50_653.0 / 116_160.0,
    299_693.0 / 1_626_240.0,
    3_375.0 / 11_648.0,
    0.0,
];
const OWREN_ZEN4_E: &[f64] = &[
    1_185.0 / 6_292.0,
    0.0,
    -4_107.0 / 7_744.0,
    68_493.0 / 108_416.0,
    -3_375.0 / 11_648.0,
    0.0,
];
const OWREN_ZEN4_C: &[f64] = &[0.0, 1.0 / 6.0, 11.0 / 37.0, 11.0 / 17.0, 13.0 / 15.0, 1.0];

// Owren-Zennaro 5/4 pair.
const OWREN_ZEN5_A2: &[f64] = &[1.0 / 6.0];
const OWREN_ZEN5_A3: &[f64] = &[1.0 / 16.0, 3.0 / 16.0];
const OWREN_ZEN5_A4: &[f64] = &[1.0 / 4.0, -3.0 / 4.0, 1.0];
const OWREN_ZEN5_A5: &[f64] = &[-3.0 / 4.0, 15.0 / 4.0, -3.0, 1.0 / 2.0];
const OWREN_ZEN5_A6: &[f64] = &[
    369.0 / 1_372.0,
    -243.0 / 343.0,
    297.0 / 343.0,
    1_485.0 / 9_604.0,
    297.0 / 4_802.0,
];
const OWREN_ZEN5_A7: &[f64] = &[
    -133.0 / 4_512.0,
    1_113.0 / 6_016.0,
    7_945.0 / 16_544.0,
    -12_845.0 / 24_064.0,
    -315.0 / 24_064.0,
    156_065.0 / 198_528.0,
];
const OWREN_ZEN5_A8: &[f64] = &[
    83.0 / 945.0,
    0.0,
    248.0 / 825.0,
    41.0 / 180.0,
    1.0 / 36.0,
    2_401.0 / 38_610.0,
    6_016.0 / 20_475.0,
];
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
const OWREN_ZEN5_B: &[f64] = &[
    83.0 / 945.0,
    0.0,
    248.0 / 825.0,
    41.0 / 180.0,
    1.0 / 36.0,
    2_401.0 / 38_610.0,
    6_016.0 / 20_475.0,
    0.0,
];
const OWREN_ZEN5_E: &[f64] = &[
    -188.0 / 945.0,
    0.0,
    752.0 / 825.0,
    -89.0 / 45.0,
    -1.0 / 9.0,
    32_242.0 / 19_305.0,
    -6_016.0 / 20_475.0,
    0.0,
];
const OWREN_ZEN5_C: &[f64] = &[
    0.0,
    1.0 / 6.0,
    1.0 / 4.0,
    1.0 / 2.0,
    1.0 / 2.0,
    9.0 / 14.0,
    7.0 / 8.0,
    1.0,
];

// Bogacki-Shampine 5/4 pair. Its controller uses the maximum of two embedded
// estimators, represented by ERROR_WEIGHTS and SECOND_ERROR_WEIGHTS.
const BS5_A2: &[f64] = &[1.0 / 6.0];
const BS5_A3: &[f64] = &[2.0 / 27.0, 4.0 / 27.0];
const BS5_A4: &[f64] = &[183.0 / 1_372.0, -162.0 / 343.0, 1_053.0 / 1_372.0];
const BS5_A5: &[f64] = &[68.0 / 297.0, -4.0 / 11.0, 42.0 / 143.0, 1_960.0 / 3_861.0];
const BS5_A6: &[f64] = &[
    597.0 / 22_528.0,
    81.0 / 352.0,
    63_099.0 / 585_728.0,
    58_653.0 / 366_080.0,
    4_617.0 / 20_480.0,
];
const BS5_A7: &[f64] = &[
    174_197.0 / 959_244.0,
    -30_942.0 / 79_937.0,
    8_152_137.0 / 19_744_439.0,
    666_106.0 / 1_039_181.0,
    -29_421.0 / 29_068.0,
    482_048.0 / 414_219.0,
];
const BS5_A8: &[f64] = &[
    587.0 / 8_064.0,
    0.0,
    4_440_339.0 / 15_491_840.0,
    24_353.0 / 124_800.0,
    387.0 / 44_800.0,
    2_152.0 / 5_985.0,
    7_267.0 / 94_080.0,
];
const BS5_A: &[&[f64]] = &[
    EMPTY, BS5_A2, BS5_A3, BS5_A4, BS5_A5, BS5_A6, BS5_A7, BS5_A8,
];
const BS5_B: &[f64] = &[
    587.0 / 8_064.0,
    0.0,
    4_440_339.0 / 15_491_840.0,
    24_353.0 / 124_800.0,
    387.0 / 44_800.0,
    2_152.0 / 5_985.0,
    7_267.0 / 94_080.0,
    0.0,
];
const BS5_E1: &[f64] = &[
    -3.0 / 1_280.0,
    0.0,
    6_561.0 / 632_320.0,
    -343.0 / 20_800.0,
    243.0 / 12_800.0,
    -1.0 / 95.0,
    0.0,
    0.0,
];
const BS5_E2: &[f64] = &[
    -3_817.0 / 1_959_552.0,
    0.0,
    140_181.0 / 15_491_840.0,
    -4_224_731.0 / 272_937_600.0,
    8_557.0 / 403_200.0,
    -57_928.0 / 4_363_065.0,
    -23_930_231.0 / 4_366_535_040.0,
    3_293.0 / 556_956.0,
];
const BS5_C: &[f64] = &[
    0.0,
    1.0 / 6.0,
    2.0 / 9.0,
    3.0 / 7.0,
    2.0 / 3.0,
    3.0 / 4.0,
    1.0,
    1.0,
];

const SSPRK22_A: &[&[f64]] = &[EMPTY, HEUN_A2];
const SSPRK22_B: &[f64] = HEUN_B;
const SSPRK22_C: &[f64] = HEUN_C;

const SSPRK33_A2: &[f64] = &[1.0];
const SSPRK33_A3: &[f64] = &[0.25, 0.25];
const SSPRK33_A: &[&[f64]] = &[EMPTY, SSPRK33_A2, SSPRK33_A3];
const SSPRK33_B: &[f64] = &[1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0];
const SSPRK33_C: &[f64] = &[0.0, 1.0, 0.5];

const SSPRK43_A2: &[f64] = &[0.5];
const SSPRK43_A3: &[f64] = &[0.5, 0.5];
const SSPRK43_A4: &[f64] = &[1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0];
const SSPRK43_A: &[&[f64]] = &[EMPTY, SSPRK43_A2, SSPRK43_A3, SSPRK43_A4];
const SSPRK43_B: &[f64] = &[1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 0.5];
const SSPRK43_E: &[f64] = &[-1.0 / 12.0, -1.0 / 12.0, -1.0 / 12.0, 0.25];
const SSPRK43_C: &[f64] = &[0.0, 0.5, 1.0, 0.5];

macro_rules! algorithm {
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
        #[doc = $documentation]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        pub struct $name;

        impl ButcherTableau for $name {
            const NODES: &'static [f64] = $nodes;
            const COEFFICIENTS: &'static [&'static [f64]] = $coefficients;
            const WEIGHTS: &'static [f64] = $weights;
            const ERROR_WEIGHTS: Option<&'static [f64]> = $error_weights;
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

algorithm!(
    Euler,
    "The fixed-step forward Euler method.",
    nodes = EULER_C,
    coefficients = EULER_A,
    weights = EULER_B,
    error_weights = None,
    order = 1,
    fsal = false
);
algorithm!(
    Midpoint,
    "The adaptive second-order explicit midpoint method with an embedded Euler estimate.",
    nodes = MIDPOINT_C,
    coefficients = MIDPOINT_A,
    weights = MIDPOINT_B,
    error_weights = Some(MIDPOINT_E),
    order = 2,
    fsal = false
);
algorithm!(
    Heun,
    "The adaptive second-order explicit trapezoid (Heun) method.",
    nodes = HEUN_C,
    coefficients = HEUN_A,
    weights = HEUN_B,
    error_weights = Some(HEUN_E),
    order = 2,
    fsal = false
);
algorithm!(
    Ralston,
    "Ralston's adaptive second-order explicit Runge–Kutta method.",
    nodes = RALSTON_C,
    coefficients = RALSTON_A,
    weights = RALSTON_B,
    error_weights = Some(RALSTON_E),
    order = 2,
    fsal = false
);
algorithm!(
    Rk4,
    "The fixed-step classical fourth-order Runge–Kutta method.",
    nodes = RK4_C,
    coefficients = RK4_A,
    weights = RK4_B,
    error_weights = None,
    order = 4,
    fsal = false
);
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
    fsal = false
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
algorithm!(
    Alshina2,
    "The adaptive optimal two-stage, second-order Alshina method.",
    nodes = RALSTON_C,
    coefficients = RALSTON_A,
    weights = RALSTON_B,
    error_weights = Some(ALSHINA2_E),
    order = 2,
    fsal = false
);
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
    order = 3,
    fsal = false
);

struct Workspace {
    // Flat stage-major storage: every stage is one contiguous component array.
    // The other work vectors remain separate arrays rather than per-component
    // structs, keeping the hot saxpy-style loops friendly to SIMD.
    stages: Vec<f64>,
    stage_count: usize,
    dimension: usize,
    temporary: Vec<f64>,
}

impl Workspace {
    fn new(stage_count: usize, dimension: usize) -> Self {
        Self {
            stages: vec![0.0; stage_count * dimension],
            stage_count,
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
    let coefficients_finite = T::NODES.iter().all(|value| value.is_finite())
        && T::WEIGHTS.iter().all(|value| value.is_finite())
        && T::COEFFICIENTS
            .iter()
            .flat_map(|row| row.iter())
            .all(|value| value.is_finite())
        && T::ERROR_WEIGHTS.is_none_or(|weights| weights.iter().all(|value| value.is_finite()));
    let second_error_estimator_finite =
        T::SECOND_ERROR_WEIGHTS.is_none_or(|weights| weights.iter().all(|value| value.is_finite()));
    let fsal_valid = !T::FSAL
        || (stage_count > 0
            && T::NODES.last() == Some(&1.0)
            && T::WEIGHTS.last() == Some(&0.0)
            && T::COEFFICIENTS
                .last()
                .is_some_and(|last_row| *last_row == &T::WEIGHTS[..stage_count - 1]));

    (structurally_valid
        && error_estimators_valid
        && coefficients_finite
        && second_error_estimator_finite
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
    marker: PhantomData<fn() -> T>,
}

impl<T> ExplicitKernel<T> {
    fn new(dimension: usize) -> Self
    where
        T: ButcherTableau,
    {
        Self {
            workspace: Workspace::new(T::WEIGHTS.len(), dimension),
            stage_zero_is_current: false,
            marker: PhantomData,
        }
    }
}

impl<F, P, T> StepKernel<F, P> for ExplicitKernel<T>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
    T: ButcherTableau,
{
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
        ensure_finite(candidate)?;
        let error = if options.adaptive {
            let primary_error = error_norm(
                &self.workspace.stages,
                self.workspace.dimension,
                (state, candidate),
                step,
                options,
                T::ERROR_WEIGHTS.expect("driver checked adaptive capability"),
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

    fn record_dense_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        previous_state: &[f64],
        state: &[f64],
        previous_time: f64,
        time: f64,
        final_time: bool,
        recorder: &mut TrajectoryRecorder<'_>,
        stats: &mut SolverStats,
    ) -> Result<bool, SolveError> {
        // The initial stage is the derivative at the accepted step's left
        // endpoint. Reuse the workspace error scratch for the right-endpoint
        // derivative so dense save-at adds no per-step allocation.
        evaluate(problem, &mut self.workspace.temporary, state, time, stats);
        ensure_finite(&self.workspace.temporary)?;
        let segment = BorrowedHermiteSegment::new(
            previous_time,
            time,
            previous_state,
            state,
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
            self.workspace
                .swap_stages(0, self.workspace.stage_count - 1);
            self.stage_zero_is_current = true;
        } else {
            self.stage_zero_is_current = false;
        }
        Ok(())
    }

    fn reject_step(&mut self) {
        self.stage_zero_is_current = true;
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
    for stage_index in 1..workspace.stage_count {
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
        workspace.stage_count,
        T::WEIGHTS,
    );
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

    use super::{Bs5, ButcherTableau, ExplicitRungeKutta, OwrenZen3, OwrenZen4, OwrenZen5};
    use crate::{
        Alshina2, Alshina3, Alshina6, Bs3, Dp5, Euler, Heun, Midpoint, OdeProblem, Ralston,
        Ralston4, Rk4, Rkm, SaveMode, SolveError, SolveOptions, SspRk22, SspRk33, SspRk43, solve,
    };

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
