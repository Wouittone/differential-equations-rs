//! Canonical compile-time coefficient fixtures used by the coefficient generator.
//!
//! The machine-readable method records below generate the checked-in manifest.
//! Run `scripts/generate_coefficients.ps1` after changing this file and use
//! `scripts/generate_coefficients.ps1 -Check` to detect drift.

#![allow(
    dead_code,
    reason = "not every canonical fixture is wired into a solver yet"
)]
#![allow(
    clippy::excessive_precision,
    reason = "canonical upstream f64 coefficient literals"
)]

// coefficient-method: method=AB3|family=multistep|order=3|variable-step=false
// coefficient-method: method=ABDF2|family=multistep|order=2|variable-step=true
// coefficient-method: method=BS3|family=explicit|order=3|embedded-order=2|fsal=true
// coefficient-method: method=DP5|family=explicit|order=5|embedded-order=4|fsal=true
// coefficient-method: method=Euler|family=explicit|order=1|embedded-order=none
// coefficient-method: method=Heun|family=explicit|order=2|embedded-order=1
// coefficient-method: method=Midpoint|family=explicit|order=2|embedded-order=1
// coefficient-method: method=SDIRK2|family=sdirk|order=2|embedded-order=1
// coefficient-method: method=VelocityVerlet|family=symplectic|order=2|embedded-order=none
// coefficient-method: method=Vern6|family=explicit|order=6|embedded-order=5|fsal=true
// coefficient-method: method=Vern7|family=explicit|order=7|embedded-order=6|fsal=true
// coefficient-method: method=Vern8|family=explicit|order=8|embedded-order=7|fsal=true
// coefficient-method: method=Vern9|family=explicit|order=9|embedded-order=8|fsal=true

pub(crate) const EULER_STAGE_TIMES: [f64; 1] = [0.0];
pub(crate) const EULER_A: [[f64; 1]; 1] = [[0.0]];
pub(crate) const EULER_EMPTY: &[f64] = &[];
pub(crate) const EULER_A_ROWS: &[&[f64]] = &[EULER_EMPTY];
pub(crate) const EULER_B: [f64; 1] = [1.0];

pub(crate) const HEUN_STAGE_TIMES: [f64; 2] = [0.0, 1.0];
pub(crate) const HEUN_A: [[f64; 2]; 2] = [[0.0, 0.0], [1.0, 0.0]];
pub(crate) const HEUN_EMPTY: &[f64] = &[];
pub(crate) const HEUN_A2: &[f64] = &[1.0];
pub(crate) const HEUN_A_ROWS: &[&[f64]] = &[HEUN_EMPTY, HEUN_A2];
pub(crate) const HEUN_B: [f64; 2] = [0.5, 0.5];
pub(crate) const HEUN_ERROR: [f64; 2] = [-0.5, 0.5];

pub(crate) const MIDPOINT_STAGE_TIMES: [f64; 2] = [0.0, 0.5];
pub(crate) const MIDPOINT_A: [[f64; 2]; 2] = [[0.0, 0.0], [0.5, 0.0]];
pub(crate) const MIDPOINT_EMPTY: &[f64] = &[];
pub(crate) const MIDPOINT_A2: &[f64] = &[0.5];
pub(crate) const MIDPOINT_A_ROWS: &[&[f64]] = &[MIDPOINT_EMPTY, MIDPOINT_A2];
pub(crate) const MIDPOINT_B: [f64; 2] = [0.0, 1.0];
pub(crate) const MIDPOINT_ERROR: [f64; 2] = [-1.0, 1.0];

/// Bogacki-Shampine 3(2) explicit tableau from OrdinaryDiffEqLowOrderRK.
/// The fourth stage is FSAL and is retained in the generated rows so the
/// shared explicit driver can reuse it after accepted steps.
pub(crate) const BS3_STAGE_TIMES: [f64; 4] = [0.0, 0.5, 0.75, 1.0];
pub(crate) const BS3_A_EMPTY: &[f64] = &[];
pub(crate) const BS3_A2: &[f64] = &[0.5];
pub(crate) const BS3_A3: &[f64] = &[0.0, 0.75];
pub(crate) const BS3_A4: &[f64] = &[2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0];
pub(crate) const BS3_A_ROWS: &[&[f64]] = &[BS3_A_EMPTY, BS3_A2, BS3_A3, BS3_A4];
pub(crate) const BS3_B: [f64; 4] = [2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0];
pub(crate) const BS3_E: [f64; 4] = [-5.0 / 72.0, 1.0 / 12.0, 1.0 / 9.0, -1.0 / 8.0];

/// Dormand-Prince 5(4) tableau from OrdinaryDiffEqLowOrderRK. The final
/// stage is FSAL; the generated record retains the embedded defect weights
/// used by the adaptive shared driver.
pub(crate) const DP5_STAGE_TIMES: [f64; 7] =
    [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];
pub(crate) const DP5_A_EMPTY: &[f64] = &[];
pub(crate) const DP5_A2: &[f64] = &[1.0 / 5.0];
pub(crate) const DP5_A3: &[f64] = &[3.0 / 40.0, 9.0 / 40.0];
pub(crate) const DP5_A4: &[f64] = &[44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0];
pub(crate) const DP5_A5: &[f64] = &[
    19_372.0 / 6_561.0,
    -25_360.0 / 2_187.0,
    64_448.0 / 6_561.0,
    -212.0 / 729.0,
];
pub(crate) const DP5_A6: &[f64] = &[
    9_017.0 / 3_168.0,
    -355.0 / 33.0,
    46_732.0 / 5_247.0,
    49.0 / 176.0,
    -5_103.0 / 18_656.0,
];
pub(crate) const DP5_A7: &[f64] = &[
    35.0 / 384.0,
    0.0,
    500.0 / 1_113.0,
    125.0 / 192.0,
    -2_187.0 / 6_784.0,
    11.0 / 84.0,
];
pub(crate) const DP5_A_ROWS: &[&[f64]] =
    &[DP5_A_EMPTY, DP5_A2, DP5_A3, DP5_A4, DP5_A5, DP5_A6, DP5_A7];
pub(crate) const DP5_B: [f64; 7] = [
    35.0 / 384.0,
    0.0,
    500.0 / 1_113.0,
    125.0 / 192.0,
    -2_187.0 / 6_784.0,
    11.0 / 84.0,
    0.0,
];
pub(crate) const DP5_E: [f64; 7] = [
    35.0 / 384.0 - 5_179.0 / 57_600.0,
    0.0,
    500.0 / 1_113.0 - 7_571.0 / 16_695.0,
    125.0 / 192.0 - 393.0 / 640.0,
    -2_187.0 / 6_784.0 + 92_097.0 / 339_200.0,
    11.0 / 84.0 - 187.0 / 2_100.0,
    -1.0 / 40.0,
];

/// Vern6 6(5) tableau from OrdinaryDiffEqVerner's pinned Float64 cache.
pub(crate) const VERN6_STAGE_TIMES: [f64; 9] = [
    0.0,
    0.06,
    0.09593333333333333,
    0.1439,
    0.4973,
    0.9725,
    0.9995,
    1.0,
    1.0,
];
pub(crate) const VERN6_A_EMPTY: &[f64] = &[];
pub(crate) const VERN6_A2: &[f64] = &[0.06];
pub(crate) const VERN6_A3: &[f64] = &[0.019239962962962962, 0.07669337037037037];
pub(crate) const VERN6_A4: &[f64] = &[0.035975, 0.0, 0.107925];
pub(crate) const VERN6_A5: &[f64] = &[
    1.3186834152331484,
    0.0,
    -5.042058063628562,
    4.220674648395414,
];
pub(crate) const VERN6_A6: &[f64] = &[
    -41.87259166432751,
    0.0,
    159.43256216313748,
    -122.11921356501004,
    5.531743066200053,
];
pub(crate) const VERN6_A7: &[f64] = &[
    -54.430156935316504,
    0.0,
    207.06725136501848,
    -158.61081378459,
    6.991816585950242,
    -0.01859723106220323,
];
pub(crate) const VERN6_A8: &[f64] = &[
    -54.66374178728198,
    0.0,
    207.95280625538936,
    -159.2889574744995,
    7.018743740796944,
    -0.018338785905045722,
    -0.0005119484997882099,
];
pub(crate) const VERN6_A9: &[f64] = &[
    0.03438957868357036,
    0.0,
    0.0,
    0.25826245556335037,
    0.4209371189673537,
    4.40539646966931,
    -176.48311902429865,
    172.36413340141507,
];
pub(crate) const VERN6_A_ROWS: &[&[f64]] = &[
    VERN6_A_EMPTY,
    VERN6_A2,
    VERN6_A3,
    VERN6_A4,
    VERN6_A5,
    VERN6_A6,
    VERN6_A7,
    VERN6_A8,
    VERN6_A9,
];
pub(crate) const VERN6_B: [f64; 9] = [
    0.03438957868357036,
    0.0,
    0.0,
    0.25826245556335037,
    0.4209371189673537,
    4.40539646966931,
    -176.48311902429865,
    172.36413340141507,
    0.0,
];
pub(crate) const VERN6_E: [f64; 9] = [
    0.008623404282200854,
    0.0,
    0.0,
    -0.019434029953152708,
    0.028450072588037983,
    -2.1097110610652914,
    103.45854289996397,
    -101.39980461914912,
    0.03333333333333333,
];

/// Vern7 7(6) tableau from OrdinaryDiffEqVerner's pinned Float64 cache.
pub(crate) const VERN7_STAGE_TIMES: [f64; 10] = [
    0.0,
    0.005,
    0.10888888888888888,
    0.16333333333333333,
    0.4555,
    0.6095094489978381,
    0.884,
    0.925,
    1.0,
    1.0,
];
pub(crate) const VERN7_A_EMPTY: &[f64] = &[];
pub(crate) const VERN7_A2: &[f64] = &[0.005];
pub(crate) const VERN7_A3: &[f64] = &[-1.07679012345679, 1.185679012345679];
pub(crate) const VERN7_A4: &[f64] = &[0.04083333333333333, 0.0, 0.1225];
pub(crate) const VERN7_A5: &[f64] = &[
    0.6389139236255726,
    0.0,
    -2.455672638223657,
    2.272258714598084,
];
pub(crate) const VERN7_A6: &[f64] = &[
    -2.6615773750187572,
    0.0,
    10.804513886456137,
    -8.3539146573962,
    0.820487594956657,
];
pub(crate) const VERN7_A7: &[f64] = &[
    6.067741434696772,
    0.0,
    -24.711273635911088,
    20.427517930788895,
    -1.9061579788166472,
    1.006172249242068,
];
pub(crate) const VERN7_A8: &[f64] = &[
    12.054670076253203,
    0.0,
    -49.75478495046899,
    41.142888638604674,
    -4.461760149974004,
    2.042334822239175,
    -0.09834843665406107,
];
pub(crate) const VERN7_A9: &[f64] = &[
    10.138146522881808,
    0.0,
    -42.6411360317175,
    35.76384003992257,
    -4.3480228403929075,
    2.0098622683770357,
    0.3487490460338272,
    -0.27143900510483127,
];
pub(crate) const VERN7_A10: &[f64] = &[
    -45.030072034298676,
    0.0,
    187.3272437654589,
    -154.02882369350186,
    18.56465306347536,
    -7.141809679295079,
    1.3088085781613787,
    0.0,
    0.0,
];
pub(crate) const VERN7_A_ROWS: &[&[f64]] = &[
    VERN7_A_EMPTY,
    VERN7_A2,
    VERN7_A3,
    VERN7_A4,
    VERN7_A5,
    VERN7_A6,
    VERN7_A7,
    VERN7_A8,
    VERN7_A9,
    VERN7_A10,
];
pub(crate) const VERN7_B: [f64; 10] = [
    0.04715561848627222,
    0.0,
    0.0,
    0.25750564298434153,
    0.26216653977412624,
    0.15216092656738558,
    0.4939969170032485,
    -0.29430311714032503,
    0.08131747232495111,
    0.0,
];
pub(crate) const VERN7_E: [f64; 10] = [
    0.002547011879931045,
    0.0,
    0.0,
    -0.00965839487279575,
    0.04206470975639691,
    -0.0666822437469301,
    0.2650097464621281,
    -0.29430311714032503,
    0.08131747232495111,
    -0.02029518466335628,
];

/// Vern8 8(7) tableau from OrdinaryDiffEqVerner's pinned Float64 cache.
pub(crate) const VERN8_STAGE_TIMES: [f64; 13] = [
    0.0,
    0.05,
    0.1065625,
    0.15984375,
    0.39,
    0.465,
    0.155,
    0.943,
    0.901802041735857,
    0.909,
    0.94,
    1.0,
    1.0,
];
pub(crate) const VERN8_A_EMPTY: &[f64] = &[];
pub(crate) const VERN8_A2: &[f64] = &[0.05];
pub(crate) const VERN8_A3: &[f64] = &[-0.0069931640625, 0.1135556640625];
pub(crate) const VERN8_A4: &[f64] = &[0.0399609375, 0.0, 0.1198828125];
pub(crate) const VERN8_A5: &[f64] = &[
    0.36139756280045754,
    0.0,
    -1.3415240667004928,
    1.3701265039000352,
];
pub(crate) const VERN8_A6: &[f64] = &[
    0.049047202797202795,
    0.0,
    0.0,
    0.23509720422144048,
    0.18085559298135673,
];
pub(crate) const VERN8_A7: &[f64] = &[
    0.06169289044289044,
    0.0,
    0.0,
    0.11236568314640277,
    -0.03885046071451367,
    0.01979188712522046,
];
pub(crate) const VERN8_A8: &[f64] = &[
    -1.767630240222327,
    0.0,
    0.0,
    -62.5,
    -6.061889377376669,
    5.6508231982227635,
    65.62169641937624,
];
pub(crate) const VERN8_A9: &[f64] = &[
    -1.1809450665549708,
    0.0,
    0.0,
    -41.50473441114321,
    -4.434438319103725,
    4.260408188586133,
    43.75364022446172,
    0.00787142548991231,
];
pub(crate) const VERN8_A10: &[f64] = &[
    -1.2814059994414884,
    0.0,
    0.0,
    -45.047139960139866,
    -4.731362069449576,
    4.514967016593808,
    47.44909557172985,
    0.01059228297111661,
    -0.0057468422638446166,
];
pub(crate) const VERN8_A11: &[f64] = &[
    -1.7244701342624853,
    0.0,
    0.0,
    -60.92349008483054,
    -5.951518376222392,
    5.556523730698456,
    63.98301198033305,
    0.014642028250414961,
    0.06460408772358203,
    -0.0793032316900888,
];
pub(crate) const VERN8_A12: &[f64] = &[
    -3.301622667747079,
    0.0,
    0.0,
    -118.01127235975251,
    -10.141422388456112,
    9.139311332232058,
    123.37594282840426,
    4.62324437887458,
    -3.3832777380682018,
    4.527592100324618,
    -5.828495485811623,
];
pub(crate) const VERN8_A13: &[f64] = &[
    -3.039515033766309,
    0.0,
    0.0,
    -109.26086808941763,
    -9.290642497400293,
    8.43050498176491,
    114.20100103783314,
    -0.9637271342145479,
    -5.0348840888021895,
    5.958130824002923,
    0.0,
    0.0,
];
pub(crate) const VERN8_A_ROWS: &[&[f64]] = &[
    VERN8_A_EMPTY,
    VERN8_A2,
    VERN8_A3,
    VERN8_A4,
    VERN8_A5,
    VERN8_A6,
    VERN8_A7,
    VERN8_A8,
    VERN8_A9,
    VERN8_A10,
    VERN8_A11,
    VERN8_A12,
    VERN8_A13,
];
pub(crate) const VERN8_B: [f64; 13] = [
    0.04427989419007951,
    0.0,
    0.0,
    0.0,
    0.0,
    0.3541049391724449,
    0.24796921549564377,
    -15.694202038838085,
    25.084064965558564,
    -31.738367786260277,
    22.938283273988784,
    -0.2361324633071542,
    0.0,
];
pub(crate) const VERN8_E: [f64; 13] = [
    -3.272103901028138e-5,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.0005046250618777704,
    0.0001211723589784759,
    -20.142336771313868,
    5.2371785994398286,
    -8.156744408794658,
    22.938283273988784,
    -0.2361324633071542,
    0.36016794372897754,
];

/// Vern9 9(8) tableau from OrdinaryDiffEqVerner's pinned Float64 cache.
pub(crate) const VERN9_STAGE_TIMES: [f64; 16] = [
    0.0,
    0.03462,
    0.09702435063878045,
    0.14553652595817068,
    0.561,
    0.22900791159048503,
    0.544992088409515,
    0.645,
    0.48375,
    0.06757,
    0.25,
    0.6590650618730999,
    0.8206,
    0.9012,
    1.0,
    1.0,
];
pub(crate) const VERN9_A_EMPTY: &[f64] = &[];
pub(crate) const VERN9_A2: &[f64] = &[0.03462];
pub(crate) const VERN9_A3: &[f64] = &[-0.03893354388572875, 0.13595789452450918];
pub(crate) const VERN9_A4: &[f64] = &[0.03638413148954267, 0.0, 0.10915239446862801];
pub(crate) const VERN9_A5: &[f64] = &[
    2.0257639143939694,
    0.0,
    -7.638023836496291,
    6.173259922102322,
];
pub(crate) const VERN9_A6: &[f64] = &[
    0.05112275589406061,
    0.0,
    0.0,
    0.17708237945550218,
    0.0008027762409222536,
];
pub(crate) const VERN9_A7: &[f64] = &[
    0.13160063579752163,
    0.0,
    0.0,
    -0.2957276252669636,
    0.08781378035642955,
    0.6213052975225274,
];
pub(crate) const VERN9_A8: &[f64] = &[
    0.07166666666666667,
    0.0,
    0.0,
    0.0,
    0.0,
    0.33055335789153195,
    0.2427799754418014,
];
pub(crate) const VERN9_A9: &[f64] = &[
    0.071806640625,
    0.0,
    0.0,
    0.0,
    0.0,
    0.3294380283228177,
    0.1165190029271823,
    -0.034013671875,
];
pub(crate) const VERN9_A10: &[f64] = &[
    0.04836757646340646,
    0.0,
    0.0,
    0.0,
    0.0,
    0.03928989925676164,
    0.10547409458903446,
    -0.021438652846483126,
    -0.10412291746271944,
];
pub(crate) const VERN9_A11: &[f64] = &[
    -0.026645614872014785,
    0.0,
    0.0,
    0.0,
    0.0,
    0.03333333333333333,
    -0.1631072244872467,
    0.03396081684127761,
    0.1572319413814626,
    0.21522674780318796,
];
pub(crate) const VERN9_A12: &[f64] = &[
    0.03689009248708622,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.1465181576725543,
    0.2242577768172024,
    0.02294405717066073,
    -0.0035850052905728597,
    0.08669223316444385,
    0.43838406519683376,
];
pub(crate) const VERN9_A13: &[f64] = &[
    -0.4866012215113341,
    0.0,
    0.0,
    0.0,
    0.0,
    -6.304602650282853,
    -0.2812456182894729,
    -2.679019236219849,
    0.5188156639241577,
    1.3653531876033418,
    5.8850910885039465,
    2.8028087862720628,
];
pub(crate) const VERN9_A14: &[f64] = &[
    0.4185367457753472,
    0.0,
    0.0,
    0.0,
    0.0,
    6.724547581906459,
    -0.42544428016461133,
    3.3432791530012653,
    0.6170816631175374,
    -0.9299661239399329,
    -6.099948804751011,
    -3.002206187889399,
    0.2553202529443446,
];
pub(crate) const VERN9_A15: &[f64] = &[
    -0.7793740861228848,
    0.0,
    0.0,
    0.0,
    0.0,
    -13.937342538107776,
    1.2520488533793563,
    -14.691500408016868,
    -0.494705058533141,
    2.2429749091462368,
    13.367893803828643,
    14.396650486650687,
    -0.79758133317768,
    0.4409353709534278,
];
pub(crate) const VERN9_A16: &[f64] = &[
    2.0580513374668867,
    0.0,
    0.0,
    0.0,
    0.0,
    22.357937727968032,
    0.9094981099755646,
    35.89110098240264,
    -3.442515027624454,
    -4.865481358036369,
    -18.909803813543427,
    -34.26354448030452,
    1.2647565216956427,
    0.0,
    0.0,
];
pub(crate) const VERN9_A_ROWS: &[&[f64]] = &[
    VERN9_A_EMPTY,
    VERN9_A2,
    VERN9_A3,
    VERN9_A4,
    VERN9_A5,
    VERN9_A6,
    VERN9_A7,
    VERN9_A8,
    VERN9_A9,
    VERN9_A10,
    VERN9_A11,
    VERN9_A12,
    VERN9_A13,
    VERN9_A14,
    VERN9_A15,
    VERN9_A16,
];
pub(crate) const VERN9_B: [f64; 16] = [
    0.014611976858423152,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.3915211862331339,
    0.23109325002895065,
    0.12747667699928525,
    0.2246434176204158,
    0.5684352689748513,
    0.058258715572158275,
    0.13643174034822156,
    0.030570139830827976,
    0.0,
];
pub(crate) const VERN9_E: [f64; 16] = [
    -0.005357988290444578,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -2.583020491182464,
    0.14252253154686625,
    0.013420653512688676,
    -0.02867296291409493,
    2.624999655215792,
    -0.2825509643291537,
    0.13643174034822156,
    0.030570139830827976,
    -0.04834231373823958,
];

pub(crate) const AB3_HISTORY: [f64; 3] = [23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0];

/// Variable-step ABDF2 fixed-leading-coefficient constants.  The history
/// ratio enters the alpha terms at runtime; beta coefficients are invariant
/// except for the linear `(rho - 1)` correction.
pub(crate) const ABDF2_BETA_ZERO: f64 = 2.0 / 3.0;
pub(crate) const ABDF2_BETA_ONE_SCALE: f64 = -1.0 / 3.0;
pub(crate) const ABDF2_ALPHA_ONE_BASE: f64 = 1.0;
pub(crate) const ABDF2_ALPHA_HISTORY_SCALE: f64 = 1.0 / 3.0;

pub(crate) const VELOCITY_VERLET_COMPOSITION: [f64; 2] = [0.5, 0.5];

/// Pinned two-stage SDIRK2 ESDIRK tableau from OrdinaryDiffEqSDIRK.
///
/// The first stage is at `c₁ = 1`, the second at `c₂ = 0`; both stages have
/// unit diagonal and the second-stage explicit coupling is `a₂₁ = -1`.
pub(crate) const SDIRK2_A: [[f64; 2]; 2] = [[1.0, 0.0], [-1.0, 1.0]];
pub(crate) const SDIRK2_B: [f64; 2] = [0.5, 0.5];
pub(crate) const SDIRK2_B_EMBEDDED: [f64; 2] = [0.5, -0.5];
pub(crate) const SDIRK2_STAGE_TIMES: [f64; 2] = [1.0, 0.0];

#[cfg(test)]
mod tests {
    use super::{
        AB3_HISTORY, BS3_A_ROWS, BS3_B, BS3_E, BS3_STAGE_TIMES, DP5_A_ROWS, DP5_B, DP5_E,
        DP5_STAGE_TIMES, EULER_A, EULER_A_ROWS, EULER_B, EULER_STAGE_TIMES, HEUN_A, HEUN_A_ROWS,
        HEUN_B, HEUN_ERROR, HEUN_STAGE_TIMES, MIDPOINT_A, MIDPOINT_A_ROWS, MIDPOINT_B,
        MIDPOINT_ERROR, MIDPOINT_STAGE_TIMES, SDIRK2_A, SDIRK2_B, SDIRK2_B_EMBEDDED,
        SDIRK2_STAGE_TIMES, VELOCITY_VERLET_COMPOSITION, VERN6_A_ROWS, VERN6_B, VERN6_E,
        VERN6_STAGE_TIMES, VERN7_A_ROWS, VERN7_B, VERN7_E, VERN7_STAGE_TIMES, VERN8_A_ROWS,
        VERN8_B, VERN8_E, VERN8_STAGE_TIMES, VERN9_A_ROWS, VERN9_B, VERN9_E, VERN9_STAGE_TIMES,
    };

    #[test]
    fn generated_fixtures_have_expected_shapes() {
        assert_eq!(AB3_HISTORY.len(), 3);
        assert_eq!(BS3_A_ROWS.len(), BS3_STAGE_TIMES.len());
        assert_eq!(BS3_B.len(), BS3_STAGE_TIMES.len());
        assert_eq!(BS3_E.len(), BS3_STAGE_TIMES.len());
        assert!((BS3_B.iter().sum::<f64>() - 1.0).abs() < 1.0e-15);
        assert_eq!(DP5_A_ROWS.len(), DP5_STAGE_TIMES.len());
        assert_eq!(DP5_B.len(), DP5_STAGE_TIMES.len());
        assert_eq!(DP5_E.len(), DP5_STAGE_TIMES.len());
        assert!((DP5_B.iter().sum::<f64>() - 1.0).abs() < 1.0e-15);
        assert_eq!(VERN6_A_ROWS.len(), VERN6_STAGE_TIMES.len());
        assert_eq!(VERN6_B.len(), VERN6_STAGE_TIMES.len());
        assert_eq!(VERN6_E.len(), VERN6_STAGE_TIMES.len());
        assert!((VERN6_B.iter().sum::<f64>() - 1.0).abs() < 1.0e-13);
        assert_eq!(VERN7_A_ROWS.len(), VERN7_STAGE_TIMES.len());
        assert_eq!(VERN7_B.len(), VERN7_STAGE_TIMES.len());
        assert_eq!(VERN7_E.len(), VERN7_STAGE_TIMES.len());
        assert!((VERN7_B.iter().sum::<f64>() - 1.0).abs() < 1.0e-13);
        assert_eq!(VERN8_A_ROWS.len(), VERN8_STAGE_TIMES.len());
        assert_eq!(VERN8_B.len(), VERN8_STAGE_TIMES.len());
        assert_eq!(VERN8_E.len(), VERN8_STAGE_TIMES.len());
        assert!((VERN8_B.iter().sum::<f64>() - 1.0).abs() < 1.0e-13);
        assert_eq!(VERN9_A_ROWS.len(), VERN9_STAGE_TIMES.len());
        assert_eq!(VERN9_B.len(), VERN9_STAGE_TIMES.len());
        assert_eq!(VERN9_E.len(), VERN9_STAGE_TIMES.len());
        assert!((VERN9_B.iter().sum::<f64>() - 1.0).abs() < 1.0e-13);
        assert_eq!(VELOCITY_VERLET_COMPOSITION, [0.5, 0.5]);
        assert_eq!(SDIRK2_A.len(), SDIRK2_STAGE_TIMES.len());
        assert_eq!(SDIRK2_B.len(), SDIRK2_STAGE_TIMES.len());
        assert_eq!(SDIRK2_B_EMBEDDED.len(), SDIRK2_STAGE_TIMES.len());
        assert!((SDIRK2_B.iter().sum::<f64>() - 1.0).abs() < 1.0e-15);
    }

    #[test]
    fn generated_low_order_explicit_fixtures_match_their_tableau_shapes() {
        fn validate<const STAGES: usize>(
            dense_rows: &[[f64; STAGES]],
            lower_rows: &[&[f64]],
            weights: &[f64],
            stage_times: &[f64],
        ) {
            assert_eq!(dense_rows.len(), stage_times.len());
            assert_eq!(lower_rows.len(), stage_times.len());
            assert_eq!(weights.len(), stage_times.len());
            assert!(
                lower_rows
                    .iter()
                    .enumerate()
                    .all(|(stage, row)| row.len() == stage)
            );
            assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1.0e-15);
        }
        validate(&EULER_A, EULER_A_ROWS, &EULER_B, &EULER_STAGE_TIMES);
        validate(&HEUN_A, HEUN_A_ROWS, &HEUN_B, &HEUN_STAGE_TIMES);
        validate(
            &MIDPOINT_A,
            MIDPOINT_A_ROWS,
            &MIDPOINT_B,
            &MIDPOINT_STAGE_TIMES,
        );
        assert_eq!(HEUN_ERROR, [-0.5, 0.5]);
        assert_eq!(MIDPOINT_ERROR, [-1.0, 1.0]);
    }
}
