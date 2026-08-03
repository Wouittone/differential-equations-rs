//! Verner's efficient embedded explicit Runge--Kutta pairs.
//!
//! The coefficients are the compiled-`Float64` default-step coefficients from
//! `OrdinaryDiffEqVerner` at commit
//! `211142263781255a9aa2f910f6760b9f18ec29c8`. OrdinaryDiffEq's additional
//! stages for the method-specific dense interpolants are intentionally not part
//! of these tableaus.

use crate::explicit_rk::{ButcherTableau, ExplicitRungeKutta};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions};

const VERN6_C: &[f64] = &[
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
const VERN6_A1: &[f64] = &[];
const VERN6_A2: &[f64] = &[0.06];
const VERN6_A3: &[f64] = &[0.019239962962962962, 0.07669337037037037];
const VERN6_A4: &[f64] = &[0.035975, 0.0, 0.107925];
const VERN6_A5: &[f64] = &[
    1.3186834152331484,
    0.0,
    -5.042058063628562,
    4.220674648395414,
];
const VERN6_A6: &[f64] = &[
    -41.87259166432751,
    0.0,
    159.43256216313748,
    -122.11921356501004,
    5.531743066200053,
];
const VERN6_A7: &[f64] = &[
    -54.430156935316504,
    0.0,
    207.06725136501848,
    -158.61081378459,
    6.991816585950242,
    -0.01859723106220323,
];
const VERN6_A8: &[f64] = &[
    -54.66374178728198,
    0.0,
    207.95280625538936,
    -159.2889574744995,
    7.018743740796944,
    -0.018338785905045722,
    -0.0005119484997882099,
];
const VERN6_A9: &[f64] = &[
    0.03438957868357036,
    0.0,
    0.0,
    0.25826245556335037,
    0.4209371189673537,
    4.40539646966931,
    -176.48311902429865,
    172.36413340141507,
];
const VERN6_A: &[&[f64]] = &[
    VERN6_A1, VERN6_A2, VERN6_A3, VERN6_A4, VERN6_A5, VERN6_A6, VERN6_A7, VERN6_A8, VERN6_A9,
];
const VERN6_B: &[f64] = &[
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
const VERN6_E: &[f64] = &[
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

const VERN7_C: &[f64] = &[
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
const VERN7_A1: &[f64] = &[];
const VERN7_A2: &[f64] = &[0.005];
const VERN7_A3: &[f64] = &[-1.07679012345679, 1.185679012345679];
const VERN7_A4: &[f64] = &[0.04083333333333333, 0.0, 0.1225];
const VERN7_A5: &[f64] = &[
    0.6389139236255726,
    0.0,
    -2.455672638223657,
    2.272258714598084,
];
const VERN7_A6: &[f64] = &[
    -2.6615773750187572,
    0.0,
    10.804513886456137,
    -8.3539146573962,
    0.820487594956657,
];
const VERN7_A7: &[f64] = &[
    6.067741434696772,
    0.0,
    -24.711273635911088,
    20.427517930788895,
    -1.9061579788166472,
    1.006172249242068,
];
const VERN7_A8: &[f64] = &[
    12.054670076253203,
    0.0,
    -49.75478495046899,
    41.142888638604674,
    -4.461760149974004,
    2.042334822239175,
    -0.09834843665406107,
];
const VERN7_A9: &[f64] = &[
    10.138146522881808,
    0.0,
    -42.6411360317175,
    35.76384003992257,
    -4.3480228403929075,
    2.0098622683770357,
    0.3487490460338272,
    -0.27143900510483127,
];
const VERN7_A10: &[f64] = &[
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
const VERN7_A: &[&[f64]] = &[
    VERN7_A1, VERN7_A2, VERN7_A3, VERN7_A4, VERN7_A5, VERN7_A6, VERN7_A7, VERN7_A8, VERN7_A9,
    VERN7_A10,
];
const VERN7_B: &[f64] = &[
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
const VERN7_E: &[f64] = &[
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

const VERN8_C: &[f64] = &[
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
const VERN8_A1: &[f64] = &[];
const VERN8_A2: &[f64] = &[0.05];
const VERN8_A3: &[f64] = &[-0.0069931640625, 0.1135556640625];
const VERN8_A4: &[f64] = &[0.0399609375, 0.0, 0.1198828125];
const VERN8_A5: &[f64] = &[
    0.36139756280045754,
    0.0,
    -1.3415240667004928,
    1.3701265039000352,
];
const VERN8_A6: &[f64] = &[
    0.049047202797202795,
    0.0,
    0.0,
    0.23509720422144048,
    0.18085559298135673,
];
const VERN8_A7: &[f64] = &[
    0.06169289044289044,
    0.0,
    0.0,
    0.11236568314640277,
    -0.03885046071451367,
    0.01979188712522046,
];
const VERN8_A8: &[f64] = &[
    -1.767630240222327,
    0.0,
    0.0,
    -62.5,
    -6.061889377376669,
    5.6508231982227635,
    65.62169641937624,
];
const VERN8_A9: &[f64] = &[
    -1.1809450665549708,
    0.0,
    0.0,
    -41.50473441114321,
    -4.434438319103725,
    4.260408188586133,
    43.75364022446172,
    0.00787142548991231,
];
const VERN8_A10: &[f64] = &[
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
const VERN8_A11: &[f64] = &[
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
const VERN8_A12: &[f64] = &[
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
const VERN8_A13: &[f64] = &[
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
const VERN8_A: &[&[f64]] = &[
    VERN8_A1, VERN8_A2, VERN8_A3, VERN8_A4, VERN8_A5, VERN8_A6, VERN8_A7, VERN8_A8, VERN8_A9,
    VERN8_A10, VERN8_A11, VERN8_A12, VERN8_A13,
];
const VERN8_B: &[f64] = &[
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
const VERN8_E: &[f64] = &[
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

const VERN9_C: &[f64] = &[
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
const VERN9_A1: &[f64] = &[];
const VERN9_A2: &[f64] = &[0.03462];
const VERN9_A3: &[f64] = &[-0.03893354388572875, 0.13595789452450918];
const VERN9_A4: &[f64] = &[0.03638413148954267, 0.0, 0.10915239446862801];
const VERN9_A5: &[f64] = &[
    2.0257639143939694,
    0.0,
    -7.638023836496291,
    6.173259922102322,
];
const VERN9_A6: &[f64] = &[
    0.05112275589406061,
    0.0,
    0.0,
    0.17708237945550218,
    0.0008027762409222536,
];
const VERN9_A7: &[f64] = &[
    0.13160063579752163,
    0.0,
    0.0,
    -0.2957276252669636,
    0.08781378035642955,
    0.6213052975225274,
];
const VERN9_A8: &[f64] = &[
    0.07166666666666667,
    0.0,
    0.0,
    0.0,
    0.0,
    0.33055335789153195,
    0.2427799754418014,
];
const VERN9_A9: &[f64] = &[
    0.071806640625,
    0.0,
    0.0,
    0.0,
    0.0,
    0.3294380283228177,
    0.1165190029271823,
    -0.034013671875,
];
const VERN9_A10: &[f64] = &[
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
const VERN9_A11: &[f64] = &[
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
const VERN9_A12: &[f64] = &[
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
const VERN9_A13: &[f64] = &[
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
const VERN9_A14: &[f64] = &[
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
const VERN9_A15: &[f64] = &[
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
const VERN9_A16: &[f64] = &[
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
const VERN9_A: &[&[f64]] = &[
    VERN9_A1, VERN9_A2, VERN9_A3, VERN9_A4, VERN9_A5, VERN9_A6, VERN9_A7, VERN9_A8, VERN9_A9,
    VERN9_A10, VERN9_A11, VERN9_A12, VERN9_A13, VERN9_A14, VERN9_A15, VERN9_A16,
];
const VERN9_B: &[f64] = &[
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
const VERN9_E: &[f64] = &[
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

macro_rules! verner_algorithm {
    ($name:ident, $documentation:literal, $order:literal, $nodes:ident, $coefficients:ident, $weights:ident, $error:ident, $fsal:literal) => {
        #[doc = $documentation]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        pub struct $name;

        impl ButcherTableau for $name {
            const NODES: &'static [f64] = $nodes;
            const COEFFICIENTS: &'static [&'static [f64]] = $coefficients;
            const WEIGHTS: &'static [f64] = $weights;
            const ERROR_WEIGHTS: Option<&'static [f64]> = Some($error);
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

verner_algorithm!(
    Vern6,
    "Verner's efficient embedded order-6 explicit Runge--Kutta method.",
    6,
    VERN6_C,
    VERN6_A,
    VERN6_B,
    VERN6_E,
    true
);
verner_algorithm!(
    Vern7,
    "Verner's efficient embedded order-7 explicit Runge--Kutta method.",
    7,
    VERN7_C,
    VERN7_A,
    VERN7_B,
    VERN7_E,
    false
);
verner_algorithm!(
    Vern8,
    "Verner's efficient embedded order-8 explicit Runge--Kutta method.",
    8,
    VERN8_C,
    VERN8_A,
    VERN8_B,
    VERN8_E,
    false
);
verner_algorithm!(
    Vern9,
    "Verner's efficient embedded order-9 explicit Runge--Kutta method.",
    9,
    VERN9_C,
    VERN9_A,
    VERN9_B,
    VERN9_E,
    false
);

#[cfg(test)]
mod tests {
    use super::{Vern6, Vern7, Vern8, Vern9};
    use crate::explicit_rk::ButcherTableau;
    use crate::{OdeAlgorithm, OdeProblem, SaveMode, SolveOptions, solve};

    type TestProblem = OdeProblem<fn(&mut [f64], &[f64], &(), f64), ()>;

    fn exponential() -> TestProblem {
        fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
            du[0] = u[0];
        }
        OdeProblem::new(rhs, vec![1.0], (0.0, 2.0), ())
    }

    fn fixed_endpoint<A: OdeAlgorithm>(algorithm: A, step: f64) -> f64 {
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

    fn convergence_ratio<A: OdeAlgorithm + Copy>(algorithm: A) -> f64 {
        let exact = 2.0_f64.exp();
        let coarse = (fixed_endpoint(algorithm, 0.5) - exact).abs();
        let fine = (fixed_endpoint(algorithm, 0.25) - exact).abs();
        coarse / fine
    }

    #[test]
    fn fixed_step_methods_have_their_expected_orders() {
        let ratios = [
            convergence_ratio(Vern6),
            convergence_ratio(Vern7),
            convergence_ratio(Vern8),
            convergence_ratio(Vern9),
        ];
        assert!(ratios[0] > 45.0);
        assert!(ratios[1] > 90.0);
        assert!(ratios[2] > 170.0);
        assert!(ratios[3] > 300.0);
    }

    #[test]
    fn adaptive_methods_reach_a_tight_tolerance() {
        let options = SolveOptions {
            absolute_tolerance: 1.0e-11,
            relative_tolerance: 1.0e-11,
            initial_step: Some(0.5),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };
        let exact = 2.0_f64.exp();

        for endpoint in [
            solve(&exponential(), Vern6, &options).unwrap().last_state()[0],
            solve(&exponential(), Vern7, &options).unwrap().last_state()[0],
            solve(&exponential(), Vern8, &options).unwrap().last_state()[0],
            solve(&exponential(), Vern9, &options).unwrap().last_state()[0],
        ] {
            assert!((endpoint - exact).abs() < 2.0e-9);
        }
    }

    fn assert_consistent<T: ButcherTableau>() {
        for (row, &node) in T::COEFFICIENTS.iter().zip(T::NODES) {
            assert!((row.iter().sum::<f64>() - node).abs() < 2.0e-12);
        }
        assert!((T::WEIGHTS.iter().sum::<f64>() - 1.0).abs() < 2.0e-13);
        assert!(T::ERROR_WEIGHTS.unwrap().iter().sum::<f64>().abs() < 2.0e-13);
    }

    fn is_fsal<T: ButcherTableau>() -> bool {
        T::FSAL
    }

    #[test]
    fn pinned_tableaus_are_internally_consistent() {
        assert_consistent::<Vern6>();
        assert_consistent::<Vern7>();
        assert_consistent::<Vern8>();
        assert_consistent::<Vern9>();
        assert!(is_fsal::<Vern6>());
        assert!(!is_fsal::<Vern7>() && !is_fsal::<Vern8>() && !is_fsal::<Vern9>());
    }
}
