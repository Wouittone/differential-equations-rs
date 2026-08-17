// Preserve the pinned source's decimal coefficient literals exactly. Precision
// lint exceptions are attached only to the associated constants that contain
// coefficient data; the integration kernels remain fully linted.

use std::marker::PhantomData;

use crate::integrator::{
    KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

trait LowStorage2N {
    const A: &'static [f64];
    const B: &'static [f64];
    const C: &'static [f64];
}

trait LowStorage2C {
    const A: &'static [f64];
    const B: &'static [f64];
    const C: &'static [f64];
}

trait LowStorage3S {
    const GAMMA1: &'static [f64];
    const GAMMA2: &'static [f64];
    const GAMMA3: &'static [f64];
    const DELTA: &'static [f64];
    const BETA1: f64;
    const BETA2: &'static [f64];
    const C: &'static [f64];
    const EVALUATE_ENDPOINT: bool = true;
}

trait LowStorageAlternating2N {
    const A1: &'static [f64];
    const B1: &'static [f64];
    const C1: &'static [f64];
    const A2: &'static [f64];
    const B2: &'static [f64];
    const C2: &'static [f64];
}

trait LowStorageRP {
    const A: &'static [&'static [f64]];
    const B: &'static [f64];
    const B_FINAL: f64;
    const C: &'static [f64];
    const HISTORY_STATES: usize;
}

macro_rules! rp_final {
    (CKLLSRK43_2) => {
        -101169746363290.0 / 37734290219643.0
    };
    (CKLLSRK54_3C) => {
        5198255086312.0 / 14908931495163.0
    };
    (CKLLSRK95_4S) => {
        3559252274877.0 / 14424734981077.0
    };
    (CKLLSRK95_4C) => {
        2993490409874.0 / 13266828321767.0
    };
    (CKLLSRK95_4M) => {
        -453873186647.0 / 15285235680030.0
    };
    (CKLLSRK54_3C_3R) => {
        707644755468.0 / 5028292464395.0
    };
    (CKLLSRK54_3M_3R) => {
        -436008689643.0 / 9453681332953.0
    };
    (CKLLSRK54_3N_3R) => {
        5597675544274.0 / 18784428342765.0
    };
    (CKLLSRK85_4C_3R) => {
        2987336121747.0 / 15645656703944.0
    };
    (CKLLSRK85_4M_3R) => {
        517396786175.0 / 6104475356879.0
    };
    (CKLLSRK85_4P_3R) => {
        1886338382073.0 / 9981671730680.0
    };
    (CKLLSRK54_3N_4R) => {
        2131913067577.0 / 7868783702050.0
    };
    (CKLLSRK54_3M_4R) => {
        -2927.0 / 546.0
    };
    (CKLLSRK65_4M_4R) => {
        2571845656138.0 / 6012342010435.0
    };
    (CKLLSRK85_4FM_4R) => {
        0.0
    };
    (CKLLSRK75_4M_5R) => {
        599706619333.0 / 7161178965783.0
    };
}

macro_rules! method {
    ($name:ident, $coefficients:ident, $doc:literal, $a:expr, $b:expr, $c:expr) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        #[allow(
            non_camel_case_types,
            reason = "preserve the upstream low-storage algorithm name"
        )]
        pub struct $name;

        #[allow(
            non_camel_case_types,
            reason = "coefficient type follows the upstream algorithm name"
        )]
        struct $coefficients;

        impl LowStorage2N for $coefficients {
            #[allow(
                clippy::excessive_precision,
                reason = "pinned upstream f64 coefficient"
            )]
            const A: &'static [f64] = $a;
            #[allow(
                clippy::excessive_precision,
                reason = "pinned upstream f64 coefficient"
            )]
            const B: &'static [f64] = $b;
            #[allow(
                clippy::excessive_precision,
                reason = "pinned upstream f64 coefficient"
            )]
            const C: &'static [f64] = $c;
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
                integrate::<F, P, $coefficients>(problem, options)
            }
        }
    };
}

macro_rules! method_3s {
    ($name:ident, $coefficients:ident, $doc:literal, $gamma1:expr, $gamma2:expr, $gamma3:expr, $delta:expr, $beta1:expr, $beta2:expr, $c:expr) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        #[allow(
            non_camel_case_types,
            reason = "preserve the upstream low-storage algorithm name"
        )]
        pub struct $name;

        #[allow(
            non_camel_case_types,
            reason = "coefficient type follows the upstream algorithm name"
        )]
        struct $coefficients;

        impl LowStorage3S for $coefficients {
            #[allow(
                clippy::excessive_precision,
                reason = "pinned upstream f64 coefficient"
            )]
            const GAMMA1: &'static [f64] = $gamma1;
            #[allow(
                clippy::excessive_precision,
                reason = "pinned upstream f64 coefficient"
            )]
            const GAMMA2: &'static [f64] = $gamma2;
            #[allow(
                clippy::excessive_precision,
                reason = "pinned upstream f64 coefficient"
            )]
            const GAMMA3: &'static [f64] = $gamma3;
            #[allow(
                clippy::excessive_precision,
                reason = "pinned upstream f64 coefficient"
            )]
            const DELTA: &'static [f64] = $delta;
            #[allow(
                clippy::excessive_precision,
                reason = "pinned upstream f64 coefficient"
            )]
            const BETA1: f64 = $beta1;
            #[allow(
                clippy::excessive_precision,
                reason = "pinned upstream f64 coefficient"
            )]
            const BETA2: &'static [f64] = $beta2;
            #[allow(
                clippy::excessive_precision,
                reason = "pinned upstream f64 coefficient"
            )]
            const C: &'static [f64] = $c;
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
                integrate_3s::<F, P, $coefficients>(problem, options)
            }
        }
    };
}

macro_rules! method_2c {
    ($name:ident, $coefficients:ident, $doc:literal, $a:expr, $b:expr, $c:expr) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        #[allow(
            non_camel_case_types,
            reason = "preserve the upstream low-storage algorithm name"
        )]
        pub struct $name;

        #[allow(
            non_camel_case_types,
            reason = "coefficient type follows the upstream algorithm name"
        )]
        struct $coefficients;

        impl LowStorage2C for $coefficients {
            #[allow(
                clippy::excessive_precision,
                reason = "pinned upstream f64 coefficient"
            )]
            const A: &'static [f64] = $a;
            #[allow(
                clippy::excessive_precision,
                reason = "pinned upstream f64 coefficient"
            )]
            const B: &'static [f64] = $b;
            #[allow(
                clippy::excessive_precision,
                reason = "pinned upstream f64 coefficient"
            )]
            const C: &'static [f64] = $c;
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
                integrate_2c::<F, P, $coefficients>(problem, options)
            }
        }
    };
}

macro_rules! method_3sp {
    ($name:ident, $coefficients:ident, $doc:literal, $endpoint:expr, $gamma1:expr, $gamma2:expr, $gamma3:expr, $delta:expr, $beta1:expr, $beta2:expr, $c:expr) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        #[allow(
            non_camel_case_types,
            reason = "preserve the upstream low-storage algorithm name"
        )]
        pub struct $name;

        #[allow(
            non_camel_case_types,
            reason = "coefficient type follows the upstream algorithm name"
        )]
        struct $coefficients;

        impl LowStorage3S for $coefficients {
            #[allow(
                clippy::excessive_precision,
                reason = "pinned upstream f64 coefficient"
            )]
            const GAMMA1: &'static [f64] = $gamma1;
            #[allow(
                clippy::excessive_precision,
                reason = "pinned upstream f64 coefficient"
            )]
            const GAMMA2: &'static [f64] = $gamma2;
            #[allow(
                clippy::excessive_precision,
                reason = "pinned upstream f64 coefficient"
            )]
            const GAMMA3: &'static [f64] = $gamma3;
            #[allow(
                clippy::excessive_precision,
                reason = "pinned upstream f64 coefficient"
            )]
            const DELTA: &'static [f64] = $delta;
            #[allow(
                clippy::excessive_precision,
                reason = "pinned upstream f64 coefficient"
            )]
            const BETA1: f64 = $beta1;
            #[allow(
                clippy::excessive_precision,
                reason = "pinned upstream f64 coefficient"
            )]
            const BETA2: &'static [f64] = $beta2;
            #[allow(
                clippy::excessive_precision,
                reason = "pinned upstream f64 coefficient"
            )]
            const C: &'static [f64] = $c;
            const EVALUATE_ENDPOINT: bool = $endpoint;
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
                integrate_3s::<F, P, $coefficients>(problem, options)
            }
        }
    };
}

macro_rules! method_rp {
    ($name:ident, $coefficients:ident, $doc:literal, $history:expr, $a:expr, $b:expr, $c:expr) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        #[allow(
            non_camel_case_types,
            reason = "preserve the upstream low-storage algorithm name"
        )]
        pub struct $name;

        #[allow(
            non_camel_case_types,
            reason = "coefficient type follows the upstream algorithm name"
        )]
        struct $coefficients;

        impl LowStorageRP for $coefficients {
            #[allow(
                clippy::excessive_precision,
                reason = "pinned upstream f64 coefficient"
            )]
            const A: &'static [&'static [f64]] = $a;
            #[allow(
                clippy::excessive_precision,
                reason = "pinned upstream f64 coefficient"
            )]
            const B: &'static [f64] = $b;
            #[allow(
                clippy::excessive_precision,
                reason = "pinned upstream f64 coefficient"
            )]
            const B_FINAL: f64 = rp_final!($name);
            #[allow(
                clippy::excessive_precision,
                reason = "pinned upstream f64 coefficient"
            )]
            const C: &'static [f64] = $c;
            const HISTORY_STATES: usize = $history;
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
                integrate_rp::<F, P, $coefficients>(problem, options)
            }
        }
    };
}

macro_rules! method_alternating_2n {
    ($name:ident, $coefficients:ident, $doc:literal, $a1:expr, $b1:expr, $c1:expr, $a2:expr, $b2:expr, $c2:expr) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        #[allow(
            non_camel_case_types,
            reason = "preserve the upstream low-storage algorithm name"
        )]
        pub struct $name;

        #[allow(
            non_camel_case_types,
            reason = "coefficient type follows the upstream algorithm name"
        )]
        struct $coefficients;

        impl LowStorageAlternating2N for $coefficients {
            #[allow(
                clippy::excessive_precision,
                reason = "pinned upstream f64 coefficient"
            )]
            const A1: &'static [f64] = $a1;
            #[allow(
                clippy::excessive_precision,
                reason = "pinned upstream f64 coefficient"
            )]
            const B1: &'static [f64] = $b1;
            #[allow(
                clippy::excessive_precision,
                reason = "pinned upstream f64 coefficient"
            )]
            const C1: &'static [f64] = $c1;
            #[allow(
                clippy::excessive_precision,
                reason = "pinned upstream f64 coefficient"
            )]
            const A2: &'static [f64] = $a2;
            #[allow(
                clippy::excessive_precision,
                reason = "pinned upstream f64 coefficient"
            )]
            const B2: &'static [f64] = $b2;
            #[allow(
                clippy::excessive_precision,
                reason = "pinned upstream f64 coefficient"
            )]
            const C2: &'static [f64] = $c2;
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
                integrate_alternating_2n::<F, P, $coefficients>(problem, options)
            }
        }
    };
}

method!(
    Ork256,
    Ork256Coefficients,
    "Five-stage, second-order low-storage method for wave propagation.",
    &[-1.0, -1.55798, -1.0, -0.45031],
    &[0.2, 0.83204, 0.6, 0.35394, 0.2],
    &[0.2, 0.2, 0.8, 0.8]
);

method!(
    CarpenterKennedy2N54,
    CarpenterKennedy2N54Coefficients,
    "Five-stage, fourth-order Carpenter--Kennedy 2N-storage method.",
    &[
        -567_301_805_773.0 / 1_357_537_059_087.0,
        -2_404_267_990_393.0 / 2_016_746_695_238.0,
        -3_550_918_686_646.0 / 2_091_501_179_385.0,
        -1_275_806_237_668.0 / 842_570_457_699.0,
    ],
    &[
        1_432_997_174_477.0 / 9_575_080_441_755.0,
        5_161_836_677_717.0 / 13_612_068_292_357.0,
        1_720_146_321_549.0 / 2_090_206_949_498.0,
        3_134_564_353_537.0 / 4_481_467_310_338.0,
        2_277_821_191_437.0 / 14_882_151_754_819.0,
    ],
    &[
        1_432_997_174_477.0 / 9_575_080_441_755.0,
        2_526_269_341_429.0 / 6_820_363_962_896.0,
        2_006_345_519_317.0 / 3_224_310_063_776.0,
        2_802_321_613_138.0 / 2_924_317_926_251.0,
    ]
);

method!(
    Shlddrk64,
    Shlddrk64Coefficients,
    "Six-stage, fourth-order low-dissipation and low-dispersion method.",
    &[-0.4919575, -0.8946264, -1.5526678, -3.4077973, -1.074264],
    &[0.1453095, 0.4653797, 0.4675397, 0.7795279, 0.3574327, 0.15],
    &[0.1453095, 0.3817422, 0.6367813, 0.7560744, 0.9271047]
);

method!(
    Dglddrk73C,
    Dglddrk73CCoefficients,
    "Seven-stage, third-order low-dissipation and low-dispersion method.",
    &[
        -0.808316387498383,
        -1.503407858773331,
        -1.053064525050744,
        -1.463149119280508,
        -0.659288128108783,
        -1.667891931891068,
    ],
    &[
        0.0119705267309784,
        0.8886897793820711,
        0.4578382089261419,
        0.5790045253338471,
        0.3160214638138484,
        0.2483525368264122,
        0.0677123095940884,
    ],
    &[
        0.0119705267309784,
        0.182317794036199,
        0.5082168062551849,
        0.653203122014859,
        0.853440138567825,
        0.998046608462379,
    ]
);

method!(
    Dglddrk84C,
    Dglddrk84CCoefficients,
    "Eight-stage, fourth-order low-dissipation and low-dispersion method.",
    &[
        -0.721296248227924,
        -0.0107733657161298,
        -0.516258469893097,
        -1.730100286632201,
        -5.200129304403076,
        0.783705894541642,
        -0.544583609433219,
    ],
    &[
        0.2165936736758085,
        0.1773950826411583,
        0.0180253861162329,
        0.0847347637254149,
        0.8129106974622483,
        1.90341603042276,
        0.1314841743399048,
        0.2082583170674149,
    ],
    &[
        0.2165936736758085,
        0.266034348753817,
        0.284005612252272,
        0.325126684378857,
        0.455514959918753,
        0.771321931710117,
        0.919902896453866,
    ]
);

method!(
    Dglddrk84F,
    Dglddrk84FCoefficients,
    "Eight-stage, fourth-order low-dissipation and low-dispersion method.",
    &[
        -0.5534431294501569,
        0.0106598757020349,
        -0.5515812888932,
        -1.885790377558741,
        -5.701295742793264,
        2.113903965664793,
        -0.533957882667528,
    ],
    &[
        0.0803793688273695,
        0.5388497458569843,
        0.0197497440903196,
        0.0991184129733997,
        0.7466920411064123,
        1.679584245618894,
        0.2433728067008188,
        0.1422730459001373,
    ],
    &[
        0.0803793688273695,
        0.321006425033843,
        0.340850182660466,
        0.385036482428547,
        0.50400524775341,
        0.657897756116854,
        0.9484087623348481,
    ]
);

method!(
    Ndblsrk124,
    Ndblsrk124Coefficients,
    "Twelve-stage, fourth-order low-storage method for advection-dominated problems.",
    &[
        -0.0923311242368072,
        -0.9441056581158819,
        -4.3271273247576394,
        -2.1557771329026072,
        -0.9770727190189062,
        -0.7581835342571139,
        -1.7977525470825499,
        -2.6915667972700770,
        -4.6466798960268143,
        -0.1539613783825189,
        -0.5943293901830616,
    ],
    &[
        0.0650008435125904,
        0.0161459902249842,
        0.5758627178358159,
        0.1649758848361671,
        0.3934619494248182,
        0.0443509641602719,
        0.2074504268408778,
        0.6914247433015102,
        0.3766646883450449,
        0.0757190350155483,
        0.2027862031054088,
        0.2167029365631842,
    ],
    &[
        0.0650008435125904,
        0.0796560563081853,
        0.1620416710085376,
        0.2248877362907778,
        0.2952293985641261,
        0.3318332506149405,
        0.4094724050198658,
        0.6356954475753369,
        0.6806551557645497,
        0.714377371241835,
        0.9032588871651854,
    ]
);

method!(
    Ndblsrk134,
    Ndblsrk134Coefficients,
    "Thirteen-stage, fourth-order low-storage method for advection-dominated problems.",
    &[
        -0.6160178650170565,
        -0.4449487060774118,
        -1.0952033345276178,
        -1.2256030785959187,
        -0.2740182222332805,
        -0.0411952089052647,
        -0.179708489915356,
        -1.1771530652064288,
        -0.4078831463120878,
        -0.8295636426191777,
        -4.7895970584252288,
        -0.6606671432964504,
    ],
    &[
        0.0271990297818803,
        0.1772488819905108,
        0.0378528418949694,
        0.6086431830142991,
        0.21543139743161,
        0.2066152563885843,
        0.0415864076069797,
        0.0219891884310925,
        0.9893081222650993,
        0.0063199019859826,
        0.3749640721105318,
        1.6080235151003195,
        0.0961209123818189,
    ],
    &[
        0.0271990297818803,
        0.0952594339119365,
        0.1266450286591127,
        0.1825883045699772,
        0.3737511439063931,
        0.5301279418422206,
        0.5704177433952291,
        0.5885784947099155,
        0.6160769826246714,
        0.6223252334314046,
        0.6897593128753419,
        0.9126827615920843,
    ]
);

method!(
    Ndblsrk144,
    Ndblsrk144Coefficients,
    "Fourteen-stage, fourth-order low-storage method for advection-dominated problems.",
    &[
        -0.718801210867241,
        -0.778533117342157,
        -0.0053282796654044,
        -0.8552979934029281,
        -3.9564138245774565,
        -1.5780575380587385,
        -2.0837094552574054,
        -0.748333418276161,
        -0.7032861106563359,
        0.0013917096117681,
        -0.093207536963746,
        -0.9514200470875948,
        -7.1151571693922548,
    ],
    &[
        0.0367762454319673,
        0.3136296607553959,
        0.1531848691869027,
        0.0030097086818182,
        0.332629379064611,
        0.2440251405350864,
        0.3718879239592277,
        0.6204126221582444,
        0.1524043173028741,
        0.0760894927419266,
        0.0077604214040978,
        0.0024647284755382,
        0.0780348340049386,
        5.5059777270269628,
    ],
    &[
        0.0367762454319673,
        0.1249685262725025,
        0.2446177702277698,
        0.247614953107042,
        0.2969311120382472,
        0.3978149645802642,
        0.5270854589440328,
        0.6981269994175695,
        0.8190890835352128,
        0.8527059887098624,
        0.8604711817462826,
        0.8627060376969976,
        0.8734213127600976,
    ]
);

method_3s!(
    ParsaniKetchesonDeconinck3S32,
    ParsaniKetchesonDeconinck3S32Coefficients,
    "Three-stage, second-order 3S low-storage method optimized for spectral-difference wave propagation.",
    &[-1.2664395576322218e-1, 1.1426980685848858e+0],
    &[6.542778259940647e-1, -8.2869287683723744e-2],
    &[0.0e+0, 0.0e+0],
    &[7.2196567116037724e-1, 0.0e+0],
    7.2366074728360086e-1,
    &[3.4217876502651023e-1, 3.6640216242653251e-1],
    &[7.2366074728360086e-1, 5.9236433182015646e-1]
);

method_3s!(
    ParsaniKetchesonDeconinck3S53,
    ParsaniKetchesonDeconinck3S53Coefficients,
    "Five-stage, third-order 3S low-storage method optimized for spectral-difference wave propagation.",
    &[
        2.5876919610938998e-1,
        -1.3243708384977859e-1,
        5.0556648948362981e-2,
        5.6705507883024708e-1,
    ],
    &[
        5.5284013909611196e-1,
        6.7318513326032769e-1,
        2.8031054965521607e-1,
        5.5215115815918758e-1,
    ],
    &[0.0, 0.0, 2.7525797946334213e-1, -8.9505445022148511e-1],
    &[
        3.4076878915216791e-1,
        3.4143871647890728e-1,
        7.2292984084963252e-1,
        0.0,
    ],
    2.3002859824852059e-1,
    &[
        3.0214498165167158e-1,
        8.0256010238856679e-1,
        4.3621618871511753e-1,
        1.1292705979513513e-1,
    ],
    &[
        2.3002859824852059e-1,
        4.0500453764839639e-1,
        8.9478204142351003e-1,
        7.2351146275625733e-1,
    ]
);

method!(
    RK46NL,
    Rk46NlCoefficients,
    "Six-stage, fourth-order low-storage method with nonlinear stability properties.",
    &[
        -0.737101392796,
        -1.634740794343,
        -0.74473900378,
        -1.469897351522,
        -2.813971388035
    ],
    &[
        0.032918605146,
        0.8232569982,
        0.3815309489,
        0.200092213184,
        1.718581042715,
        0.27
    ],
    &[
        0.032918605146,
        0.249351723343,
        0.466911705055,
        0.582030414044,
        0.847252983783
    ]
);

method_2c!(
    CFRLDDRK64,
    Cfrlddrk64Coefficients,
    "Six-stage, fourth-order low-dissipation and low-dispersion 2C method.",
    &[
        0.17985400977138,
        0.14081893152111,
        0.08255631629428,
        0.65804425034331,
        0.31862993413251
    ],
    &[
        0.10893125722541,
        0.13201701492152,
        0.38911623225517,
        -0.59203884581148,
        0.47385028714844,
        0.48812405426094
    ],
    &[
        0.28878526699679,
        0.38176720366804,
        0.71262082069639,
        0.69606990893393,
        0.83050587987157
    ]
);

method_2c!(
    TSLDDRK74,
    Tslddrk74Coefficients,
    "Seven-stage, fourth-order low-dissipation and low-dispersion 2C method.",
    &[
        0.241566650129646868,
        0.0423866513027719953,
        0.215602732678803776,
        0.232328007537583987,
        0.256223412574146438,
        0.097869410214269723
    ],
    &[
        0.0941840925477795334,
        0.149683694803496998,
        0.285204742060440058,
        -0.122201846148053668,
        0.0605151571191401122,
        0.345986987898399296,
        0.18662717171879767
    ],
    &[
        0.335750742677426401,
        0.286254438654048527,
        0.744675262090520366,
        0.639198690801246909,
        0.723609252956949472,
        0.91124223849547205
    ]
);

method!(
    SHLDDRK52,
    Shlddrk52Coefficients,
    "Five-stage, second-order low-dissipation and low-dispersion method.",
    &[-0.6913065, -2.655155, -0.8147688, -0.6686587],
    &[0.1, 0.75, 0.7, 0.479313, 0.310392],
    &[0.1, 0.3315201, 0.4577796, 0.8666528]
);

method_alternating_2n!(
    SHLDDRK_2N,
    Shlddrk2nCoefficients,
    "Alternating five- and six-stage fourth-order low-dissipation and low-dispersion method.",
    &[-0.6051226, -2.0437564, -0.7406999, -4.4231765],
    &[0.2687454, 0.8014706, 0.505157, 0.5623568, 0.0590065],
    &[0.2687454, 0.585228, 0.6827066, 1.1646854],
    &[-0.4412737, -1.073982, -1.706357, -2.7979293, -4.0913537],
    &[0.1158488, 0.3728769, 0.7379536, 0.579811, 1.0312849, 0.15],
    &[0.1158485, 0.324185, 0.6193208, 0.8034472, 0.9184166]
);

method_3s!(
    ParsaniKetchesonDeconinck3S82,
    ParsaniKetchesonDeconinck3S82Coefficients,
    "Eight-stage, second-order 3S low-storage method optimized for spectral-difference wave propagation.",
    &[
        4.2397552118208004e-1,
        -2.3528852074619033e-1,
        7.9598685017877846e-1,
        -1.3205224623823270e0,
        2.1452956294251941e0,
        -9.5532770501880648e-1,
        2.5361391125131094e-1,
    ],
    &[
        4.4390665802303775e-1,
        7.5333732286056154e-1,
        6.5885460813015481e-2,
        6.3976199384289623e-1,
        -7.3823030755143193e-1,
        7.0177211879534529e-1,
        4.0185379950224559e-1,
    ],
    &[
        0.0e0,
        0.0e0,
        5.8415358412023582e-2,
        6.4219008773865116e-1,
        6.8770305706885126e-1,
        6.3729822311671305e-2,
        -3.3679429978131387e-1,
    ],
    &[
        2.9762522910396538e-1,
        3.4212961014330662e-1,
        5.7010739154759105e-1,
        4.1350769551529132e-1,
        -1.4040672669058066e-1,
        2.1249567092409008e-1,
        0.0e0,
    ],
    9.9292229393265474e-1,
    &[
        5.2108385130005974e-1,
        3.8505327083543915e-3,
        7.9714199213087467e-1,
        -8.182246027664912e-2,
        8.4604310411858186e-1,
        -1.0191166090841246e-1,
        6.3190236038107500e-2,
    ],
    &[
        9.9292229393265474e-1,
        1.0732413280565014e0,
        2.5057060509809409e-1,
        1.0496674928979783e0,
        -6.7488037049720317e-1,
        -1.5868411612120166e0,
        2.1138242369563969e0,
    ]
);

method_3s!(
    ParsaniKetchesonDeconinck3S173,
    ParsaniKetchesonDeconinck3S173Coefficients,
    "Seventeen-stage, third-order 3S low-storage method optimized for spectral-difference wave propagation.",
    &[
        7.9377023961829174e-1,
        -8.3475116244241754e-2,
        -1.6706337980062214e-2,
        3.6410691500331427e-1,
        6.917825518154278e-1,
        1.4887115004739182e0,
        4.5336125560871188e-1,
        -1.2705776046458739e-1,
        8.3749845457747696e-1,
        1.5709218393361746e-1,
        -5.7768207086288348e-1,
        -5.7340394122375393e-1,
        -1.205073484651447e0,
        -2.8100719513641002e0,
        1.6142798657609492e-1,
        -2.5801264756641613e0,
    ],
    &[
        3.285786194081125e-1,
        1.1276843361180819e0,
        1.3149447395238016e0,
        5.2062891534209055e-1,
        8.8127462325164985e-1,
        4.2020606445856712e-1,
        7.6532635739246124e-2,
        4.4386734924685722e-1,
        6.6503093955199682e-2,
        1.5850209163184039e0,
        1.1521721573462576e0,
        1.1172750819374575e0,
        7.7630223917584007e-1,
        1.0046657060652295e0,
        -1.9795868964959054e-1,
        1.3350583594705518e0,
    ],
    &[
        0.0e0,
        0.0e0,
        8.4034574578399479e-1,
        8.5047738439705145e-1,
        1.4082448501410852e-1,
        -3.2678802469519369e-1,
        5.3716357620635535e-1,
        9.0228922115199051e-1,
        1.5960226946983552e-1,
        1.1038153140686748e0,
        1.0843516423068365e-1,
        4.6212710442787724e-1,
        -3.3448312125108398e-1,
        1.1153826567096696e0,
        1.5503248734613539e0,
        -1.2200245424704212e0,
    ],
    &[
        -3.7235794357769936e-1,
        3.3315440189685536e-1,
        -8.266763033840252e-1,
        -5.4628377681035534e-1,
        6.0210777634642887e-1,
        -5.7528717894031067e-1,
        5.0914861529202782e-1,
        3.8258114767897194e-1,
        -4.627906322118529e-1,
        -2.0820434288562648e-1,
        1.4398056081552713e0,
        -2.8056600927348752e-1,
        2.2767189929551406e0,
        -5.8917530100546356e-1,
        9.1328651048418164e-1,
        0.0e0,
    ],
    4.9565403010221741e-2,
    &[
        9.7408718698159397e-2,
        -1.762073797680187e-1,
        1.485206917546025e-1,
        -3.3127657103714951e-2,
        4.8294609330498492e-2,
        4.9622612199980112e-2,
        8.7340766269850378e-1,
        -2.869280439908537e-1,
        1.2679897532256112e0,
        -1.0217436118953449e-2,
        8.466557003259835e-2,
        2.8253854742588246e-2,
        -9.2936733010804407e-2,
        -8.4798124766803512e-2,
        -1.6923145636158564e-2,
        -4.7305106233879957e-2,
    ],
    &[
        4.9565403010221741e-2,
        1.3068799001687578e-1,
        -1.5883063460310493e-1,
        3.5681144740196935e-1,
        7.6727123317642698e-2,
        1.0812579255374613e-1,
        1.8767228084815801e-1,
        9.6162976936182631e-1,
        -2.2760719867560897e-1,
        1.1115681606027146e0,
        6.126684542767652e-1,
        1.0729473245077408e0,
        3.7824186468104548e-1,
        7.904189134764672e-1,
        -1.0406955693161675e0,
        -2.4607146824557105e-1,
    ]
);

method_3s!(
    ParsaniKetchesonDeconinck3S184,
    ParsaniKetchesonDeconinck3S184Coefficients,
    "Eighteen-stage, fourth-order 3S low-storage method optimized for spectral-difference wave propagation.",
    &[
        1.1750819811951678e+0,
        3.0909017892654811e-1,
        1.4409117788115862e+0,
        -4.3563049445694069e-1,
        2.0341503014683893e-1,
        4.9828356971917692e-1,
        3.5307737157745489e+0,
        -7.9318790975894626e-1,
        8.9120513355345166e-1,
        5.7091009196320974e-1,
        1.6912188575015419e-2,
        1.0077912519329719e+0,
        -6.8532953752099512e-1,
        1.0488165551884063e+0,
        8.3647761371829943e-1,
        1.308790983044571e+0,
        9.0419681700177323e-1,
    ],
    &[
        -1.2891068509748144e-1,
        3.5609406666728954e-1,
        -4.0648075226104241e-1,
        6.0714786995207426e-1,
        1.0253501186236846e+0,
        2.4411240760769423e-1,
        -1.2813606970134104e+0,
        8.1625711892373898e-1,
        1.0171269354643386e-1,
        1.9379378662711269e-1,
        7.4408643544851782e-1,
        -1.2591764563430008e-1,
        1.1996463179654226e+0,
        4.5772068865370406e-2,
        8.3622292077033844e-1,
        -1.4179124272450148e+0,
        1.3661459065331649e-1,
    ],
    &[
        0.0e+0,
        0.0e+0,
        2.5583378537249163e-1,
        5.2676794366988289e-1,
        -2.5648375621792202e-1,
        3.1932438003236391e-1,
        -3.1106815010852862e-1,
        4.7631196164025996e-1,
        -9.8853727938895783e-2,
        1.9274726276883622e-1,
        3.2389860855971508e-2,
        7.5923980038397509e-2,
        2.0635456088664017e-1,
        -8.9741032556032857e-2,
        2.689993250567619e-2,
        4.1882069379552307e-2,
        6.2016148912381761e-2,
    ],
    &[
        3.5816500441970289e-1,
        5.8208024465093577e-1,
        -2.2615285894283538e-1,
        -2.1715466578266213e-1,
        -4.6990441450888265e-1,
        -2.7986911594744995e-1,
        9.8513926355272197e-1,
        -1.1899324232814899e-1,
        4.2821073124370562e-1,
        -8.2196355299900403e-1,
        5.8113997057675074e-2,
        -6.1283024325436919e-1,
        5.6800136190634054e-1,
        -3.3874970570335106e-1,
        -7.3071238125137772e-1,
        8.3936016960374532e-2,
        0.0e+0,
    ],
    1.2384169480626298e-1,
    &[
        1.0176262534280349e+0,
        -6.9732026387527429e-2,
        3.4239356067806476e-1,
        1.8177707207807942e-2,
        -6.1188746289480445e-3,
        7.8242308902580354e-2,
        -3.7642864750532951e-1,
        -4.5078383666690258e-2,
        -7.5734228201432585e-1,
        -2.7149222760935121e-1,
        1.1833684341657344e-3,
        2.8858319979308041e-2,
        4.6005267586974657e-1,
        1.8014887068775631e-2,
        -1.5508175395461857e-2,
        -4.0095737929274988e-1,
        1.4949678367038011e-1,
    ],
    &[
        1.2384169480626298e-1,
        1.1574324659554065e+0,
        5.4372099141546926e-1,
        8.8394666834280744e-1,
        -1.2212042176605774e-1,
        4.4125685133082082e-1,
        3.8039092095473748e-1,
        5.4591107347528367e-2,
        4.8731855535356028e-1,
        -2.3007964303896034e-1,
        -1.8907656662915873e-1,
        8.1059805668623763e-1,
        7.7080875997868803e-1,
        1.1712158507200179e+0,
        1.2755351018003545e+0,
        8.0422507946168564e-1,
        9.7508680250761848e-1,
    ]
);

method_3s!(
    ParsaniKetchesonDeconinck3S94,
    ParsaniKetchesonDeconinck3S94Coefficients,
    "Nine-stage, fourth-order 3S low-storage method optimized for spectral-difference wave propagation.",
    &[
        -4.6556413837561301e+0,
        -7.7202649689034453e-1,
        -4.0244202720632174e+0,
        -2.1296873883702272e-2,
        -2.4350219407769953e+0,
        1.9856336960249132e-2,
        -2.8107894116913812e-1,
        1.68943543736779e-1,
    ],
    &[
        2.4992627683300688e+0,
        5.8668202764174726e-1,
        1.2051419816240785e+0,
        3.4747937498564541e-1,
        1.3213458736302766e+0,
        3.1196363453264964e-1,
        4.3514189245414447e-1,
        2.3596980658341213e-1,
    ],
    &[
        0.0e+0,
        0.0e+0,
        7.6209857891449362e-1,
        -1.981181783296552e-1,
        -6.2289587091629484e-1,
        -3.7522475499063573e-1,
        -3.3554373281046146e-1,
        -4.5609629702116454e-2,
    ],
    &[
        1.2629238731608268e+0,
        7.5749675232391733e-1,
        5.1635907196195419e-1,
        -2.7463346616574083e-2,
        -4.3826743572318672e-1,
        1.2735870231839268e+0,
        -6.294738221773023e-1,
        0.0e+0,
    ],
    2.8363432481011769e-1,
    &[
        9.7364980747486463e-1,
        3.3823592364196498e-1,
        -3.5849518935750763e-1,
        -4.1139587569859462e-3,
        1.4279689871485013e+0,
        1.8084680519536503e-2,
        1.6057708856060501e-1,
        2.9522267863254809e-1,
    ],
    &[
        2.8363432481011769e-1,
        5.4840742446661772e-1,
        3.6872298094969475e-1,
        -6.8061183026103156e-1,
        3.5185265855105619e-1,
        1.6659419385562171e+0,
        9.7152778807463247e-1,
        9.0515694340066954e-1,
    ]
);

method_3s!(
    ParsaniKetchesonDeconinck3S105,
    ParsaniKetchesonDeconinck3S105Coefficients,
    "Ten-stage, fifth-order 3S low-storage method optimized for spectral-difference wave propagation.",
    &[
        4.0436600785287713e-1,
        -8.5034274641295027e-1,
        -6.9508941671218478e0,
        9.2387652252320684e-1,
        -2.5631780399589106e0,
        2.5457448699988827e-1,
        3.1258317336761454e-1,
        -7.0071148003175443e-1,
        4.8396209710057070e-1,
    ],
    &[
        6.8714670697294733e-1,
        1.0930247604585732e0,
        3.2259753823377983e0,
        1.0411537008416110e0,
        1.2928214888638039e0,
        7.3914627692888835e-1,
        1.2391292570651462e-1,
        1.8427534793568445e-1,
        5.7127889427161162e-2,
    ],
    &[
        0.0e0,
        0.0e0,
        -2.3934051593398129e0,
        -1.9028544220991284e0,
        -2.8200422105835639e0,
        -1.8326984641282289e0,
        -2.1990945108072310e-1,
        -4.0824306603783045e-1,
        -1.3776697911236280e-1,
    ],
    &[
        -1.3317784091400336e-1,
        8.2604227852898304e-1,
        1.5137004305165804e0,
        -1.3058100631721905e0,
        3.0366787893355149e0,
        -1.4494582670831953e0,
        3.8343138733685103e0,
        4.1222939718018692e0,
        0.0e0,
    ],
    2.5978835757039448e-1,
    &[
        1.7770088002098183e-2,
        2.4816366373161344e-1,
        7.9417368275785671e-1,
        3.8853912968701337e-1,
        1.4550516642704694e-1,
        1.5875173794655811e-1,
        1.6506056315937651e-1,
        2.1180932999328042e-1,
        1.5593923403495016e-1,
    ],
    &[
        2.5978835757039448e-1,
        9.9045731158085557e-2,
        2.1555118823045644e-1,
        5.007950078415504e-1,
        5.5922519148547800e-1,
        5.4499869734044426e-1,
        7.6152246625852738e-1,
        8.4270620830633836e-1,
        9.1522098071770008e-1,
    ]
);

method_3s!(
    ParsaniKetchesonDeconinck3S205,
    ParsaniKetchesonDeconinck3S205Coefficients,
    "Twenty-stage, fifth-order 3S low-storage method optimized for spectral-difference wave propagation.",
    &[
        -1.168247970322938e0,
        -2.5112155037089772e0,
        -5.5259960154735988e-1,
        2.924303350951174e-3,
        -4.7948973385386493e0,
        -5.3095533497183016e0,
        -2.3624194456630736e0,
        2.0068995756589547e-1,
        -1.498580866159771e0,
        4.8941228502377687e-1,
        -1.0387512755259576e-1,
        -1.3287664273288191e-1,
        7.5858678822837511e-1,
        -4.3321586294096939e0,
        4.8199700138402146e-1,
        -7.0924756614960671e-3,
        -8.8422252029506054e-1,
        -8.9129367099545231e-1,
        1.5297157134040762e0,
    ],
    &[
        8.8952052154583572e-1,
        8.8988129100385194e-1,
        3.5701564494677057e-1,
        2.4232462479216824e-1,
        1.2727083024258155e0,
        1.1126977210342681e0,
        5.1360709645409097e-1,
        1.1181089682044856e-1,
        2.7881272382085232e-1,
        4.9032886260666715e-2,
        4.187105106589787e-2,
        4.4602463796686219e-2,
        1.489727125115475e-2,
        2.6244269699436817e-1,
        -4.7486056986590294e-3,
        2.3219312682036197e-2,
        6.2852588972458059e-2,
        5.4473719351268962e-2,
        2.4345446089014514e-2,
    ],
    &[
        0.0e0,
        0.0e0,
        1.9595487007932735e-1,
        -6.9871675039100595e-5,
        1.059223116981005e-1,
        1.0730426871909635e0,
        8.9257826744389124e-1,
        -1.4078912484894415e-1,
        -2.6869890558434262e-1,
        -6.5175753568318007e-2,
        4.9177812903108553e-1,
        4.6017684776493678e-1,
        -6.4689512947008251e-3,
        4.4034728024115377e-1,
        6.1086885767527943e-1,
        5.0546454457410162e-1,
        5.4668509293072887e-1,
        7.1414182420995431e-1,
        -1.0558095282893749e0,
    ],
    &[
        1.4375468781258596e0,
        1.5081653637261594e0,
        -1.4575347066062688e-1,
        3.1495761082838158e-1,
        3.5505919368536931e-1,
        2.361638937456696e-1,
        1.0267488547302055e-1,
        3.5991243524519438e0,
        1.5172890003890782e0,
        1.8171662741779953e0,
        2.8762263521436831e0,
        4.6350154228218754e-1,
        1.557312211072722e0,
        2.0001066778080254e0,
        9.1690694855534305e-1,
        2.0474618401365854e0,
        -3.2336329115436924e-1,
        3.2899060754742177e-1,
        0.0e0,
    ],
    1.7342385375780556e-1,
    &[
        2.8569004728564801e-1,
        6.8727044379779589e-1,
        1.2812121060977319e-1,
        4.9137180740403122e-4,
        4.7033584446956857e-2,
        4.4539998128170821e-1,
        1.225982488734372e0,
        2.0616463985024421e-2,
        1.5941162575324802e-1,
        1.2953803678226099e0,
        1.7287352967302603e-3,
        1.1660483420536467e-1,
        7.7997036621815521e-2,
        3.2563250234418012e-1,
        1.0611520488333197e0,
        6.5891625628040993e-4,
        8.3534647700054046e-2,
        9.8972579458252483e-2,
        4.301011614509704e-2,
    ],
    &[
        1.7342385375780556e-1,
        3.0484982420032158e-1,
        5.5271395645729193e-1,
        4.7079204549750037e-2,
        1.5652540451324129e-1,
        1.8602224049074517e-1,
        2.8426620035751449e-1,
        9.5094727548792268e-1,
        6.804650107009601e-1,
        5.9705366562360063e-1,
        1.8970821645077285e0,
        2.9742664004529606e-1,
        6.081346370013494e-1,
        7.3080004188477765e-1,
        9.1656999044951792e-1,
        1.430968755461453e0,
        4.1043824968249148e-1,
        8.4898255952298962e-1,
        3.3543896258348421e-1,
    ]
);

method_rp!(
    CKLLSRK43_2,
    CKLLSRK43_2Coefficients,
    "Pinned CKLLSRK43_2 low-storage register-pipeline method.",
    1,
    &[&[
        11847461282814.0 / 36547543011857.0,
        3943225443063.0 / 7078155732230.0,
        -346793006927.0 / 4029903576067.0
    ]],
    &[
        1017324711453.0 / 9774461848756.0,
        8237718856693.0 / 13685301971492.0,
        57731312506979.0 / 19404895981398.0
    ],
    &[
        11847461282814.0 / 36547543011857.0,
        2079258608735161403527719.0 / 3144780143828896577027540.0,
        41775191021672206476512620310545281003.0 / 67383242951014563804622635478530729598.0
    ]
);
method_rp!(
    CKLLSRK54_3C,
    CKLLSRK54_3CCoefficients,
    "Pinned CKLLSRK54_3C low-storage register-pipeline method.",
    1,
    &[&[
        970286171893.0 / 4311952581923.0,
        6584761158862.0 / 12103376702013.0,
        2251764453980.0 / 15575788980749.0,
        26877169314380.0 / 34165994151039.0
    ]],
    &[
        1153189308089.0 / 22510343858157.0,
        1772645290293.0 / 4653164025191.0,
        -1672844663538.0 / 4480602732383.0,
        2114624349019.0 / 3568978502595.0
    ],
    &[
        970286171893.0 / 4311952581923.0,
        18020302501594987297224499.0 / 30272352378568762325374449.0,
        940957347754451928235896289983310398260.0 / 1631475460071027605339136597003329167263.0,
        8054848232572758807908657851968985615984276476412066.0
            / 8139155613487734148190408375391604039319069461908135.0
    ]
);
method_rp!(CKLLSRK95_4S, CKLLSRK95_4SCoefficients, "Pinned CKLLSRK95_4S low-storage register-pipeline method.", 1, &[&[1107026461565.0 / 5417078080134.0, 38141181049399.0 / 41724347789894.0, 493273079041.0 / 11940823631197.0, 1851571280403.0 / 6147804934346.0, 11782306865191.0 / 62590030070788.0, 9452544825720.0 / 13648368537481.0, 4435885630781.0 / 26285702406235.0, 2357909744247.0 / 11371140753790.0]], &[2274579626619.0 / 23610510767302.0, 693987741272.0 / 12394497460941.0, -347131529483.0 / 15096185902911.0, 1144057200723.0 / 32081666971178.0, 1562491064753.0 / 11797114684756.0, 13113619727965.0 / 44346030145118.0, 393957816125.0 / 7825732611452.0, 720647959663.0 / 6565743875477.0], &[1107026461565.0 / 5417078080134.0, 248859529315327119359384971.0 / 246283290687986423455311497.0, 676645811244741430568548054467096184193.0 / 3494367591912647069105975861901917224854.0, 974370561662349106845723178377944301517533305964589.0 / 2263290880944514209862892217007179742168288737673791.0, 23738915426186839814576142955255044211724736499516359049188590711.0 / 67203160149331519751012175988216621571869262839903428488408759604.0, 1882683585832901544671586749377753597775777511029847145277760106172106584376955.0 / 1901663903553486696887572033100456166564493852721284994300276200102719954709068.0, 61872982955093233917984290421186995265732234396821660871734841970091372539489172106504162637.0 / 81207728164913218881758751120099941603350662788460257311895072645631357391473675997419584220.0, 197565042693102647130189450792520184956129841555961940530192020871289515369046683661585184411130637357.0 / 232196202198018941876505157326935602816917261769279531369710269478309137067357703513986211472070374865.0]);
method_rp!(CKLLSRK95_4C, CKLLSRK95_4CCoefficients, "Pinned CKLLSRK95_4C low-storage register-pipeline method.", 1, &[&[2756167973529.0 / 16886029417639.0, 11436141375279.0 / 13592993952163.0, 88551658327.0 / 2352971381260.0, 1882111988787.0 / 5590444193957.0, 846820081679.0 / 4754706910573.0, 4475289710031.0 / 6420120086209.0, 118394748311.0 / 9144450320350.0, 3307377157135.0 / 13111544596386.0]], &[1051460336009.0 / 14326298067773.0, 930517604889.0 / 7067438519321.0, -311910530565.0 / 11769786407153.0, -410144036239.0 / 7045999268647.0, 16692278975653.0 / 83604524739127.0, 3777666801280.0 / 13181243438959.0, 286682614203.0 / 12966190094317.0, 3296161604512.0 / 22629905347183.0], &[2756167973529.0 / 16886029417639.0, 178130064075748009421121134.0 / 194737282992122861693942999.0, 57818276708998807530478158133449099851.0 / 238238895426494403638887583424360627580.0, 3432454166457135667348375590572529790194124848059104.0 / 6662096512485931545803670383440459769502981926779993.0, 11915126765643872062053118401193741919814944004335534493046474237.0 / 39923715169802034300462756237193519081954994679332637422466438119.0, 4583883621300589683158355859163890943947800555246686854224916208836514024614442.0 / 4506922925096139856045533451931734406235454975594364558624038359246205017801029.0, 52423219056629312880725209686636192777075511202228566787042655312097949192300218484424118619.0 / 84615702680158836756876794083943762639542619835321175569533203672153042594634924742431352650.0, 1385843715228499555828057735261132084759031703937678116167963792224108372724503731226480538087331079769069.0 / 1573111845759510782008384284066606688388217112071821912231287750254246452350240904652428530379336814559998.0]);
method_rp!(CKLLSRK95_4M, CKLLSRK95_4MCoefficients, "Pinned CKLLSRK95_4M low-storage register-pipeline method.", 1, &[&[5573095071601.0 / 11304125995793.0, 315581365608.0 / 4729744040249.0, 8734064225157.0 / 30508564569118.0, 6457785058448.0 / 14982850401353.0, 5771559441664.0 / 18187997215013.0, 1906712129266.0 / 6681214991155.0, 311585568784.0 / 2369973437185.0, -4840285693886.0 / 7758383361725.0]], &[549666665015.0 / 5899839355879.0, -548816778320.0 / 9402908589133.0, 1672704946363.0 / 13015471661974.0, 1025420337373.0 / 5970204766762.0, 1524419752016.0 / 6755273790179.0, -10259399787359.0 / 43440802207630.0, 4242280279850.0 / 10722460893763.0, 1887552771913.0 / 6099058196803.0], &[5573095071601.0 / 11304125995793.0, 4461661993774357683398167.0 / 27904730031895199210773871.0, 543425730194107827015264404954831354769.0 / 1692482454734045499140692116457071506026.0, 6429586327013850295560537918723231687699697140756067.0 / 10818243561353065593628044468492745774799533452459554.0, 555984804780268998022260997164198311752115182012221553157164786.0 / 852213854337283773231630192518719827415190771786411558523853399.0, 1789345671284476461332539715762783748132668223013904373945129499237446392572.0 / 2114764997945705573761804541148983827155257005191540481884326639410208291635.0, 2972211964132922642906704796208250552795647483819924111704054115070043529037601892705217.0 / 6517454043294174770082798998332814729652497865130816822916618330047242844192616374937270.0, 22038106775746116973750004935225594022265950105933360206617843987546593773108577078867914238620973639.0 / 228770596964454885481304478061363897900267080665965044117230250287302271092811814450282133504194141850.0]);
method_rp!(
    CKLLSRK54_3C_3R,
    CKLLSRK54_3C_3RCoefficients,
    "Pinned CKLLSRK54_3C_3R low-storage register-pipeline method.",
    2,
    &[
        &[
            2365592473904.0 / 8146167614645.0,
            4278267785271.0 / 6823155464066.0,
            2789585899612.0 / 8986505720531.0,
            15310836689591.0 / 24358012670437.0
        ],
        &[
            0.0 / 1.0,
            -722262345248.0 / 10870640012513.0,
            1365858020701.0 / 8494387045469.0,
            3819021186.0 / 2763618202291.0
        ]
    ],
    &[
        846876320697.0 / 6523801458457.0,
        3032295699695.0 / 12397907741132.0,
        612618101729.0 / 6534652265123.0,
        1155491934595.0 / 2954287928812.0
    ],
    &[
        2365592473904.0 / 8146167614645.0,
        41579400703344293287237655.0 / 74172066799272566561857858.0,
        299308060739053880467044545349561265546.0 / 497993456493513966629488516767096447823.0,
        5468330126750791548369684419304733938034170906513585.0
            / 5444638279732761024893610553331663911104849888809108.0
    ]
);
method_rp!(
    CKLLSRK54_3M_3R,
    CKLLSRK54_3M_3RCoefficients,
    "Pinned CKLLSRK54_3M_3R low-storage register-pipeline method.",
    2,
    &[
        &[
            17396840518954.0 / 49788467287365.0,
            21253110367599.0 / 14558944785238.0,
            4293647616769.0 / 14519312872408.0,
            -8941886866937.0 / 7464816931160.0
        ],
        &[
            0.0 / 1.0,
            -12587430488023.0 / 11977319897242.0,
            6191878339181.0 / 13848262311063.0,
            19121624165801.0 / 12321025968027.0
        ]
    ],
    &[
        1977388745448.0 / 17714523675943.0,
        6528140725453.0 / 14879534818174.0,
        4395900531415.0 / 55649460397719.0,
        6567440254656.0 / 15757960182571.0
    ],
    &[
        17396840518954.0 / 49788467287365.0,
        2546271293606266795002053.0 / 6227754966395669782804057.0,
        3043453778831534771251734214272440269577.0 / 3561810617861654942925591050154818470872.0,
        10963106193663894855575270257133723083246622141340761.0
            / 12121458300971454511596914396147459030814063072954120.0
    ]
);
method_rp!(
    CKLLSRK54_3N_3R,
    CKLLSRK54_3N_3RCoefficients,
    "Pinned CKLLSRK54_3N_3R low-storage register-pipeline method.",
    2,
    &[
        &[
            4745337637855.0 / 22386579876409.0,
            6808157035527.0 / 13197844641179.0,
            4367509502613.0 / 10454198590847.0,
            1236962429870.0 / 3429868089329.0
        ],
        &[
            0.0 / 1.0,
            546509042554.0 / 9152262712923.0,
            625707605167.0 / 5316659119056.0,
            582400652113.0 / 7078426004906.0
        ]
    ],
    &[
        314199625218.0 / 7198350928319.0,
        6410344372641.0 / 17000082738695.0,
        292278564125.0 / 5593752632744.0,
        5010207514426.0 / 21876007855139.0
    ],
    &[
        4745337637855.0 / 22386579876409.0,
        6320253019873211389522417.0 / 10980921945492108365568747.0,
        231699760563456147635097088564862719039.0 / 400094496217566390613617613962197753808.0,
        2565873674791335200443549967376635530873909687156071.0
            / 2970969302106648098855751120425897741072516011514170.0
    ]
);
method_rp!(
    CKLLSRK85_4C_3R,
    CKLLSRK85_4C_3RCoefficients,
    "Pinned CKLLSRK85_4C_3R low-storage register-pipeline method.",
    2,
    &[
        &[
            141236061735.0 / 3636543850841.0,
            7367658691349.0 / 25881828075080.0,
            6185269491390.0 / 13597512850793.0,
            2669739616339.0 / 18583622645114.0,
            42158992267337.0 / 9664249073111.0,
            970532350048.0 / 4459675494195.0,
            1415616989537.0 / 7108576874996.0
        ],
        &[
            0.0 / 1.0,
            -343061178215.0 / 2523150225462.0,
            -4057757969325.0 / 18246604264081.0,
            1415180642415.0 / 13311741862438.0,
            -93461894168145.0 / 25333855312294.0,
            7285104933991.0 / 14106269434317.0,
            -4825949463597.0 / 16828400578907.0
        ]
    ],
    &[
        514862045033.0 / 4637360145389.0,
        0.0 / 1.0,
        0.0 / 1.0,
        0.0 / 1.0,
        2561084526938.0 / 7959061818733.0,
        4857652849.0 / 7350455163355.0,
        1059943012790.0 / 2822036905401.0
    ],
    &[
        141236061735.0 / 3636543850841.0,
        4855329627204641469273019.0 / 32651870171503411731843480.0,
        395246570619540395679764439681768625174.0 / 1150568172675067443707820382013045349637.0,
        103533040647279909858308372897770021461.0 / 286797987459862321650077169609703051387.0,
        890342029406775514852349518244920625309.0 / 1135377348321966192554675673174478190626.0,
        82180664649829640456237722943611531408.0 / 97244490215364259564723087293866304345.0,
        1524044277359326675923410465291452002169116939509651.0
            / 4415279581486844959297591640758696961331751174567964.0
    ]
);
method_rp!(
    CKLLSRK85_4M_3R,
    CKLLSRK85_4M_3RCoefficients,
    "Pinned CKLLSRK85_4M_3R low-storage register-pipeline method.",
    2,
    &[
        &[
            967290102210.0 / 6283494269639.0,
            852959821520.0 / 5603806251467.0,
            8043261511347.0 / 8583649637008.0,
            -115941139189.0 / 8015933834062.0,
            2151445634296.0 / 7749920058933.0,
            15619711431787.0 / 74684159414562.0,
            12444295717883.0 / 11188327299274.0
        ],
        &[
            0.0 / 1.0,
            475331134681.0 / 7396070923784.0,
            -8677837986029.0 / 16519245648862.0,
            2224500752467.0 / 10812521810777.0,
            1245361422071.0 / 3717287139065.0,
            1652079198131.0 / 3788458824028.0,
            -5225103653628.0 / 8584162722535.0
        ]
    ],
    &[
        83759458317.0 / 1018970565139.0,
        0.0 / 1.0,
        0.0 / 1.0,
        0.0 / 1.0,
        6968891091250.0 / 16855527649349.0,
        783521911849.0 / 8570887289572.0,
        3686104854613.0 / 11232032898210.0
    ],
    &[
        967290102210.0 / 6283494269639.0,
        8972214919142352493858707.0 / 41446148478994088895191128.0,
        35682660731882055122214991891899678815.0 / 72242678055272695781813348615158920272.0,
        24151963894889409757443700144610337197.0 / 88316684951621554188239538678367088186.0,
        20396803294876689925555603189127802602.0 / 29355195069529377650856010387665377655.0,
        104860372573190455963699691732496938387.0 / 144152676952392296448858925279884773652.0,
        1648260218501227913212294426176971326433416596592133.0
            / 1649556119556299790473636959153132604082083356090490.0
    ]
);
method_rp!(
    CKLLSRK85_4P_3R,
    CKLLSRK85_4P_3RCoefficients,
    "Pinned CKLLSRK85_4P_3R low-storage register-pipeline method.",
    2,
    &[
        &[
            1298271176151.0 / 60748409385661.0,
            14078610000243.0 / 41877490110127.0,
            553998884433.0 / 1150223130613.0,
            15658478150918.0 / 92423611770207.0,
            18843935397718.0 / 7227975568851.0,
            6206560082614.0 / 27846110321329.0,
            2841125392315.0 / 14844217636077.0
        ],
        &[
            0.0 / 1.0,
            -2491873887327.0 / 11519757507826.0,
            -3833614938189.0 / 14183712281236.0,
            628609886693.0 / 8177399110319.0,
            -4943723744483.0 / 2558074780976.0,
            1024000837540.0 / 1998038638351.0,
            -2492809296391.0 / 9064568868273.0
        ]
    ],
    &[
        346820227625.0 / 3124407780749.0,
        0.0 / 1.0,
        0.0 / 1.0,
        0.0 / 1.0,
        814249513470.0 / 2521483007009.0,
        195246859987.0 / 15831935944600.0,
        3570596951509.0 / 9788921605312.0
    ],
    &[
        1298271176151.0 / 60748409385661.0,
        57828749177833338114741189.0 / 482418531105044571804353902.0,
        16431909216114342992530887716659137419.0 / 50972944352640941110022041298448213332.0,
        843711271601954807241466442429582743082.0 / 2361379786784371499429045948205315798717.0,
        45377346645618697840609101263059649515.0 / 57769368855607143441437855651622233424.0,
        147132600561369761792017800077859262701.0 / 173834563932749284125206995856250290771.0,
        123785620236259768586332555932209432529705897037921.0
            / 353351523019265026737831367789312912172448045683187.0
    ]
);
method_rp!(
    CKLLSRK54_3N_4R,
    CKLLSRK54_3N_4RCoefficients,
    "Pinned CKLLSRK54_3N_4R low-storage register-pipeline method.",
    3,
    &[
        &[
            9435338793489.0 / 32856462503258.0,
            6195609865473.0 / 14441396468602.0,
            7502925572378.0 / 28098850972003.0,
            4527781290407.0 / 9280887680514.0
        ],
        &[
            0.0 / 1.0,
            2934593324920.0 / 16923654741811.0,
            16352725096886.0 / 101421723321009.0,
            3004243580591.0 / 16385320447374.0
        ],
        &[
            0.0 / 1.0,
            0.0 / 1.0,
            390352446067.0 / 5989890148791.0,
            902830387041.0 / 8154716972155.0
        ]
    ],
    &[
        929310922418.0 / 8329727308495.0,
        4343420149496.0 / 15735497610667.0,
        885252399220.0 / 9490460854667.0,
        3341719902227.0 / 13464012733180.0
    ],
    &[
        9435338793489.0 / 32856462503258.0,
        147231987957505837822553443.0 / 244401207824228867478118222.0,
        401086457089554669663078760253749450489.0 / 812866282711293513804077001645679258017.0,
        153823244836258719400905156342054669945035476219421.0
            / 172160249040778711548900853819650745575758693592285.0
    ]
);
method_rp!(
    CKLLSRK54_3M_4R,
    CKLLSRK54_3M_4RCoefficients,
    "Pinned CKLLSRK54_3M_4R low-storage register-pipeline method.",
    3,
    &[
        &[
            7142524119.0 / 20567653057.0,
            20567653057.0 / 89550000000.0,
            7407775.0 / 2008982.0,
            -4577300.0 / 867302297.0
        ],
        &[
            0.0 / 1.0,
            15198616943.0 / 89550000000.0,
            -226244183627.0 / 80359280000.0,
            33311687500.0 / 8703531091.0
        ],
        &[
            0.0 / 1.0,
            0.0 / 1.0,
            9890667227.0 / 80359280000.0,
            -20567653057.0 / 6979191486.0
        ]
    ],
    &[
        297809.0 / 2384418.0,
        0.0 / 1.0,
        156250000.0 / 270591503.0,
        5030000.0 / 888933.0
    ],
    &[
        7142524119.0 / 20567653057.0,
        1997.0 / 5000.0,
        199.0 / 200.0,
        1.0
    ]
);
method_rp!(
    CKLLSRK65_4M_4R,
    CKLLSRK65_4M_4RCoefficients,
    "Pinned CKLLSRK65_4M_4R low-storage register-pipeline method.",
    3,
    &[
        &[
            1811061732419.0 / 6538712036350.0,
            936386506953.0 / 6510757757683.0,
            8253430823511.0 / 9903985211908.0,
            4157325866175.0 / 11306150349782.0,
            3299942024581.0 / 13404534943033.0
        ],
        &[
            0.0 / 1.0,
            968127049827.0 / 6993254963231.0,
            -4242729801665.0 / 12001587034923.0,
            1960956671631.0 / 3017447659538.0,
            2088737530132.0 / 14638867961951.0
        ],
        &[
            0.0 / 1.0,
            0.0 / 1.0,
            332803037697.0 / 7529436905221.0,
            -19590089343957.0 / 51581831082203.0,
            3811366828049.0 / 10653298326636.0
        ]
    ],
    &[
        1437717300581.0 / 14622899446031.0,
        0.0 / 1.0,
        3070006287879.0 / 9321175678070.0,
        2276970273632.0 / 7940670647385.0,
        -1056149936631.0 / 7427907425983.0
    ],
    &[
        1811061732419.0 / 6538712036350.0,
        12851630287335503073915984.0 / 45531389003311376172753773.0,
        468994575306978457607500930904657513641.0 / 894975528626103930282351283769588361564.0,
        4735520442856752193881763097298943558246492547269018.0
            / 6433166018040288425494806218280078848936316641536447.0,
        25828983228256103590265182981008154883102570637999497.0
            / 30568689961801519095090666149791133914967119469889228.0
    ]
);
method_rp!(
    CKLLSRK85_4FM_4R,
    CKLLSRK85_4FM_4RCoefficients,
    "Pinned CKLLSRK85_4FM_4R low-storage register-pipeline method.",
    3,
    &[
        &[
            319960152914.0 / 39034091721739.0,
            16440040368765.0 / 7252463661539.0,
            1381950791880.0 / 6599155371617.0,
            18466735994895.0 / 7394178462407.0,
            2786140924985.0 / 14262827431161.0,
            28327099865656.0 / 21470840267743.0,
            0.0 / 1.0
        ],
        &[
            0.0 / 1.0,
            -16195115415565.0 / 7808461210678.0,
            -1316066362688.0 / 10261382634081.0,
            -23893000145797.0 / 9614512377075.0,
            6556893593075.0 / 12530787773541.0,
            -5015572218207.0 / 5719938983072.0,
            0.0 / 1.0
        ],
        &[
            0.0 / 1.0,
            0.0 / 1.0,
            334167490531.0 / 1677017272502.0,
            4579492417936.0 / 7930641522963.0,
            -2255846922213.0 / 30066310003000.0,
            3212719728776.0 / 7037340048693.0,
            0.0 / 1.0
        ]
    ],
    &[
        1147876221211.0 / 13910763665259.0,
        0.0 / 1.0,
        182134362610.0 / 9852075053293.0,
        3396705055007.0 / 8495597747463.0,
        363006049056.0 / 22366003978609.0,
        6078825123673.0 / 15200143133108.0,
        583593328277.0 / 7028929464160.0
    ],
    &[
        319960152914.0 / 39034091721739.0,
        10916931475666701983218135.0 / 56630581182979020764713442.0,
        31845189551971545944223680050155078355.0 / 113561670251926090809438891701398790454.0,
        585892393366635581491792016142825500310911249371223.0
            / 871432942801472160798333604371480303171919616321325.0,
        6030664727234996630401450278844701818157369618311237.0
            / 8305630304762506786823923305099106403075216590053000.0,
        1.0,
        194373043039840208108258122050794558876.0 / 388106905684556737922360607016380520227.0
    ]
);
method_rp!(
    CKLLSRK75_4M_5R,
    CKLLSRK75_4M_5RCoefficients,
    "Pinned CKLLSRK75_4M_5R low-storage register-pipeline method.",
    4,
    &[
        &[
            984894634849.0 / 6216792334776.0,
            984894634849.0 / 5526037630912.0,
            13256335809797.0 / 10977774807827.0,
            5386479425293.0 / 11045691190948.0,
            -1717767168952.0 / 11602237717369.0,
            -10054679524430.0 / 10306851287569.0
        ],
        &[
            0.0 / 1.0,
            890852251480.0 / 14995156510369.0,
            -18544705752398.0 / 18426539884027.0,
            1115398761892.0 / 28058504699217.0,
            5538441135605.0 / 13014942352969.0,
            23855853001162.0 / 20968156556405.0
        ],
        &[
            0.0 / 1.0,
            0.0 / 1.0,
            1722683259617.0 / 5669183367476.0,
            342961171087.0 / 6505721096888.0,
            -14472869285404.0 / 19736045536601.0,
            -8169744035288.0 / 5424738459363.0
        ],
        &[
            0.0 / 1.0,
            0.0 / 1.0,
            0.0 / 1.0,
            762111618422.0 / 5198184381557.0,
            2896263505307.0 / 6364015805096.0,
            60049403517654.0 / 26787923986853.0
        ]
    ],
    &[
        1008141064049.0 / 9867084721348.0,
        0.0 / 1.0,
        8222186491841.0 / 18352662300888.0,
        514621697208.0 / 8712119383831.0,
        1808964136873.0 / 4546032443428.0,
        -362754645297.0 / 3989911846061.0
    ],
    &[
        984894634849.0 / 6216792334776.0,
        19691532261044641782999041.0 / 82863799157714161922926528.0,
        579140763944732527715749105230082493541.0 / 1146776047854201324825397010814855303604.0,
        1904235205010770769196995566618512437342488019008993.0
            / 2620260981179174237577004881164696841381017975634264.0,
        4745866356039511505795256436748010529615723318082554645080208661.0
            / 46784744516176933667763632070461960177241008032286254911869725672.0,
        1.0
    ]
);

method_3sp!(
    RDPK3Sp35,
    RDPK3Sp35Coefficients,
    "Pinned RDPK3Sp35 3S-plus low-storage method.",
    false,
    &[
        2.587669070352079020144955303389306026e-01,
        -1.324366873994502973977035353758550057e-01,
        5.055601231460399101814291350373559483e-02,
        5.670552807902877312521811889846000976e-01
    ],
    &[
        5.528418745102160639901976698795928733e-01,
        6.731844400389673824374042790213570079e-01,
        2.803103804507635075215805236096803381e-01,
        5.521508873507393276457754945308880998e-01
    ],
    &[
        0.000000000000000000000000000000000000e+00,
        0.000000000000000000000000000000000000e+00,
        2.752585813446636957256614568573008811e-01,
        -8.950548709279785077579454232514633376e-01
    ],
    &[
        3.407687209321455242558804921815861422e-01,
        3.414399280584625023244387687873774697e-01,
        7.229302732875589702087936723400941329e-01,
        0.000000000000000000000000000000000000e+00
    ],
    2.300285062878154351930669430512780706e-01,
    &[
        3.021457892454169700189445968126242994e-01,
        8.025601039472704213300183888573974531e-01,
        4.362158997637629844305216319994356355e-01,
        1.129268494470295369172265188216779157e-01
    ],
    &[
        2.300285062878154351930669430512780706e-01,
        4.050049049262914975700372321130661410e-01,
        8.947823877926760224705450466361360720e-01,
        7.235108137218888081489570284485201518e-01
    ]
);
method_3sp!(
    RDPK3Sp49,
    RDPK3Sp49Coefficients,
    "Pinned RDPK3Sp49 3S-plus low-storage method.",
    false,
    &[
        -4.655641301259180308677051498071354582e+00,
        -7.720264924836063859141482018013692338e-01,
        -4.024423213419724605695005429153112050e+00,
        -2.129685246739018613087466942802498152e-02,
        -2.435022519234470128602335652131234586e+00,
        1.985627480986167686791439120784668251e-02,
        -2.810790112885283952929218377438668784e-01,
        1.689434895835535695524003319503844110e-01
    ],
    &[
        2.499262752607825957145627300817258023e+00,
        5.866820365436136799319929406678132638e-01,
        1.205141365412670762568835277881144391e+00,
        3.474793796700868848597960521248007941e-01,
        1.321346140128723105871355808477092220e+00,
        3.119636324379370564023292317172847140e-01,
        4.351419055894087609560896967082486864e-01,
        2.359698299440788299161958168555704234e-01
    ],
    &[
        0.000000000000000000000000000000000000e+00,
        0.000000000000000000000000000000000000e+00,
        7.621037111138170045618771082985664430e-01,
        -1.981182159087218433914909510116664154e-01,
        -6.228960706317566993192689455719570179e-01,
        -3.752246993432626328289874575355102038e-01,
        -3.355436539000946543242869676125143358e-01,
        -4.560963110717484359015342341157302403e-02
    ],
    &[
        1.262923854387806460989545005598562667e+00,
        7.574967177560872438940839460448329992e-01,
        5.163591158111222863455531895152351544e-01,
        -2.746333792042827389548936599648122146e-02,
        -4.382674653941770848797864513655752318e-01,
        1.273587103668392811985704533534301656e+00,
        -6.294740045442794829622796613103492913e-01,
        0.000000000000000000000000000000000000e+00
    ],
    2.836343531977826022543660465926414772e-01,
    &[
        9.736497978646965372894268287659773644e-01,
        3.382358566377620380505126936670933370e-01,
        -3.584937820217850715182820651063453804e-01,
        -4.113955814725134294322006403954822487e-03,
        1.427968962196019024010757034274849198e+00,
        1.808467712038743032991177525728915926e-02,
        1.605771316794521018947553625079465692e-01,
        2.952226811394310028003810072027839487e-01
    ],
    &[
        2.836343531977826022543660465926414772e-01,
        5.484073767552486705240014599676811834e-01,
        3.687229456675706936558667052479014150e-01,
        -6.806119916032093175251948474173648331e-01,
        3.518526451892056368706593492732753284e-01,
        1.665941920204672094647868254892387293e+00,
        9.715276989307335935187466054546761665e-01,
        9.051569554420045339601721625247585643e-01
    ]
);
method_3sp!(
    RDPK3Sp510,
    RDPK3Sp510Coefficients,
    "Pinned RDPK3Sp510 3S-plus low-storage method.",
    false,
    &[
        4.043660078504695837542588769963326988e-01,
        -8.503427464263185087039788184485627962e-01,
        -6.950894167072419998080989313353063399e+00,
        9.238765225328278557805080247596562995e-01,
        -2.563178039957404359875124580586147888e+00,
        2.545744869966347362604059848503340890e-01,
        3.125831733863168874151935287174374515e-01,
        -7.007114800567584871263283872289072079e-01,
        4.839620970980726631935174740648996010e-01
    ],
    &[
        6.871467069752345566001768382316915820e-01,
        1.093024760468898686510433898645775908e+00,
        3.225975382330161123625348062949430509e+00,
        1.041153700841396427100436517666787823e+00,
        1.292821488864702752767390075072674807e+00,
        7.391462769297006312785029455392854586e-01,
        1.239129257039300081860496157739352186e-01,
        1.842753479366766790220633908793933781e-01,
        5.712788942697077644959290025755003720e-02
    ],
    &[
        0.000000000000000000000000000000000000e+00,
        0.000000000000000000000000000000000000e+00,
        -2.393405159342139386425044844626597490e+00,
        -1.902854422095986544338294743445530533e+00,
        -2.820042210583207174321941694153843259e+00,
        -1.832698464130564949123807896975136336e+00,
        -2.199094510750697865007677774395365522e-01,
        -4.082430660384876496971887725512427800e-01,
        -1.377669791121207993339861855818881150e-01
    ],
    &[
        -1.331778409133849616712007380176762548e-01,
        8.260422785246030254485064732649153253e-01,
        1.513700430513332405798616943654007796e+00,
        -1.305810063177048110528482211982726539e+00,
        3.036678789342507704281817524408221954e+00,
        -1.449458267074592489788800461540171106e+00,
        3.834313873320957483471400258279635203e+00,
        4.122293971923324492772059928094971199e+00,
        0.000000000000000000000000000000000000e+00
    ],
    2.597883575710995826783320802193635406e-01,
    &[
        1.777008800169541694837687556103565007e-02,
        2.481636637328140606807905234325691851e-01,
        7.941736827560429420202759490815682546e-01,
        3.885391296871822541486945325814526190e-01,
        1.455051664264339366757555740296587660e-01,
        1.587517379462528932413419955691782412e-01,
        1.650605631567659573994022720500446501e-01,
        2.118093299943235065178000892467421832e-01,
        1.559392340339606299335442956580114440e-01
    ],
    &[
        2.597883575710995826783320802193635406e-01,
        9.904573115730917688557891428202061598e-02,
        2.155511882303785204133426661931565216e-01,
        5.007950078421880417512789524851012021e-01,
        5.592251914858131230054392022144328176e-01,
        5.449986973408778242805929551952000165e-01,
        7.615224662599497796472095353126697300e-01,
        8.427062083059167761623893618875787414e-01,
        9.152209807185253394871325258038753352e-01
    ]
);
method_3sp!(
    RDPK3SpFSAL35,
    RDPK3SpFSAL35Coefficients,
    "Pinned RDPK3SpFSAL35 3S-plus low-storage method.",
    true,
    &[
        2.587771979725733308135192812685323706e-01,
        -1.324380360140723382965420909764953437e-01,
        5.056033948190826045833606441415585735e-02,
        5.670532000739313812633197158607642990e-01
    ],
    &[
        5.528354909301389892439698870483746541e-01,
        6.731871608203061824849561782794643600e-01,
        2.803103963297672407841316576323901761e-01,
        5.521525447020610386070346724931300367e-01
    ],
    &[
        0.000000000000000000000000000000000000e+00,
        0.000000000000000000000000000000000000e+00,
        2.752563273304676380891217287572780582e-01,
        -8.950526174674033822276061734289327568e-01
    ],
    &[
        3.407655879334525365094815965895763636e-01,
        3.414382655003386206551709871126405331e-01,
        7.229275366787987419692007421895451953e-01,
        0.000000000000000000000000000000000000e+00
    ],
    2.300298624518076223899418286314123354e-01,
    &[
        3.021434166948288809034402119555380003e-01,
        8.025606185416310937583009085873554681e-01,
        4.362158943603440930655148245148766471e-01,
        1.129272530455059129782111662594436580e-01
    ],
    &[
        2.300298624518076223899418286314123354e-01,
        4.050046072094990912268498160116125481e-01,
        8.947822893693433545220710894560512805e-01,
        7.235136928826589010272834603680114769e-01
    ]
);
method_3sp!(
    RDPK3SpFSAL49,
    RDPK3SpFSAL49Coefficients,
    "Pinned RDPK3SpFSAL49 3S-plus low-storage method.",
    true,
    &[
        -4.655641447335068552684422206224169103e+00,
        -7.720265099645871829248487209517314217e-01,
        -4.024436690519806086742256154738379161e+00,
        -2.129676284018530966221583708648634733e-02,
        -2.435022509790109546199372365866450709e+00,
        1.985627297131987000579523283542615256e-02,
        -2.810791146791038566946663374735713961e-01,
        1.689434168754859644351230590422137972e-01
    ],
    &[
        2.499262792574495009336242992898153462e+00,
        5.866820377718875577451517985847920081e-01,
        1.205146086523094569925592464380295241e+00,
        3.474793722186732780030762737753849272e-01,
        1.321346060965113109321230804210670518e+00,
        3.119636464694193615946633676950358444e-01,
        4.351419539684379261368971206040518552e-01,
        2.359698130028753572503744518147537768e-01
    ],
    &[
        0.000000000000000000000000000000000000e+00,
        0.000000000000000000000000000000000000e+00,
        7.621006678721315291614677352949377871e-01,
        -1.981182504339400567765766904309673119e-01,
        -6.228959218699007450469629366684127462e-01,
        -3.752248380775956442989480369774937099e-01,
        -3.355438309135169811915662336248989661e-01,
        -4.560955005031121479972862973705108039e-02
    ],
    &[
        1.262923876648114432874834923838556100e+00,
        7.574967189685911558308119415539596711e-01,
        5.163589453140728104667573195005629833e-01,
        -2.746327421802609557034437892013640319e-02,
        -4.382673178127944142238606608356542890e-01,
        1.273587294602656522645691372699677063e+00,
        -6.294740283927400326554066998751383342e-01,
        0.000000000000000000000000000000000000e+00
    ],
    2.836343005184365275160654678626695428e-01,
    &[
        9.736500104654741223716056170419660217e-01,
        3.382359225242515288768487569778320563e-01,
        -3.584943611106183357043212309791897386e-01,
        -4.113944068471528211627210454497620358e-03,
        1.427968894048586363415504654313371031e+00,
        1.808470948394314017665968411915568633e-02,
        1.605770645946802213926893453819236685e-01,
        2.952227015964591648775833803635147962e-01
    ],
    &[
        2.836343005184365275160654678626695428e-01,
        5.484076570002894365286665352032296535e-01,
        3.687228761669438493478872632332010073e-01,
        -6.806126440140844191258463830024463902e-01,
        3.518526124230705801739919476290327750e-01,
        1.665941994879593315477304663913129942e+00,
        9.715279295934715835299192116436237065e-01,
        9.051569840159589594903399929316959062e-01
    ]
);
method_3sp!(
    RDPK3SpFSAL510,
    RDPK3SpFSAL510Coefficients,
    "Pinned RDPK3SpFSAL510 3S-plus low-storage method.",
    true,
    &[
        4.043660121685749695640462197806189975e-01,
        -8.503427289575839690883191973980814832e-01,
        -6.950894175262117526410215315179482885e+00,
        9.238765192731084931855438934978371889e-01,
        -2.563178056509891340215942413817786020e+00,
        2.545744879365226143946122067064118430e-01,
        3.125831707411998258746812355492206137e-01,
        -7.007114414440507927791249989236719346e-01,
        4.839621016023833375810172323297465039e-01
    ],
    &[
        6.871467028161416909922221357014564412e-01,
        1.093024748914750833700799552463885117e+00,
        3.225975379607193001678365742708874597e+00,
        1.041153702510101386914019859778740444e+00,
        1.292821487912164945157744726076279306e+00,
        7.391462755788122847651304143259254381e-01,
        1.239129251371800313941948224441873274e-01,
        1.842753472370123193132193302369345580e-01,
        5.712788998796583446479387686662738843e-02
    ],
    &[
        0.000000000000000000000000000000000000e+00,
        0.000000000000000000000000000000000000e+00,
        -2.393405133244194727221124311276648940e+00,
        -1.902854422421760920850597670305403139e+00,
        -2.820042207399977261483046412236557428e+00,
        -1.832698465277380999601896111079977378e+00,
        -2.199094483084671192328083958346519535e-01,
        -4.082430635847870963724591602173546218e-01,
        -1.377669797880289713535665985132703979e-01
    ],
    &[
        -1.331778419508803397033287009506932673e-01,
        8.260422814750207498262063505871077303e-01,
        1.513700425755728332485300719652378197e+00,
        -1.305810059935023735972298885749903694e+00,
        3.036678802924163246003321318996156380e+00,
        -1.449458274398895177922690618003584514e+00,
        3.834313899176362315089976408899373409e+00,
        4.122293760012985409330881631526514714e+00,
        0.000000000000000000000000000000000000e+00
    ],
    2.597883554788674084039539165398464630e-01,
    &[
        1.777008889438867858759149597539211023e-02,
        2.481636629715501931294746189266601496e-01,
        7.941736871152005775821844297293296135e-01,
        3.885391285642019129575902994397298066e-01,
        1.455051657916305055730603387469193768e-01,
        1.587517385964749337690916959584348979e-01,
        1.650605617880053419242434594242509601e-01,
        2.118093284937153836908655490906875007e-01,
        1.559392342362059886106995325687547506e-01
    ],
    &[
        2.597883554788674084039539165398464630e-01,
        9.904573247592460887087003212056568980e-02,
        2.155511890524058691860390281856497503e-01,
        5.007950088969676776844289399972611534e-01,
        5.592251911688643533787800688765883636e-01,
        5.449986978853637084972622392134732553e-01,
        7.615224694532590139829150720490417596e-01,
        8.427062083267360939805493320684741215e-01,
        9.152209805057669959657927210873423883e-01
    ]
);

fn integrate<F, P, T>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
) -> Result<Solution, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
    T: LowStorage2N,
{
    validate_recurrence::<T>()?;
    drive_integration(
        problem,
        options,
        LowStorageKernel::<T>::new(problem.initial_state().len()),
    )
}

fn integrate_3s<F, P, T>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
) -> Result<Solution, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
    T: LowStorage3S,
{
    validate_recurrence_3s::<T>()?;
    drive_integration(
        problem,
        options,
        LowStorage3SKernel::<T>::new(problem.initial_state().len()),
    )
}

fn integrate_2c<F, P, T>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
) -> Result<Solution, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
    T: LowStorage2C,
{
    validate_recurrence_2c::<T>()?;
    drive_integration(
        problem,
        options,
        LowStorage2CKernel::<T>::new(problem.initial_state().len()),
    )
}

fn integrate_alternating_2n<F, P, T>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
) -> Result<Solution, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
    T: LowStorageAlternating2N,
{
    validate_alternating_recurrence::<T>()?;
    drive_integration(
        problem,
        options,
        LowStorageAlternating2NKernel::<T>::new(problem.initial_state().len()),
    )
}

fn integrate_rp<F, P, T>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
) -> Result<Solution, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
    T: LowStorageRP,
{
    validate_recurrence_rp::<T>()?;
    drive_integration(
        problem,
        options,
        LowStorageRPKernel::<T>::new(problem.initial_state().len()),
    )
}

fn validate_recurrence<T: LowStorage2N>() -> Result<(), SolveError> {
    if T::A.len() + 1 != T::B.len() || T::A.len() != T::C.len() {
        return Err(SolveError::InvalidTableau);
    }
    Ok(())
}

fn validate_recurrence_3s<T: LowStorage3S>() -> Result<(), SolveError> {
    let stages = T::GAMMA1.len();
    if stages == 0
        || T::GAMMA2.len() != stages
        || T::GAMMA3.len() != stages
        || T::DELTA.len() != stages
        || T::BETA2.len() != stages
        || T::C.len() != stages
    {
        return Err(SolveError::InvalidTableau);
    }
    Ok(())
}

fn validate_recurrence_2c<T: LowStorage2C>() -> Result<(), SolveError> {
    if T::A.len() + 1 != T::B.len() || T::A.len() != T::C.len() {
        return Err(SolveError::InvalidTableau);
    }
    Ok(())
}

fn validate_alternating_recurrence<T: LowStorageAlternating2N>() -> Result<(), SolveError> {
    for (a, b, c) in [(T::A1, T::B1, T::C1), (T::A2, T::B2, T::C2)] {
        if a.len() + 1 != b.len() || a.len() != c.len() {
            return Err(SolveError::InvalidTableau);
        }
    }
    Ok(())
}

fn validate_recurrence_rp<T: LowStorageRP>() -> Result<(), SolveError> {
    let stages = T::C.len();
    if T::B.len() != stages
        || T::C.len() != stages
        || T::A.len() != T::HISTORY_STATES
        || T::A.iter().any(|a| a.len() != stages)
        || T::HISTORY_STATES == 0
    {
        return Err(SolveError::InvalidTableau);
    }
    Ok(())
}

struct LowStorageKernel<T> {
    derivative: Vec<f64>,
    residual: Vec<f64>,
    marker: PhantomData<fn() -> T>,
}

struct LowStorage3SKernel<T> {
    derivative: Vec<f64>,
    temporary: Vec<f64>,
    marker: PhantomData<fn() -> T>,
}

struct LowStorage2CKernel<T> {
    derivative: Vec<f64>,
    temporary: Vec<f64>,
    marker: PhantomData<fn() -> T>,
}

struct LowStorageAlternating2NKernel<T> {
    derivative: Vec<f64>,
    residual: Vec<f64>,
    second_tableau: bool,
    marker: PhantomData<fn() -> T>,
}

struct LowStorageRPKernel<T> {
    derivative: Vec<f64>,
    gprev: Vec<f64>,
    history_states: Vec<Vec<f64>>,
    history_derivatives: Vec<Vec<f64>>,
    marker: PhantomData<fn() -> T>,
}

impl<T> LowStorage3SKernel<T> {
    fn new(dimension: usize) -> Self {
        Self {
            derivative: vec![0.0; dimension],
            temporary: vec![0.0; dimension],
            marker: PhantomData,
        }
    }
}

impl<T> LowStorage2CKernel<T> {
    fn new(dimension: usize) -> Self {
        Self {
            derivative: vec![0.0; dimension],
            temporary: vec![0.0; dimension],
            marker: PhantomData,
        }
    }
}

impl<T> LowStorageAlternating2NKernel<T> {
    fn new(dimension: usize) -> Self {
        Self {
            derivative: vec![0.0; dimension],
            residual: vec![0.0; dimension],
            second_tableau: false,
            marker: PhantomData,
        }
    }
}

impl<T> LowStorageRPKernel<T>
where
    T: LowStorageRP,
{
    fn new(dimension: usize) -> Self {
        Self {
            derivative: vec![0.0; dimension],
            gprev: vec![0.0; dimension],
            history_states: (0..T::HISTORY_STATES)
                .map(|_| vec![0.0; dimension])
                .collect(),
            history_derivatives: (0..T::HISTORY_STATES.saturating_sub(1))
                .map(|_| vec![0.0; dimension])
                .collect(),
            marker: PhantomData,
        }
    }
}

impl<T> LowStorageKernel<T> {
    fn new(dimension: usize) -> Self {
        Self {
            derivative: vec![0.0; dimension],
            residual: vec![0.0; dimension],
            marker: PhantomData,
        }
    }
}

impl<F, P, T> StepKernel<F, P> for LowStorageKernel<T>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
    T: LowStorage2N,
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::new(false, 1)
    }

    fn initialize(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: f64,
        _: &mut SolverStats,
    ) -> Result<(), SolveError> {
        // Stage zero evaluates the current derivative on every attempt.
        Ok(())
    }

    fn estimate_initial_step(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: f64,
        _: f64,
        _: f64,
        _: &mut [f64],
        _: &SolveOptions,
        _: &mut SolverStats,
    ) -> Result<f64, SolveError> {
        Err(SolveError::InitialStepRequired)
    }

    fn attempt_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        _: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError> {
        candidate.copy_from_slice(state);
        evaluate(problem, &mut self.derivative, state, time, stats)?;
        for ((residual, candidate), derivative) in self
            .residual
            .iter_mut()
            .zip(&mut *candidate)
            .zip(&self.derivative)
        {
            *residual = step * derivative;
            *candidate += T::B[0] * *residual;
        }
        for stage in 0..T::A.len() {
            evaluate(
                problem,
                &mut self.derivative,
                candidate,
                time + T::C[stage] * step,
                stats,
            )?;
            for ((residual, candidate), derivative) in self
                .residual
                .iter_mut()
                .zip(&mut *candidate)
                .zip(&self.derivative)
            {
                *residual = T::A[stage] * *residual + step * derivative;
                *candidate += T::B[stage + 1] * *residual;
            }
        }
        ensure_finite(candidate)?;
        Ok(StepEstimate::new(0.0))
    }

    fn accept_step(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: &[f64],
        _: f64,
        _: f64,
        _: bool,
        _: &mut SolverStats,
    ) -> Result<(), SolveError> {
        Ok(())
    }

    fn reject_step(&mut self) {}
}

impl<F, P, T> StepKernel<F, P> for LowStorage3SKernel<T>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
    T: LowStorage3S,
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::new(false, 1)
    }

    fn initialize(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: f64,
        _: &mut SolverStats,
    ) -> Result<(), SolveError> {
        Ok(())
    }

    fn estimate_initial_step(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: f64,
        _: f64,
        _: f64,
        _: &mut [f64],
        _: &SolveOptions,
        _: &mut SolverStats,
    ) -> Result<f64, SolveError> {
        Err(SolveError::InitialStepRequired)
    }

    fn attempt_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        _: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError> {
        candidate.copy_from_slice(state);
        self.temporary.copy_from_slice(state);
        evaluate(problem, &mut self.derivative, state, time, stats)?;
        for (candidate, derivative) in candidate.iter_mut().zip(&self.derivative) {
            *candidate += T::BETA1 * step * *derivative;
        }
        for stage in 0..T::GAMMA1.len() {
            evaluate(
                problem,
                &mut self.derivative,
                candidate,
                time + T::C[stage] * step,
                stats,
            )?;
            for (((candidate, temporary), derivative), state_value) in candidate
                .iter_mut()
                .zip(&mut self.temporary)
                .zip(&self.derivative)
                .zip(state)
            {
                *temporary += T::DELTA[stage] * *candidate;
                *candidate = T::GAMMA1[stage] * *candidate
                    + T::GAMMA2[stage] * *temporary
                    + T::GAMMA3[stage] * *state_value
                    + T::BETA2[stage] * step * *derivative;
            }
        }
        if T::EVALUATE_ENDPOINT {
            // The pinned implementation evaluates the endpoint derivative for
            // FSAL/interpolation bookkeeping even though this fixed-step driver
            // does not reuse it.
            evaluate(problem, &mut self.derivative, candidate, time + step, stats)?;
        }
        ensure_finite(candidate)?;
        Ok(StepEstimate::new(0.0))
    }

    fn accept_step(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: &[f64],
        _: f64,
        _: f64,
        _: bool,
        _: &mut SolverStats,
    ) -> Result<(), SolveError> {
        Ok(())
    }

    fn reject_step(&mut self) {}
}

impl<F, P, T> StepKernel<F, P> for LowStorage2CKernel<T>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
    T: LowStorage2C,
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::new(false, 1)
    }

    fn initialize(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: f64,
        _: &mut SolverStats,
    ) -> Result<(), SolveError> {
        Ok(())
    }

    fn estimate_initial_step(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: f64,
        _: f64,
        _: f64,
        _: &mut [f64],
        _: &SolveOptions,
        _: &mut SolverStats,
    ) -> Result<f64, SolveError> {
        Err(SolveError::InitialStepRequired)
    }

    fn attempt_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        _: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError> {
        candidate.copy_from_slice(state);
        evaluate(problem, &mut self.derivative, state, time, stats)?;
        for (candidate, derivative) in candidate.iter_mut().zip(&self.derivative) {
            *candidate += T::B[0] * step * *derivative;
        }
        for stage in 0..T::A.len() {
            self.temporary.copy_from_slice(candidate);
            for (temporary, derivative) in self.temporary.iter_mut().zip(&self.derivative) {
                *temporary += T::A[stage] * step * *derivative;
            }
            evaluate(
                problem,
                &mut self.derivative,
                &self.temporary,
                time + T::C[stage] * step,
                stats,
            )?;
            for (candidate, derivative) in candidate.iter_mut().zip(&self.derivative) {
                *candidate += T::B[stage + 1] * step * *derivative;
            }
        }
        ensure_finite(candidate)?;
        Ok(StepEstimate::new(0.0))
    }

    fn accept_step(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: &[f64],
        _: f64,
        _: f64,
        _: bool,
        _: &mut SolverStats,
    ) -> Result<(), SolveError> {
        Ok(())
    }

    fn reject_step(&mut self) {}
}

impl<F, P, T> StepKernel<F, P> for LowStorageAlternating2NKernel<T>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
    T: LowStorageAlternating2N,
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::new(false, 1)
    }

    fn initialize(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: f64,
        _: &mut SolverStats,
    ) -> Result<(), SolveError> {
        Ok(())
    }

    fn estimate_initial_step(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: f64,
        _: f64,
        _: f64,
        _: &mut [f64],
        _: &SolveOptions,
        _: &mut SolverStats,
    ) -> Result<f64, SolveError> {
        Err(SolveError::InitialStepRequired)
    }

    fn attempt_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        _: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError> {
        let (a, b, c) = if self.second_tableau {
            (T::A2, T::B2, T::C2)
        } else {
            (T::A1, T::B1, T::C1)
        };
        candidate.copy_from_slice(state);
        evaluate(problem, &mut self.derivative, state, time, stats)?;
        for ((residual, candidate), derivative) in self
            .residual
            .iter_mut()
            .zip(&mut *candidate)
            .zip(&self.derivative)
        {
            *residual = step * derivative;
            *candidate += b[0] * *residual;
        }
        for stage in 0..a.len() {
            evaluate(
                problem,
                &mut self.derivative,
                candidate,
                time + c[stage] * step,
                stats,
            )?;
            for ((residual, candidate), derivative) in self
                .residual
                .iter_mut()
                .zip(&mut *candidate)
                .zip(&self.derivative)
            {
                *residual = a[stage] * *residual + step * derivative;
                *candidate += b[stage + 1] * *residual;
            }
        }
        evaluate(problem, &mut self.derivative, candidate, time + step, stats)?;
        ensure_finite(candidate)?;
        Ok(StepEstimate::new(0.0))
    }

    fn accept_step(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: &[f64],
        _: f64,
        _: f64,
        _: bool,
        _: &mut SolverStats,
    ) -> Result<(), SolveError> {
        self.second_tableau = !self.second_tableau;
        Ok(())
    }

    fn reject_step(&mut self) {}
}

impl<F, P, T> StepKernel<F, P> for LowStorageRPKernel<T>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
    T: LowStorageRP,
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::new(false, 1)
    }

    fn initialize(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: f64,
        _: &mut SolverStats,
    ) -> Result<(), SolveError> {
        Ok(())
    }

    fn estimate_initial_step(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: f64,
        _: f64,
        _: f64,
        _: &mut [f64],
        _: &SolveOptions,
        _: &mut SolverStats,
    ) -> Result<f64, SolveError> {
        Err(SolveError::InitialStepRequired)
    }

    fn attempt_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        _: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError> {
        candidate.copy_from_slice(state);
        for history in &mut self.history_states {
            history.copy_from_slice(state);
        }
        for history in &mut self.history_derivatives {
            history.fill(0.0);
        }
        evaluate(problem, &mut self.derivative, state, time, stats)?;
        for stage in 0..T::C.len() {
            self.gprev.copy_from_slice(
                self.history_states
                    .last()
                    .expect("validated register pipeline"),
            );
            for (value, derivative) in self.gprev.iter_mut().zip(&self.derivative) {
                *value += T::A[0][stage] * step * *derivative;
            }
            for (register, coefficients) in self.history_derivatives.iter().zip(T::A.iter().skip(1))
            {
                for ((value, derivative), coefficient) in self
                    .gprev
                    .iter_mut()
                    .zip(register)
                    .zip(std::iter::repeat(&coefficients[stage]))
                {
                    *value += *derivative * *coefficient * step;
                }
            }
            for (candidate, derivative) in candidate.iter_mut().zip(&self.derivative) {
                *candidate += T::B[stage] * step * *derivative;
            }
            for index in (1..self.history_derivatives.len()).rev() {
                let (head, tail) = self.history_derivatives.split_at_mut(index);
                tail[0].copy_from_slice(&head[index - 1]);
            }
            if let Some(history) = self.history_derivatives.first_mut() {
                history.copy_from_slice(&self.derivative);
            }
            for index in (1..self.history_states.len()).rev() {
                let (head, tail) = self.history_states.split_at_mut(index);
                tail[0].copy_from_slice(&head[index - 1]);
            }
            self.history_states[0].copy_from_slice(candidate);
            evaluate(
                problem,
                &mut self.derivative,
                &self.gprev,
                time + T::C[stage] * step,
                stats,
            )?;
        }
        for (candidate, derivative) in candidate.iter_mut().zip(&self.derivative) {
            *candidate += T::B_FINAL * step * *derivative;
        }
        evaluate(problem, &mut self.derivative, candidate, time + step, stats)?;
        ensure_finite(candidate)?;
        Ok(StepEstimate::new(0.0))
    }

    fn accept_step(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: &[f64],
        _: f64,
        _: f64,
        _: bool,
        _: &mut SolverStats,
    ) -> Result<(), SolveError> {
        Ok(())
    }

    fn reject_step(&mut self) {}
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
    (problem.rhs)(derivative, state, problem.parameters(), time);
    stats.rhs_evaluations += 1;
    ensure_finite(derivative)
}

fn ensure_finite(values: &[f64]) -> Result<(), SolveError> {
    values
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteDerivative)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use super::{
        CFRLDDRK64, CKLLSRK43_2, CKLLSRK54_3C, CKLLSRK54_3C_3R, CKLLSRK54_3M_3R, CKLLSRK54_3M_4R,
        CKLLSRK54_3N_3R, CKLLSRK54_3N_4R, CKLLSRK65_4M_4R, CKLLSRK75_4M_5R, CKLLSRK85_4C_3R,
        CKLLSRK85_4FM_4R, CKLLSRK85_4M_3R, CKLLSRK85_4P_3R, CKLLSRK95_4C, CKLLSRK95_4M,
        CKLLSRK95_4S, CarpenterKennedy2N54, Dglddrk73C, Dglddrk84C, Dglddrk84F, Ndblsrk124,
        Ndblsrk134, Ndblsrk144, Ork256, ParsaniKetchesonDeconinck3S32,
        ParsaniKetchesonDeconinck3S53, ParsaniKetchesonDeconinck3S82,
        ParsaniKetchesonDeconinck3S94, ParsaniKetchesonDeconinck3S105,
        ParsaniKetchesonDeconinck3S173, ParsaniKetchesonDeconinck3S184,
        ParsaniKetchesonDeconinck3S205, RDPK3Sp35, RDPK3Sp49, RDPK3Sp510, RDPK3SpFSAL35,
        RDPK3SpFSAL49, RDPK3SpFSAL510, RK46NL, SHLDDRK_2N, SHLDDRK52, Shlddrk64, TSLDDRK74,
        integrate,
    };

    struct Malformed3S;

    impl super::LowStorage3S for Malformed3S {
        const GAMMA1: &'static [f64] = &[0.0];
        const GAMMA2: &'static [f64] = &[];
        const GAMMA3: &'static [f64] = &[0.0];
        const DELTA: &'static [f64] = &[0.0];
        const BETA1: f64 = 1.0;
        const BETA2: &'static [f64] = &[1.0];
        const C: &'static [f64] = &[0.0];
    }
    use crate::{
        CallbackAction, OdeAlgorithm, OdeProblem, SaveMode, SolveError, SolveOptions, solve,
    };

    type TestRhs = fn(&mut [f64], &[f64], &(), f64);

    fn problem(time_span: (f64, f64), initial: f64) -> OdeProblem<TestRhs, ()> {
        fn rhs(du: &mut [f64], u: &[f64], _: &(), time: f64) {
            du[0] = u[0] + time;
        }
        OdeProblem::new(rhs, vec![initial], time_span, ())
    }

    fn options(step: f64) -> SolveOptions {
        SolveOptions {
            adaptive: false,
            initial_step: Some(step),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        }
    }

    fn endpoint<A: OdeAlgorithm>(algorithm: A, step: f64) -> f64 {
        solve(&problem((0.0, 1.0), 1.0), algorithm, &options(step))
            .unwrap()
            .last_state()[0]
    }

    fn order<A: OdeAlgorithm + Copy>(algorithm: A) -> f64 {
        let exact = 2.0 * std::f64::consts::E - 2.0;
        let coarse = (endpoint(algorithm, 0.1) - exact).abs();
        let fine = (endpoint(algorithm, 0.05) - exact).abs();
        (coarse / fine).log2()
    }

    #[test]
    fn methods_recover_their_design_orders() {
        assert!(order(Ork256) > 1.9);
        assert!(order(ParsaniKetchesonDeconinck3S32) > 1.8);
        assert!(order(ParsaniKetchesonDeconinck3S53) > 2.8);
        assert!(order(ParsaniKetchesonDeconinck3S173) > 2.8);
        assert!(order(ParsaniKetchesonDeconinck3S105) > 4.7);
        assert!(order(ParsaniKetchesonDeconinck3S82) > 1.8);
        assert!(order(ParsaniKetchesonDeconinck3S94) > 3.75);
        assert!(order(ParsaniKetchesonDeconinck3S184) > 3.75);
        assert!(order(ParsaniKetchesonDeconinck3S205) > 4.7);
        assert!(order(Dglddrk73C) > 2.9);
        for (name, observed) in [
            ("CarpenterKennedy2N54", order(CarpenterKennedy2N54)),
            ("DGLDDRK84_C", order(Dglddrk84C)),
            ("DGLDDRK84_F", order(Dglddrk84F)),
            ("NDBLSRK124", order(Ndblsrk124)),
            ("NDBLSRK134", order(Ndblsrk134)),
            ("NDBLSRK144", order(Ndblsrk144)),
        ] {
            assert!(observed > 3.75, "{name} observed order was {observed}");
        }

        // The pinned upstream suite marks SHLDDRK64's order checks broken due
        // to the published coefficients' limited precision. Keep its exact
        // recurrence covered without asserting an order upstream cannot meet.
        assert!(endpoint(Shlddrk64, 0.01).is_finite());
    }

    #[test]
    fn remaining_low_storage_families_execute_one_step() {
        macro_rules! exercise {
            ($($algorithm:expr),+ $(,)?) => {
                $(assert!(endpoint($algorithm, 0.01).is_finite(), stringify!($algorithm));)+
            };
        }

        exercise!(
            RK46NL,
            CFRLDDRK64,
            TSLDDRK74,
            SHLDDRK52,
            SHLDDRK_2N,
            RDPK3Sp35,
            RDPK3Sp49,
            RDPK3Sp510,
            RDPK3SpFSAL35,
            RDPK3SpFSAL49,
            RDPK3SpFSAL510,
            CKLLSRK43_2,
            CKLLSRK54_3C,
            CKLLSRK95_4S,
            CKLLSRK95_4C,
            CKLLSRK95_4M,
            CKLLSRK54_3C_3R,
            CKLLSRK54_3M_3R,
            CKLLSRK54_3N_3R,
            CKLLSRK85_4C_3R,
            CKLLSRK85_4M_3R,
            CKLLSRK85_4P_3R,
            CKLLSRK54_3N_4R,
            CKLLSRK54_3M_4R,
            CKLLSRK65_4M_4R,
            CKLLSRK85_4FM_4R,
            CKLLSRK75_4M_5R,
        );
    }

    #[test]
    fn callbacks_save_at_and_backward_integration_use_shared_semantics() {
        let backward = problem((1.0, 0.0), 2.0 * std::f64::consts::E - 2.0);
        let backward_options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.01),
            save_at: vec![1.0, 0.5, 0.0],
            ..SolveOptions::default()
        };
        let solution = solve(&backward, CarpenterKennedy2N54, &backward_options).unwrap();
        assert_eq!(solution.times(), &[1.0, 0.5, 0.0]);
        assert!((solution.last_state()[0] - 1.0).abs() < 1.0e-8);

        let solution = solve(&backward, ParsaniKetchesonDeconinck3S32, &backward_options).unwrap();
        assert_eq!(solution.times(), &[1.0, 0.5, 0.0]);
        assert!((solution.last_state()[0] - 1.0).abs() < 2.0e-3);

        let solution = solve(&backward, ParsaniKetchesonDeconinck3S53, &backward_options).unwrap();
        assert_eq!(solution.times(), &[1.0, 0.5, 0.0]);
        assert!((solution.last_state()[0] - 1.0).abs() < 2.0e-5);

        let solution = solve(&backward, ParsaniKetchesonDeconinck3S173, &backward_options).unwrap();
        assert_eq!(solution.times(), &[1.0, 0.5, 0.0]);
        assert!((solution.last_state()[0] - 1.0).abs() < 2.0e-5);

        let solution = solve(&backward, ParsaniKetchesonDeconinck3S105, &backward_options).unwrap();
        assert_eq!(solution.times(), &[1.0, 0.5, 0.0]);
        assert!((solution.last_state()[0] - 1.0).abs() < 2.0e-8);

        let solution = solve(&backward, ParsaniKetchesonDeconinck3S82, &backward_options).unwrap();
        assert_eq!(solution.times(), &[1.0, 0.5, 0.0]);
        assert!((solution.last_state()[0] - 1.0).abs() < 2.0e-3);

        let solution = solve(&backward, ParsaniKetchesonDeconinck3S94, &backward_options).unwrap();
        assert_eq!(solution.times(), &[1.0, 0.5, 0.0]);
        assert!((solution.last_state()[0] - 1.0).abs() < 2.0e-5);

        let solution = solve(&backward, ParsaniKetchesonDeconinck3S184, &backward_options).unwrap();
        assert_eq!(solution.times(), &[1.0, 0.5, 0.0]);
        assert!((solution.last_state()[0] - 1.0).abs() < 2.0e-5);

        let solution = solve(&backward, ParsaniKetchesonDeconinck3S205, &backward_options).unwrap();
        assert_eq!(solution.times(), &[1.0, 0.5, 0.0]);
        assert!((solution.last_state()[0] - 1.0).abs() < 2.0e-8);

        let terminating = problem((0.0, 1.0), 1.0)
            .with_continuous_callback(|_, _, time| time - 0.5, |_, _, _| CallbackAction::Terminate);
        let solution = solve(&terminating, Dglddrk73C, &options(0.1)).unwrap();
        assert!((solution.times().last().unwrap() - 0.5).abs() < 1.0e-14);
        assert_eq!(solution.stats().callback_invocations, 1);
    }

    #[test]
    fn malformed_three_register_coefficients_are_rejected() {
        assert_eq!(
            super::validate_recurrence_3s::<Malformed3S>(),
            Err(SolveError::InvalidTableau)
        );
    }

    #[test]
    fn three_register_callbacks_terminate_at_the_accepted_endpoint() {
        let problem = problem((0.0, 1.0), 1.0).with_discrete_callback(
            |_, _, time| time >= 0.25,
            |_, _, _| CallbackAction::Terminate,
        );
        let solution = solve(&problem, ParsaniKetchesonDeconinck3S32, &options(0.25)).unwrap();
        assert!((solution.times().last().unwrap() - 0.25).abs() < 1.0e-14);
        assert_eq!(solution.stats().callback_invocations, 1);

        let solution = solve(&problem, ParsaniKetchesonDeconinck3S53, &options(0.25)).unwrap();
        assert!((solution.times().last().unwrap() - 0.25).abs() < 1.0e-14);
        assert_eq!(solution.stats().callback_invocations, 1);

        let solution = solve(&problem, ParsaniKetchesonDeconinck3S173, &options(0.25)).unwrap();
        assert!((solution.times().last().unwrap() - 0.25).abs() < 1.0e-14);
        assert_eq!(solution.stats().callback_invocations, 1);

        let solution = solve(&problem, ParsaniKetchesonDeconinck3S82, &options(0.25)).unwrap();
        assert!((solution.times().last().unwrap() - 0.25).abs() < 1.0e-14);
        assert_eq!(solution.stats().callback_invocations, 1);

        let solution = solve(&problem, ParsaniKetchesonDeconinck3S94, &options(0.25)).unwrap();
        assert!((solution.times().last().unwrap() - 0.25).abs() < 1.0e-14);
        assert_eq!(solution.stats().callback_invocations, 1);

        let solution = solve(&problem, ParsaniKetchesonDeconinck3S184, &options(0.25)).unwrap();
        assert!((solution.times().last().unwrap() - 0.25).abs() < 1.0e-14);
        assert_eq!(solution.stats().callback_invocations, 1);

        let solution = solve(&problem, ParsaniKetchesonDeconinck3S205, &options(0.25)).unwrap();
        assert!((solution.times().last().unwrap() - 0.25).abs() < 1.0e-14);
        assert_eq!(solution.stats().callback_invocations, 1);

        let solution = solve(&problem, ParsaniKetchesonDeconinck3S105, &options(0.25)).unwrap();
        assert!((solution.times().last().unwrap() - 0.25).abs() < 1.0e-14);
        assert_eq!(solution.stats().callback_invocations, 1);
    }

    #[test]
    fn malformed_recurrence_is_rejected_before_driver_dispatch() {
        struct MalformedRecurrence;

        impl super::LowStorage2N for MalformedRecurrence {
            const A: &'static [f64] = &[0.0];
            const B: &'static [f64] = &[1.0];
            const C: &'static [f64] = &[0.0];
        }

        assert_eq!(
            integrate::<_, _, MalformedRecurrence>(&problem((0.0, 1.0), 1.0), &options(0.1))
                .unwrap_err(),
            SolveError::InvalidTableau
        );
    }

    #[test]
    fn terminating_callbacks_do_not_trigger_post_effect_rhs_work() {
        let rhs_calls = Rc::new(Cell::new(0));
        let rhs_counter = Rc::clone(&rhs_calls);
        let problem = OdeProblem::new(
            move |derivative: &mut [f64], state: &[f64], _: &(), _: f64| {
                rhs_counter.set(rhs_counter.get() + 1);
                derivative[0] = state[0];
            },
            vec![1.0],
            (0.0, 1.0),
            (),
        )
        .with_discrete_callback(
            |_, _, time| time >= 0.25,
            |_, _, _| CallbackAction::Terminate,
        );
        let solution = solve(&problem, Dglddrk73C, &options(0.25)).unwrap();
        assert_eq!(solution.stats().rhs_evaluations, 7);
        assert_eq!(rhs_calls.get(), 7);

        let initial_rhs_calls = Rc::new(Cell::new(0));
        let initial_rhs_counter = Rc::clone(&initial_rhs_calls);
        let initially_terminating = OdeProblem::new(
            move |derivative: &mut [f64], state: &[f64], _: &(), _: f64| {
                initial_rhs_counter.set(initial_rhs_counter.get() + 1);
                derivative[0] = state[0];
            },
            vec![1.0],
            (0.0, 1.0),
            (),
        )
        .with_discrete_callback(|_, _, _| true, |_, _, _| CallbackAction::Terminate);
        let solution = solve(&initially_terminating, Ork256, &options(0.25)).unwrap();
        assert_eq!(solution.stats().rhs_evaluations, 0);
        assert_eq!(initial_rhs_calls.get(), 0);
    }
}
