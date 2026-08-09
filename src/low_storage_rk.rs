//! Fixed-step two-register low-storage Runge--Kutta methods.
//!
//! This module implements the Williamson 2N recurrence used by the pinned
//! `OrdinaryDiffEqLowStorageRK` source. The numerical recurrence and stage
//! times are preserved. OrdinaryDiffEq's stage/step limiter, fused-array
//! `williamson_condition`, and threading configuration are not exposed.

// Preserve the pinned source's decimal coefficient literals exactly.
#![allow(clippy::excessive_precision)]

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

trait LowStorage3S {
    const GAMMA1: &'static [f64];
    const GAMMA2: &'static [f64];
    const GAMMA3: &'static [f64];
    const DELTA: &'static [f64];
    const BETA1: f64;
    const BETA2: &'static [f64];
    const C: &'static [f64];
}

macro_rules! method {
    ($name:ident, $coefficients:ident, $doc:literal, $a:expr, $b:expr, $c:expr) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        pub struct $name;

        struct $coefficients;

        impl LowStorage2N for $coefficients {
            const A: &'static [f64] = $a;
            const B: &'static [f64] = $b;
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
        pub struct $name;

        struct $coefficients;

        impl LowStorage3S for $coefficients {
            const GAMMA1: &'static [f64] = $gamma1;
            const GAMMA2: &'static [f64] = $gamma2;
            const GAMMA3: &'static [f64] = $gamma3;
            const DELTA: &'static [f64] = $delta;
            const BETA1: f64 = $beta1;
            const BETA2: &'static [f64] = $beta2;
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
        -2.691566797270077,
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

impl<T> LowStorage3SKernel<T> {
    fn new(dimension: usize) -> Self {
        Self {
            derivative: vec![0.0; dimension],
            temporary: vec![0.0; dimension],
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
        // The pinned implementation evaluates the endpoint derivative for
        // FSAL/interpolation bookkeeping even though this fixed-step driver
        // does not reuse it.
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
        CarpenterKennedy2N54, Dglddrk73C, Dglddrk84C, Dglddrk84F, Ndblsrk124, Ndblsrk134,
        Ndblsrk144, Ork256, ParsaniKetchesonDeconinck3S32, ParsaniKetchesonDeconinck3S53,
        ParsaniKetchesonDeconinck3S82, ParsaniKetchesonDeconinck3S94,
        ParsaniKetchesonDeconinck3S105, ParsaniKetchesonDeconinck3S173,
        ParsaniKetchesonDeconinck3S184, ParsaniKetchesonDeconinck3S205, Shlddrk64, integrate,
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
