// Preserve the pinned source's decimal coefficient literals exactly. Precision
// lint exceptions are attached only to the associated constants that contain
// coefficient data; the integration kernels remain fully linted.

use std::marker::PhantomData;

use crate::integrator::{
    KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

mod coefficient_data {
    use differential_equations_tableau_macros::define_coefficients_from_file;

    define_coefficients_from_file!(
        pub(super),
        "coefficients/explicit/low_storage.toml",
        crate = crate
    );
}

use coefficient_data::*;

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
            fn solve_validated<F, P>(
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
            fn solve_validated<F, P>(
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
            fn solve_validated<F, P>(
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
            fn solve_validated<F, P>(
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
    ($name:ident, $coefficients:ident, $doc:literal, $history:expr, $a:expr, $b:expr, $b_final:expr, $c:expr) => {
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
            const B_FINAL: f64 = $b_final;
            #[allow(
                clippy::excessive_precision,
                reason = "pinned upstream f64 coefficient"
            )]
            const C: &'static [f64] = $c;
            const HISTORY_STATES: usize = $history;
        }

        impl OdeAlgorithm for $name {
            fn solve_validated<F, P>(
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
            fn solve_validated<F, P>(
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
    LS_ORK256_A,
    LS_ORK256_B,
    LS_ORK256_C
);

method!(
    CarpenterKennedy2N54,
    CarpenterKennedy2N54Coefficients,
    "Five-stage, fourth-order Carpenter--Kennedy 2N-storage method.",
    LS_CARPENTERKENNEDY2N54_A,
    LS_CARPENTERKENNEDY2N54_B,
    LS_CARPENTERKENNEDY2N54_C
);

method!(
    Shlddrk64,
    Shlddrk64Coefficients,
    "Six-stage, fourth-order low-dissipation and low-dispersion method.",
    LS_SHLDDRK64_A,
    LS_SHLDDRK64_B,
    LS_SHLDDRK64_C
);

method!(
    Dglddrk73C,
    Dglddrk73CCoefficients,
    "Seven-stage, third-order low-dissipation and low-dispersion method.",
    LS_DGLDDRK73C_A,
    LS_DGLDDRK73C_B,
    LS_DGLDDRK73C_C
);

method!(
    Dglddrk84C,
    Dglddrk84CCoefficients,
    "Eight-stage, fourth-order low-dissipation and low-dispersion method.",
    LS_DGLDDRK84C_A,
    LS_DGLDDRK84C_B,
    LS_DGLDDRK84C_C
);

method!(
    Dglddrk84F,
    Dglddrk84FCoefficients,
    "Eight-stage, fourth-order low-dissipation and low-dispersion method.",
    LS_DGLDDRK84F_A,
    LS_DGLDDRK84F_B,
    LS_DGLDDRK84F_C
);

method!(
    Ndblsrk124,
    Ndblsrk124Coefficients,
    "Twelve-stage, fourth-order low-storage method for advection-dominated problems.",
    LS_NDBLSRK124_A,
    LS_NDBLSRK124_B,
    LS_NDBLSRK124_C
);

method!(
    Ndblsrk134,
    Ndblsrk134Coefficients,
    "Thirteen-stage, fourth-order low-storage method for advection-dominated problems.",
    LS_NDBLSRK134_A,
    LS_NDBLSRK134_B,
    LS_NDBLSRK134_C
);

method!(
    Ndblsrk144,
    Ndblsrk144Coefficients,
    "Fourteen-stage, fourth-order low-storage method for advection-dominated problems.",
    LS_NDBLSRK144_A,
    LS_NDBLSRK144_B,
    LS_NDBLSRK144_C
);

method_3s!(
    ParsaniKetchesonDeconinck3S32,
    ParsaniKetchesonDeconinck3S32Coefficients,
    "Three-stage, second-order 3S low-storage method optimized for spectral-difference wave propagation.",
    LS_PARSANIKETCHESONDECONINCK3S32_GAMMA1,
    LS_PARSANIKETCHESONDECONINCK3S32_GAMMA2,
    LS_PARSANIKETCHESONDECONINCK3S32_GAMMA3,
    LS_PARSANIKETCHESONDECONINCK3S32_DELTA,
    LS_PARSANIKETCHESONDECONINCK3S32_BETA1,
    LS_PARSANIKETCHESONDECONINCK3S32_BETA2,
    LS_PARSANIKETCHESONDECONINCK3S32_C
);

method_3s!(
    ParsaniKetchesonDeconinck3S53,
    ParsaniKetchesonDeconinck3S53Coefficients,
    "Five-stage, third-order 3S low-storage method optimized for spectral-difference wave propagation.",
    LS_PARSANIKETCHESONDECONINCK3S53_GAMMA1,
    LS_PARSANIKETCHESONDECONINCK3S53_GAMMA2,
    LS_PARSANIKETCHESONDECONINCK3S53_GAMMA3,
    LS_PARSANIKETCHESONDECONINCK3S53_DELTA,
    LS_PARSANIKETCHESONDECONINCK3S53_BETA1,
    LS_PARSANIKETCHESONDECONINCK3S53_BETA2,
    LS_PARSANIKETCHESONDECONINCK3S53_C
);

method!(
    RK46NL,
    Rk46NlCoefficients,
    "Six-stage, fourth-order low-storage method with nonlinear stability properties.",
    LS_RK46NL_A,
    LS_RK46NL_B,
    LS_RK46NL_C
);

method_2c!(
    CFRLDDRK64,
    Cfrlddrk64Coefficients,
    "Six-stage, fourth-order low-dissipation and low-dispersion 2C method.",
    LS_CFRLDDRK64_A,
    LS_CFRLDDRK64_B,
    LS_CFRLDDRK64_C
);

method_2c!(
    TSLDDRK74,
    Tslddrk74Coefficients,
    "Seven-stage, fourth-order low-dissipation and low-dispersion 2C method.",
    LS_TSLDDRK74_A,
    LS_TSLDDRK74_B,
    LS_TSLDDRK74_C
);

method!(
    SHLDDRK52,
    Shlddrk52Coefficients,
    "Five-stage, second-order low-dissipation and low-dispersion method.",
    LS_SHLDDRK52_A,
    LS_SHLDDRK52_B,
    LS_SHLDDRK52_C
);

method_alternating_2n!(
    SHLDDRK_2N,
    Shlddrk2nCoefficients,
    "Alternating five- and six-stage fourth-order low-dissipation and low-dispersion method.",
    LS_SHLDDRK_2N_A1,
    LS_SHLDDRK_2N_B1,
    LS_SHLDDRK_2N_C1,
    LS_SHLDDRK_2N_A2,
    LS_SHLDDRK_2N_B2,
    LS_SHLDDRK_2N_C2
);

method_3s!(
    ParsaniKetchesonDeconinck3S82,
    ParsaniKetchesonDeconinck3S82Coefficients,
    "Eight-stage, second-order 3S low-storage method optimized for spectral-difference wave propagation.",
    LS_PARSANIKETCHESONDECONINCK3S82_GAMMA1,
    LS_PARSANIKETCHESONDECONINCK3S82_GAMMA2,
    LS_PARSANIKETCHESONDECONINCK3S82_GAMMA3,
    LS_PARSANIKETCHESONDECONINCK3S82_DELTA,
    LS_PARSANIKETCHESONDECONINCK3S82_BETA1,
    LS_PARSANIKETCHESONDECONINCK3S82_BETA2,
    LS_PARSANIKETCHESONDECONINCK3S82_C
);

method_3s!(
    ParsaniKetchesonDeconinck3S173,
    ParsaniKetchesonDeconinck3S173Coefficients,
    "Seventeen-stage, third-order 3S low-storage method optimized for spectral-difference wave propagation.",
    LS_PARSANIKETCHESONDECONINCK3S173_GAMMA1,
    LS_PARSANIKETCHESONDECONINCK3S173_GAMMA2,
    LS_PARSANIKETCHESONDECONINCK3S173_GAMMA3,
    LS_PARSANIKETCHESONDECONINCK3S173_DELTA,
    LS_PARSANIKETCHESONDECONINCK3S173_BETA1,
    LS_PARSANIKETCHESONDECONINCK3S173_BETA2,
    LS_PARSANIKETCHESONDECONINCK3S173_C
);

method_3s!(
    ParsaniKetchesonDeconinck3S184,
    ParsaniKetchesonDeconinck3S184Coefficients,
    "Eighteen-stage, fourth-order 3S low-storage method optimized for spectral-difference wave propagation.",
    LS_PARSANIKETCHESONDECONINCK3S184_GAMMA1,
    LS_PARSANIKETCHESONDECONINCK3S184_GAMMA2,
    LS_PARSANIKETCHESONDECONINCK3S184_GAMMA3,
    LS_PARSANIKETCHESONDECONINCK3S184_DELTA,
    LS_PARSANIKETCHESONDECONINCK3S184_BETA1,
    LS_PARSANIKETCHESONDECONINCK3S184_BETA2,
    LS_PARSANIKETCHESONDECONINCK3S184_C
);

method_3s!(
    ParsaniKetchesonDeconinck3S94,
    ParsaniKetchesonDeconinck3S94Coefficients,
    "Nine-stage, fourth-order 3S low-storage method optimized for spectral-difference wave propagation.",
    LS_PARSANIKETCHESONDECONINCK3S94_GAMMA1,
    LS_PARSANIKETCHESONDECONINCK3S94_GAMMA2,
    LS_PARSANIKETCHESONDECONINCK3S94_GAMMA3,
    LS_PARSANIKETCHESONDECONINCK3S94_DELTA,
    LS_PARSANIKETCHESONDECONINCK3S94_BETA1,
    LS_PARSANIKETCHESONDECONINCK3S94_BETA2,
    LS_PARSANIKETCHESONDECONINCK3S94_C
);

method_3s!(
    ParsaniKetchesonDeconinck3S105,
    ParsaniKetchesonDeconinck3S105Coefficients,
    "Ten-stage, fifth-order 3S low-storage method optimized for spectral-difference wave propagation.",
    LS_PARSANIKETCHESONDECONINCK3S105_GAMMA1,
    LS_PARSANIKETCHESONDECONINCK3S105_GAMMA2,
    LS_PARSANIKETCHESONDECONINCK3S105_GAMMA3,
    LS_PARSANIKETCHESONDECONINCK3S105_DELTA,
    LS_PARSANIKETCHESONDECONINCK3S105_BETA1,
    LS_PARSANIKETCHESONDECONINCK3S105_BETA2,
    LS_PARSANIKETCHESONDECONINCK3S105_C
);

method_3s!(
    ParsaniKetchesonDeconinck3S205,
    ParsaniKetchesonDeconinck3S205Coefficients,
    "Twenty-stage, fifth-order 3S low-storage method optimized for spectral-difference wave propagation.",
    LS_PARSANIKETCHESONDECONINCK3S205_GAMMA1,
    LS_PARSANIKETCHESONDECONINCK3S205_GAMMA2,
    LS_PARSANIKETCHESONDECONINCK3S205_GAMMA3,
    LS_PARSANIKETCHESONDECONINCK3S205_DELTA,
    LS_PARSANIKETCHESONDECONINCK3S205_BETA1,
    LS_PARSANIKETCHESONDECONINCK3S205_BETA2,
    LS_PARSANIKETCHESONDECONINCK3S205_C
);

method_rp!(
    CKLLSRK43_2,
    CKLLSRK43_2Coefficients,
    "Pinned CKLLSRK43_2 low-storage register-pipeline method.",
    LS_CKLLSRK43_2_HISTORY,
    LS_CKLLSRK43_2_A,
    LS_CKLLSRK43_2_B,
    LS_CKLLSRK43_2_B_FINAL,
    LS_CKLLSRK43_2_C
);
method_rp!(
    CKLLSRK54_3C,
    CKLLSRK54_3CCoefficients,
    "Pinned CKLLSRK54_3C low-storage register-pipeline method.",
    LS_CKLLSRK54_3C_HISTORY,
    LS_CKLLSRK54_3C_A,
    LS_CKLLSRK54_3C_B,
    LS_CKLLSRK54_3C_B_FINAL,
    LS_CKLLSRK54_3C_C
);
method_rp!(
    CKLLSRK95_4S,
    CKLLSRK95_4SCoefficients,
    "Pinned CKLLSRK95_4S low-storage register-pipeline method.",
    LS_CKLLSRK95_4S_HISTORY,
    LS_CKLLSRK95_4S_A,
    LS_CKLLSRK95_4S_B,
    LS_CKLLSRK95_4S_B_FINAL,
    LS_CKLLSRK95_4S_C
);
method_rp!(
    CKLLSRK95_4C,
    CKLLSRK95_4CCoefficients,
    "Pinned CKLLSRK95_4C low-storage register-pipeline method.",
    LS_CKLLSRK95_4C_HISTORY,
    LS_CKLLSRK95_4C_A,
    LS_CKLLSRK95_4C_B,
    LS_CKLLSRK95_4C_B_FINAL,
    LS_CKLLSRK95_4C_C
);
method_rp!(
    CKLLSRK95_4M,
    CKLLSRK95_4MCoefficients,
    "Pinned CKLLSRK95_4M low-storage register-pipeline method.",
    LS_CKLLSRK95_4M_HISTORY,
    LS_CKLLSRK95_4M_A,
    LS_CKLLSRK95_4M_B,
    LS_CKLLSRK95_4M_B_FINAL,
    LS_CKLLSRK95_4M_C
);
method_rp!(
    CKLLSRK54_3C_3R,
    CKLLSRK54_3C_3RCoefficients,
    "Pinned CKLLSRK54_3C_3R low-storage register-pipeline method.",
    LS_CKLLSRK54_3C_3R_HISTORY,
    LS_CKLLSRK54_3C_3R_A,
    LS_CKLLSRK54_3C_3R_B,
    LS_CKLLSRK54_3C_3R_B_FINAL,
    LS_CKLLSRK54_3C_3R_C
);
method_rp!(
    CKLLSRK54_3M_3R,
    CKLLSRK54_3M_3RCoefficients,
    "Pinned CKLLSRK54_3M_3R low-storage register-pipeline method.",
    LS_CKLLSRK54_3M_3R_HISTORY,
    LS_CKLLSRK54_3M_3R_A,
    LS_CKLLSRK54_3M_3R_B,
    LS_CKLLSRK54_3M_3R_B_FINAL,
    LS_CKLLSRK54_3M_3R_C
);
method_rp!(
    CKLLSRK54_3N_3R,
    CKLLSRK54_3N_3RCoefficients,
    "Pinned CKLLSRK54_3N_3R low-storage register-pipeline method.",
    LS_CKLLSRK54_3N_3R_HISTORY,
    LS_CKLLSRK54_3N_3R_A,
    LS_CKLLSRK54_3N_3R_B,
    LS_CKLLSRK54_3N_3R_B_FINAL,
    LS_CKLLSRK54_3N_3R_C
);
method_rp!(
    CKLLSRK85_4C_3R,
    CKLLSRK85_4C_3RCoefficients,
    "Pinned CKLLSRK85_4C_3R low-storage register-pipeline method.",
    LS_CKLLSRK85_4C_3R_HISTORY,
    LS_CKLLSRK85_4C_3R_A,
    LS_CKLLSRK85_4C_3R_B,
    LS_CKLLSRK85_4C_3R_B_FINAL,
    LS_CKLLSRK85_4C_3R_C
);
method_rp!(
    CKLLSRK85_4M_3R,
    CKLLSRK85_4M_3RCoefficients,
    "Pinned CKLLSRK85_4M_3R low-storage register-pipeline method.",
    LS_CKLLSRK85_4M_3R_HISTORY,
    LS_CKLLSRK85_4M_3R_A,
    LS_CKLLSRK85_4M_3R_B,
    LS_CKLLSRK85_4M_3R_B_FINAL,
    LS_CKLLSRK85_4M_3R_C
);
method_rp!(
    CKLLSRK85_4P_3R,
    CKLLSRK85_4P_3RCoefficients,
    "Pinned CKLLSRK85_4P_3R low-storage register-pipeline method.",
    LS_CKLLSRK85_4P_3R_HISTORY,
    LS_CKLLSRK85_4P_3R_A,
    LS_CKLLSRK85_4P_3R_B,
    LS_CKLLSRK85_4P_3R_B_FINAL,
    LS_CKLLSRK85_4P_3R_C
);
method_rp!(
    CKLLSRK54_3N_4R,
    CKLLSRK54_3N_4RCoefficients,
    "Pinned CKLLSRK54_3N_4R low-storage register-pipeline method.",
    LS_CKLLSRK54_3N_4R_HISTORY,
    LS_CKLLSRK54_3N_4R_A,
    LS_CKLLSRK54_3N_4R_B,
    LS_CKLLSRK54_3N_4R_B_FINAL,
    LS_CKLLSRK54_3N_4R_C
);
method_rp!(
    CKLLSRK54_3M_4R,
    CKLLSRK54_3M_4RCoefficients,
    "Pinned CKLLSRK54_3M_4R low-storage register-pipeline method.",
    LS_CKLLSRK54_3M_4R_HISTORY,
    LS_CKLLSRK54_3M_4R_A,
    LS_CKLLSRK54_3M_4R_B,
    LS_CKLLSRK54_3M_4R_B_FINAL,
    LS_CKLLSRK54_3M_4R_C
);
method_rp!(
    CKLLSRK65_4M_4R,
    CKLLSRK65_4M_4RCoefficients,
    "Pinned CKLLSRK65_4M_4R low-storage register-pipeline method.",
    LS_CKLLSRK65_4M_4R_HISTORY,
    LS_CKLLSRK65_4M_4R_A,
    LS_CKLLSRK65_4M_4R_B,
    LS_CKLLSRK65_4M_4R_B_FINAL,
    LS_CKLLSRK65_4M_4R_C
);
method_rp!(
    CKLLSRK85_4FM_4R,
    CKLLSRK85_4FM_4RCoefficients,
    "Pinned CKLLSRK85_4FM_4R low-storage register-pipeline method.",
    LS_CKLLSRK85_4FM_4R_HISTORY,
    LS_CKLLSRK85_4FM_4R_A,
    LS_CKLLSRK85_4FM_4R_B,
    LS_CKLLSRK85_4FM_4R_B_FINAL,
    LS_CKLLSRK85_4FM_4R_C
);
method_rp!(
    CKLLSRK75_4M_5R,
    CKLLSRK75_4M_5RCoefficients,
    "Pinned CKLLSRK75_4M_5R low-storage register-pipeline method.",
    LS_CKLLSRK75_4M_5R_HISTORY,
    LS_CKLLSRK75_4M_5R_A,
    LS_CKLLSRK75_4M_5R_B,
    LS_CKLLSRK75_4M_5R_B_FINAL,
    LS_CKLLSRK75_4M_5R_C
);

method_3sp!(
    RDPK3Sp35,
    RDPK3Sp35Coefficients,
    "Pinned RDPK3Sp35 3S-plus low-storage method.",
    LS_RDPK3SP35_ENDPOINT,
    LS_RDPK3SP35_GAMMA1,
    LS_RDPK3SP35_GAMMA2,
    LS_RDPK3SP35_GAMMA3,
    LS_RDPK3SP35_DELTA,
    LS_RDPK3SP35_BETA1,
    LS_RDPK3SP35_BETA2,
    LS_RDPK3SP35_C
);
method_3sp!(
    RDPK3Sp49,
    RDPK3Sp49Coefficients,
    "Pinned RDPK3Sp49 3S-plus low-storage method.",
    LS_RDPK3SP49_ENDPOINT,
    LS_RDPK3SP49_GAMMA1,
    LS_RDPK3SP49_GAMMA2,
    LS_RDPK3SP49_GAMMA3,
    LS_RDPK3SP49_DELTA,
    LS_RDPK3SP49_BETA1,
    LS_RDPK3SP49_BETA2,
    LS_RDPK3SP49_C
);
method_3sp!(
    RDPK3Sp510,
    RDPK3Sp510Coefficients,
    "Pinned RDPK3Sp510 3S-plus low-storage method.",
    LS_RDPK3SP510_ENDPOINT,
    LS_RDPK3SP510_GAMMA1,
    LS_RDPK3SP510_GAMMA2,
    LS_RDPK3SP510_GAMMA3,
    LS_RDPK3SP510_DELTA,
    LS_RDPK3SP510_BETA1,
    LS_RDPK3SP510_BETA2,
    LS_RDPK3SP510_C
);
method_3sp!(
    RDPK3SpFSAL35,
    RDPK3SpFSAL35Coefficients,
    "Pinned RDPK3SpFSAL35 3S-plus low-storage method.",
    LS_RDPK3SPFSAL35_ENDPOINT,
    LS_RDPK3SPFSAL35_GAMMA1,
    LS_RDPK3SPFSAL35_GAMMA2,
    LS_RDPK3SPFSAL35_GAMMA3,
    LS_RDPK3SPFSAL35_DELTA,
    LS_RDPK3SPFSAL35_BETA1,
    LS_RDPK3SPFSAL35_BETA2,
    LS_RDPK3SPFSAL35_C
);
method_3sp!(
    RDPK3SpFSAL49,
    RDPK3SpFSAL49Coefficients,
    "Pinned RDPK3SpFSAL49 3S-plus low-storage method.",
    LS_RDPK3SPFSAL49_ENDPOINT,
    LS_RDPK3SPFSAL49_GAMMA1,
    LS_RDPK3SPFSAL49_GAMMA2,
    LS_RDPK3SPFSAL49_GAMMA3,
    LS_RDPK3SPFSAL49_DELTA,
    LS_RDPK3SPFSAL49_BETA1,
    LS_RDPK3SPFSAL49_BETA2,
    LS_RDPK3SPFSAL49_C
);
method_3sp!(
    RDPK3SpFSAL510,
    RDPK3SpFSAL510Coefficients,
    "Pinned RDPK3SpFSAL510 3S-plus low-storage method.",
    LS_RDPK3SPFSAL510_ENDPOINT,
    LS_RDPK3SPFSAL510_GAMMA1,
    LS_RDPK3SPFSAL510_GAMMA2,
    LS_RDPK3SPFSAL510_GAMMA3,
    LS_RDPK3SPFSAL510_DELTA,
    LS_RDPK3SPFSAL510_BETA1,
    LS_RDPK3SPFSAL510_BETA2,
    LS_RDPK3SPFSAL510_C
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
            let previous_register = self
                .history_states
                .last()
                .ok_or(SolveError::InvalidTableau)?;
            self.gprev.copy_from_slice(previous_register);
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
