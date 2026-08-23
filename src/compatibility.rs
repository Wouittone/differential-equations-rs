//! Organized access to the algorithms implemented by this crate.
//!
//! This module deliberately contains re-exports only. A name appears here
//! only when the crate has an implementation of that algorithm; no algorithm
//! is substituted for a numerically unrelated method.

/// Value constructor for the genuine [`tyalias@crate::Tsit5DA`] spelling alias.
///
/// Rust type aliases do not inherit a unit struct's value constructor, so the
/// matching constant preserves `solve(problem, Tsit5DA, options)` while using
/// the real [`crate::HybridExplicitImplicitRK`] implementation.
#[allow(non_upper_case_globals)]
pub const Tsit5DA: crate::HybridExplicitImplicitRK = crate::HybridExplicitImplicitRK;

/// Solver algorithms grouped by numerical family.
///
/// All concrete algorithms remain available at the crate root for backwards
/// compatibility. These namespaces provide a discoverable alternative for
/// new code without introducing duplicate wrapper types.
pub mod algorithms {
    /// Automatic and default algorithm selectors.
    pub mod automatic {
        pub use crate::{
            AutoDP5, AutoDp5, AutoTsit5, AutoVern6, AutoVern7, AutoVern8, AutoVern9,
            DefaultImplicitODEAlgorithm, DefaultODEAlgorithm,
        };
    }

    /// Explicit Runge--Kutta algorithms.
    pub mod explicit {
        /// General-purpose explicit Runge--Kutta algorithms.
        pub mod general {
            pub use crate::{
                Alshina2, Alshina3, Alshina6, Bs3, Bs5, Dp5, Euler, ExplicitRK, ExplicitRungeKutta,
                Heun, KuttaPRK2p5, Midpoint, Msrk5, Msrk6, OwrenZen3, OwrenZen4, OwrenZen5,
                Psrk3p5q4, Psrk3p6q5, Psrk4p7q6, QPRK98, Ralston, Ralston4, Rk4, Rkm, Rko65, Sir54,
                SplitEuler, Stepanov5, Tsit5,
            };
        }

        /// Higher-order explicit Runge--Kutta algorithms.
        pub mod high_order {
            pub use crate::{
                Anas5, DP8, Feagin10, Feagin12, Feagin14, Frk65, PFRK87, RKV76IIa, TanYam7,
                TsitPap8, Vern6, Vern7, Vern8, Vern9,
            };
        }

        /// Explicit low-storage Runge--Kutta algorithms.
        pub mod low_storage {
            pub use crate::{
                CFRLDDRK64, CKLLSRK43_2, CKLLSRK54_3C, CKLLSRK54_3C_3R, CKLLSRK54_3M_3R,
                CKLLSRK54_3M_4R, CKLLSRK54_3N_3R, CKLLSRK54_3N_4R, CKLLSRK65_4M_4R,
                CKLLSRK75_4M_5R, CKLLSRK85_4C_3R, CKLLSRK85_4FM_4R, CKLLSRK85_4M_3R,
                CKLLSRK85_4P_3R, CKLLSRK95_4C, CKLLSRK95_4M, CKLLSRK95_4S, CarpenterKennedy2N54,
                Dglddrk73C, Dglddrk84C, Dglddrk84F, Ndblsrk124, Ndblsrk134, Ndblsrk144, Ork256,
                ParsaniKetchesonDeconinck3S32, ParsaniKetchesonDeconinck3S53,
                ParsaniKetchesonDeconinck3S82, ParsaniKetchesonDeconinck3S94,
                ParsaniKetchesonDeconinck3S105, ParsaniKetchesonDeconinck3S173,
                ParsaniKetchesonDeconinck3S184, ParsaniKetchesonDeconinck3S205, RDPK3Sp35,
                RDPK3Sp49, RDPK3Sp510, RDPK3SpFSAL35, RDPK3SpFSAL49, RDPK3SpFSAL510, RK46NL,
                SHLDDRK_2N, SHLDDRK52, Shlddrk64, TSLDDRK74,
            };
        }

        /// Strong-stability-preserving and positivity-preserving algorithms.
        pub mod ssp {
            pub use crate::{
                KYKSSPRK42, Kyk2014DgSsprk3S2, KykSsprk42, Prrk22, Prrk33, Prrk54, SSPRKMSVS32,
                SSPRKMSVS43, SspRk22, SspRk33, SspRk43, SspRk53, SspRk53H, SspRk53TwoN1,
                SspRk53TwoN2, SspRk54, SspRk63, SspRk73, SspRk83, SspRk104, SspRk432, SspRk932,
                SspRkMsvs32, SspRkMsvs43, pRRK22, pRRK33, pRRK54,
            };
        }

        pub use general::*;
        pub use high_order::*;
        pub use low_storage::*;
        pub use ssp::*;
    }

    /// Implicit Runge--Kutta algorithms.
    pub mod implicit {
        /// Basic implicit one-step algorithms.
        pub mod general {
            pub use crate::{ImplicitEuler, ImplicitMidpoint, Trapezoid};
        }

        /// Singly diagonally implicit and additive implicit algorithms.
        pub mod diagonally_implicit {
            pub use crate::{
                Ars222, Ars232, Ars343, Ars443, Bhr553, Cash4, Cfnlirk3, Esdirk54I8L2Sa,
                Esdirk325L2Sa, Esdirk436L2Sa2, Esdirk437L2Sa, Esdirk547L2Sa2, Esdirk659L2Sa,
                Hairer4, Hairer42, ImexSsp222, ImexSsp2322, ImexSsp3332, ImexSsp3433, KenCarp3,
                KenCarp4, KenCarp5, KenCarp47, KenCarp58, Kvaerno3, Kvaerno4, Kvaerno5, Sdirk2,
                Sdirk22, Sfsdirk4, Sfsdirk5, Sfsdirk6, Sfsdirk7, Sfsdirk8, SspSdirk2,
            };
        }

        pub use diagonally_implicit::*;
        pub use general::*;
    }

    /// Linear multistep and multiderivative algorithms.
    pub mod multistep {
        pub use crate::{
            Ab3, Ab4, Ab5, Abdf2, Abm32, Abm43, Abm54, FBDF, Fbdf, Mebdf2, QBDF, QNDF, Qbdf, Qbdf1,
            Qbdf2, Qndf, Qndf1, Qndf2, Trbdf2, Vcab3, Vcab4, Vcab5, Vcabm3, Vcabm4, Vcabm5,
        };
    }

    /// Rosenbrock and Rosenbrock--W algorithms.
    pub mod rosenbrock {
        pub use crate::{
            Grk4a, Grk4t, HybridExplicitImplicitRK, Rodas3, Rodas3P, Rodas3d, Rodas4, Rodas4P,
            Rodas4P2, Rodas4PW, Rodas5, Rodas5P, Rodas5Pe, Rodas5Pr, Rodas6P, Rodas23W, Rodas42,
            Rok4a, Ros2, Ros2Pr, Ros2S, Ros3, Ros3Pr, Ros3Prl, Ros3Prl2, Ros3p, Ros4LStab,
            Ros34Prw, Ros34Pw1a, Ros34Pw1b, Ros34Pw2, Ros34Pw3, RosShamp4, Rosenbrock23,
            Rosenbrock32, RosenbrockW6S4OS, Scholz4_7, Tsit5DA, Veldd4, Velds4,
        };
    }

    /// Explicit methods with extended real-axis stability intervals.
    pub mod stabilized {
        pub use crate::{
            ESERK4, ESERK5, RKC, RKG1, RKG2, RKL1, RKL2, RKMC2, ROCK2, ROCK4, SERK2, TSRKC2, TSRKC3,
        };
    }

    /// Algorithms for partitioned second-order problems.
    pub mod second_order {
        pub use crate::{
            DPRKN4, DPRKN5, DPRKN6, DPRKN6FM, DPRKN8, DPRKN12, Dprkn4, Dprkn5, Dprkn6, Dprkn6Fm,
            Dprkn8, Dprkn12, ERKN4, ERKN5, ERKN7, Erkn4, Erkn5, Erkn7, FineRKN4, FineRKN5,
            FineRkn4, FineRkn5, IRKN3, IRKN4, Irkn3, Irkn4, LeapfrogDriftKickDrift, Nystrom4,
            Nystrom4VelocityIndependent, Nystrom5VelocityIndependent, Rkn4, SymplecticEuler,
            VelocityVerlet, VerletLeapfrog,
        };
    }
}
