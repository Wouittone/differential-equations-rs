//! Organized access to the algorithms implemented by this crate.
//!
//! Concrete methods live here instead of at the crate root. Family modules
//! provide focused imports, while `algorithms::*` remains convenient for code
//! that intentionally works with many solver families.

/// Solver algorithms grouped by numerical family.
pub mod algorithms {
    /// Automatic and default algorithm selectors.
    pub mod automatic {
        pub use crate::autodp5::{AutoDP5, AutoDp5};
        pub use crate::composites::{
            AutoTsit5, AutoVern6, AutoVern7, AutoVern8, AutoVern9, DefaultImplicitODEAlgorithm,
            DefaultODEAlgorithm,
        };
    }

    /// Explicit Runge--Kutta algorithms.
    pub mod explicit {
        /// General-purpose explicit Runge--Kutta algorithms and tableau API.
        pub mod general {
            pub use crate::explicit_rk::{
                Alshina2, Alshina3, Alshina6, Bs3, Bs5, ButcherTableau, Dp5, Euler, ExplicitRK,
                ExplicitRungeKutta, Heun, Midpoint, Msrk5, Msrk6, OwrenZen3, OwrenZen4, OwrenZen5,
                Psrk3p5q4, Psrk3p6q5, Psrk4p7q6, Ralston, Ralston4, Rk4, Rkm, Rko65, Sir54,
                Stepanov5,
            };
            pub use crate::prk::{KuttaPRK2p5, KuttaPrk2p5Tableau};
            pub use crate::qprk::{QPRK98, Qprk98Tableau};
            pub use crate::split_euler::SplitEuler;
            pub use crate::tsit5::Tsit5;
        }

        /// Higher-order explicit Runge--Kutta algorithms.
        pub mod high_order {
            pub use crate::anas5::Anas5;
            pub use crate::frk65::Frk65;
            pub use crate::high_order::{
                DP8, Feagin10, Feagin12, Feagin14, PFRK87, RKV76IIa, TanYam7, TsitPap8,
            };
            pub use crate::verner::{Vern6, Vern7, Vern8, Vern9};
        }

        /// Explicit low-storage Runge--Kutta algorithms.
        pub mod low_storage {
            pub use crate::low_storage_rk::{
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
            pub use crate::explicit_rk::{SspRk22, SspRk33, SspRk43};
            pub use crate::ssprk_extended::{
                Prrk22, Prrk33, Prrk54, SspRk53, SspRk53H, SspRk53TwoN1, SspRk53TwoN2, SspRk54,
                SspRk63, SspRk73, SspRk83, SspRk104, SspRk432, SspRk932, pRRK22, pRRK33, pRRK54,
            };
            pub use crate::ssprk_kyk42::{KYKSSPRK42, KykSsprk42};
            pub use crate::ssprk_kyk2014::Kyk2014DgSsprk3S2;
            pub use crate::ssprk_msvs::{SSPRKMSVS32, SSPRKMSVS43, SspRkMsvs32, SspRkMsvs43};
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
            pub use crate::implicit::{ImplicitEuler, ImplicitMidpoint, Trapezoid};
        }

        /// Singly diagonally implicit and additive implicit algorithms.
        pub mod diagonally_implicit {
            pub use crate::pdirk::{PDIRK44, Pdirk44};
            pub use crate::sdirk::{
                Ars222, Ars232, Ars343, Ars443, Bhr553, Cfnlirk3, Esdirk54I8L2Sa, Esdirk325L2Sa,
                Esdirk436L2Sa2, Esdirk437L2Sa, Esdirk547L2Sa2, Esdirk659L2Sa, Hairer4, Hairer42,
                ImexSsp222, ImexSsp2322, ImexSsp3332, ImexSsp3433, KenCarp3, KenCarp4, KenCarp5,
                KenCarp47, KenCarp58, Kvaerno3, Kvaerno4, Kvaerno5, Sdirk2, Sdirk22, Sfsdirk4,
                Sfsdirk5, Sfsdirk6, Sfsdirk7, Sfsdirk8, SspSdirk2,
            };
            pub use crate::sdirk_cash4::Cash4;
        }

        pub use diagonally_implicit::*;
        pub use general::*;
    }

    /// Linear multistep and multiderivative algorithms.
    pub mod multistep {
        pub use crate::abdf2::Abdf2;
        pub use crate::adams::{Ab3, Ab4, Ab5, Abm32, Abm43, Abm54};
        pub use crate::bdf::{FBDF, Fbdf, QBDF, QNDF, Qbdf, Qndf};
        pub use crate::mebdf2::Mebdf2;
        pub use crate::qndf1::{Qbdf1, Qndf1};
        pub use crate::qndf2::{Qbdf2, Qndf2};
        pub use crate::trbdf2::Trbdf2;
        pub use crate::variable_adams::{Vcab3, Vcab4, Vcab5, Vcabm3, Vcabm4, Vcabm5};
    }

    /// Rosenbrock and Rosenbrock--W algorithms.
    pub mod rosenbrock {
        pub use crate::rosenbrock::Rosenbrock23;
        pub use crate::rosenbrock_extended::{
            Grk4a, Grk4t, HybridExplicitImplicitRK, Rodas3, Rodas3P, Rodas3d, Rodas4, Rodas4P,
            Rodas4P2, Rodas4PW, Rodas5, Rodas5P, Rodas5Pe, Rodas5Pr, Rodas6P, Rodas23W, Rodas42,
            Rok4a, Ros2, Ros2Pr, Ros2S, Ros3, Ros3Pr, Ros3Prl, Ros3Prl2, Ros3p, Ros4LStab,
            Ros34Prw, Ros34Pw1a, Ros34Pw1b, Ros34Pw2, Ros34Pw3, RosShamp4, Rosenbrock32,
            RosenbrockW6S4OS, Scholz4_7, Tsit5DA, Veldd4, Velds4,
        };

        /// Value constructor for the genuine `Tsit5DA` spelling alias.
        #[allow(non_upper_case_globals)]
        pub const Tsit5DA: HybridExplicitImplicitRK = HybridExplicitImplicitRK;
    }

    /// Explicit methods with extended real-axis stability intervals.
    pub mod stabilized {
        pub use crate::stabilized::{
            ESERK4, ESERK5, RKC, RKG1, RKG2, RKL1, RKL2, RKMC2, ROCK2, ROCK4, SERK2, TSRKC2, TSRKC3,
        };
    }

    /// Algorithms for partitioned second-order problems.
    pub mod second_order {
        /// Runge--Kutta--Nyström algorithms.
        pub mod rkn {
            pub use crate::second_order::{
                DPRKN4, DPRKN5, DPRKN6, DPRKN6FM, DPRKN8, DPRKN12, Dprkn4, Dprkn5, Dprkn6,
                Dprkn6Fm, Dprkn8, Dprkn12, ERKN4, ERKN5, ERKN7, Erkn4, Erkn5, Erkn7, FineRKN4,
                FineRKN5, FineRkn4, FineRkn5, IRKN3, IRKN4, Irkn3, Irkn4, Nystrom4,
                Nystrom4VelocityIndependent, Nystrom5VelocityIndependent, Rkn4,
            };
        }

        /// Symplectic and partitioned algorithms.
        pub mod symplectic {
            pub use crate::second_order::{
                LeapfrogDriftKickDrift, SymplecticEuler, VelocityVerlet, VerletLeapfrog,
            };
            pub use crate::symplectic::{
                CalvoSanz4, CandyRoz4, KahanLi6, KahanLi8, McAte2, McAte3, McAte4, McAte5, McAte8,
                McAte42, PseudoVerletLeapfrog, Ruth3, SofSpa10, Yoshida6,
            };
        }

        /// Implicit structural-dynamics algorithms.
        pub mod structural {
            pub use crate::second_order::{GeneralizedAlpha, NewmarkBeta};
        }

        pub use rkn::*;
        pub use structural::*;
        pub use symplectic::*;
    }

    pub use automatic::*;
    pub use explicit::{general::*, high_order::*, low_storage::*, ssp::*};
    pub use implicit::{diagonally_implicit::*, general::*};
    pub use multistep::*;
    pub use rosenbrock::*;
    pub use second_order::*;
    pub use stabilized::*;
}
