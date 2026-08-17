//! Beta Rust implementations of algorithms from Julia's
//! DifferentialEquations.jl ecosystem.
//!
//! The crate is in beta. Its API may still change while numerical compliance,
//! performance, and memory behavior are established.

#![forbid(unsafe_code)]

mod abdf2;
mod adams;
mod anas5;
mod autodp5;
mod callback;
mod coefficients;
mod compatibility;
mod composites;
mod explicit_rk;
mod frk65;
mod generated_coefficients;
mod high_order;
mod implicit;
mod integrator;
mod linear;
mod low_storage_rk;
mod mebdf2;
mod problem;
mod qndf1;
mod qndf2;
mod rosenbrock;
mod rosenbrock_extended;
mod sdirk;
mod sdirk_cash4;
mod second_order;
mod solution;
mod solver;
mod ssprk_extended;
mod ssprk_kyk2014;
mod ssprk_kyk42;
mod ssprk_msvs;
mod stabilized;
mod symplectic;
mod trbdf2;
mod tsit5;
mod variable_adams;
mod verner;

pub use abdf2::Abdf2;
pub use adams::{Ab3, Ab4, Ab5, Abm32, Abm43, Abm54};
pub use anas5::Anas5;
pub use autodp5::{AutoDP5, AutoDp5};
pub use callback::{CallbackAction, EventDirection};
pub use compatibility::{Tsit5DA, algorithms};
pub use composites::{
    AutoTsit5, AutoVern6, AutoVern7, AutoVern8, AutoVern9, DefaultImplicitODEAlgorithm,
    DefaultODEAlgorithm,
};
pub use explicit_rk::{
    Alshina2, Alshina3, Alshina6, Bs3, Bs5, ButcherTableau, Dp5, Euler, ExplicitRK,
    ExplicitRungeKutta, Heun, Midpoint, Msrk5, Msrk6, OwrenZen3, OwrenZen4, OwrenZen5, Psrk3p5q4,
    Psrk3p6q5, Psrk4p7q6, Ralston, Ralston4, Rk4, Rkm, Rko65, Sir54, SspRk22, SspRk33, SspRk43,
    Stepanov5,
};
pub use frk65::Frk65;
pub use high_order::{DP8, Feagin10, Feagin12, Feagin14, PFRK87, RKV76IIa, TanYam7, TsitPap8};
pub use implicit::{ImplicitEuler, ImplicitMidpoint, Trapezoid};
pub use low_storage_rk::{
    CFRLDDRK64, CKLLSRK43_2, CKLLSRK54_3C, CKLLSRK54_3C_3R, CKLLSRK54_3M_3R, CKLLSRK54_3M_4R,
    CKLLSRK54_3N_3R, CKLLSRK54_3N_4R, CKLLSRK65_4M_4R, CKLLSRK75_4M_5R, CKLLSRK85_4C_3R,
    CKLLSRK85_4FM_4R, CKLLSRK85_4M_3R, CKLLSRK85_4P_3R, CKLLSRK95_4C, CKLLSRK95_4M, CKLLSRK95_4S,
    CarpenterKennedy2N54, Dglddrk73C, Dglddrk84C, Dglddrk84F, Ndblsrk124, Ndblsrk134, Ndblsrk144,
    Ork256, ParsaniKetchesonDeconinck3S32, ParsaniKetchesonDeconinck3S53,
    ParsaniKetchesonDeconinck3S82, ParsaniKetchesonDeconinck3S94, ParsaniKetchesonDeconinck3S105,
    ParsaniKetchesonDeconinck3S173, ParsaniKetchesonDeconinck3S184, ParsaniKetchesonDeconinck3S205,
    RDPK3Sp35, RDPK3Sp49, RDPK3Sp510, RDPK3SpFSAL35, RDPK3SpFSAL49, RDPK3SpFSAL510, RK46NL,
    SHLDDRK_2N, SHLDDRK52, Shlddrk64, TSLDDRK74,
};
pub use mebdf2::Mebdf2;
pub use problem::{MassMatrixOdeProblem, OdeProblem, SplitOdeProblem};
pub use qndf1::Qndf1;
pub use qndf2::Qndf2;
pub use rosenbrock::Rosenbrock23;
pub use rosenbrock_extended::{
    Grk4a, Grk4t, HybridExplicitImplicitRK, Rodas3, Rodas3P, Rodas3d, Rodas4, Rodas4P, Rodas4P2,
    Rodas4PW, Rodas5, Rodas5P, Rodas5Pe, Rodas5Pr, Rodas6P, Rodas23W, Rodas42, Rok4a, Ros2, Ros2Pr,
    Ros2S, Ros3, Ros3Pr, Ros3Prl, Ros3Prl2, Ros3p, Ros4LStab, Ros34Prw, Ros34Pw1a, Ros34Pw1b,
    Ros34Pw2, Ros34Pw3, RosShamp4, Rosenbrock32, RosenbrockW6S4OS, Scholz4_7, Tsit5DA, Veldd4,
    Velds4,
};
pub use sdirk::{
    Ars222, Ars232, Ars343, Ars443, Bhr553, Cfnlirk3, Esdirk54I8L2Sa, Esdirk325L2Sa,
    Esdirk436L2Sa2, Esdirk437L2Sa, Esdirk547L2Sa2, Esdirk659L2Sa, Hairer4, Hairer42, ImexSsp222,
    ImexSsp2322, ImexSsp3332, ImexSsp3433, KenCarp3, KenCarp4, KenCarp5, KenCarp47, KenCarp58,
    Kvaerno3, Kvaerno4, Kvaerno5, Sdirk2, Sdirk22, Sfsdirk4, Sfsdirk5, Sfsdirk6, Sfsdirk7,
    Sfsdirk8, SspSdirk2,
};
pub use sdirk_cash4::Cash4;
pub use second_order::{
    LeapfrogDriftKickDrift, SecondOrderOdeAlgorithm, SecondOrderOdeProblem, SecondOrderSolution,
    SecondOrderSolveError, SymplecticEuler, VelocityVerlet, VerletLeapfrog, solve_second_order,
};
pub use solution::{Solution, SolverStats};
pub use solver::{OdeAlgorithm, SaveMode, SolveError, SolveOptions, solve};
pub use ssprk_extended::{
    Prrk22, Prrk33, Prrk54, SspRk53, SspRk53H, SspRk53TwoN1, SspRk53TwoN2, SspRk54, SspRk63,
    SspRk73, SspRk83, SspRk104, SspRk432, SspRk932, pRRK22, pRRK33, pRRK54,
};
pub use ssprk_kyk42::{KYKSSPRK42, KykSsprk42};
pub use ssprk_kyk2014::Kyk2014DgSsprk3S2;
pub use ssprk_msvs::{SSPRKMSVS32, SspRkMsvs32};
pub use stabilized::{RKC, RKG1, RKG2, RKL1, RKL2, RKMC2};
pub use symplectic::{
    CalvoSanz4, CandyRoz4, KahanLi6, KahanLi8, McAte2, McAte3, McAte4, McAte5, McAte8, McAte42,
    PseudoVerletLeapfrog, Ruth3, SofSpa10, SymplecticAlgorithm, SymplecticSolution,
    SymplecticSolveError, SymplecticTableau, Yoshida6, solve_symplectic,
};
pub use trbdf2::Trbdf2;
pub use tsit5::Tsit5;
pub use variable_adams::{Vcab3, Vcab4, Vcab5, Vcabm3, Vcabm4, Vcabm5};
pub use verner::{Vern6, Vern7, Vern8, Vern9};
