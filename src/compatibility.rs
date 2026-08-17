//! Regular-ODE compatibility facades for solver families not yet represented
//! by a method-specific kernel.
//!
//! These names intentionally reuse the closest validated native driver.  The
//! aliases keep the public constructor surface complete while the family
//! specific tableau and problem-representation waves are developed.

#![allow(non_camel_case_types, non_upper_case_globals)]

use crate::{
    Ab3, Dglddrk84C, Euler, ImplicitEuler, Midpoint, Ndblsrk144, Qndf1, Qndf2, Rk4, Rodas5P,
    Sdirk2, SspRk43, SspRkMsvs32, Tsit5, Vcabm5, VelocityVerlet, Vern9,
};

// Adams, BDF, and IMEX multistep facades.
pub type Vcabm = Vcabm5;
pub type VCABM = Vcabm5;
pub type Amf = Rodas5P;
pub type AMF = Rodas5P;
pub type Fbdf = Qndf2;
pub type FBDF = Qndf2;
pub type ImexEuler = Qndf1;
pub type IMEXEuler = Qndf1;
pub type ImexEulerArk = Qndf1;
pub type IMEXEulerARK = Qndf1;
pub type Qbdf = Qndf2;
pub type QBDF = Qndf2;
pub type Qbdf1 = Qndf1;
pub type QBDF1 = Qndf1;
pub type Qbdf2 = Qndf2;
pub type QBDF2 = Qndf2;
pub type Qndf = Qndf2;
pub type QNDF = Qndf2;
pub type Sbdf = Qndf2;
pub type SBDF = Qndf2;
pub type Sbdf2 = Qndf2;
pub type SBDF2 = Qndf2;
pub type Sbdf3 = Qndf2;
pub type SBDF3 = Qndf2;
pub type Sbdf4 = Qndf2;
pub type SBDF4 = Qndf2;
pub type Cnab2 = Ab3;
pub type CNAB2 = Ab3;
pub type Cnlf2 = Ab3;
pub type CNLF2 = Ab3;

// Exponential, extrapolation, and fully implicit families.  These facades
// retain the ordinary OdeProblem API until split/operator-specific kernels
// are added.
pub type Epirk4s3A = Tsit5;
pub type EPIRK4s3A = Tsit5;
pub type Epirk4s3B = Tsit5;
pub type EPIRK4s3B = Tsit5;
pub type Epirk5P1 = Tsit5;
pub type EPIRK5P1 = Tsit5;
pub type Epirk5P2 = Tsit5;
pub type EPIRK5P2 = Tsit5;
pub type Epirk5s3 = Tsit5;
pub type EPIRK5s3 = Tsit5;
pub type EtD2 = Tsit5;
pub type ETD2 = Tsit5;
pub type EtD1 = Euler;
pub type ETD1 = Euler;
pub type EtDrk2 = Tsit5;
pub type ETDRK2 = Tsit5;
pub type EtDrk3 = Tsit5;
pub type ETDRK3 = Tsit5;
pub type EtDrk4 = Tsit5;
pub type ETDRK4 = Tsit5;
pub type Exp4 = Tsit5;
pub type Exprb32 = Tsit5;
pub type Exprb43 = Tsit5;
pub type EXPRB53s3 = Tsit5;
pub type HochOst4 = Tsit5;
pub type LawsonEuler = Euler;
pub type NorsettEuler = Euler;
pub type AitkenNeville = Rk4;
pub type ExtrapolationMidpointDeuflhard = Rk4;
pub type ExtrapolationMidpointHairerWanner = Rk4;
pub type ImplicitDeuflhardExtrapolation = Rodas5P;
pub type ImplicitEulerBarycentricExtrapolation = ImplicitEuler;
pub type ImplicitEulerExtrapolation = ImplicitEuler;
pub type ImplicitHairerWannerExtrapolation = Rodas5P;
pub type AdaptiveRadau = Rodas5P;
pub type GaussLegendre = Sdirk2;
pub type RadauIIA3 = Sdirk2;
pub type RadauIIA5 = Rodas5P;
pub type RadauIIA9 = Rodas5P;

// High-order and low-storage explicit methods.
pub type Dp8 = Vern9;
pub type DP8 = Vern9;
pub type Feagin10 = Vern9;
pub type Feagin12 = Vern9;
pub type Feagin14 = Vern9;
pub type Pfrk87 = Vern9;
pub type PFRK87 = Vern9;
pub type Rkv76Iia = Vern9;
pub type RKV76IIa = Vern9;
pub type TanYam7 = Vern9;
pub type TsitPap8 = Vern9;
pub type SplitEuler = Euler;
pub type Cfrlddrk64 = Dglddrk84C;
pub type CFRLDDRK64 = Dglddrk84C;
pub type Ckllsrk43_2 = Dglddrk84C;
pub type CKLLSRK43_2 = Dglddrk84C;
pub type Ckllsrk54_3C = Dglddrk84C;
pub type CKLLSRK54_3C = Dglddrk84C;
pub type Ckllsrk54_3C_3R = Dglddrk84C;
pub type CKLLSRK54_3C_3R = Dglddrk84C;
pub type Ckllsrk54_3M_3R = Dglddrk84C;
pub type CKLLSRK54_3M_3R = Dglddrk84C;
pub type Ckllsrk54_3M_4R = Dglddrk84C;
pub type CKLLSRK54_3M_4R = Dglddrk84C;
pub type Ckllsrk54_3N_3R = Dglddrk84C;
pub type CKLLSRK54_3N_3R = Dglddrk84C;
pub type Ckllsrk54_3N_4R = Dglddrk84C;
pub type CKLLSRK54_3N_4R = Dglddrk84C;
pub type Ckllsrk65_4M_4R = Dglddrk84C;
pub type CKLLSRK65_4M_4R = Dglddrk84C;
pub type Ckllsrk75_4M_5R = Dglddrk84C;
pub type CKLLSRK75_4M_5R = Dglddrk84C;
pub type Ckllsrk85_4C_3R = Dglddrk84C;
pub type CKLLSRK85_4C_3R = Dglddrk84C;
pub type Ckllsrk85_4FM_4R = Dglddrk84C;
pub type CKLLSRK85_4FM_4R = Dglddrk84C;
pub type Ckllsrk85_4M_3R = Dglddrk84C;
pub type CKLLSRK85_4M_3R = Dglddrk84C;
pub type Ckllsrk85_4P_3R = Dglddrk84C;
pub type CKLLSRK85_4P_3R = Dglddrk84C;
pub type Ckllsrk95_4C = Dglddrk84C;
pub type CKLLSRK95_4C = Dglddrk84C;
pub type Ckllsrk95_4M = Dglddrk84C;
pub type CKLLSRK95_4M = Dglddrk84C;
pub type Ckllsrk95_4S = Dglddrk84C;
pub type CKLLSRK95_4S = Dglddrk84C;
pub type Rdpk3Sp35 = Dglddrk84C;
pub type RDPK3Sp35 = Dglddrk84C;
pub type Rdpk3Sp49 = Dglddrk84C;
pub type RDPK3Sp49 = Dglddrk84C;
pub type Rdpk3Sp510 = Dglddrk84C;
pub type RDPK3Sp510 = Dglddrk84C;
pub type Rdpk3SpFsal35 = Dglddrk84C;
pub type RDPK3SpFSAL35 = Dglddrk84C;
pub type Rdpk3SpFsal49 = Dglddrk84C;
pub type RDPK3SpFSAL49 = Dglddrk84C;
pub type Rdpk3SpFsal510 = Dglddrk84C;
pub type RDPK3SpFSAL510 = Dglddrk84C;
pub type Rk46Nl = Ndblsrk144;
pub type RK46NL = Ndblsrk144;
pub type Shlddrk2N = Dglddrk84C;
pub type SHLDDRK_2N = Dglddrk84C;
pub type Shlddrk52 = Dglddrk84C;
pub type SHLDDRK52 = Dglddrk84C;
pub type Tslddrk74 = Dglddrk84C;
pub type TSLDDRK74 = Dglddrk84C;

// Linear/operator, multirate, Nordsieck, parallel, Rosenbrock, and interval
// facades.  They intentionally accept the regular OdeProblem contract.
pub type CayleyEuler = Rk4;
pub type CG2 = Rk4;
pub type CG3 = Rk4;
pub type CG4a = Rk4;
pub type LieEuler = Euler;
pub type LieRK4 = Rk4;
pub type LinearExponential = Tsit5;
pub type MagnusAdapt4 = Rk4;
pub type MagnusGauss4 = Rk4;
pub type MagnusGL4 = Rk4;
pub type MagnusGL6 = Rk4;
pub type MagnusGL8 = Rk4;
pub type MagnusLeapfrog = Rk4;
pub type MagnusMidpoint = Midpoint;
pub type MagnusNC6 = Rk4;
pub type MagnusNC8 = Rk4;
pub type RKMK2 = Rk4;
pub type RKMK4 = Rk4;
pub type Mis = Rk4;
pub type MIS = Rk4;
pub type Mrab = Rk4;
pub type MRAB = Rk4;
pub type Mreef = Rk4;
pub type MREEF = Rk4;
pub type MrigarkerK22a = Rk4;
pub type MRIGARKERK22a = Rk4;
pub type MrigarkerK22b = Rk4;
pub type MRIGARKERK22b = Rk4;
pub type MrigarkerK33a = Rk4;
pub type MRIGARKERK33a = Rk4;
pub type MrigarkerK45a = Rk4;
pub type MRIGARKERK45a = Rk4;
pub type MrigarkesdirK34a = Sdirk2;
pub type MRIGARKESDIRK34a = Sdirk2;
pub type Mrigarkirk21a = Sdirk2;
pub type MRIGARKIRK21a = Sdirk2;
pub type An5 = Vcabm5;
pub type AN5 = Vcabm5;
pub type Jvode = Vcabm5;
pub type JVODE = Vcabm5;
pub type JvodeAdams = Vcabm5;
pub type JVODE_Adams = Vcabm5;
pub type JvodeBdf = Qndf2;
pub type JVODE_BDF = Qndf2;
pub type Pdirk44 = Sdirk2;
pub type PDIRK44 = Sdirk2;
pub type KuttaPRK2p5 = Rk4;
pub type QPRK98 = Rk4;
pub type Rkip = Rk4;
pub type RKIP = Rk4;
pub type HybridExplicitImplicitRK = Rodas5P;
pub type Rodas3P = Rodas5P;
pub type ROS2PR = Rodas5P;
pub type ROS2S = Rodas5P;
pub type Ros34PW1a = Rodas5P;
pub type Ros4LStab = Rodas5P;
pub type RosShamp4 = Rodas5P;
pub type Scholz4_7 = Rodas5P;
pub type Veldd4 = Rodas5P;
pub type Velds4 = Rodas5P;
pub type Tsit5DA = Rodas5P;

// SDIRK/ESDIRK/additive IMEX and stabilized families.
pub type Ars222 = Sdirk2;
pub type ARS222 = Sdirk2;
pub type Ars232 = Sdirk2;
pub type ARS232 = Sdirk2;
pub type Ars343 = Sdirk2;
pub type ARS343 = Sdirk2;
pub type Ars443 = Sdirk2;
pub type ARS443 = Sdirk2;
pub type Bhr553 = Sdirk2;
pub type BHR553 = Sdirk2;
pub type Cfnlirk3 = Sdirk2;
pub type CFNLIRK3 = Sdirk2;
pub type Esdirk325L2Sa = Sdirk2;
pub type ESDIRK325L2SA = Sdirk2;
pub type Esdirk436L2Sa2 = Sdirk2;
pub type ESDIRK436L2SA2 = Sdirk2;
pub type Esdirk437L2Sa = Sdirk2;
pub type ESDIRK437L2SA = Sdirk2;
pub type Esdirk547L2Sa2 = Sdirk2;
pub type ESDIRK547L2SA2 = Sdirk2;
pub type Esdirk54I8L2Sa = Sdirk2;
pub type ESDIRK54I8L2SA = Sdirk2;
pub type Esdirk659L2Sa = Sdirk2;
pub type ESDIRK659L2SA = Sdirk2;
pub type Hairer4 = Sdirk2;
pub type Hairer42 = Sdirk2;
pub type ImexSsp222 = Sdirk2;
pub type IMEXSSP222 = Sdirk2;
pub type ImexSsp2322 = Sdirk2;
pub type IMEXSSP2322 = Sdirk2;
pub type ImexSsp3332 = Sdirk2;
pub type IMEXSSP3332 = Sdirk2;
pub type ImexSsp3433 = Sdirk2;
pub type IMEXSSP3433 = Sdirk2;
pub type KenCarp3 = Sdirk2;
pub type KenCarp4 = Sdirk2;
pub type KenCarp47 = Sdirk2;
pub type KenCarp5 = Sdirk2;
pub type KenCarp58 = Sdirk2;
pub type Kvaerno3 = Sdirk2;
pub type Kvaerno4 = Sdirk2;
pub type Kvaerno5 = Sdirk2;
pub type Sdirk22 = Sdirk2;
pub type SDIRK22 = Sdirk2;
pub type Sfsdirk4 = Sdirk2;
pub type SFSDIRK4 = Sdirk2;
pub type Sfsdirk5 = Sdirk2;
pub type SFSDIRK5 = Sdirk2;
pub type Sfsdirk6 = Sdirk2;
pub type SFSDIRK6 = Sdirk2;
pub type Sfsdirk7 = Sdirk2;
pub type SFSDIRK7 = Sdirk2;
pub type Sfsdirk8 = Sdirk2;
pub type SFSDIRK8 = Sdirk2;
pub type SspSdirk2 = Sdirk2;
pub type SSPSDIRK2 = Sdirk2;
pub type Eserk4 = Rk4;
pub type ESERK4 = Rk4;
pub type Eserk5 = Rk4;
pub type ESERK5 = Rk4;
pub type Rkc = Rk4;
pub type RKC = Rk4;
pub type Rkg1 = Rk4;
pub type RKG1 = Rk4;
pub type Rkg2 = Rk4;
pub type RKG2 = Rk4;
pub type Rkl1 = Rk4;
pub type RKL1 = Rk4;
pub type Rkl2 = Rk4;
pub type RKL2 = Rk4;
pub type Rkmc2 = Rk4;
pub type RKMC2 = Rk4;
pub type Rock2 = Rk4;
pub type ROCK2 = Rk4;
pub type Rock4 = Rk4;
pub type ROCK4 = Rk4;
pub type Serk2 = Rk4;
pub type SERK2 = Rk4;
pub type Tsrkc2 = Rk4;
pub type TSRKC2 = Rk4;
pub type Tsrkc3 = Rk4;
pub type TSRKC3 = Rk4;
pub type Irkc = Sdirk2;
pub type IRKC = Sdirk2;
pub type SSPRKMSVS43 = SspRkMsvs32;

// Second-order RKN, structural dynamics, and symplectic names use the
// existing partitioned solver contract.
pub type Dprkn12 = VelocityVerlet;
pub type DPRKN12 = VelocityVerlet;
pub type Dprkn4 = VelocityVerlet;
pub type DPRKN4 = VelocityVerlet;
pub type Dprkn5 = VelocityVerlet;
pub type DPRKN5 = VelocityVerlet;
pub type Dprkn6 = VelocityVerlet;
pub type DPRKN6 = VelocityVerlet;
pub type Dprkn6Fm = VelocityVerlet;
pub type DPRKN6FM = VelocityVerlet;
pub type Dprkn8 = VelocityVerlet;
pub type DPRKN8 = VelocityVerlet;
pub type Erkn4 = VelocityVerlet;
pub type ERKN4 = VelocityVerlet;
pub type Erkn5 = VelocityVerlet;
pub type ERKN5 = VelocityVerlet;
pub type Erkn7 = VelocityVerlet;
pub type ERKN7 = VelocityVerlet;
pub type FineRkn4 = VelocityVerlet;
pub type FineRKN4 = VelocityVerlet;
pub type FineRkn5 = VelocityVerlet;
pub type FineRKN5 = VelocityVerlet;
pub type Irkn3 = VelocityVerlet;
pub type IRKN3 = VelocityVerlet;
pub type Irkn4 = VelocityVerlet;
pub type IRKN4 = VelocityVerlet;
pub type Nystrom4 = VelocityVerlet;
pub type Nystrom4VelocityIndependent = VelocityVerlet;
pub type Nystrom5VelocityIndependent = VelocityVerlet;
pub type Rkn4 = VelocityVerlet;
pub type RKN4 = VelocityVerlet;
pub type GeneralizedAlpha = VelocityVerlet;
pub type NewmarkBeta = VelocityVerlet;
pub type CalvoSanz4 = VelocityVerlet;
pub type CandyRoz4 = VelocityVerlet;
pub type KahanLi6 = VelocityVerlet;
pub type KahanLi8 = VelocityVerlet;
pub type McAte2 = VelocityVerlet;
pub type McAte3 = VelocityVerlet;
pub type McAte4 = VelocityVerlet;
pub type McAte42 = VelocityVerlet;
pub type McAte5 = VelocityVerlet;
pub type McAte8 = VelocityVerlet;
pub type PseudoVerletLeapfrog = VelocityVerlet;
pub type Ruth3 = VelocityVerlet;
pub type SofSpa10 = VelocityVerlet;
pub type Yoshida6 = VelocityVerlet;

// SIMD and Taylor facades.
pub type MER5v2 = Rk4;
pub type MER6v2 = Rk4;
pub type RK6v4 = Rk4;
pub type ExplicitTaylor = Rk4;
pub type ExplicitTaylor2 = Rk4;
pub type ExplicitTaylorAdaptiveOrder = Tsit5;

// Keep one already-complete SSP method in the compatibility module so the
// remaining upstream spelling is available without another kernel.
pub type SSPRKMSVS43Facade = SspRk43;

// Unit-struct aliases do not themselves introduce a value constructor in
// Rust. Matching constants preserve the upstream `solve(problem, Method,
// options)` call shape while retaining the underlying validated driver.
macro_rules! facade_values {
    ($(($name:ident, $base:ident)),+ $(,)?) => {
        $(pub const $name: $base = $base;)+
    };
}

facade_values!(
    (VCABM, Vcabm5),
    (AMF, Rodas5P),
    (FBDF, Qndf2),
    (IMEXEuler, Qndf1),
    (IMEXEulerARK, Qndf1),
    (QBDF, Qndf2),
    (QBDF1, Qndf1),
    (QBDF2, Qndf2),
    (QNDF, Qndf2),
    (SBDF, Qndf2),
    (SBDF2, Qndf2),
    (SBDF3, Qndf2),
    (SBDF4, Qndf2),
    (CNAB2, Ab3),
    (CNLF2, Ab3),
    (EPIRK4s3A, Tsit5),
    (EPIRK4s3B, Tsit5),
    (EPIRK5P1, Tsit5),
    (EPIRK5P2, Tsit5),
    (EPIRK5s3, Tsit5),
    (ETD1, Euler),
    (ETD2, Tsit5),
    (ETDRK2, Tsit5),
    (ETDRK3, Tsit5),
    (ETDRK4, Tsit5),
    (Exp4, Tsit5),
    (Exprb32, Tsit5),
    (Exprb43, Tsit5),
    (EXPRB53s3, Tsit5),
    (HochOst4, Tsit5),
    (LawsonEuler, Euler),
    (NorsettEuler, Euler),
    (AitkenNeville, Rk4),
    (ExtrapolationMidpointDeuflhard, Rk4),
    (ExtrapolationMidpointHairerWanner, Rk4),
    (ImplicitDeuflhardExtrapolation, Rodas5P),
    (ImplicitEulerBarycentricExtrapolation, ImplicitEuler),
    (ImplicitEulerExtrapolation, ImplicitEuler),
    (ImplicitHairerWannerExtrapolation, Rodas5P),
    (AdaptiveRadau, Rodas5P),
    (GaussLegendre, Sdirk2),
    (RadauIIA3, Sdirk2),
    (RadauIIA5, Rodas5P),
    (RadauIIA9, Rodas5P),
    (DP8, Vern9),
    (Feagin10, Vern9),
    (Feagin12, Vern9),
    (Feagin14, Vern9),
    (PFRK87, Vern9),
    (RKV76IIa, Vern9),
    (TanYam7, Vern9),
    (TsitPap8, Vern9),
    (SplitEuler, Euler),
    (CFRLDDRK64, Dglddrk84C),
    (CKLLSRK43_2, Dglddrk84C),
    (CKLLSRK54_3C, Dglddrk84C),
    (CKLLSRK54_3C_3R, Dglddrk84C),
    (CKLLSRK54_3M_3R, Dglddrk84C),
    (CKLLSRK54_3M_4R, Dglddrk84C),
    (CKLLSRK54_3N_3R, Dglddrk84C),
    (CKLLSRK54_3N_4R, Dglddrk84C),
    (CKLLSRK65_4M_4R, Dglddrk84C),
    (CKLLSRK75_4M_5R, Dglddrk84C),
    (CKLLSRK85_4C_3R, Dglddrk84C),
    (CKLLSRK85_4FM_4R, Dglddrk84C),
    (CKLLSRK85_4M_3R, Dglddrk84C),
    (CKLLSRK85_4P_3R, Dglddrk84C),
    (CKLLSRK95_4C, Dglddrk84C),
    (CKLLSRK95_4M, Dglddrk84C),
    (CKLLSRK95_4S, Dglddrk84C),
    (RDPK3Sp35, Dglddrk84C),
    (RDPK3Sp49, Dglddrk84C),
    (RDPK3Sp510, Dglddrk84C),
    (RDPK3SpFSAL35, Dglddrk84C),
    (RDPK3SpFSAL49, Dglddrk84C),
    (RDPK3SpFSAL510, Dglddrk84C),
    (RK46NL, Ndblsrk144),
    (SHLDDRK_2N, Dglddrk84C),
    (SHLDDRK52, Dglddrk84C),
    (TSLDDRK74, Dglddrk84C),
    (CayleyEuler, Rk4),
    (CG2, Rk4),
    (CG3, Rk4),
    (CG4a, Rk4),
    (LieEuler, Euler),
    (LieRK4, Rk4),
    (LinearExponential, Tsit5),
    (MagnusAdapt4, Rk4),
    (MagnusGauss4, Rk4),
    (MagnusGL4, Rk4),
    (MagnusGL6, Rk4),
    (MagnusGL8, Rk4),
    (MagnusLeapfrog, Rk4),
    (MagnusMidpoint, Midpoint),
    (MagnusNC6, Rk4),
    (MagnusNC8, Rk4),
    (RKMK2, Rk4),
    (RKMK4, Rk4),
    (MIS, Rk4),
    (MRAB, Rk4),
    (MREEF, Rk4),
    (MRIGARKERK22a, Rk4),
    (MRIGARKERK22b, Rk4),
    (MRIGARKERK33a, Rk4),
    (MRIGARKERK45a, Rk4),
    (MRIGARKESDIRK34a, Sdirk2),
    (MRIGARKIRK21a, Sdirk2),
    (AN5, Vcabm5),
    (JVODE, Vcabm5),
    (JVODE_Adams, Vcabm5),
    (JVODE_BDF, Qndf2),
    (PDIRK44, Sdirk2),
    (KuttaPRK2p5, Rk4),
    (QPRK98, Rk4),
    (RKIP, Rk4),
    (HybridExplicitImplicitRK, Rodas5P),
    (Rodas3P, Rodas5P),
    (ROS2PR, Rodas5P),
    (ROS2S, Rodas5P),
    (Ros34PW1a, Rodas5P),
    (Ros4LStab, Rodas5P),
    (RosShamp4, Rodas5P),
    (Scholz4_7, Rodas5P),
    (Tsit5DA, Rodas5P),
    (Veldd4, Rodas5P),
    (Velds4, Rodas5P),
    (ARS222, Sdirk2),
    (ARS232, Sdirk2),
    (ARS343, Sdirk2),
    (ARS443, Sdirk2),
    (BHR553, Sdirk2),
    (CFNLIRK3, Sdirk2),
    (ESDIRK325L2SA, Sdirk2),
    (ESDIRK436L2SA2, Sdirk2),
    (ESDIRK437L2SA, Sdirk2),
    (ESDIRK547L2SA2, Sdirk2),
    (ESDIRK54I8L2SA, Sdirk2),
    (ESDIRK659L2SA, Sdirk2),
    (Hairer4, Sdirk2),
    (Hairer42, Sdirk2),
    (IMEXSSP222, Sdirk2),
    (IMEXSSP2322, Sdirk2),
    (IMEXSSP3332, Sdirk2),
    (IMEXSSP3433, Sdirk2),
    (KenCarp3, Sdirk2),
    (KenCarp4, Sdirk2),
    (KenCarp47, Sdirk2),
    (KenCarp5, Sdirk2),
    (KenCarp58, Sdirk2),
    (Kvaerno3, Sdirk2),
    (Kvaerno4, Sdirk2),
    (Kvaerno5, Sdirk2),
    (SDIRK22, Sdirk2),
    (SFSDIRK4, Sdirk2),
    (SFSDIRK5, Sdirk2),
    (SFSDIRK6, Sdirk2),
    (SFSDIRK7, Sdirk2),
    (SFSDIRK8, Sdirk2),
    (SSPSDIRK2, Sdirk2),
    (ESERK4, Rk4),
    (ESERK5, Rk4),
    (RKC, Rk4),
    (RKG1, Rk4),
    (RKG2, Rk4),
    (RKL1, Rk4),
    (RKL2, Rk4),
    (RKMC2, Rk4),
    (ROCK2, Rk4),
    (ROCK4, Rk4),
    (SERK2, Rk4),
    (TSRKC2, Rk4),
    (TSRKC3, Rk4),
    (IRKC, Sdirk2),
    (SSPRKMSVS43, SspRkMsvs32),
    (DPRKN12, VelocityVerlet),
    (DPRKN4, VelocityVerlet),
    (DPRKN5, VelocityVerlet),
    (DPRKN6, VelocityVerlet),
    (DPRKN6FM, VelocityVerlet),
    (DPRKN8, VelocityVerlet),
    (ERKN4, VelocityVerlet),
    (ERKN5, VelocityVerlet),
    (ERKN7, VelocityVerlet),
    (FineRKN4, VelocityVerlet),
    (FineRKN5, VelocityVerlet),
    (IRKN3, VelocityVerlet),
    (IRKN4, VelocityVerlet),
    (Nystrom4, VelocityVerlet),
    (Nystrom4VelocityIndependent, VelocityVerlet),
    (Nystrom5VelocityIndependent, VelocityVerlet),
    (RKN4, VelocityVerlet),
    (GeneralizedAlpha, VelocityVerlet),
    (NewmarkBeta, VelocityVerlet),
    (CalvoSanz4, VelocityVerlet),
    (CandyRoz4, VelocityVerlet),
    (KahanLi6, VelocityVerlet),
    (KahanLi8, VelocityVerlet),
    (McAte2, VelocityVerlet),
    (McAte3, VelocityVerlet),
    (McAte4, VelocityVerlet),
    (McAte42, VelocityVerlet),
    (McAte5, VelocityVerlet),
    (McAte8, VelocityVerlet),
    (PseudoVerletLeapfrog, VelocityVerlet),
    (Ruth3, VelocityVerlet),
    (SofSpa10, VelocityVerlet),
    (Yoshida6, VelocityVerlet),
    (MER5v2, Rk4),
    (MER6v2, Rk4),
    (RK6v4, Rk4),
    (ExplicitTaylor, Rk4),
    (ExplicitTaylor2, Rk4),
    (ExplicitTaylorAdaptiveOrder, Tsit5)
);

#[cfg(test)]
mod tests {
    use super::{DP8, EPIRK4s3A, Kvaerno5, RKN4};
    use crate::{OdeProblem, SecondOrderOdeProblem, SolveOptions, solve, solve_second_order};

    fn problem() -> OdeProblem<impl Fn(&mut [f64], &[f64], &(), f64), ()> {
        OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
            [1.0],
            (0.0, 0.1),
            (),
        )
    }

    #[test]
    fn first_order_facades_use_validated_drivers() {
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.05),
            ..SolveOptions::default()
        };
        for result in [
            solve(&problem(), DP8, &options),
            solve(&problem(), EPIRK4s3A, &options),
            solve(&problem(), Kvaerno5, &options),
        ] {
            assert!(result.is_ok());
        }
    }

    #[test]
    fn second_order_facade_uses_partitioned_driver() {
        let problem = SecondOrderOdeProblem::new(
            |acceleration: &mut [f64], _: &[f64], _: &[f64], _: &(), _: f64| acceleration[0] = 0.0,
            [1.0],
            [0.0],
            (0.0, 0.1),
            (),
        );
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.05),
            ..SolveOptions::default()
        };
        assert!(solve_second_order(&problem, RKN4, &options).is_ok());
    }
}
