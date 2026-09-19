use crate::tableau::LazyTableau;
use differential_equations_tableau_macros::define_implicit_rk_tableau_from_file;

define_implicit_rk_tableau_from_file!(pub(super) ARS222_TABLEAU, "Ars222", "src/tableau/resources/implicit/ars222.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) ARS232_TABLEAU, "Ars232", "src/tableau/resources/implicit/ars232.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) ARS343_TABLEAU, "Ars343", "src/tableau/resources/implicit/ars343.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) ARS443_TABLEAU, "Ars443", "src/tableau/resources/implicit/ars443.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) BHR553_TABLEAU, "Bhr553", "src/tableau/resources/implicit/bhr553.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) CFNLIRK3_TABLEAU, "Cfnlirk3", "src/tableau/resources/implicit/cfnlirk3.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) ESDIRK325_TABLEAU, "Esdirk325L2Sa", "src/tableau/resources/implicit/esdirk325_l2_sa.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) ESDIRK436_TABLEAU, "Esdirk436L2Sa2", "src/tableau/resources/implicit/esdirk436_l2_sa2.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) ESDIRK437_TABLEAU, "Esdirk437L2Sa", "src/tableau/resources/implicit/esdirk437_l2_sa.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) ESDIRK547_TABLEAU, "Esdirk547L2Sa2", "src/tableau/resources/implicit/esdirk547_l2_sa2.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) ESDIRK54_TABLEAU, "Esdirk54I8L2Sa", "src/tableau/resources/implicit/esdirk54_i8_l2_sa.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) ESDIRK659_TABLEAU, "Esdirk659L2Sa", "src/tableau/resources/implicit/esdirk659_l2_sa.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) HAIRER4_TABLEAU, "Hairer4", "src/tableau/resources/implicit/hairer4.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) HAIRER42_TABLEAU, "Hairer42", "src/tableau/resources/implicit/hairer42.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) IMEX_SSP222_TABLEAU, "ImexSsp222", "src/tableau/resources/implicit/imex_ssp222.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) IMEX_SSP2322_TABLEAU, "ImexSsp2322", "src/tableau/resources/implicit/imex_ssp2322.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) IMEX_SSP3332_TABLEAU, "ImexSsp3332", "src/tableau/resources/implicit/imex_ssp3332.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) IMEX_SSP3433_TABLEAU, "ImexSsp3433", "src/tableau/resources/implicit/imex_ssp3433.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) KENCARP3_TABLEAU, "KenCarp3", "src/tableau/resources/implicit/ken_carp3.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) KENCARP4_TABLEAU, "KenCarp4", "src/tableau/resources/implicit/ken_carp4.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) KENCARP47_TABLEAU, "KenCarp47", "src/tableau/resources/implicit/ken_carp47.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) KENCARP5_TABLEAU, "KenCarp5", "src/tableau/resources/implicit/ken_carp5.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) KENCARP58_TABLEAU, "KenCarp58", "src/tableau/resources/implicit/ken_carp58.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) KVAERNO3_TABLEAU, "Kvaerno3", "src/tableau/resources/implicit/kvaerno3.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) KVAERNO4_TABLEAU, "Kvaerno4", "src/tableau/resources/implicit/kvaerno4.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) KVAERNO5_TABLEAU, "Kvaerno5", "src/tableau/resources/implicit/kvaerno5.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) SDIRK22_TABLEAU, "Sdirk22", "src/tableau/resources/implicit/sdirk22.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) SFSDIRK4_TABLEAU, "Sfsdirk4", "src/tableau/resources/implicit/sfsdirk4.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) SFSDIRK5_TABLEAU, "Sfsdirk5", "src/tableau/resources/implicit/sfsdirk5.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) SFSDIRK6_TABLEAU, "Sfsdirk6", "src/tableau/resources/implicit/sfsdirk6.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) SFSDIRK7_TABLEAU, "Sfsdirk7", "src/tableau/resources/implicit/sfsdirk7.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) SFSDIRK8_TABLEAU, "Sfsdirk8", "src/tableau/resources/implicit/sfsdirk8.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) SSP_SDIRK2_TABLEAU, "SspSdirk2", "src/tableau/resources/implicit/ssp_sdirk2.json", crate = crate);
define_implicit_rk_tableau_from_file!(pub(super) SDIRK2_TABLEAU, "Sdirk2", "src/tableau/resources/implicit/sdirk2.json", crate = crate);

#[derive(Clone, Copy)]
pub(super) enum ExtendedKind {
    Ars222,
    Ars232,
    Ars343,
    Ars443,
    Bhr553,
    Cfnlirk3,
    Esdirk325,
    Esdirk436,
    Esdirk437,
    Esdirk547,
    Esdirk54,
    Esdirk659,
    Hairer4,
    Hairer42,
    ImexSsp222,
    ImexSsp2322,
    ImexSsp3332,
    ImexSsp3433,
    KenCarp3,
    KenCarp4,
    KenCarp47,
    KenCarp5,
    KenCarp58,
    Kvaerno3,
    Kvaerno4,
    Kvaerno5,
    Sdirk22,
    Sfsdirk4,
    Sfsdirk5,
    Sfsdirk6,
    Sfsdirk7,
    Sfsdirk8,
    SspSdirk2,
}

pub(super) fn extended_resource(kind: ExtendedKind) -> &'static LazyTableau {
    match kind {
        ExtendedKind::Ars222 => &ARS222_TABLEAU,
        ExtendedKind::Ars232 => &ARS232_TABLEAU,
        ExtendedKind::Ars343 => &ARS343_TABLEAU,
        ExtendedKind::Ars443 => &ARS443_TABLEAU,
        ExtendedKind::Bhr553 => &BHR553_TABLEAU,
        ExtendedKind::Cfnlirk3 => &CFNLIRK3_TABLEAU,
        ExtendedKind::Esdirk325 => &ESDIRK325_TABLEAU,
        ExtendedKind::Esdirk436 => &ESDIRK436_TABLEAU,
        ExtendedKind::Esdirk437 => &ESDIRK437_TABLEAU,
        ExtendedKind::Esdirk547 => &ESDIRK547_TABLEAU,
        ExtendedKind::Esdirk54 => &ESDIRK54_TABLEAU,
        ExtendedKind::Esdirk659 => &ESDIRK659_TABLEAU,
        ExtendedKind::Hairer4 => &HAIRER4_TABLEAU,
        ExtendedKind::Hairer42 => &HAIRER42_TABLEAU,
        ExtendedKind::ImexSsp222 => &IMEX_SSP222_TABLEAU,
        ExtendedKind::ImexSsp2322 => &IMEX_SSP2322_TABLEAU,
        ExtendedKind::ImexSsp3332 => &IMEX_SSP3332_TABLEAU,
        ExtendedKind::ImexSsp3433 => &IMEX_SSP3433_TABLEAU,
        ExtendedKind::KenCarp3 => &KENCARP3_TABLEAU,
        ExtendedKind::KenCarp4 => &KENCARP4_TABLEAU,
        ExtendedKind::KenCarp47 => &KENCARP47_TABLEAU,
        ExtendedKind::KenCarp5 => &KENCARP5_TABLEAU,
        ExtendedKind::KenCarp58 => &KENCARP58_TABLEAU,
        ExtendedKind::Kvaerno3 => &KVAERNO3_TABLEAU,
        ExtendedKind::Kvaerno4 => &KVAERNO4_TABLEAU,
        ExtendedKind::Kvaerno5 => &KVAERNO5_TABLEAU,
        ExtendedKind::Sdirk22 => &SDIRK22_TABLEAU,
        ExtendedKind::Sfsdirk4 => &SFSDIRK4_TABLEAU,
        ExtendedKind::Sfsdirk5 => &SFSDIRK5_TABLEAU,
        ExtendedKind::Sfsdirk6 => &SFSDIRK6_TABLEAU,
        ExtendedKind::Sfsdirk7 => &SFSDIRK7_TABLEAU,
        ExtendedKind::Sfsdirk8 => &SFSDIRK8_TABLEAU,
        ExtendedKind::SspSdirk2 => &SSP_SDIRK2_TABLEAU,
    }
}

pub(super) fn sdirk2_resource() -> &'static LazyTableau {
    &SDIRK2_TABLEAU
}
