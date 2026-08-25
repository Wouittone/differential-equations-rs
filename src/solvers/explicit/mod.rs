//! Explicit Runge--Kutta algorithms and coefficient-backed variants.

pub mod anas5;
pub mod frk65;
pub mod general;
pub mod high_order;
pub mod low_storage_rk;
pub mod prk;
pub mod qprk;
pub mod simd_rk;
pub mod split_euler;
pub mod ssprk_extended;
pub mod ssprk_kyk2014;
pub mod ssprk_kyk42;
pub mod ssprk_msvs;
pub mod tsit5;
pub mod verner;

pub(crate) mod coefficient_data {
    #![allow(clippy::excessive_precision)]

    use differential_equations_tableau_macros::define_coefficients_from_file;

    define_coefficients_from_file!(pub(crate), "coefficients/explicit/core.toml", crate = crate);
    define_coefficients_from_file!(pub(crate), "coefficients/explicit/dense.toml", crate = crate);
}

#[cfg(test)]
mod resource_tests {
    use super::coefficient_data::{
        BS3_A_ROWS, BS3_B, BS3_E, BS3_STAGE_TIMES, DP5_A_ROWS, DP5_B, DP5_E, DP5_STAGE_TIMES,
        SDIRK2_A, SDIRK2_B, SDIRK2_B_EMBEDDED, SDIRK2_STAGE_TIMES, VERN6_A_ROWS, VERN6_B, VERN6_E,
        VERN6_STAGE_TIMES, VERN7_A_ROWS, VERN7_B, VERN7_E, VERN7_STAGE_TIMES, VERN8_A_ROWS,
        VERN8_B, VERN8_E, VERN8_STAGE_TIMES, VERN9_A_ROWS, VERN9_B, VERN9_E, VERN9_STAGE_TIMES,
    };

    #[test]
    fn coefficient_resources_have_expected_shapes() {
        assert_tableau_shape(BS3_A_ROWS, &BS3_B, &BS3_E, &BS3_STAGE_TIMES, 1.0e-15);
        assert_tableau_shape(DP5_A_ROWS, &DP5_B, &DP5_E, &DP5_STAGE_TIMES, 1.0e-15);
        assert_tableau_shape(
            VERN6_A_ROWS,
            &VERN6_B,
            &VERN6_E,
            &VERN6_STAGE_TIMES,
            1.0e-13,
        );
        assert_tableau_shape(
            VERN7_A_ROWS,
            &VERN7_B,
            &VERN7_E,
            &VERN7_STAGE_TIMES,
            1.0e-13,
        );
        assert_tableau_shape(
            VERN8_A_ROWS,
            &VERN8_B,
            &VERN8_E,
            &VERN8_STAGE_TIMES,
            1.0e-13,
        );
        assert_tableau_shape(
            VERN9_A_ROWS,
            &VERN9_B,
            &VERN9_E,
            &VERN9_STAGE_TIMES,
            1.0e-13,
        );
        assert_eq!(SDIRK2_A.len(), SDIRK2_STAGE_TIMES.len());
        assert_eq!(SDIRK2_B.len(), SDIRK2_STAGE_TIMES.len());
        assert_eq!(SDIRK2_B_EMBEDDED.len(), SDIRK2_STAGE_TIMES.len());
        assert!((SDIRK2_B.iter().sum::<f64>() - 1.0).abs() < 1.0e-15);
    }

    fn assert_tableau_shape(
        rows: &[&[f64]],
        weights: &[f64],
        error_weights: &[f64],
        nodes: &[f64],
        tolerance: f64,
    ) {
        assert_eq!(rows.len(), nodes.len());
        assert_eq!(weights.len(), nodes.len());
        assert_eq!(error_weights.len(), nodes.len());
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < tolerance);
    }
}

pub use anas5::Anas5;
pub use frk65::Frk65;
pub use general::*;
pub use high_order::*;
pub use low_storage_rk::*;
pub use prk::{KuttaPRK2p5, KuttaPrk2p5Tableau};
pub use qprk::{QPRK98, Qprk98Tableau};
pub use simd_rk::{MER5v2, MER6v2, RK6v4};
pub use split_euler::SplitEuler;
pub use ssprk_extended::*;
pub use ssprk_kyk42::{KYKSSPRK42, KykSsprk42};
pub use ssprk_kyk2014::Kyk2014DgSsprk3S2;
pub use ssprk_msvs::{SSPRKMSVS32, SSPRKMSVS43, SspRkMsvs32, SspRkMsvs43};
pub use tsit5::Tsit5;
pub use verner::{Vern6, Vern7, Vern8, Vern9};

/// Historical name for the low-storage implementation module.
pub mod low_storage {
    pub use super::low_storage_rk::*;
}

/// Strong-stability-preserving and positivity-preserving algorithms.
pub mod ssp {
    pub use super::general::{SspRk22, SspRk33, SspRk43};
    pub use super::ssprk_extended::*;
    pub use super::ssprk_kyk42::*;
    pub use super::ssprk_kyk2014::*;
    pub use super::ssprk_msvs::*;
}

pub mod prelude {
    pub use super::anas5::Anas5;
    pub use super::frk65::Frk65;
    pub use super::general::*;
    pub use super::high_order::*;
    pub use super::low_storage_rk::*;
    pub use super::prk::*;
    pub use super::qprk::*;
    pub use super::simd_rk::*;
    pub use super::split_euler::SplitEuler;
    pub use super::ssprk_extended::*;
    pub use super::ssprk_kyk42::*;
    pub use super::ssprk_kyk2014::*;
    pub use super::ssprk_msvs::*;
    pub use super::tsit5::Tsit5;
    pub use super::verner::*;
}
