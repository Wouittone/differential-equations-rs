//! Explicit Runge--Kutta algorithms and coefficient-backed variants.

pub mod anas5;
mod dense_coefficients;
pub mod frk65;
pub mod general;
pub(crate) mod generated_coefficients;
pub mod high_order;
pub mod low_storage_rk;
pub mod prk;
pub mod qprk;
pub mod simd_rk;
mod simd_rk_coefficients;
pub mod split_euler;
pub mod ssprk_extended;
pub mod ssprk_kyk2014;
pub mod ssprk_kyk42;
pub mod ssprk_msvs;
pub mod tsit5;
pub mod verner;

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
