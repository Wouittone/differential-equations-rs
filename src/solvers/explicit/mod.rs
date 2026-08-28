//! Explicit Runge--Kutta algorithms and resource-backed variants.

/// Configurable fifth-order Ananthakrishnan explicit method.
pub mod anas5;
pub mod frk65;
/// General explicit Runge--Kutta tableaus and their shared driver.
pub mod general;
pub mod high_order;
/// Low-storage Runge--Kutta algorithm families.
pub mod low_storage_rk;
pub mod prk;
pub mod qprk;
pub mod split_euler;
pub mod ssprk_extended;
pub mod ssprk_kyk2014;
pub mod ssprk_kyk42;
pub mod ssprk_msvs;
/// Tsitouras 5/4 explicit Runge--Kutta method.
pub mod tsit5;
pub mod verner;

pub use anas5::Anas5;
pub use frk65::Frk65;
pub use general::*;
pub use high_order::*;
pub use low_storage_rk::*;
pub use prk::KuttaPRK2p5;
pub use qprk::QPRK98;
pub use split_euler::{SplitEuler, SplitOdeAlgorithm, solve_split, solve_split_euler};
pub use ssprk_extended::*;
pub use ssprk_kyk42::{KYKSSPRK42, KykSsprk42};
pub use ssprk_kyk2014::Kyk2014DgSsprk3S2;
pub use ssprk_msvs::{SSPRKMSVS32, SSPRKMSVS43, SspRkMsvs32, SspRkMsvs43};
pub use tsit5::Tsit5;
pub use verner::{Vern6, Vern7, Vern8, Vern9};
