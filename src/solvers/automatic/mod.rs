//! Automatic and composite algorithm selectors.

/// Nonstiff-first Dormand--Prince selector with a configurable stiff fallback.
pub mod autodp5;
pub mod composites;

pub use autodp5::{AutoDP5, AutoDp5};
pub use composites::{
    AutoTsit5, AutoVern6, AutoVern7, AutoVern8, AutoVern9, DefaultImplicitODEAlgorithm,
    DefaultODEAlgorithm,
};
