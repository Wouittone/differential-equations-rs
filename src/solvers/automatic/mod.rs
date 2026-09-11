//! Automatic algorithms that switch between non-stiff and stiff kernels.
//!
//! Both kernels run inside one integration driver. Branch transitions occur
//! only at accepted-state boundaries, so callbacks, exact stops, saved output,
//! dense interpolation, and cumulative statistics are never restarted or
//! replayed. Method-specific caches are reinitialized at the accepted handoff
//! state, and controller error history is deliberately reset so one method's
//! estimates cannot bias the other. [`AutoSwitchConfig`] exposes the detector
//! thresholds, confirmation counts, residence rule, initial branch, and
//! switch-back policy.
//!
//! ```
//! use differential_equations::solvers::{
//!     automatic::{AutoSwitchConfig, AutoTsit5, AutomaticBranch},
//!     rosenbrock::Rodas5P,
//! };
//! use differential_equations::{OdeProblem, SolveOptions, solve};
//!
//! let problem = OdeProblem::new(
//!     |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -u[0],
//!     [1.0],
//!     (0.0, 1.0),
//!     (),
//! );
//! let policy = AutoSwitchConfig::new().with_stiff_confirmations(4)?;
//! let solution = solve(
//!     &problem,
//!     AutoTsit5::new(Rodas5P).with_switch_config(policy)?,
//!     &SolveOptions::default(),
//! )?;
//! assert!(solution.stats().final_automatic_branch.is_some());
//! assert!(matches!(
//!     solution.stats().final_automatic_branch,
//!     Some(AutomaticBranch::NonStiff | AutomaticBranch::Stiff)
//! ));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod config;
mod policy;
mod switching;

/// Nonstiff-first Dormand--Prince selector with a configurable stiff fallback.
pub mod autodp5;
pub mod composites;

pub use autodp5::{AutoDP5, AutoDp5};
pub use composites::{
    AutoTsit5, AutoVern6, AutoVern7, AutoVern8, AutoVern9, DefaultImplicitODEAlgorithm,
    DefaultODEAlgorithm,
};
pub use config::{AutoSwitchConfig, AutoSwitchConfigError, AutomaticBranch};
pub use switching::AutomaticStiffAlgorithm;
