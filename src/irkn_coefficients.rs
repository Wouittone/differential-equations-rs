//! Generated IRKN3/IRKN4 fixed-step coefficients.
//! Source: SciML/OrdinaryDiffEq.jl at `211142263781255a9aa2f910f6760b9f18ec29c8`.
//! Regenerate with `scripts/generate_irkn_coefficients.jl`.

pub(super) const IRKN3_ORDER: usize = 3;
pub(super) const IRKN3_BOOTSTRAP_ORDER: usize = 4;
pub(super) const IRKN3_INTERNAL_STAGES: usize = 1;
pub(super) const IRKN3_RETAINED_ENDPOINT_ACCELERATIONS: usize = 2;
pub(super) const IRKN3_RETAINED_INTERNAL_STAGES: usize = 1;

pub(super) const IRKN3_VELOCITY_HISTORY: [f64; 2] = [3.0 / 2.0, -1.0 / 2.0];

pub(super) const IRKN3_C: [f64; 1] = [1.0 / 2.0];

pub(super) const IRKN3_A: [f64; 1] = [1.0 / 8.0];

pub(super) const IRKN3_VELOCITY_WEIGHTS: [f64; 2] = [2.0 / 3.0, 5.0 / 6.0];

pub(super) const IRKN3_HISTORY_WEIGHTS: [f64; 2] = [1.0 / 3.0, 5.0 / 12.0];

pub(super) const IRKN4_ORDER: usize = 4;
pub(super) const IRKN4_BOOTSTRAP_ORDER: usize = 4;
pub(super) const IRKN4_INTERNAL_STAGES: usize = 2;
pub(super) const IRKN4_RETAINED_ENDPOINT_ACCELERATIONS: usize = 2;
pub(super) const IRKN4_RETAINED_INTERNAL_STAGES: usize = 2;

pub(super) const IRKN4_VELOCITY_HISTORY: [f64; 2] = [3.0 / 2.0, -1.0 / 2.0];

pub(super) const IRKN4_C: [f64; 2] = [1.0 / 4.0, 3.0 / 4.0];

pub(super) const IRKN4_A: [f64; 2] = [1.0 / 32.0, 9.0 / 32.0];

pub(super) const IRKN4_VELOCITY_WEIGHTS: [f64; 3] = [19.0 / 18.0, -1.0 / 6.0, 11.0 / 18.0];

pub(super) const IRKN4_HISTORY_WEIGHTS: [f64; 3] = [-1.0 / 18.0, 7.0 / 24.0, 1.0 / 8.0];
