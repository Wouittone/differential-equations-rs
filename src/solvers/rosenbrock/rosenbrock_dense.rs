//! Pinned low-order Rosenbrock dense-output coefficient rows.
//!
//! The specialized Rosenbrock23/32 polynomial from OrdinaryDiffEqRosenbrock at
//! `211142263781255a9aa2f910f6760b9f18ec29c8`, stored row-major.

pub(crate) const ROSENBROCK_SPECIAL_1: &[f64] = &[
    1.0 / (1.0 - 2.0 / (2.0 + std::f64::consts::SQRT_2)),
    -1.0 / (1.0 - 2.0 / (2.0 + std::f64::consts::SQRT_2)),
];
pub(crate) const ROSENBROCK_SPECIAL_2: &[f64] = &[
    -(2.0 / (2.0 + std::f64::consts::SQRT_2)) / (1.0 - 2.0 / (2.0 + std::f64::consts::SQRT_2)),
    1.0 / (1.0 - 2.0 / (2.0 + std::f64::consts::SQRT_2)),
];
pub(crate) const ROSENBROCK_SPECIAL: &[&[f64]] = &[ROSENBROCK_SPECIAL_1, ROSENBROCK_SPECIAL_2];
