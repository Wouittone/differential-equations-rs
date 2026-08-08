//! Deterministic compile-time coefficient fixtures emitted by the Phase 4 generator.

#![allow(dead_code)]

pub(crate) const RK4_STAGE_TIMES: [f64; 4] = [0.0, 0.5, 0.5, 1.0];
pub(crate) const RK4_A: [[f64; 4]; 4] = [
    [0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
];
pub(crate) const RK4_B: [f64; 4] = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0];

pub(crate) const AB3_HISTORY: [f64; 3] = [23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0];

pub(crate) const VELOCITY_VERLET_COMPOSITION: [f64; 2] = [0.5, 0.5];

#[cfg(test)]
mod tests {
    use super::{AB3_HISTORY, RK4_A, RK4_B, RK4_STAGE_TIMES, VELOCITY_VERLET_COMPOSITION};

    #[test]
    fn generated_fixtures_have_expected_shapes() {
        assert_eq!(RK4_A.len(), RK4_STAGE_TIMES.len());
        assert_eq!(RK4_B.len(), RK4_STAGE_TIMES.len());
        assert_eq!(AB3_HISTORY.len(), 3);
        assert_eq!(VELOCITY_VERLET_COMPOSITION, [0.5, 0.5]);
        assert!((RK4_B.iter().sum::<f64>() - 1.0).abs() < 1.0e-15);
    }
}
