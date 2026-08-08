//! Deterministic compile-time coefficient fixtures emitted by the Phase 4 generator.

#![allow(dead_code)]

pub(crate) const RK4_STAGE_TIMES: [f64; 4] = [0.0, 0.5, 0.5, 1.0];
pub(crate) const RK4_A: [[f64; 4]; 4] = [
    [0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
];
pub(crate) const RK4_EMPTY: &[f64] = &[];
pub(crate) const RK4_A2: &[f64] = &[0.5];
pub(crate) const RK4_A3: &[f64] = &[0.0, 0.5];
pub(crate) const RK4_A4: &[f64] = &[0.0, 0.0, 1.0];
pub(crate) const RK4_A_ROWS: &[&[f64]] = &[RK4_EMPTY, RK4_A2, RK4_A3, RK4_A4];
pub(crate) const RK4_B: [f64; 4] = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0];

pub(crate) const AB3_HISTORY: [f64; 3] = [23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0];

pub(crate) const VELOCITY_VERLET_COMPOSITION: [f64; 2] = [0.5, 0.5];

/// Pinned two-stage SDIRK2 ESDIRK tableau from OrdinaryDiffEqSDIRK.
///
/// The first stage is at `c₁ = 1`, the second at `c₂ = 0`; both stages have
/// unit diagonal and the second-stage explicit coupling is `a₂₁ = -1`.
pub(crate) const SDIRK2_A: [[f64; 2]; 2] = [[1.0, 0.0], [-1.0, 1.0]];
pub(crate) const SDIRK2_B: [f64; 2] = [0.5, 0.5];
pub(crate) const SDIRK2_B_EMBEDDED: [f64; 2] = [0.5, -0.5];
pub(crate) const SDIRK2_STAGE_TIMES: [f64; 2] = [1.0, 0.0];

#[cfg(test)]
mod tests {
    use super::{
        AB3_HISTORY, RK4_A, RK4_B, RK4_STAGE_TIMES, SDIRK2_A, SDIRK2_B, SDIRK2_B_EMBEDDED,
        SDIRK2_STAGE_TIMES, VELOCITY_VERLET_COMPOSITION,
    };

    #[test]
    fn generated_fixtures_have_expected_shapes() {
        assert_eq!(RK4_A.len(), RK4_STAGE_TIMES.len());
        assert_eq!(RK4_B.len(), RK4_STAGE_TIMES.len());
        assert_eq!(AB3_HISTORY.len(), 3);
        assert_eq!(VELOCITY_VERLET_COMPOSITION, [0.5, 0.5]);
        assert!((RK4_B.iter().sum::<f64>() - 1.0).abs() < 1.0e-15);
        assert_eq!(SDIRK2_A.len(), SDIRK2_STAGE_TIMES.len());
        assert_eq!(SDIRK2_B.len(), SDIRK2_STAGE_TIMES.len());
        assert_eq!(SDIRK2_B_EMBEDDED.len(), SDIRK2_STAGE_TIMES.len());
        assert!((SDIRK2_B.iter().sum::<f64>() - 1.0).abs() < 1.0e-15);
    }
}
