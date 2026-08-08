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

/// Bogacki-Shampine 3(2) explicit tableau from OrdinaryDiffEqLowOrderRK.
/// The fourth stage is FSAL and is retained in the generated rows so the
/// shared explicit driver can reuse it after accepted steps.
pub(crate) const BS3_STAGE_TIMES: [f64; 4] = [0.0, 0.5, 0.75, 1.0];
pub(crate) const BS3_A_EMPTY: &[f64] = &[];
pub(crate) const BS3_A2: &[f64] = &[0.5];
pub(crate) const BS3_A3: &[f64] = &[0.0, 0.75];
pub(crate) const BS3_A4: &[f64] = &[2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0];
pub(crate) const BS3_A_ROWS: &[&[f64]] = &[BS3_A_EMPTY, BS3_A2, BS3_A3, BS3_A4];
pub(crate) const BS3_B: [f64; 4] = [2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0];
pub(crate) const BS3_E: [f64; 4] = [-5.0 / 72.0, 1.0 / 12.0, 1.0 / 9.0, -1.0 / 8.0];

/// Dormand-Prince 5(4) tableau from OrdinaryDiffEqLowOrderRK. The final
/// stage is FSAL; the generated record retains the embedded defect weights
/// used by the adaptive shared driver.
pub(crate) const DP5_STAGE_TIMES: [f64; 7] =
    [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];
pub(crate) const DP5_A_EMPTY: &[f64] = &[];
pub(crate) const DP5_A2: &[f64] = &[1.0 / 5.0];
pub(crate) const DP5_A3: &[f64] = &[3.0 / 40.0, 9.0 / 40.0];
pub(crate) const DP5_A4: &[f64] = &[44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0];
pub(crate) const DP5_A5: &[f64] = &[
    19_372.0 / 6_561.0,
    -25_360.0 / 2_187.0,
    64_448.0 / 6_561.0,
    -212.0 / 729.0,
];
pub(crate) const DP5_A6: &[f64] = &[
    9_017.0 / 3_168.0,
    -355.0 / 33.0,
    46_732.0 / 5_247.0,
    49.0 / 176.0,
    -5_103.0 / 18_656.0,
];
pub(crate) const DP5_A7: &[f64] = &[
    35.0 / 384.0,
    0.0,
    500.0 / 1_113.0,
    125.0 / 192.0,
    -2_187.0 / 6_784.0,
    11.0 / 84.0,
];
pub(crate) const DP5_A_ROWS: &[&[f64]] =
    &[DP5_A_EMPTY, DP5_A2, DP5_A3, DP5_A4, DP5_A5, DP5_A6, DP5_A7];
pub(crate) const DP5_B: [f64; 7] = [
    35.0 / 384.0,
    0.0,
    500.0 / 1_113.0,
    125.0 / 192.0,
    -2_187.0 / 6_784.0,
    11.0 / 84.0,
    0.0,
];
pub(crate) const DP5_E: [f64; 7] = [
    35.0 / 384.0 - 5_179.0 / 57_600.0,
    0.0,
    500.0 / 1_113.0 - 7_571.0 / 16_695.0,
    125.0 / 192.0 - 393.0 / 640.0,
    -2_187.0 / 6_784.0 + 92_097.0 / 339_200.0,
    11.0 / 84.0 - 187.0 / 2_100.0,
    -1.0 / 40.0,
];

pub(crate) const AB3_HISTORY: [f64; 3] = [23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0];

/// Variable-step ABDF2 fixed-leading-coefficient constants.  The history
/// ratio enters the alpha terms at runtime; beta coefficients are invariant
/// except for the linear `(rho - 1)` correction.
pub(crate) const ABDF2_BETA_ZERO: f64 = 2.0 / 3.0;
pub(crate) const ABDF2_BETA_ONE_SCALE: f64 = -1.0 / 3.0;
pub(crate) const ABDF2_ALPHA_ONE_BASE: f64 = 1.0;
pub(crate) const ABDF2_ALPHA_HISTORY_SCALE: f64 = 1.0 / 3.0;

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
        AB3_HISTORY, BS3_A_ROWS, BS3_B, BS3_E, BS3_STAGE_TIMES, DP5_A_ROWS, DP5_B, DP5_E,
        DP5_STAGE_TIMES, RK4_A, RK4_B, RK4_STAGE_TIMES, SDIRK2_A, SDIRK2_B, SDIRK2_B_EMBEDDED,
        SDIRK2_STAGE_TIMES, VELOCITY_VERLET_COMPOSITION,
    };

    #[test]
    fn generated_fixtures_have_expected_shapes() {
        assert_eq!(RK4_A.len(), RK4_STAGE_TIMES.len());
        assert_eq!(RK4_B.len(), RK4_STAGE_TIMES.len());
        assert_eq!(AB3_HISTORY.len(), 3);
        assert_eq!(BS3_A_ROWS.len(), BS3_STAGE_TIMES.len());
        assert_eq!(BS3_B.len(), BS3_STAGE_TIMES.len());
        assert_eq!(BS3_E.len(), BS3_STAGE_TIMES.len());
        assert!((BS3_B.iter().sum::<f64>() - 1.0).abs() < 1.0e-15);
        assert_eq!(DP5_A_ROWS.len(), DP5_STAGE_TIMES.len());
        assert_eq!(DP5_B.len(), DP5_STAGE_TIMES.len());
        assert_eq!(DP5_E.len(), DP5_STAGE_TIMES.len());
        assert!((DP5_B.iter().sum::<f64>() - 1.0).abs() < 1.0e-15);
        assert_eq!(VELOCITY_VERLET_COMPOSITION, [0.5, 0.5]);
        assert!((RK4_B.iter().sum::<f64>() - 1.0).abs() < 1.0e-15);
        assert_eq!(SDIRK2_A.len(), SDIRK2_STAGE_TIMES.len());
        assert_eq!(SDIRK2_B.len(), SDIRK2_STAGE_TIMES.len());
        assert_eq!(SDIRK2_B_EMBEDDED.len(), SDIRK2_STAGE_TIMES.len());
        assert!((SDIRK2_B.iter().sum::<f64>() - 1.0).abs() < 1.0e-15);
    }
}
