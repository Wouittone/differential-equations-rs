use super::{ETD1, NorsettEuler, matrix_exp, phi_action};
use std::any::TypeId;

#[test]
fn matrix_functions_match_scalar_definitions() {
    let exponential = matrix_exp(&[1.0], 1)[0];
    let phi1 = phi_action(&[1.0], 1.0, 1, &[1.0])[0];
    let phi2 = phi_action(&[1.0], 1.0, 2, &[1.0])[0];
    assert!((exponential - std::f64::consts::E).abs() < 1.0e-14);
    assert!((phi1 - (std::f64::consts::E - 1.0)).abs() < 1.0e-14);
    assert!((phi2 - (std::f64::consts::E - 2.0)).abs() < 1.0e-14);
}

#[test]
fn etd1_is_an_exact_type_alias() {
    assert_eq!(TypeId::of::<ETD1>(), TypeId::of::<NorsettEuler>());
}
