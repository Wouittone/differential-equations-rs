use differential_equations::solvers::second_order::*;

// FNV-1a fingerprints of the previous Rust f64 coefficient sequences, in
// a-then-b order with little-endian bytes. No duplicate coefficient bank.
#[test]
fn resource_coefficients_preserve_legacy_bit_patterns() {
    for (tableau, expected) in [
        (
            PseudoVerletLeapfrog::tableau().unwrap(),
            0xb59c66f05a917fd8_u64,
        ),
        (McAte2::tableau().unwrap(), 0x08d6942d8d29467e),
        (Ruth3::tableau().unwrap(), 0x92013c583071c421),
        (McAte3::tableau().unwrap(), 0xe816344ded4cec35),
        (CandyRoz4::tableau().unwrap(), 0x905f0a9221e8417f),
        (McAte4::tableau().unwrap(), 0x184295a7d900391a),
        (CalvoSanz4::tableau().unwrap(), 0xfc27a22be0607ab9),
        (McAte42::tableau().unwrap(), 0xa0ea055258a2b2e2),
        (McAte5::tableau().unwrap(), 0x44bd8fe5b4478d1f),
        (Yoshida6::tableau().unwrap(), 0xb05f97f884fadf96),
        (KahanLi6::tableau().unwrap(), 0xe1684394b96c65f6),
        (McAte8::tableau().unwrap(), 0xca8066f31fdbc79b),
        (KahanLi8::tableau().unwrap(), 0x1a5af54963293012),
        (SofSpa10::tableau().unwrap(), 0xb00fb1b80589712a),
    ] {
        let mut hash = 0xcbf29ce484222325_u64;
        for value in tableau.a().iter().chain(tableau.b()) {
            for byte in value.to_bits().to_le_bytes() {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
            }
        }
        assert_eq!(hash, expected, "{} coefficients changed", tableau.name());
    }
}

#[test]
fn builtin_tableaus_are_cached_and_share_storage_across_threads() {
    let first = Yoshida6::tableau().unwrap();
    assert_eq!(first.name(), "Yoshida6");
    assert_eq!(first.order(), 6);
    assert!(!first.description().is_empty());
    assert!(std::ptr::eq(first, Yoshida6::tableau().unwrap()));
    let other = std::thread::spawn(|| Yoshida6::tableau().unwrap())
        .join()
        .unwrap();
    assert!(std::ptr::eq(first, other));
}
