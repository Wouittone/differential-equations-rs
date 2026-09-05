use differential_equations::solvers::second_order::{
    Dprkn4, Dprkn5, Dprkn6, Dprkn6Fm, Dprkn8, Dprkn12, Erkn4, Erkn5, Erkn7, FineRkn4, FineRkn5,
    Irkn3, Irkn4, Nystrom4, Nystrom4VelocityIndependent, Nystrom5VelocityIndependent, Rkn4,
};
use differential_equations::tableau::{
    IrknBootstrapSeed, IrknTableau, RungeKuttaNystromKind, RungeKuttaNystromTableau,
};

fn extend_fingerprint<'a>(hash: &mut u64, values: impl IntoIterator<Item = &'a f64>) {
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            *hash = (*hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
    }
}

fn rkn_fingerprint(tableau: &RungeKuttaNystromTableau) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    extend_fingerprint(&mut hash, tableau.a().iter().flatten());
    if let Some(matrix) = tableau.a_velocity() {
        extend_fingerprint(&mut hash, matrix.iter().flatten());
    }
    extend_fingerprint(&mut hash, tableau.b());
    extend_fingerprint(&mut hash, tableau.b_velocity());
    extend_fingerprint(&mut hash, tableau.c());
    if let Some(values) = tableau.error() {
        extend_fingerprint(&mut hash, values);
    }
    if let Some(values) = tableau.velocity_error() {
        extend_fingerprint(&mut hash, values);
    }
    if let Some(matrix) = tableau.dense() {
        extend_fingerprint(&mut hash, matrix.iter().flatten());
    }
    if let Some(matrix) = tableau.velocity_dense() {
        extend_fingerprint(&mut hash, matrix.iter().flatten());
    }
    hash
}

fn irkn_fingerprint(tableau: &IrknTableau) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    extend_fingerprint(&mut hash, tableau.velocity_history());
    extend_fingerprint(&mut hash, tableau.c());
    extend_fingerprint(&mut hash, tableau.a());
    extend_fingerprint(&mut hash, tableau.velocity_weights());
    extend_fingerprint(&mut hash, tableau.history_weights());
    hash
}

fn assert_square(tableau: &RungeKuttaNystromTableau) {
    assert_eq!(tableau.a().len(), tableau.stages());
    assert!(tableau.a().iter().all(|row| row.len() == tableau.stages()));
    assert_eq!(tableau.b().len(), tableau.stages());
    assert_eq!(tableau.b_velocity().len(), tableau.stages());
    assert_eq!(tableau.c().len(), tableau.stages());
}

#[test]
fn every_fixed_rkn_algorithm_exposes_its_own_validated_tableau() {
    let tableaus = [
        Nystrom4.tableau().unwrap(),
        Nystrom4VelocityIndependent.tableau().unwrap(),
        Nystrom5VelocityIndependent.tableau().unwrap(),
        Rkn4.tableau().unwrap(),
    ];
    assert_eq!(
        tableaus.map(RungeKuttaNystromTableau::name),
        [
            "Nystrom4",
            "Nystrom4VelocityIndependent",
            "Nystrom5VelocityIndependent",
            "Rkn4",
        ]
    );
    for tableau in tableaus {
        assert_eq!(tableau.kind(), RungeKuttaNystromKind::Fixed);
        assert!(tableau.error().is_none());
        assert_square(tableau);
    }
    assert!(Nystrom4.tableau().unwrap().a_velocity().is_some());
    assert!(
        Nystrom4VelocityIndependent
            .tableau()
            .unwrap()
            .a_velocity()
            .is_none()
    );
}

#[test]
fn every_adaptive_rkn_algorithm_exposes_estimator_metadata() {
    let tableaus = [
        Dprkn4.tableau().unwrap(),
        Dprkn5.tableau().unwrap(),
        Dprkn6.tableau().unwrap(),
        Dprkn6Fm.tableau().unwrap(),
        Dprkn8.tableau().unwrap(),
        Dprkn12.tableau().unwrap(),
        Erkn4.tableau().unwrap(),
        Erkn5.tableau().unwrap(),
        Erkn7.tableau().unwrap(),
        FineRkn4.tableau().unwrap(),
        FineRkn5.tableau().unwrap(),
    ];
    assert_eq!(
        tableaus.map(RungeKuttaNystromTableau::name),
        [
            "Dprkn4", "Dprkn5", "Dprkn6", "Dprkn6Fm", "Dprkn8", "Dprkn12", "Erkn4", "Erkn5",
            "Erkn7", "FineRkn4", "FineRkn5",
        ]
    );
    for tableau in tableaus {
        assert_eq!(tableau.kind(), RungeKuttaNystromKind::Adaptive);
        assert_eq!(tableau.error().unwrap().len(), tableau.stages());
        assert_square(tableau);
    }
    let erkn5 = Erkn5.tableau().unwrap();
    assert!(erkn5.position_only_error());
    assert!(erkn5.velocity_error().is_none());
    let dprkn6 = Dprkn6.tableau().unwrap();
    assert_eq!(dprkn6.dense().unwrap().len(), dprkn6.stages());
    assert_eq!(dprkn6.velocity_dense().unwrap().len(), dprkn6.stages());
}

#[test]
fn improved_rkn_algorithms_expose_history_and_bootstrap_policy() {
    let third = Irkn3.tableau().unwrap();
    let fourth = Irkn4.tableau().unwrap();
    assert_eq!(third.name(), "Irkn3");
    assert_eq!(third.order(), 3);
    assert_eq!(third.stages(), 1);
    assert_eq!(third.bootstrap_order(), 4);
    assert_eq!(third.bootstrap_seed(), IrknBootstrapSeed::PreviousEndpoint);
    assert_eq!(fourth.name(), "Irkn4");
    assert_eq!(fourth.order(), 4);
    assert_eq!(fourth.stages(), 2);
    assert_eq!(fourth.bootstrap_seed(), IrknBootstrapSeed::NewEndpoint);
}

#[test]
fn resource_coefficients_preserve_legacy_bit_patterns() {
    // FNV-1a over little-endian f64::to_bits bytes. RKN fields are ordered as
    // A, A_velocity, b, b_velocity, c, error, velocity_error, dense, and
    // velocity_dense; absent optional fields contribute no values.
    for (tableau, expected) in [
        (Nystrom4.tableau().unwrap(), 0x3961e1031f5d454e_u64),
        (
            Nystrom4VelocityIndependent.tableau().unwrap(),
            0x1ece3063f4664cff,
        ),
        (
            Nystrom5VelocityIndependent.tableau().unwrap(),
            0xb0f44adc9c9b9b8f,
        ),
        (Rkn4.tableau().unwrap(), 0x7ac851ffa3211a2f),
        (Dprkn4.tableau().unwrap(), 0x381369ed98506941),
        (Dprkn5.tableau().unwrap(), 0xaacd062f74caccdd),
        (Dprkn6.tableau().unwrap(), 0x2fdd1b17f21379cd),
        (Dprkn6Fm.tableau().unwrap(), 0xb454b70e0321e027),
        (Dprkn8.tableau().unwrap(), 0x7d0db6bb9ece7247),
        (Dprkn12.tableau().unwrap(), 0x44bb900902b677e9),
        (Erkn4.tableau().unwrap(), 0x6b8371eb260cf2c6),
        (Erkn5.tableau().unwrap(), 0x0ca19384cc82f7d3),
        (Erkn7.tableau().unwrap(), 0x0388612cc2a5e82a),
        (FineRkn4.tableau().unwrap(), 0x45b89dd70a807dc1),
        (FineRkn5.tableau().unwrap(), 0xe3004d3a1fc4a557),
    ] {
        let actual = rkn_fingerprint(tableau);
        assert_eq!(actual, expected, "{} coefficients changed", tableau.name());
    }

    // IRKN fields are ordered as velocity_history, c, A, velocity_weights,
    // and history_weights.
    for (tableau, expected) in [
        (Irkn3.tableau().unwrap(), 0x5ea2a4ee11b40c4d_u64),
        (Irkn4.tableau().unwrap(), 0x46986c3cfb6ba74a),
    ] {
        let actual = irkn_fingerprint(tableau);
        assert_eq!(actual, expected, "{} coefficients changed", tableau.name());
    }
}
