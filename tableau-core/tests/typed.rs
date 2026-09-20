use differential_equations_tableau_core::{
    RknCoefficients, RungeKuttaCoefficients, TableauErrorKind, parse_rkn_tableau, parse_tableau,
};
#[test]
fn typed_rk_matches_resource_coefficients_and_dense() {
    let original = parse_tableau(
        include_str!("../../src/tableau/resources/explicit/tsit5.json"),
        "Tsit5",
    )
    .unwrap();
    let rows: Vec<_> = original.a().iter().map(Vec::as_slice).collect();
    let dense: Option<Vec<_>> = original
        .dense()
        .map(|rows| rows.iter().map(Vec::as_slice).collect());
    let mut input = RungeKuttaCoefficients::explicit(
        original.name(),
        original.order(),
        &rows,
        original.b(),
        original.c(),
    );
    input.embedded_order = original.embedded_order();
    input.error = original.error();
    input.dense = dense.as_deref();
    input.fsal = original.fsal();
    let typed = input.build().unwrap();
    assert_eq!(typed.a(), original.a());
    assert_eq!(typed.b(), original.b());
    assert_eq!(typed.c(), original.c());
    assert_eq!(typed.error(), original.error());
    assert_eq!(typed.dense(), original.dense());
    assert_eq!(typed.fsal(), original.fsal());
}
#[test]
fn typed_rkn_matches_resource() {
    let original = parse_rkn_tableau(
        include_str!("../../src/tableau/resources/second_order/dprkn6.json"),
        "Dprkn6",
    )
    .unwrap();
    let a: Vec<_> = original.a().iter().map(Vec::as_slice).collect();
    let av: Option<Vec<_>> = original
        .a_velocity()
        .map(|rows| rows.iter().map(Vec::as_slice).collect());
    let dense: Option<Vec<_>> = original
        .dense()
        .map(|rows| rows.iter().map(Vec::as_slice).collect());
    let vdense: Option<Vec<_>> = original
        .velocity_dense()
        .map(|rows| rows.iter().map(Vec::as_slice).collect());
    let mut input = RknCoefficients::fixed(
        original.name(),
        original.order(),
        &a,
        original.b(),
        original.b_velocity(),
        original.c(),
    );
    input.a_velocity = av.as_deref();
    input.error = original.error();
    input.velocity_error = original.velocity_error();
    input.position_only_error = original.position_only_error();
    input.dense = dense.as_deref();
    input.velocity_dense = vdense.as_deref();
    let typed = input.build().unwrap();
    assert_eq!(typed.a(), original.a());
    assert_eq!(typed.error(), original.error());
    assert_eq!(typed.dense(), original.dense());
    assert_eq!(typed.velocity_dense(), original.velocity_dense());
}
#[test]
fn typed_rejects_bad_shapes_and_identifies_nonfinite_index() {
    let err = RungeKuttaCoefficients::explicit("Bad", 1, &[&[0.0, 0.0]], &[1.0], &[0.0])
        .build()
        .unwrap_err();
    assert!(err.to_string().contains("square"));
    let err = RungeKuttaCoefficients::explicit("Bad", 1, &[&[f64::NAN]], &[1.0], &[0.0])
        .build()
        .unwrap_err();
    assert_eq!(err.kind(), TableauErrorKind::NonFiniteCoefficient);
    assert!(err.to_string().contains("A[0][0]"));
}
#[test]
fn description_defaults_and_missing_metadata_stays_precise() {
    let json =
        r#"{"name":"Euler","kind":"explicit-runge-kutta","order":1,"A":[[0]],"b":[1],"c":[0]}"#;
    assert!(
        !parse_tableau(json, "Euler")
            .unwrap()
            .description()
            .is_empty()
    );
    assert!(
        parse_tableau(&json.replace("\"name\":\"Euler\",", ""), "Euler")
            .unwrap_err()
            .to_string()
            .contains("name")
    );
    assert!(
        parse_tableau(&json.replace("explicit-runge-kutta", "invalid"), "Euler")
            .unwrap_err()
            .to_string()
            .contains("invalid")
    );
}
#[test]
fn typed_supports_residual_estimators_and_implicit_matrices() {
    use differential_equations_tableau_core::{ErrorEstimatorKind, RungeKuttaKind};
    let mut input = RungeKuttaCoefficients::explicit(
        "HeunResidual",
        2,
        &[&[0., 0.], &[1., 0.]],
        &[0.5, 0.5],
        &[0., 1.],
    );
    input.embedded_order = Some(1);
    input.error = Some(&[1., 0.]);
    input.second_error = Some(&[0., 1.]);
    input.error_estimator = ErrorEstimatorKind::DirectResidual;
    input.description = Some("Residual comparison");
    let method = input.build().unwrap();
    assert_eq!(
        method.error_estimator_kind(),
        ErrorEstimatorKind::DirectResidual
    );
    assert_eq!(method.second_error(), Some([0., 1.].as_slice()));
    let mut input = RungeKuttaCoefficients::explicit(
        "Trap",
        2,
        &[&[0., 0.], &[0.5, 0.5]],
        &[0.5, 0.5],
        &[0., 1.],
    );
    input.kind = RungeKuttaKind::Implicit;
    assert_eq!(input.build().unwrap().kind(), RungeKuttaKind::Implicit);
}
