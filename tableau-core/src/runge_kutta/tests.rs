use super::{
    ErrorEstimatorKind, RungeKuttaKind, TableauErrorKind, parse_numeric_expression, parse_tableau,
};
use std::error::Error as _;

const RESOURCE: &str = r#"{
      "name": "Heun",
      "description": "Heun's explicit second-order method.",
      "kind": "explicit-runge-kutta",
      "order": 2,
      "A": [[0, 0], [1, 0]],
      "b": ["1/2", "1/2"],
      "c": [0, 1]
    }"#;

#[test]
fn parses_canonical_butcher_tableau() {
    let tableau = parse_tableau(RESOURCE, "Heun").unwrap();
    assert_eq!(tableau.kind(), RungeKuttaKind::Explicit);
    assert_eq!(tableau.stages(), 2);
    assert_eq!(tableau.a(), &[vec![0.0, 0.0], vec![1.0, 0.0]]);
    assert_eq!(tableau.a_row(0), Some([0.0, 0.0].as_slice()));
    assert_eq!(tableau.a_row(1), Some([1.0, 0.0].as_slice()));
    assert_eq!(tableau.a_row(2), None);
    assert_eq!(tableau.stage_row(0), Some([].as_slice()));
    assert_eq!(tableau.stage_row(1), Some([1.0].as_slice()));
    assert_eq!(tableau.stage_row(2), None);
    assert_eq!(tableau.b(), &[0.5, 0.5]);
    assert_eq!(tableau.c(), &[0.0, 1.0]);
    assert_eq!(tableau.real_stability_radius(), None);
}

#[test]
fn explicit_stage_rows_are_not_exposed_for_implicit_tableaus() {
    let source = r#"{"name":"Trap","description":"Trapezoidal rule","kind":"implicit-runge-kutta","order":2,"A":[[0,0],["1/2","1/2"]],"b":["1/2","1/2"],"c":[0,1]}"#;
    let tableau = parse_tableau(source, "Trap").unwrap();

    assert_eq!(tableau.stages(), 2);
    assert_eq!(tableau.a_row(1), Some([0.5, 0.5].as_slice()));
    assert_eq!(tableau.stage_row(0), None);
    assert_eq!(tableau.stage_row(1), None);
    assert_eq!(tableau.stage_row(2), None);
}

#[test]
fn real_stability_radius_is_optional_positive_explicit_metadata() {
    let source = RESOURCE.replace(
        "\"order\": 2,",
        "\"order\": 2, \"real_stability_radius\": \"4 / 2\",",
    );
    assert_eq!(
        parse_tableau(&source, "Heun")
            .unwrap()
            .real_stability_radius(),
        Some(2.0)
    );

    for invalid in [
        source.replace("\"4 / 2\"", "0"),
        source.replace("\"4 / 2\"", "-1"),
        source.replace("\"4 / 2\"", "\"1 / 0\""),
        source.replace("explicit-runge-kutta", "implicit-runge-kutta"),
    ] {
        assert!(
            parse_tableau(&invalid, "Heun").is_err(),
            "accepted {invalid}"
        );
    }
}

#[test]
fn embedded_weights_and_direct_errors_materialize_identically() {
    let embedded = RESOURCE.replace(
        "\"order\": 2,",
        "\"order\": 2, \"embedded_order\": 1, \"b_hat\": [1, 0],",
    );
    let direct = embedded.replace("\"b_hat\": [1, 0]", "\"error\": [\"-1/2\", \"1/2\"]");
    let tableau = parse_tableau(&embedded, "Heun").unwrap();
    assert_eq!(tableau, parse_tableau(&direct, "Heun").unwrap());
    assert_eq!(tableau.error(), Some([-0.5, 0.5].as_slice()));
    let second = embedded.replace(
        "\"c\": [0, 1]",
        "\"c\": [0, 1], \"second_error\": [\"-1/4\", \"1/4\"]",
    );
    assert_eq!(
        parse_tableau(&second, "Heun").unwrap().second_error(),
        Some([-0.25, 0.25].as_slice())
    );
}

#[test]
fn direct_residual_estimators_are_explicitly_typed() {
    let residual = RESOURCE.replace(
            "\"order\": 2,",
            "\"order\": 2, \"embedded_order\": 1, \"error_estimator\": \"direct-residual\", \"error\": [1, 0],",
        );
    let tableau = parse_tableau(&residual, "Heun").unwrap();
    assert_eq!(
        tableau.error_estimator_kind(),
        ErrorEstimatorKind::DirectResidual
    );
    assert_eq!(tableau.error(), Some([1.0, 0.0].as_slice()));

    assert!(
        parse_tableau(
            &residual.replace(", \"error_estimator\": \"direct-residual\"", ""),
            "Heun"
        )
        .is_err()
    );
    assert!(
        parse_tableau(
            &residual.replace("\"error\": [1, 0]", "\"b_hat\": [1, 0]"),
            "Heun",
        )
        .is_err()
    );
    assert!(
        parse_tableau(
            &RESOURCE.replace(
                "\"order\": 2,",
                "\"order\": 2, \"error_estimator\": \"direct-residual\",",
            ),
            "Heun",
        )
        .is_err()
    );
}

#[test]
fn secondary_estimators_and_dense_rows_are_structurally_validated() {
    let embedded = RESOURCE.replace(
        "\"order\": 2,",
        "\"order\": 2, \"embedded_order\": 1, \"error\": [\"-1/2\", \"1/2\"],",
    );
    for invalid in [
        RESOURCE.replace("\"c\": [0, 1]", "\"c\": [0, 1], \"second_error\": [0, 0]"),
        embedded.replace("\"c\": [0, 1]", "\"c\": [0, 1], \"second_error\": [0]"),
        embedded.replace("\"-1/2\", \"1/2\"", "0, 0"),
        embedded.replace("\"-1/2\", \"1/2\"", "1, 1"),
        RESOURCE.replace("\"c\": [0, 1]", "\"c\": [0, 1], \"dense\": [[\"1/2\"], []]"),
        RESOURCE.replace(
            "\"c\": [0, 1]",
            "\"c\": [0, 1], \"dense\": [[1], [\"1/2\"], [0]]",
        ),
    ] {
        assert!(
            parse_tableau(&invalid, "Heun").is_err(),
            "accepted {invalid}"
        );
    }
}

#[test]
fn implicit_companions_and_stage_predictors_are_validated() {
    // First-order endpoint update with a second-order trapezoidal companion.
    let source = r#"{"name":"Pair","description":"Implicit pair","kind":"implicit-runge-kutta","order":1,"embedded_order":2,"A":[[0,0],["1/2","1/2"]],"b":[0,1],"c":[0,1],"error":["-1/2","1/2"],"stage_predictors":[[],[1]]}"#;
    let tableau = parse_tableau(source, "Pair").unwrap();
    assert_eq!(tableau.embedded_order(), Some(2));
    assert_eq!(tableau.stage_predictor(1), Some([1.0].as_slice()));
    assert_eq!(tableau.stage_predictor(0), None);
    assert_eq!(tableau.stage_predictor(2), None);
    for invalid in [
        source.replace("[[],[1]]", "[[]]"),
        source.replace("[[],[1]]", "[[1],[1]]"),
        source.replace("[[],[1]]", "[[],[1,0]]"),
        source.replace("[[],[1]]", "[[],[2]]"),
        source.replace("[[],[1]]", "[[],[\"1/0\"]]"),
        source.replace("[[],[1]]", "null"),
        source.replace("\"embedded_order\":2", "\"embedded_order\":0"),
        source.replace("implicit-runge-kutta", "explicit-runge-kutta"),
    ] {
        assert!(
            parse_tableau(&invalid, "Pair").is_err(),
            "accepted {invalid}"
        );
    }
    let defaults = source.replace("[[],[1]]", "[[],[]]");
    assert_eq!(
        parse_tableau(&defaults, "Pair").unwrap().stage_predictor(1),
        None
    );
    let explicit = RESOURCE.replace(
        "\"order\": 2,",
        "\"order\": 2, \"stage_predictors\": [[],[1]],",
    );
    assert!(parse_tableau(&explicit, "Heun").is_err());
}

#[test]
fn fsal_checks_both_endpoint_stages_for_explicit_and_implicit_tableaus() {
    let implicit = r#"{"name":"Trap","description":"Trapezoidal rule","kind":"implicit-runge-kutta","order":2,"fsal":true,"A":[[0,0],["1/2","1/2"]],"b":["1/2","1/2"],"c":[0,1]}"#;
    assert!(parse_tableau(implicit, "Trap").unwrap().fsal());
    for invalid in [
        implicit.replace("[0,1]", "[0.1,1]"),
        implicit.replace("[0,1]", "[0,0.5]"),
        implicit.replace("[0,0]", "[\"1/2\",\"-1/2\"]"),
        implicit.replace("\"b\":[\"1/2\",\"1/2\"]", "\"b\":[1,0]"),
    ] {
        assert!(
            parse_tableau(&invalid, "Trap").is_err(),
            "accepted {invalid}"
        );
    }
    let explicit = r#"{"name":"Heun","description":"Heun with a final FSAL stage","kind":"explicit-runge-kutta","order":2,"fsal":true,"A":[[0,0,0],[1,0,0],["1/2","1/2",0]],"b":["1/2","1/2",0],"c":[0,1,1]}"#;
    assert!(parse_tableau(explicit, "Heun").unwrap().fsal());
}

#[test]
fn rejects_invalid_or_ambiguous_embedded_weights() {
    let embedded = RESOURCE.replace(
        "\"order\": 2,",
        "\"order\": 2, \"embedded_order\": 1, \"b_hat\": [1, 0],",
    );
    for invalid in [
        embedded.replace("\"b_hat\": [1, 0]", "\"b_hat\": [1]"),
        embedded.replace("\"b_hat\": [1, 0]", "\"b_hat\": [1, 1]"),
        embedded.replace("\"b_hat\": [1, 0]", "\"b_hat\": [\"1/0\", 0]"),
        embedded.replace("\"b_hat\": [1, 0]", "\"b_hat\": [1e308, 1e308]"),
        embedded.replace("\"b_hat\": [1, 0]", "\"b_hat\": [1, 0], \"error\": [0, 0]"),
        embedded.replace("\"embedded_order\": 1,", ""),
        embedded.replace("\"b_hat\": [1, 0],", ""),
    ] {
        assert!(
            parse_tableau(&invalid, "Heun").is_err(),
            "accepted {invalid}"
        );
    }
    // Each weight sum is finite and one, but subtraction can still overflow.
    let overflow = r#"{"name":"Overflow","description":"Subtraction overflow","kind":"explicit-runge-kutta","order":2,"embedded_order":1,"A":[[0,0,0],[0,0,0],[0,0,0]],"b":[1e308,-1e308,1],"b_hat":[-1e308,1e308,1],"c":[0,0,0]}"#;
    let error = parse_tableau(overflow, "Overflow").unwrap_err();
    assert!(error.to_string().contains("derived error[0] is not finite"));
}

#[test]
fn overflowing_coefficient_sums_are_not_approximately_consistent() {
    let weights = RESOURCE.replace("[\"1/2\", \"1/2\"]", "[1e308, 1e308]");
    assert!(parse_tableau(&weights, "Heun").is_err());
    let dense = RESOURCE.replace(
        "\"c\": [0, 1]",
        "\"c\": [0, 1], \"dense\": [[1e308, 1e308], [\"1/2\"]]",
    );
    assert!(parse_tableau(&dense, "Heun").is_err());
}

#[test]
fn expression_parser_supports_exact_style_coefficients() {
    assert_eq!(
        parse_numeric_expression("(3 - sqrt(3)) / 6").unwrap(),
        (3.0 - 3.0_f64.sqrt()) / 6.0
    );
    assert_eq!(parse_numeric_expression("1_000 / 4").unwrap(), 250.0);
    assert_eq!(parse_numeric_expression("-3.25e-7").unwrap(), -3.25e-7);
}

#[test]
fn expression_parser_exposes_only_the_tableau_math_context() {
    assert!(parse_numeric_expression("pi").is_err());
    assert!(parse_numeric_expression("sin(1)").is_err());
    assert!(parse_numeric_expression("coefficient + 1").is_err());
}

#[test]
fn tableau_failures_expose_stable_categories_and_parser_sources() {
    let json = parse_tableau("{", "Heun").unwrap_err();
    assert_eq!(json.kind(), TableauErrorKind::JsonSyntax);
    assert!(json.source().is_some());

    let name = parse_tableau(RESOURCE, "Other").unwrap_err();
    assert_eq!(name.kind(), TableauErrorKind::NameMismatch);
    assert!(name.source().is_none());

    let expression = parse_numeric_expression("1+").unwrap_err();
    assert_eq!(expression.kind(), TableauErrorKind::NumericExpression);
    assert!(expression.source().is_some());

    let non_finite = parse_numeric_expression("1e999").unwrap_err();
    assert_eq!(non_finite.kind(), TableauErrorKind::NonFiniteCoefficient);

    let invalid = RESOURCE.replace("[\"1/2\", \"1/2\"]", "[1, 1]");
    assert_eq!(
        parse_tableau(&invalid, "Heun").unwrap_err().kind(),
        TableauErrorKind::Validation
    );
}

#[test]
fn rejects_structurally_invalid_resources() {
    let nonsquare = RESOURCE.replace("[1, 0]", "[1]");
    assert!(parse_tableau(&nonsquare, "Heun").is_err());

    let nonexplicit = RESOURCE.replace("[1, 0]", "[1, 1]");
    assert!(parse_tableau(&nonexplicit, "Heun").is_err());

    let bad_weights = RESOURCE.replace("[\"1/2\", \"1/2\"]", "[1, 1]");
    assert!(parse_tableau(&bad_weights, "Heun").is_err());

    let unknown = RESOURCE.replace("\"order\": 2,", "\"order\": 2, \"mystery\": 1,");
    assert!(parse_tableau(&unknown, "Heun").is_err());
}

#[test]
fn rejects_invalid_expressions_before_runtime_use() {
    let division_by_zero = RESOURCE.replace("\"1/2\"", "\"1/0\"");
    assert!(parse_tableau(&division_by_zero, "Heun").is_err());
}

#[test]
fn schema_reference_is_ignored() {
    let source = RESOURCE.replace("{\n", "{\n      \"$schema\": \"../schema.json\",\n");
    assert!(parse_tableau(&source, "Heun").is_ok());
}

#[test]
fn schema_reference_is_typed_without_weakening_unknown_field_checks() {
    let invalid_schema = RESOURCE.replace("{\n", "{\n      \"$schema\": {\"unexpected\": true},\n");
    let error = parse_tableau(&invalid_schema, "Heun").unwrap_err();
    assert!(error.to_string().contains("invalid type"));

    let unknown = RESOURCE.replace("\"order\": 2,", "\"order\": 2, \"typo\": 1,");
    let error = parse_tableau(&unknown, "Heun").unwrap_err();
    assert!(error.to_string().contains("unknown field `typo`"));
}

#[test]
fn parses_and_validates_runtime_fitted_weights() {
    let fitted = RESOURCE.replace(
        "\"c\": [0, 1]",
        "\"c\": [0, 1], \"fitted_weights\": [{\"stage\": 0, \
             \"numerator\": [\"1/2\", 1], \"denominator\": [1, \"1/2\"]}]",
    );
    let tableau = parse_tableau(&fitted, "Heun").unwrap();
    let weight = tableau.fitted_weight(0).unwrap();
    assert_eq!(weight.stage(), 0);
    assert_eq!(weight.evaluate(2.0), Some(1.25));

    let wrong_zero_fit = fitted.replace("[\"1/2\", 1]", "[\"1/3\", 1]");
    assert!(parse_tableau(&wrong_zero_fit, "Heun").is_err());

    let duplicate = fitted.replace(
        "]}",
        "]}, {\"stage\": 0, \"numerator\": [\"1/2\"], \
             \"denominator\": [1]}]",
    );
    assert!(parse_tableau(&duplicate, "Heun").is_err());
}
