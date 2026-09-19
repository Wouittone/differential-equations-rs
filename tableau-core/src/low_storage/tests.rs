use super::*;

const TWO_N: &str = r#"{"name":"Williamson2","description":"two-stage 2N method","kind":"low-storage-runge-kutta","layout":"two-n","order":2,"A":["-1/2"],"b":["1/2",1],"c":["1/2"]}"#;
const EULER_2N: &str = r#"{"name":"Euler2N","description":"one-stage 2N method","kind":"low-storage-runge-kutta","layout":"two-n","order":1,"A":[],"b":[1],"c":[]}"#;
const TWO_C: &str = r#"{"name":"Midpoint2C","description":"two-stage 2C method","kind":"low-storage-runge-kutta","layout":"two-c","order":2,"A":["1/2"],"b":[0,1],"c":["1/2"]}"#;
const THREE_S: &str = r#"{"name":"Euler3S","description":"one-stage 3S recurrence","kind":"low-storage-runge-kutta","layout":"three-s","order":1,"gamma1":[0],"gamma2":[0],"gamma3":[1],"delta":[0],"beta1":0,"beta2":[1],"c":[0],"endpoint_evaluation":"omit"}"#;
const ALTERNATING: &str = r#"{"name":"Alternating","description":"alternating midpoint recurrences","kind":"low-storage-runge-kutta","layout":"alternating-two-n","order":2,"A1":["-1/2"],"b1":["1/2",1],"c1":["1/2"],"A2":["-1/2"],"b2":["1/2",1],"c2":["1/2"]}"#;
const PIPELINE: &str = r#"{"name":"Pipeline","description":"two-stage register pipeline","kind":"low-storage-runge-kutta","layout":"register-pipeline","order":2,"history_states":1,"A":[["1/2"]],"b":[0],"b_final":1,"c":["1/2"]}"#;

#[test]
fn parses_every_recurrence_layout() {
    let euler = parse_low_storage_tableau(EULER_2N, "Euler2N").unwrap();
    assert_eq!(euler.layout().stages(), 1);

    let two_n = parse_low_storage_tableau(TWO_N, "Williamson2").unwrap();
    assert_eq!(two_n.order(), 2);
    let LowStorageRungeKuttaLayout::TwoN(two_n) = two_n.layout() else {
        panic!("expected 2N layout")
    };
    assert_eq!(two_n.stages(), 2);
    assert_eq!(two_n.a(), [-0.5]);

    let two_c = parse_low_storage_tableau(TWO_C, "Midpoint2C").unwrap();
    assert!(matches!(
        two_c.layout(),
        LowStorageRungeKuttaLayout::TwoC(_)
    ));

    let three_s = parse_low_storage_tableau(THREE_S, "Euler3S").unwrap();
    let LowStorageRungeKuttaLayout::ThreeS(three_s) = three_s.layout() else {
        panic!("expected 3S layout")
    };
    assert_eq!(three_s.stages(), 2);
    assert_eq!(
        three_s.endpoint_evaluation(),
        LowStorageEndpointEvaluation::Omit
    );

    let alternating = parse_low_storage_tableau(ALTERNATING, "Alternating").unwrap();
    assert_eq!(alternating.layout().stages(), 2);
    assert_eq!(alternating.layout().alternate_stages(), Some(2));

    let pipeline = parse_low_storage_tableau(PIPELINE, "Pipeline").unwrap();
    let LowStorageRungeKuttaLayout::RegisterPipeline(pipeline) = pipeline.layout() else {
        panic!("expected register-pipeline layout")
    };
    assert_eq!(pipeline.history_states(), 1);
    assert_eq!(pipeline.b_final(), 1.0);
}

#[test]
fn requires_three_s_endpoint_evaluation_policy() {
    let source = THREE_S.replace(",\"endpoint_evaluation\":\"omit\"", "");
    let error = parse_low_storage_tableau(&source, "Euler3S").unwrap_err();
    assert!(error.to_string().contains("endpoint_evaluation"), "{error}");
}

#[test]
fn independent_node_policy_is_explicit_and_typed() {
    let mismatched = PIPELINE.replace("\"c\":[\"1/2\"]", "\"c\":[\"3/4\"]");
    assert!(parse_low_storage_tableau(&mismatched, "Pipeline").is_err());

    let independent =
        mismatched.replace("\"order\":2", "\"order\":2,\"node_policy\":\"independent\"");
    let tableau = parse_low_storage_tableau(&independent, "Pipeline").unwrap();
    assert_eq!(tableau.node_policy(), LowStorageNodePolicy::Independent);
}

#[test]
fn relaxed_consistency_tolerance_is_explicit_and_bounded() {
    let rounded = TWO_N.replace("\"c\":[\"1/2\"]", "\"c\":[\"0.5000005\"]");
    assert!(parse_low_storage_tableau(&rounded, "Williamson2").is_err());

    let declared = rounded.replace(
        "\"order\":2",
        "\"order\":2,\"consistency_tolerance\":\"1e-6\"",
    );
    let tableau = parse_low_storage_tableau(&declared, "Williamson2").unwrap();
    assert_eq!(tableau.consistency_tolerance(), 1.0e-6);

    for invalid in ["0", "1e-5"] {
        let source = TWO_N.replace(
            "\"order\":2",
            &format!("\"order\":2,\"consistency_tolerance\":\"{invalid}\""),
        );
        assert!(parse_low_storage_tableau(&source, "Williamson2").is_err());
    }
}

#[test]
fn independent_nodes_still_require_finite_reconstructed_stages() {
    let source = r#"{"name":"Overflow","description":"overflowing 3S recurrence","kind":"low-storage-runge-kutta","layout":"three-s","order":1,"node_policy":"independent","gamma1":["1e308",0],"gamma2":["1e308",0],"gamma3":[0,1],"delta":[0,0],"beta1":0,"beta2":[0,1],"c":[0,0],"endpoint_evaluation":"omit"}"#;
    let error = parse_low_storage_tableau(source, "Overflow").unwrap_err();
    assert!(error.to_string().contains("not finite"), "{error}");
}

#[test]
fn rejects_invalid_metadata_and_unknown_fields() {
    for invalid in [
        TWO_N.replace("\"name\":\"Williamson2\"", "\"name\":\"Other\""),
        TWO_N.replace("two-stage 2N method", " "),
        TWO_N.replace("\"order\":2", "\"order\":0"),
        TWO_N.replace("\"order\":2", "\"order\":3"),
        TWO_N.replace("\"order\":2", "\"order\":2,\"typo\":0"),
        TWO_N.replace("low-storage-runge-kutta", "explicit-runge-kutta"),
        TWO_N.replace("\"1/2\"", "\"1/0\""),
    ] {
        assert!(
            parse_low_storage_tableau(&invalid, "Williamson2").is_err(),
            "accepted {invalid}"
        );
    }
}

#[test]
fn rejects_malformed_abc_and_inconsistent_nodes_or_weights() {
    for (source, name) in [
        (TWO_N.replace("\"A\":[\"-1/2\"]", "\"A\":[]"), "Williamson2"),
        (
            TWO_N.replace("\"b\":[\"1/2\",1]", "\"b\":[1]"),
            "Williamson2",
        ),
        (TWO_N.replace("\"c\":[\"1/2\"]", "\"c\":[0]"), "Williamson2"),
        (
            TWO_N.replace("\"b\":[\"1/2\",1]", "\"b\":[0,0]"),
            "Williamson2",
        ),
        (TWO_C.replace("\"A\":[\"1/2\"]", "\"A\":[0]"), "Midpoint2C"),
        (TWO_C.replace("\"b\":[0,1]", "\"b\":[0,2]"), "Midpoint2C"),
    ] {
        assert!(
            parse_low_storage_tableau(&source, name).is_err(),
            "accepted {source}"
        );
    }
}

#[test]
fn rejects_malformed_three_s_recurrences() {
    for invalid in [
        THREE_S.replace("\"gamma2\":[0]", "\"gamma2\":[]"),
        THREE_S.replace("\"c\":[0]", "\"c\":[1]"),
        THREE_S.replace("\"gamma3\":[1]", "\"gamma3\":[0]"),
        THREE_S.replace("\"beta2\":[1]", "\"beta2\":[2]"),
    ] {
        assert!(
            parse_low_storage_tableau(&invalid, "Euler3S").is_err(),
            "accepted {invalid}"
        );
    }
}

#[test]
fn rejects_malformed_alternating_and_pipeline_recurrences() {
    for (source, name) in [
        (
            ALTERNATING.replace("\"c2\":[\"1/2\"]", "\"c2\":[0]"),
            "Alternating",
        ),
        (
            PIPELINE.replace("\"history_states\":1", "\"history_states\":0"),
            "Pipeline",
        ),
        (
            PIPELINE.replace("\"history_states\":1", "\"history_states\":2"),
            "Pipeline",
        ),
        (
            PIPELINE.replace("\"A\":[[\"1/2\"]]", "\"A\":[[]]"),
            "Pipeline",
        ),
        (PIPELINE.replace("\"c\":[\"1/2\"]", "\"c\":[0]"), "Pipeline"),
        (
            PIPELINE.replace("\"b_final\":1", "\"b_final\":2"),
            "Pipeline",
        ),
    ] {
        assert!(
            parse_low_storage_tableau(&source, name).is_err(),
            "accepted {source}"
        );
    }
}

#[test]
fn parses_direct_and_embedded_weight_formulas() {
    let pipeline = PIPELINE.replace(
        "\"c\":[\"1/2\"]",
        "\"c\":[\"1/2\"],\"embedded\":{\"order\":1,\"b_hat\":[1],\"b_hat_final\":0}",
    );
    let tableau = parse_low_storage_tableau(&pipeline, "Pipeline").unwrap();
    let embedded = tableau.embedded().unwrap();
    assert_eq!(embedded.order(), 1);
    assert_eq!(embedded.error(), [-1.0, 1.0]);
    assert_eq!(
        embedded.controller(),
        LowStorageAdaptiveController::StandardPi
    );

    let three_s = THREE_S
            .replace("\"order\":1", "\"order\":2")
            .replace(
                "\"endpoint_evaluation\":\"omit\"",
                "\"endpoint_evaluation\":\"omit\",\"embedded\":{\"order\":1,\"error\":[-1,1],\"controller\":{\"kind\":\"pid\",\"beta\":[\"0.7\",\"-0.2\",0],\"acceptance_safety\":\"0.81\"}}",
            );
    let tableau = parse_low_storage_tableau(&three_s, "Euler3S").unwrap();
    let embedded = tableau.embedded().unwrap();
    let LowStorageAdaptiveController::Pid(controller) = embedded.controller() else {
        panic!("expected a PID controller")
    };
    assert_eq!(controller.beta(), [0.7, -0.2, 0.0]);
    assert_eq!(controller.acceptance_safety(), 0.81);
}

#[test]
fn rejects_malformed_embedded_formulas_and_controllers() {
    let embedded_pipeline = PIPELINE.replace(
        "\"c\":[\"1/2\"]",
        "\"c\":[\"1/2\"],\"embedded\":{\"order\":1,\"b_hat\":[1],\"b_hat_final\":0}",
    );
    let invalid = [
            embedded_pipeline.replace("\"order\":1,\"b_hat\"", "\"order\":2,\"b_hat\""),
            embedded_pipeline.replace(
                "\"b_hat\":[1],\"b_hat_final\":0",
                "\"b_hat\":[1]",
            ),
            embedded_pipeline.replace("\"b_hat\":[1]", "\"b_hat\":[1,0]"),
            embedded_pipeline.replace(
                "\"b_hat\":[1],\"b_hat_final\":0",
                "\"error\":[-1,1],\"b_hat\":[1],\"b_hat_final\":0",
            ),
            embedded_pipeline.replace(
                "\"b_hat\":[1],\"b_hat_final\":0",
                "\"error\":[1,1]",
            ),
            embedded_pipeline.replace(
                "\"b_hat\":[1],\"b_hat_final\":0",
                "\"error\":[-1,1],\"controller\":{\"kind\":\"pid\",\"beta\":[1,0],\"acceptance_safety\":\"0.81\"}",
            ),
            embedded_pipeline.replace(
                "\"b_hat\":[1],\"b_hat_final\":0",
                "\"error\":[-1,1],\"controller\":{\"kind\":\"pid\",\"beta\":[1,0,0],\"acceptance_safety\":2}",
            ),
        ];
    for source in invalid {
        assert!(
            parse_low_storage_tableau(&source, "Pipeline").is_err(),
            "accepted {source}"
        );
    }

    let fixed_layout = TWO_N.replace(
        "\"c\":[\"1/2\"]",
        "\"c\":[\"1/2\"],\"embedded\":{\"order\":1,\"error\":[-1,1]}",
    );
    assert!(parse_low_storage_tableau(&fixed_layout, "Williamson2").is_err());
}
