use super::{parse_rock2_tableau, parse_rock4_tableau, parse_serk2_tableau};

const RESOURCE: &str = r#"{
        "name": "ROCK2",
        "description": "Two-degree ROCK2 test recurrence",
        "kind": "rock2",
        "order": 2,
        "degree": 2,
        "recurrence": {
            "first": "0.09326607661089206",
            "stages": [["0.1268473641290642", "0.02103378190528467"]]
        },
        "finishing": { "first": "0.3889624104727243", "second": "0.4219428123056774" }
    }"#;

const ROCK4_RESOURCE: &str = r#"{
        "name":"ROCK4",
        "description":"Degree-one ROCK4 test recurrence",
        "kind":"rock4",
        "order":4,
        "embedded_order":3,
        "degree":1,
        "recurrence":{"first":"0.1762962957651941","stages":[]},
        "finishing":{
            "A":[[],["-0.149352078672699"],["0.629768962985252","-0.35520106157365"],["0.0146745996307541","-0.0558517281602565","0.590312931352706"]],
            "b":["0.934502625489809","-0.426556402801135","-0.428612609028723","0.744370090574855"],
            "b_hat":["1.1350997211054","-0.58433336098972","-0.319172911177732","0.482853558185876","0.109256697110981"]
        }
    }"#;

const SERK2_RESOURCE: &str = r#"{
        "name":"SERK2",
        "description":"Degree-two SERK2 test recurrence",
        "kind":"serk2",
        "order":2,
        "degree":2,
        "alpha":"2.5 / 4",
        "subdivisions":1,
        "weights":["1.32","-0.96","0.64"]
    }"#;

#[test]
fn parses_one_typed_degree() {
    let tableau = parse_rock2_tableau(RESOURCE, "ROCK2", 2).unwrap();
    assert_eq!(tableau.name(), "ROCK2");
    assert_eq!(tableau.order(), 2);
    assert_eq!(tableau.degree(), 2);
    assert_eq!(tableau.recurrence().first_stage(), 0.09326607661089206);
    assert_eq!(tableau.recurrence().stages()[0].mu(), 0.1268473641290642);
    assert_eq!(
        tableau.recurrence().stages()[0].kappa(),
        0.02103378190528467
    );
    assert_eq!(tableau.finish_first(), 0.3889624104727243);
    assert_eq!(tableau.finish_second(), 0.4219428123056774);
}

#[test]
fn degree_one_uses_an_empty_recurrence_tail() {
    let degree_one = r#"{
            "name":"ROCK2",
            "description":"Degree-one ROCK2 fixture",
            "kind":"rock2",
            "order":2,
            "degree":1,
            "recurrence":{"first":"0.1794612899156781","stages":[]},
            "finishing":{
                "first":"0.4102693550421609",
                "second":"0.4495196112243335"
            }
        }"#;
    let tableau = parse_rock2_tableau(degree_one, "ROCK2", 1).unwrap();
    assert!(tableau.recurrence().stages().is_empty());
}

#[test]
fn validates_resource_identity_and_shape() {
    assert!(parse_rock2_tableau(RESOURCE, "ROCK4", 2).is_err());
    assert!(parse_rock2_tableau(RESOURCE, "ROCK2", 3).is_err());

    let wrong_shape = RESOURCE.replace(
        r#""stages": [["0.1268473641290642", "0.02103378190528467"]]"#,
        r#""stages": []"#,
    );
    assert!(parse_rock2_tableau(&wrong_shape, "ROCK2", 2).is_err());

    let wrong_row = RESOURCE.replace(
        r#"["0.1268473641290642", "0.02103378190528467"]"#,
        r#"["0.1268473641290642"]"#,
    );
    assert!(parse_rock2_tableau(&wrong_row, "ROCK2", 2).is_err());

    let unknown = RESOURCE.replace(r#""degree": 2,"#, r#""degree": 2, "unexpected": true,"#);
    assert!(parse_rock2_tableau(&unknown, "ROCK2", 2).is_err());

    let nested_unknown = RESOURCE.replace(
        r#""first": "0.3889624104727243""#,
        r#""first": "0.3889624104727243", "unexpected": 0"#,
    );
    assert!(parse_rock2_tableau(&nested_unknown, "ROCK2", 2).is_err());

    for invalid in [
        RESOURCE.replace(r#""kind": "rock2""#, r#""kind": "rock4""#),
        RESOURCE.replace(r#""order": 2"#, r#""order": 3"#),
        RESOURCE.replace(
            r#""description": "Two-degree ROCK2 test recurrence""#,
            r#""description": " ""#,
        ),
        RESOURCE.replace(r#""name": "ROCK2""#, r#""name": """#),
        RESOURCE.replace(r#""first": "0.09326607661089206""#, r#""first": true"#),
    ] {
        let requested_name = if invalid.contains(r#""name": """#) {
            ""
        } else {
            "ROCK2"
        };
        assert!(parse_rock2_tableau(&invalid, requested_name, 2).is_err());
    }
}

#[test]
fn rejects_non_finite_coefficients() {
    let overflow = RESOURCE.replace("0.09326607661089206", "1e999");
    assert!(parse_rock2_tableau(&overflow, "ROCK2", 2).is_err());
}

#[test]
fn rejects_coefficients_that_break_order_two() {
    let inconsistent = RESOURCE.replace("0.3889624104727243", "0.4");
    assert!(parse_rock2_tableau(&inconsistent, "ROCK2", 2).is_err());
}

#[test]
fn parses_and_validates_a_complete_rock4_formula() {
    let tableau = parse_rock4_tableau(ROCK4_RESOURCE, "ROCK4", 1).unwrap();
    assert_eq!(tableau.name(), "ROCK4");
    assert_eq!(tableau.order(), 4);
    assert_eq!(tableau.embedded_order(), 3);
    assert_eq!(tableau.degree(), 1);
    assert!(tableau.recurrence().stages().is_empty());
    assert_eq!(tableau.finishing_a()[3].len(), 3);
    assert_eq!(tableau.b().len(), 4);
    assert_eq!(tableau.b_hat().len(), 5);
}

#[test]
fn rejects_malformed_or_inconsistent_rock4_finishing_tableaus() {
    for invalid in [
        ROCK4_RESOURCE.replace(r#""degree":1"#, r#""degree":2"#),
        ROCK4_RESOURCE.replace(r#""order":4"#, r#""order":3"#),
        ROCK4_RESOURCE.replace(r#""embedded_order":3"#, r#""embedded_order":2"#),
        ROCK4_RESOURCE.replace(r#"["-0.149352078672699"]"#, r#"["-0.149352078672699",0]"#),
        ROCK4_RESOURCE.replace(
            r#"["0.0146745996307541","-0.0558517281602565","0.590312931352706"]"#,
            r#"["0.0146745996307541","-0.0558517281602565"]"#,
        ),
        ROCK4_RESOURCE.replace(r#","0.744370090574855"]"#, r#"]"#),
        ROCK4_RESOURCE.replace(r#","0.109256697110981"]"#, r#"]"#),
        ROCK4_RESOURCE.replace(
            r#""description":"Degree-one ROCK4 test recurrence","#,
            r#""description":"Degree-one ROCK4 test recurrence","unknown":true,"#,
        ),
        ROCK4_RESOURCE.replace(r#""finishing":{"#, r#""finishing":{"unknown":true,"#),
        ROCK4_RESOURCE.replace("0.934502625489809", "0.9"),
        ROCK4_RESOURCE.replace("0.109256697110981", "1e999"),
    ] {
        assert!(parse_rock4_tableau(&invalid, "ROCK4", 1).is_err());
    }
}

#[test]
fn parses_and_validates_a_serk2_recurrence() {
    let tableau = parse_serk2_tableau(SERK2_RESOURCE, "SERK2", 2).unwrap();
    assert_eq!(tableau.name(), "SERK2");
    assert_eq!(tableau.description(), "Degree-two SERK2 test recurrence");
    assert_eq!(tableau.order(), 2);
    assert_eq!(tableau.degree(), 2);
    assert_eq!(tableau.subdivisions(), 1);
    assert_eq!(tableau.internal_degree(), 2);
    assert_eq!(tableau.weights(), [1.32, -0.96, 0.64]);
}

#[test]
fn rejects_malformed_or_inconsistent_serk2_resources() {
    for (invalid, requested_name, requested_degree) in [
        (
            SERK2_RESOURCE.replace(r#""name":"SERK2""#, r#""name":"OTHER""#),
            "SERK2",
            2,
        ),
        (
            SERK2_RESOURCE.replace(r#""degree":2"#, r#""degree":3"#),
            "SERK2",
            2,
        ),
        (
            SERK2_RESOURCE.replace(r#""kind":"serk2""#, r#""kind":"rock2""#),
            "SERK2",
            2,
        ),
        (
            SERK2_RESOURCE.replace(r#""order":2"#, r#""order":1"#),
            "SERK2",
            2,
        ),
        (
            SERK2_RESOURCE.replace(r#""subdivisions":1"#, r#""subdivisions":0"#),
            "SERK2",
            2,
        ),
        (
            SERK2_RESOURCE.replace(r#""subdivisions":1"#, r#""subdivisions":3"#),
            "SERK2",
            2,
        ),
        (SERK2_RESOURCE.replace(r#","0.64"]"#, r#"]"#), "SERK2", 2),
        (
            SERK2_RESOURCE.replace(
                r#""description":"Degree-two SERK2 test recurrence""#,
                r#""description":" ""#,
            ),
            "SERK2",
            2,
        ),
        (
            SERK2_RESOURCE.replace(r#""weights":["#, r#""unexpected":true,"weights":["#),
            "SERK2",
            2,
        ),
        (SERK2_RESOURCE.replace("1.32", "1e999"), "SERK2", 2),
        (SERK2_RESOURCE.replace("0.64", "0.63"), "SERK2", 2),
    ] {
        assert!(
            parse_serk2_tableau(&invalid, requested_name, requested_degree).is_err(),
            "accepted invalid resource: {invalid}"
        );
    }
}
