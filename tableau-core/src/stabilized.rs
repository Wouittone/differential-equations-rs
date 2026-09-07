use serde::Deserialize;

use super::{Scalar, TableauError, approximately_equal};

/// One two-term stage in a ROCK polynomial recurrence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RockRecurrenceStage {
    mu: f64,
    kappa: f64,
}

impl RockRecurrenceStage {
    /// Multiplier applied to the current derivative.
    pub fn mu(self) -> f64 {
        self.mu
    }

    /// Multiplier applied to the stage from two recurrence steps ago.
    pub fn kappa(self) -> f64 {
        self.kappa
    }
}

/// Validated recurrence shared by ROCK stabilized methods.
#[derive(Clone, Debug, PartialEq)]
pub struct RockRecurrence {
    first_stage: f64,
    stages: Vec<RockRecurrenceStage>,
}

impl RockRecurrence {
    /// Coefficient of the initial derivative stage.
    pub fn first_stage(&self) -> f64 {
        self.first_stage
    }

    /// Two-term recurrence stages after the initial derivative stage.
    pub fn stages(&self) -> &[RockRecurrenceStage] {
        &self.stages
    }
}

/// One validated, degree-specific ROCK2 tableau.
///
/// A degree-specific value deliberately owns only the recurrence selected for
/// that degree. Built-in solver catalogues can therefore keep every resource
/// behind an independent lazy initializer instead of parsing one combined
/// coefficient bank.
#[derive(Clone, Debug, PartialEq)]
pub struct Rock2Tableau {
    name: String,
    description: String,
    order: usize,
    degree: usize,
    recurrence: RockRecurrence,
    finish_first: f64,
    finish_second: f64,
}

impl Rock2Tableau {
    /// Resource method name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Human-readable method description and coefficient provenance.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Classical order verified from the reconstructed RK weights and nodes.
    pub fn order(&self) -> usize {
        self.order
    }

    /// Polynomial degree represented by this resource.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Validated ROCK recurrence coefficients.
    pub fn recurrence(&self) -> &RockRecurrence {
        &self.recurrence
    }

    /// First finishing-stage derivative coefficient.
    pub fn finish_first(&self) -> f64 {
        self.finish_first
    }

    /// Second finishing-stage derivative coefficient.
    pub fn finish_second(&self) -> f64 {
        self.finish_second
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum RawKind {
    Rock2,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRock2Tableau {
    /// Editor-only JSON Schema hint, deliberately discarded after parsing.
    #[serde(rename = "$schema", default)]
    _schema: Option<String>,
    name: String,
    description: String,
    #[serde(rename = "kind")]
    _kind: RawKind,
    order: usize,
    degree: usize,
    recurrence: RawRockRecurrence,
    finishing: RawRock2Finish,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRockRecurrence {
    first: Scalar,
    stages: Vec<[Scalar; 2]>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRock2Finish {
    first: Scalar,
    second: Scalar,
}

/// Parses and validates one degree-specific ROCK2 JSON resource.
///
/// Both the method name and degree are checked against the caller's expected
/// identity. This prevents a valid resource from being accidentally wired to
/// the wrong lazy catalogue entry.
pub fn parse_rock2_tableau(
    source: &str,
    requested_name: &str,
    requested_degree: usize,
) -> Result<Rock2Tableau, TableauError> {
    let raw: RawRock2Tableau = serde_json::from_str(source)
        .map_err(|error| TableauError::new(format!("invalid ROCK2 tableau JSON: {error}")))?;

    if raw.name != requested_name {
        return Err(TableauError::new(format!(
            "resource method `{}` does not match requested method `{requested_name}`",
            raw.name
        )));
    }
    if raw.name.trim().is_empty() {
        return Err(TableauError::new("ROCK2 tableau name must not be empty"));
    }
    if raw.degree != requested_degree {
        return Err(TableauError::new(format!(
            "resource degree {} does not match requested degree {requested_degree}",
            raw.degree
        )));
    }
    if raw.description.trim().is_empty() {
        return Err(TableauError::new(
            "ROCK2 tableau description must not be empty",
        ));
    }
    if raw.degree == 0 {
        return Err(TableauError::new("ROCK2 degree must be positive"));
    }
    if raw.order != 2 {
        return Err(TableauError::new("ROCK2 resources must declare order 2"));
    }

    let expected_stages = raw.degree - 1;
    if raw.recurrence.stages.len() != expected_stages {
        return Err(TableauError::new(format!(
            "ROCK2 degree {} requires {expected_stages} recurrence stages; found {}",
            raw.degree,
            raw.recurrence.stages.len()
        )));
    }

    let first_stage = raw
        .recurrence
        .first
        .materialize()
        .map_err(|error| TableauError::new(format!("recurrence.first: {error}")))?;
    let stages = raw
        .recurrence
        .stages
        .into_iter()
        .enumerate()
        .map(|(index, stage)| {
            Ok(RockRecurrenceStage {
                mu: stage[0].materialize().map_err(|error| {
                    TableauError::new(format!("recurrence.stages[{index}].mu: {error}"))
                })?,
                kappa: stage[1].materialize().map_err(|error| {
                    TableauError::new(format!("recurrence.stages[{index}].kappa: {error}"))
                })?,
            })
        })
        .collect::<Result<Vec<_>, TableauError>>()?;
    let finish_first = raw
        .finishing
        .first
        .materialize()
        .map_err(|error| TableauError::new(format!("finishing.first: {error}")))?;
    let finish_second = raw
        .finishing
        .second
        .materialize()
        .map_err(|error| TableauError::new(format!("finishing.second: {error}")))?;

    validate_order_two(first_stage, &stages, finish_first, finish_second)?;

    Ok(Rock2Tableau {
        name: raw.name,
        description: raw.description,
        order: raw.order,
        degree: raw.degree,
        recurrence: RockRecurrence {
            first_stage,
            stages,
        },
        finish_first,
        finish_second,
    })
}

fn validate_order_two(
    first_stage: f64,
    stages: &[RockRecurrenceStage],
    finish_first: f64,
    finish_second: f64,
) -> Result<(), TableauError> {
    let degree = stages.len() + 1;
    let derivative_count = degree + 2;
    let mut previous_two = vec![0.0; derivative_count];
    let mut previous_one = vec![0.0; derivative_count];
    let mut next = vec![0.0; derivative_count];
    previous_one[0] = first_stage;

    let mut nodes = vec![0.0; derivative_count];
    nodes[1] = first_stage;
    for (offset, stage) in stages.iter().enumerate() {
        let derivative = offset + 1;
        for index in 0..derivative_count {
            next[index] =
                (1.0 + stage.kappa) * previous_one[index] - stage.kappa * previous_two[index];
        }
        next[derivative] += stage.mu;
        nodes[derivative + 1] = next.iter().sum();
        std::mem::swap(&mut previous_two, &mut previous_one);
        std::mem::swap(&mut previous_one, &mut next);
    }

    nodes[degree + 1] = nodes[degree] + finish_first;
    let mut weights = previous_one;
    weights[degree] += finish_first - finish_second;
    weights[degree + 1] += finish_first + finish_second;
    let weight_sum = weights.iter().sum::<f64>();
    if !approximately_equal(weight_sum, 1.0) {
        return Err(TableauError::new(format!(
            "ROCK2 weights must sum to one; found {weight_sum}"
        )));
    }
    let first_moment = weights
        .iter()
        .zip(nodes)
        .map(|(weight, node)| weight * node)
        .sum::<f64>();
    if !approximately_equal(first_moment, 0.5) {
        return Err(TableauError::new(format!(
            "ROCK2 weights and nodes must satisfy the second-order condition; found {first_moment}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_rock2_tableau;

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
}
