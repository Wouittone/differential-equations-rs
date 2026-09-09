use serde::Deserialize;

use crate::{Scalar, TableauError, approximately_equal, materialize_vector};

/// One validated, degree-specific extrapolated stabilized RK tableau.
///
/// The resource owns the complete recurrence and extrapolation policy for one
/// degree. Keeping each degree independent lets solver catalogues materialize
/// only the selected formula at runtime.
#[derive(Clone, Debug, PartialEq)]
pub struct EserkTableau {
    name: String,
    description: String,
    order: usize,
    embedded_order: usize,
    degree: usize,
    internal_degree: usize,
    alpha: f64,
    subdivisions: usize,
    solution_combination: Vec<i32>,
    error_combination: Vec<i32>,
    combination_denominator: f64,
    weights: Vec<f64>,
}

impl EserkTableau {
    /// Resource method name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Human-readable method description and coefficient provenance.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Formal order of the extrapolated solution.
    pub fn order(&self) -> usize {
        self.order
    }

    /// Formal order represented by the embedded error combination.
    pub fn embedded_order(&self) -> usize {
        self.embedded_order
    }

    /// Polynomial degree represented by this resource.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Number of recurrence stages between forward-Euler restarts.
    pub fn internal_degree(&self) -> usize {
        self.internal_degree
    }

    /// Recurrence step coefficient applied to each derivative.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Number of extrapolation subdivisions.
    pub fn subdivisions(&self) -> usize {
        self.subdivisions
    }

    /// Integer combination for the primary extrapolated solution.
    pub fn solution_combination(&self) -> &[i32] {
        &self.solution_combination
    }

    /// Integer combination for the embedded error estimate.
    pub fn error_combination(&self) -> &[i32] {
        &self.error_combination
    }

    /// Shared divisor for both extrapolation combinations.
    pub fn combination_denominator(&self) -> f64 {
        self.combination_denominator
    }

    /// Initial-state and generated-stage weights for the base recurrence.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum RawKind {
    Eserk,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawEserkTableau {
    /// Editor-only JSON Schema hint, deliberately discarded after parsing.
    #[serde(rename = "$schema", default)]
    _schema: Option<String>,
    name: String,
    description: String,
    #[serde(rename = "kind")]
    _kind: RawKind,
    order: usize,
    embedded_order: usize,
    degree: usize,
    internal_degree: usize,
    alpha: Scalar,
    subdivisions: usize,
    solution_combination: Vec<i32>,
    error_combination: Vec<i32>,
    combination_denominator: Scalar,
    weights: Vec<Scalar>,
}

/// Parses and validates one degree-specific ESERK JSON resource.
///
/// Validation covers resource identity, recurrence consistency, and the
/// Richardson extrapolation moments through the declared primary and embedded
/// orders. The same parser is run during procedural-macro expansion and on the
/// selected resource's first runtime access.
pub fn parse_eserk_tableau(
    source: &str,
    requested_name: &str,
    requested_order: usize,
    requested_degree: usize,
) -> Result<EserkTableau, TableauError> {
    let raw: RawEserkTableau = serde_json::from_str(source)
        .map_err(|error| TableauError::new(format!("invalid ESERK tableau JSON: {error}")))?;
    if raw.name != requested_name {
        return Err(TableauError::new(format!(
            "resource method `{}` does not match requested method `{requested_name}`",
            raw.name
        )));
    }
    if raw.name.trim().is_empty() {
        return Err(TableauError::new("ESERK tableau name must not be empty"));
    }
    if raw.description.trim().is_empty() {
        return Err(TableauError::new(
            "ESERK tableau description must not be empty",
        ));
    }
    if raw.order != requested_order {
        return Err(TableauError::new(format!(
            "resource order {} does not match requested order {requested_order}",
            raw.order
        )));
    }
    if raw.degree != requested_degree {
        return Err(TableauError::new(format!(
            "resource degree {} does not match requested degree {requested_degree}",
            raw.degree
        )));
    }
    if !(4..=5).contains(&raw.order) {
        return Err(TableauError::new("ESERK order must be 4 or 5"));
    }
    if raw.embedded_order + 1 != raw.order {
        return Err(TableauError::new(
            "ESERK embedded_order must be one less than order",
        ));
    }
    if raw.degree == 0 {
        return Err(TableauError::new("ESERK degree must be positive"));
    }
    if raw.internal_degree == 0 {
        return Err(TableauError::new("ESERK internal_degree must be positive"));
    }
    if raw.subdivisions != raw.order {
        return Err(TableauError::new(
            "ESERK subdivisions must equal the method order",
        ));
    }
    if raw.solution_combination.len() != raw.subdivisions
        || raw.error_combination.len() != raw.subdivisions
    {
        return Err(TableauError::new(
            "ESERK extrapolation combinations require one entry per subdivision",
        ));
    }

    let alpha = raw
        .alpha
        .materialize()
        .map_err(|error| TableauError::new(format!("alpha: {error}")))?;
    if alpha <= 0.0 {
        return Err(TableauError::new("ESERK alpha must be positive"));
    }
    let combination_denominator = raw
        .combination_denominator
        .materialize()
        .map_err(|error| TableauError::new(format!("combination_denominator: {error}")))?;
    if combination_denominator == 0.0 {
        return Err(TableauError::new(
            "ESERK combination_denominator must be non-zero",
        ));
    }
    let weights = materialize_vector(&raw.weights, "weights")?;
    if weights.len() != raw.degree + 1 {
        return Err(TableauError::new(format!(
            "ESERK degree {} requires {} weights; found {}",
            raw.degree,
            raw.degree + 1,
            weights.len()
        )));
    }

    validate_base_recurrence(raw.internal_degree, alpha, &weights)?;
    validate_extrapolation(
        raw.order,
        &raw.solution_combination,
        &raw.error_combination,
        combination_denominator,
    )?;

    Ok(EserkTableau {
        name: raw.name,
        description: raw.description,
        order: raw.order,
        embedded_order: raw.embedded_order,
        degree: raw.degree,
        internal_degree: raw.internal_degree,
        alpha,
        subdivisions: raw.subdivisions,
        solution_combination: raw.solution_combination,
        error_combination: raw.error_combination,
        combination_denominator,
        weights,
    })
}

fn validate_base_recurrence(
    internal_degree: usize,
    alpha: f64,
    weights: &[f64],
) -> Result<(), TableauError> {
    let weight_sum = weights.iter().sum::<f64>();
    if !resource_close(weight_sum, 1.0) {
        return Err(TableauError::new(format!(
            "ESERK base weights must sum to one; found {weight_sum}"
        )));
    }

    let first_moment = weights
        .iter()
        .skip(1)
        .enumerate()
        .map(|(offset, weight)| {
            let stage = offset + 1;
            let block = (stage - 1) / internal_degree;
            let local_stage = (stage - 1) % internal_degree + 1;
            let node = (block * internal_degree * internal_degree + local_stage * local_stage)
                as f64
                * alpha;
            weight * node
        })
        .sum::<f64>();
    if !resource_close(first_moment, 1.0) {
        return Err(TableauError::new(format!(
            "ESERK base weights and recurrence nodes must satisfy first order; found {first_moment}"
        )));
    }
    Ok(())
}

fn validate_extrapolation(
    order: usize,
    solution: &[i32],
    error: &[i32],
    denominator: f64,
) -> Result<(), TableauError> {
    validate_moment(solution, 0, denominator, "solution")?;
    for power in 1..order {
        validate_moment(solution, power, 0.0, "solution")?;
    }
    for power in 0..order - 1 {
        validate_moment(error, power, 0.0, "error")?;
    }
    let leading_error_moment = inverse_power_moment(error, order - 1);
    if approximately_equal(leading_error_moment, 0.0) {
        return Err(TableauError::new(format!(
            "ESERK error combination must have a nonzero inverse-power moment {}; found {leading_error_moment}",
            order - 1
        )));
    }
    Ok(())
}

fn validate_moment(
    coefficients: &[i32],
    inverse_power: usize,
    expected: f64,
    label: &str,
) -> Result<(), TableauError> {
    let value = inverse_power_moment(coefficients, inverse_power);
    approximately_equal(value, expected)
        .then_some(())
        .ok_or_else(|| {
            TableauError::new(format!(
                "ESERK {label} combination violates inverse-power moment {inverse_power}: found {value}, expected {expected}"
            ))
        })
}

fn inverse_power_moment(coefficients: &[i32], inverse_power: usize) -> f64 {
    coefficients
        .iter()
        .enumerate()
        .map(|(offset, coefficient)| {
            *coefficient as f64 / ((offset + 1) as f64).powi(inverse_power as i32)
        })
        .sum()
}

// The degree-4000 ESERK4 bank has condition numbers large enough that its
// decimal coefficients retain roughly nine absolute digits in these two
// cancellation-heavy identities. This bound remains two orders of magnitude
// tighter than the smallest mutation exercised by the parser tests.
fn resource_close(left: f64, right: f64) -> bool {
    left.is_finite() && right.is_finite() && (left - right).abs() <= 1.0e-8
}

#[cfg(test)]
mod tests {
    use super::parse_eserk_tableau;

    const RESOURCE: &str = r#"{
        "name":"ESERK4",
        "description":"Synthetic degree-one extrapolated stabilized method",
        "kind":"eserk",
        "order":4,
        "embedded_order":3,
        "degree":1,
        "internal_degree":1,
        "alpha":"1",
        "subdivisions":4,
        "solution_combination":[-1,24,-81,64],
        "error_combination":[-1,12,-27,16],
        "combination_denominator":"6",
        "weights":["0","1"]
    }"#;

    #[test]
    fn parses_a_complete_extrapolated_recurrence() {
        let tableau = parse_eserk_tableau(RESOURCE, "ESERK4", 4, 1).unwrap();
        assert_eq!(tableau.order(), 4);
        assert_eq!(tableau.embedded_order(), 3);
        assert_eq!(tableau.degree(), 1);
        assert_eq!(tableau.internal_degree(), 1);
        assert_eq!(tableau.subdivisions(), 4);
        assert_eq!(tableau.solution_combination(), &[-1, 24, -81, 64]);
        assert_eq!(tableau.error_combination(), &[-1, 12, -27, 16]);
        assert_eq!(tableau.combination_denominator(), 6.0);
        assert_eq!(tableau.weights(), &[0.0, 1.0]);
    }

    #[test]
    fn rejects_identity_shape_recurrence_and_extrapolation_errors() {
        let invalid = [
            "{".to_owned(),
            RESOURCE.replace(r#""kind":"eserk""#, r#""kind":"other""#),
            RESOURCE.replace("{", r#"{"unknown":1,"#),
            RESOURCE.replace(r#""name":"ESERK4""#, r#""name":"""#),
            RESOURCE.replace("Synthetic degree-one extrapolated stabilized method", ""),
            RESOURCE.replace("ESERK4", "OTHER"),
            RESOURCE.replace(r#""order":4"#, r#""order":5"#),
            RESOURCE.replace(r#""embedded_order":3"#, r#""embedded_order":2"#),
            RESOURCE.replace(r#""degree":1"#, r#""degree":2"#),
            RESOURCE.replace(r#""degree":1"#, r#""degree":0"#),
            RESOURCE.replace(r#""internal_degree":1"#, r#""internal_degree":0"#),
            RESOURCE.replace(r#""alpha":"1""#, r#""alpha":"0""#),
            RESOURCE.replace(r#""subdivisions":4"#, r#""subdivisions":3"#),
            RESOURCE.replace("[-1,24,-81,64]", "[-1,24,-81,63]"),
            RESOURCE.replace("[-1,24,-81,64]", "[-1,24,-81]"),
            RESOURCE.replace("[-1,12,-27,16]", "[-1,12,-27,15]"),
            RESOURCE.replace("[-1,12,-27,16]", "[-1,12,-27]"),
            RESOURCE.replace("[-1,12,-27,16]", "[0,0,0,0]"),
            RESOURCE.replace(
                r#""combination_denominator":"6""#,
                r#""combination_denominator":"0""#,
            ),
            RESOURCE.replace(
                r#""combination_denominator":"6""#,
                r#""combination_denominator":"sqrt(-1)""#,
            ),
            RESOURCE.replace(r#"["0","1"]"#, r#"["1"]"#),
            RESOURCE.replace(r#"["0","1"]"#, r#"["0","0.999"]"#),
            RESOURCE.replace(r#"["0","1"]"#, r#"["0.1","0.9"]"#),
            RESOURCE.replace(r#"["0","1"]"#, r#"["0","sqrt(-1)"]"#),
        ];
        for source in invalid {
            assert!(
                parse_eserk_tableau(&source, "ESERK4", 4, 1).is_err(),
                "accepted invalid resource: {source}"
            );
        }
    }
}
