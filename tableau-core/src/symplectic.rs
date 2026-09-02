use serde::Deserialize;

use super::{Scalar, TableauError, approximately_equal, materialize_vector};

/// A validated alternating drift/kick composition.
///
/// Each stage drifts position by `b[i]`, then kicks velocity by `a[i]`.
/// Negative coefficients are valid and necessary for many higher-order methods.
#[derive(Clone, Debug, PartialEq)]
pub struct SymplecticTableau {
    name: String,
    description: String,
    order: usize,
    a: Vec<f64>,
    b: Vec<f64>,
}

impl SymplecticTableau {
    /// Resource method name.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Human-readable method description.
    pub fn description(&self) -> &str {
        &self.description
    }
    /// Declared classical order; validation does not prove higher-order conditions.
    pub fn order(&self) -> usize {
        self.order
    }
    /// Velocity (kick) coefficients in stage order.
    pub fn a(&self) -> &[f64] {
        &self.a
    }
    /// Position (drift) coefficients in stage order.
    pub fn b(&self) -> &[f64] {
        &self.b
    }
    /// Number of alternating drift/kick stages.
    pub fn stages(&self) -> usize {
        self.a.len()
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "kebab-case")]
enum Kind {
    SymplecticComposition,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSymplecticTableau {
    #[serde(rename = "$schema", default)]
    _schema: Option<String>,
    name: String,
    description: String,
    #[serde(rename = "kind")]
    _kind: Kind,
    order: usize,
    a: Vec<Scalar>,
    b: Vec<Scalar>,
}

/// Parses a JSON drift/kick tableau using the shared coefficient expression parser.
///
/// Validates the requested name, positive order, nonempty description, equal
/// nonzero stage counts, finite coefficients, and first-order consistency of
/// both coefficient sums. Used identically at compile time and first use.
pub fn parse_symplectic_tableau(
    source: &str,
    requested_name: &str,
) -> Result<SymplecticTableau, TableauError> {
    let raw: RawSymplecticTableau = serde_json::from_str(source)
        .map_err(|error| TableauError::new(format!("invalid symplectic tableau JSON: {error}")))?;
    if raw.name.trim().is_empty() || raw.name != requested_name {
        return Err(TableauError::new(format!(
            "resource method `{}` does not match requested method `{requested_name}`",
            raw.name
        )));
    }
    if raw.description.trim().is_empty() || raw.order == 0 {
        return Err(TableauError::new(
            "symplectic tableau requires a description and positive order",
        ));
    }
    let a = materialize_vector(&raw.a, "a")?;
    let b = materialize_vector(&raw.b, "b")?;
    if a.is_empty() || a.len() != b.len() {
        return Err(TableauError::new(
            "a and b must have the same non-zero stage count",
        ));
    }
    for (label, coefficients) in [("a", &a), ("b", &b)] {
        let sum = coefficients.iter().sum::<f64>();
        if !sum.is_finite() || !approximately_equal(sum, 1.0) {
            return Err(TableauError::new(format!(
                "symplectic coefficients {label} must sum to one; found {sum}"
            )));
        }
    }
    Ok(SymplecticTableau {
        name: raw.name,
        description: raw.description,
        order: raw.order,
        a,
        b,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = r#"{"name":"Test","description":"Drift/kick test","kind":"symplectic-composition","order":2,"a":[1,0],"b":["1/2","1/2"]}"#;

    #[test]
    fn parses_shared_numeric_expressions_and_metadata() {
        let tableau = parse_symplectic_tableau(SOURCE, "Test").unwrap();
        assert_eq!(tableau.name(), "Test");
        assert_eq!(tableau.order(), 2);
        assert_eq!(tableau.stages(), 2);
        assert_eq!(tableau.a(), &[1.0, 0.0]);
        assert_eq!(tableau.b(), &[0.5, 0.5]);
        let negative = SOURCE.replace("[1,0]", "[\"sqrt(4)\",-1]");
        assert_eq!(
            parse_symplectic_tableau(&negative, "Test").unwrap().a(),
            &[2.0, -1.0]
        );
    }

    #[test]
    fn rejects_invalid_metadata_shapes_and_expressions() {
        assert!(parse_symplectic_tableau(SOURCE, "Other").is_err());
        for invalid in [
            SOURCE.replace("\"Test\"", "\"\""),
            SOURCE.replace("Drift/kick test", " "),
            SOURCE.replace("\"order\":2", "\"order\":0"),
            SOURCE.replace("symplectic-composition", "explicit-runge-kutta"),
            SOURCE.replace("[1,0]", "[]"),
            SOURCE.replace("[1,0]", "[1]"),
            SOURCE.replace("[1,0]", "[2,0]"),
            SOURCE.replace("[1,0]", "[\"1/0\",0]"),
            SOURCE.replace("[1,0]", "[\"sqrt(-1)\",0]"),
            SOURCE.replace("[1,0]", "[true,0]"),
            SOURCE.replace("[1,0]", "[1e308,1e308]"),
            SOURCE.replace("\"order\":2", "\"order\":2,\"extra\":0"),
            SOURCE.replace("\"order\":2", "\"order\":2,\"schema_version\":1"),
            SOURCE.replace("\"order\":2", "\"order\":2,\"$schema\":42"),
        ] {
            assert!(
                parse_symplectic_tableau(&invalid, "Test").is_err(),
                "accepted {invalid}"
            );
        }
    }
}
