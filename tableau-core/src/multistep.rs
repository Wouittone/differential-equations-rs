use serde::Deserialize;

use super::{Scalar, TableauError, approximately_equal, materialize_vector};

/// A validated constant-step linear multistep formula.
///
/// Entries run from newest to oldest:
/// `sum(alpha[j] * y[n+1-j]) = h * sum(beta[j] * f[n+1-j])`.
/// Validation checks polynomial order conditions, not zero-stability or the
/// stability region. Solvers must also support the formula's particular form.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearMultistepTableau {
    name: String,
    description: String,
    order: usize,
    alpha: Vec<f64>,
    beta: Vec<f64>,
    ndf_kappa: Option<f64>,
}

impl LinearMultistepTableau {
    /// Resource method name.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Human-readable formula description.
    pub fn description(&self) -> &str {
        &self.description
    }
    /// Declared order, checked against the polynomial order conditions.
    pub fn order(&self) -> usize {
        self.order
    }
    /// Number of prior solution values represented by this formula.
    pub fn steps(&self) -> usize {
        self.alpha.len() - 1
    }
    /// Solution weights, with the new endpoint first.
    pub fn alpha(&self) -> &[f64] {
        &self.alpha
    }
    /// Derivative weights, with the new endpoint first.
    pub fn beta(&self) -> &[f64] {
        &self.beta
    }
    /// Whether the formula omits the new endpoint derivative.
    pub fn is_explicit(&self) -> bool {
        self.beta[0] == 0.0
    }

    /// Optional NDF modifier accompanying a canonical BDF base formula.
    ///
    /// `alpha()` and `beta()` still describe the unmodified BDF formula.
    /// An NDF driver subtracts `kappa * alpha[0] * difference^(order+1)(y)`
    /// from its left-hand side. BDF drivers ignore this modifier.
    pub fn ndf_kappa(&self) -> Option<f64> {
        self.ndf_kappa
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "kebab-case")]
enum Kind {
    LinearMultistep,
    BackwardDifferentiation,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTableau {
    #[serde(rename = "$schema", default)]
    _schema: Option<String>,
    name: String,
    description: String,
    kind: Kind,
    order: usize,
    alpha: Vec<Scalar>,
    beta: Vec<Scalar>,
    #[serde(default, deserialize_with = "optional_scalar")]
    ndf_kappa: Option<Scalar>,
}

fn optional_scalar<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Scalar>, D::Error> {
    Scalar::deserialize(deserializer).map(Some)
}

/// Parses a canonical JSON linear multistep formula.
///
/// Checks metadata, finite equally sized arrays, a nonzero leading solution
/// coefficient, and polynomial order conditions through the declared order.
/// Backward-differentiation resources also validate the normalized BDF
/// structure and their accompanying NDF modifier. The base arrays remain BDF.
/// The same parser runs during macro expansion and lazy initialization.
pub fn parse_multistep_tableau(
    source: &str,
    requested_name: &str,
) -> Result<LinearMultistepTableau, TableauError> {
    let raw: RawTableau = serde_json::from_str(source)
        .map_err(|error| TableauError::new(format!("invalid multistep tableau JSON: {error}")))?;
    if raw.name.trim().is_empty() || raw.name != requested_name {
        return Err(TableauError::new(format!(
            "resource method `{}` does not match requested method `{requested_name}`",
            raw.name
        )));
    }
    if raw.description.trim().is_empty() || raw.order == 0 {
        return Err(TableauError::new(
            "multistep tableau requires a description and positive order",
        ));
    }
    let alpha = materialize_vector(&raw.alpha, "alpha")?;
    let beta = materialize_vector(&raw.beta, "beta")?;
    if alpha.len() < 2 || beta.len() != alpha.len() {
        return Err(TableauError::new(
            "alpha and beta must have equal lengths of at least two",
        ));
    }
    if alpha[0] == 0.0 {
        return Err(TableauError::new("alpha[0] must be nonzero"));
    }
    if raw.order > 2 * (alpha.len() - 1) {
        return Err(TableauError::new(
            "multistep order cannot exceed twice the step count",
        ));
    }
    validate_order(&alpha, &beta, raw.order)?;
    let ndf_kappa = match (raw.kind, raw.ndf_kappa) {
        (Kind::LinearMultistep, None) => None,
        (Kind::BackwardDifferentiation, Some(kappa)) => {
            // The order conditions uniquely specify the normalized BDF formula
            // when it has exactly `order` steps and only the new derivative.
            if alpha.len() != raw.order + 1
                || alpha[0] <= 0.0
                || beta[0] != 1.0
                || beta[1..].iter().any(|value| *value != 0.0)
            {
                return Err(TableauError::new("invalid canonical BDF structure"));
            }
            let kappa = kappa.materialize()?;
            let leading = (1.0 - kappa) * alpha[0];
            let error = kappa * alpha[0] + 1.0 / (raw.order + 1) as f64;
            if !leading.is_finite() || leading <= 0.0 || !error.is_finite() || error <= 0.0 {
                return Err(TableauError::new(
                    "NDF modifier must give positive finite leading and error coefficients",
                ));
            }
            Some(kappa)
        }
        _ => {
            return Err(TableauError::new(
                "ndf_kappa is required only for backward-differentiation resources",
            ));
        }
    };
    Ok(LinearMultistepTableau {
        name: raw.name,
        description: raw.description,
        order: raw.order,
        alpha,
        beta,
        ndf_kappa,
    })
}

fn validate_order(alpha: &[f64], beta: &[f64], order: usize) -> Result<(), TableauError> {
    // Normalize only the validation arithmetic, preserving resource bits in
    // the returned tableau. Small scalar multiples must not evade validation.
    let normalize = |values: &[f64]| {
        values
            .iter()
            .map(|value| value / alpha[0])
            .collect::<Vec<_>>()
    };
    let beta = normalize(beta);
    let alpha = normalize(alpha);
    if alpha.iter().chain(&beta).any(|value| !value.is_finite()) {
        return Err(TableauError::new(
            "normalizing the multistep formula overflowed",
        ));
    }
    let mut powers = vec![1.0; alpha.len()];
    let mut previous_powers = vec![0.0; alpha.len()];
    for degree in 0..=order {
        let left: f64 = alpha.iter().zip(&powers).map(|(a, p)| a * p).sum();
        let right = degree as f64
            * beta
                .iter()
                .zip(&previous_powers)
                .map(|(b, p)| b * p)
                .sum::<f64>();
        if !approximately_equal(left, right) {
            return Err(TableauError::new(format!(
                "multistep order condition {degree} failed: {left} != {right}"
            )));
        }
        previous_powers.copy_from_slice(&powers);
        for (lag, power) in powers.iter_mut().enumerate() {
            *power *= -(lag as f64);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    const SOURCE: &str = r#"{"name":"AB2","description":"Second-order Adams-Bashforth","kind":"linear-multistep","order":2,"alpha":[1,-1,0],"beta":[0,"3/2","-1/2"]}"#;

    #[test]
    fn canonical_multistep_formulas_preserve_coefficients_and_metadata() {
        let tableau = parse_multistep_tableau(SOURCE, "AB2").unwrap();
        assert_eq!(tableau.name(), "AB2");
        assert_eq!(tableau.order(), 2);
        assert_eq!(tableau.steps(), 2);
        assert_eq!(tableau.alpha(), &[1.0, -1.0, 0.0]);
        assert_eq!(tableau.beta(), &[0.0, 1.5, -0.5]);
        assert!(tableau.is_explicit());
        let implicit = SOURCE
            .replace("[1,-1,0]", "[1,-1]")
            .replace("[0,\"3/2\",\"-1/2\"]", "[\"1/2\",\"1/2\"]");
        assert!(
            !parse_multistep_tableau(&implicit, "AB2")
                .unwrap()
                .is_explicit()
        );
        let scaled = SOURCE
            .replace("[1,-1,0]", "[2,-2,0]")
            .replace("[0,\"3/2\",\"-1/2\"]", "[0,3,-1]");
        assert_eq!(
            parse_multistep_tableau(&scaled, "AB2").unwrap().alpha()[0],
            2.0
        );
    }

    #[test]
    fn malformed_and_inconsistent_formulas_fail_before_use() {
        assert!(parse_multistep_tableau(SOURCE, "Other").is_err());
        for invalid in [
            SOURCE.replace("\"AB2\"", "\"\""),
            SOURCE.replace("Second-order Adams-Bashforth", " "),
            SOURCE.replace("linear-multistep", "explicit-runge-kutta"),
            SOURCE.replace("\"order\":2", "\"order\":0"),
            SOURCE.replace("\"order\":2", "\"order\":5"),
            SOURCE.replace("\"order\":2", "\"order\":3"),
            SOURCE.replace("[1,-1,0]", "[1]"),
            SOURCE.replace("[1,-1,0]", "[0,-1,0]"),
            SOURCE.replace("[1,-1,0]", "[1,1,0]"),
            SOURCE.replace("\"3/2\"", "\"sqrt(-1)\""),
            SOURCE.replace("\"3/2\"", "true"),
            SOURCE.replace("\"3/2\"", "2"),
            SOURCE.replace("[1,-1,0]", "[1e-300,-1e-300,0]"),
            SOURCE.replace("[1,-1,0]", "[1e-300,-1e300,0]"),
            SOURCE.replace("\"order\":2", "\"order\":2,\"schema_version\":1"),
            SOURCE.replace("\"order\":2", "\"order\":2,\"$schema\":42"),
            SOURCE.replace("\"order\":2", "\"order\":2,\"ndf_kappa\":null"),
            SOURCE.replace("\"order\":2", "\"order\":2,\"ndf_kappa\":0"),
        ] {
            assert!(
                parse_multistep_tableau(&invalid, "AB2").is_err(),
                "accepted {invalid}"
            );
        }
    }

    #[test]
    fn backward_differentiation_reuses_order_validation_and_checks_ndf_modifier() {
        let source = r#"{"name":"BDF2","description":"BDF with NDF modifier","kind":"backward-differentiation","order":2,"alpha":["3/2",-2,"1/2"],"beta":[1,0,0],"ndf_kappa":"-1/9"}"#;
        let tableau = parse_multistep_tableau(source, "BDF2").unwrap();
        assert_eq!(tableau.alpha(), &[1.5, -2.0, 0.5]);
        assert_eq!(tableau.ndf_kappa(), Some(-1.0 / 9.0));
        for invalid in [
            source.replace("\"-1/9\"", "1"),
            source.replace("\"-1/9\"", "-1"),
            source.replace("\"-1/9\"", "\"1/0\""),
            source.replace("\"-1/9\"", "null"),
            source.replace("backward-differentiation", "linear-multistep"),
            source.replace("[\"3/2\",-2,\"1/2\"]", "[\"3/2\",-3,\"1/2\"]"),
            source
                .replace("[\"3/2\",-2,\"1/2\"]", "[3,-4,1]")
                .replace("[1,0,0]", "[2,0,0]"),
            source
                .replace("[\"3/2\",-2,\"1/2\"]", "[\"3/2\",-2,\"1/2\",0]")
                .replace("[1,0,0]", "[1,0,0,0]"),
        ] {
            assert!(
                parse_multistep_tableau(&invalid, "BDF2").is_err(),
                "accepted {invalid}"
            );
        }
    }
}
