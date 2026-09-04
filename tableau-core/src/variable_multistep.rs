use serde::Deserialize;

use super::multistep::{validate_order, validate_order_at_nodes};
use super::{Scalar, TableauError, evaluate_polynomial, materialize_vector};

#[derive(Clone, Debug, PartialEq)]
struct RatioCoefficient {
    numerator: Vec<f64>,
    denominator: Vec<f64>,
}

impl RatioCoefficient {
    fn evaluate(&self, ratio: f64) -> Option<f64> {
        if !ratio.is_finite() || ratio <= 0.0 {
            return None;
        }
        let denominator = evaluate_polynomial(&self.denominator, ratio);
        let value = evaluate_polynomial(&self.numerator, ratio) / denominator;
        (denominator != 0.0 && value.is_finite()).then_some(value)
    }
}

/// A validated variable-step two-step linear multistep tableau.
///
/// Each `alpha` and `beta` entry is a rational polynomial in
/// `ratio = current_step / previous_step`. Entries use the canonical ordering
/// from newest to oldest:
/// `sum(alpha[j] * y[n+1-j]) = h * sum(beta[j] * f[n+1-j])`.
#[derive(Clone, Debug, PartialEq)]
pub struct VariableMultistepTableau {
    name: String,
    description: String,
    order: usize,
    alpha: Vec<RatioCoefficient>,
    beta: Vec<RatioCoefficient>,
    startup_order: usize,
    startup_alpha: Vec<f64>,
    startup_beta: Vec<f64>,
    defect_weights: Vec<RatioCoefficient>,
    defect_scale: RatioCoefficient,
}

impl VariableMultistepTableau {
    /// Resource method name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Human-readable formula description and provenance.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Declared convergence order of the variable-step formula.
    pub fn order(&self) -> usize {
        self.order
    }

    /// Number of prior solution values required after startup.
    pub fn steps(&self) -> usize {
        self.alpha.len() - 1
    }

    /// Evaluates one canonical solution coefficient at a positive step ratio.
    pub fn alpha(&self, index: usize, ratio: f64) -> Option<f64> {
        self.alpha.get(index)?.evaluate(ratio)
    }

    /// Evaluates one canonical derivative coefficient at a positive step ratio.
    pub fn beta(&self, index: usize, ratio: f64) -> Option<f64> {
        self.beta.get(index)?.evaluate(ratio)
    }

    /// Order of the fixed formula used before sufficient history exists.
    pub fn startup_order(&self) -> usize {
        self.startup_order
    }

    /// Canonical solution weights for the fixed startup formula.
    pub fn startup_alpha(&self) -> &[f64] {
        &self.startup_alpha
    }

    /// Canonical derivative weights for the fixed startup formula.
    pub fn startup_beta(&self) -> &[f64] {
        &self.startup_beta
    }

    /// Evaluates one derivative-defect weight at a positive step ratio.
    pub fn defect_weight(&self, index: usize, ratio: f64) -> Option<f64> {
        self.defect_weights.get(index)?.evaluate(ratio)
    }

    /// Evaluates the dimensionless local-defect scale at a positive step ratio.
    pub fn defect_scale(&self, ratio: f64) -> Option<f64> {
        self.defect_scale.evaluate(ratio)
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "kebab-case")]
enum Kind {
    VariableLinearMultistep,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTableau {
    #[serde(rename = "$schema", default)]
    _schema: Option<String>,
    name: String,
    description: String,
    #[serde(rename = "kind")]
    _kind: Kind,
    order: usize,
    alpha: Vec<RawRatioCoefficient>,
    beta: Vec<RawRatioCoefficient>,
    startup: RawStartup,
    defect: RawDefect,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawStartup {
    order: usize,
    alpha: Vec<Scalar>,
    beta: Vec<Scalar>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDefect {
    derivative_weights: Vec<RawRatioCoefficient>,
    scale: RawRatioCoefficient,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RawRatioCoefficient {
    Polynomial(Vec<Scalar>),
    Rational(RawRationalCoefficient),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRationalCoefficient {
    numerator: Vec<Scalar>,
    denominator: Vec<Scalar>,
}

impl RawRatioCoefficient {
    fn materialize(&self, label: &str) -> Result<RatioCoefficient, TableauError> {
        let (numerator, denominator) = match self {
            Self::Polynomial(numerator) => (materialize_vector(numerator, label)?, vec![1.0]),
            Self::Rational(raw) => (
                materialize_vector(&raw.numerator, &format!("{label}.numerator"))?,
                materialize_vector(&raw.denominator, &format!("{label}.denominator"))?,
            ),
        };
        if numerator.is_empty() || denominator.is_empty() {
            return Err(TableauError::new(format!(
                "{label} requires nonempty numerator and denominator polynomials"
            )));
        }
        if denominator.iter().all(|coefficient| *coefficient == 0.0) {
            return Err(TableauError::new(format!(
                "{label} denominator polynomial must not be identically zero"
            )));
        }
        Ok(RatioCoefficient {
            numerator,
            denominator,
        })
    }
}

/// Parses and validates a canonical variable-step multistep JSON resource.
///
/// Structural checks cover metadata, two-step coefficient dimensions, a fixed startup
/// formula, and a finite positive defect scale. Polynomial order conditions
/// are checked at representative positive step ratios; every runtime
/// evaluation still rejects nonpositive ratios, poles, and overflow.
pub fn parse_variable_multistep_tableau(
    source: &str,
    requested_name: &str,
) -> Result<VariableMultistepTableau, TableauError> {
    let raw: RawTableau = serde_json::from_str(source).map_err(|error| {
        TableauError::new(format!("invalid variable multistep tableau JSON: {error}"))
    })?;
    if raw.name.trim().is_empty() || raw.name != requested_name {
        return Err(TableauError::new(format!(
            "resource method `{}` does not match requested method `{requested_name}`",
            raw.name
        )));
    }
    if raw.description.trim().is_empty() || raw.order == 0 {
        return Err(TableauError::new(
            "variable multistep tableau requires a description and positive order",
        ));
    }
    let alpha = raw
        .alpha
        .iter()
        .enumerate()
        .map(|(index, value)| value.materialize(&format!("alpha[{index}]")))
        .collect::<Result<Vec<_>, _>>()?;
    let beta = raw
        .beta
        .iter()
        .enumerate()
        .map(|(index, value)| value.materialize(&format!("beta[{index}]")))
        .collect::<Result<Vec<_>, _>>()?;
    if alpha.len() != 3 || beta.len() != 3 || raw.order > 4 {
        return Err(TableauError::new(
            "variable two-step alpha and beta must each contain three entries and have order at most four",
        ));
    }

    let startup_alpha = materialize_vector(&raw.startup.alpha, "startup.alpha")?;
    let startup_beta = materialize_vector(&raw.startup.beta, "startup.beta")?;
    if raw.startup.order == 0
        || raw.startup.order >= raw.order
        || startup_alpha.len() != 2
        || startup_beta.len() != 2
        || startup_alpha[0] == 0.0
    {
        return Err(TableauError::new(
            "startup must be a lower-order formula with equal nonzero-leading arrays",
        ));
    }
    validate_order(&startup_alpha, &startup_beta, raw.startup.order)?;

    let defect_weights = raw
        .defect
        .derivative_weights
        .iter()
        .enumerate()
        .map(|(index, value)| value.materialize(&format!("defect.derivative_weights[{index}]")))
        .collect::<Result<Vec<_>, _>>()?;
    if defect_weights.len() != alpha.len() {
        return Err(TableauError::new(
            "the derivative defect must contain one weight per multistep entry",
        ));
    }
    let defect_scale = raw.defect.scale.materialize("defect.scale")?;
    for ratio in [0.25, 0.5, 1.0, 2.0, 4.0] {
        let evaluated_alpha = alpha
            .iter()
            .map(|coefficient| coefficient.evaluate(ratio))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| TableauError::new("alpha is undefined at a validation ratio"))?;
        let evaluated_beta = beta
            .iter()
            .map(|coefficient| coefficient.evaluate(ratio))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| TableauError::new("beta is undefined at a validation ratio"))?;
        if evaluated_alpha[0] == 0.0 {
            return Err(TableauError::new(
                "the newest-state alpha coefficient must remain nonzero",
            ));
        }
        let nodes = [0.0, -1.0, -(1.0 + 1.0 / ratio)];
        validate_order_at_nodes(&evaluated_alpha, &evaluated_beta, &nodes, raw.order)?;
        let evaluated_defect = defect_weights
            .iter()
            .map(|coefficient| coefficient.evaluate(ratio))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                TableauError::new("derivative defect is undefined at a validation ratio")
            })?;
        if !super::approximately_equal(evaluated_defect.iter().sum(), 0.0) {
            return Err(TableauError::new(
                "derivative-defect weights must cancel a constant derivative",
            ));
        }
        if defect_scale
            .evaluate(ratio)
            .is_none_or(|value| value <= 0.0)
        {
            return Err(TableauError::new(
                "defect_scale must remain finite and positive for positive step ratios",
            ));
        }
    }

    Ok(VariableMultistepTableau {
        name: raw.name,
        description: raw.description,
        order: raw.order,
        alpha,
        beta,
        startup_order: raw.startup.order,
        startup_alpha,
        startup_beta,
        defect_weights,
        defect_scale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = r#"{
      "name":"ABDF2","description":"variable BDF","kind":"variable-linear-multistep","order":2,
      "alpha":[[1],[-1,0,"-1/3"],[0,0,"1/3"]],
      "beta":[["2/3"],["1/3","-1/3"],[0]],
      "startup":{"order":1,"alpha":[1,-1],"beta":[1,0]},
      "defect":{"derivative_weights":[[1],[-1,-1],[0,1]],"scale":{"numerator":[1,1],"denominator":[0,6]}}
    }"#;

    #[test]
    fn evaluates_ratio_dependent_coefficients_without_allocating_results() {
        let tableau = parse_variable_multistep_tableau(SOURCE, "ABDF2").unwrap();
        assert_eq!(tableau.order(), 2);
        assert_eq!(tableau.steps(), 2);
        assert_eq!(tableau.alpha(1, 1.0), Some(-4.0 / 3.0));
        assert_eq!(tableau.alpha(2, 1.0), Some(1.0 / 3.0));
        assert_eq!(tableau.beta(0, 1.0), Some(2.0 / 3.0));
        assert_eq!(tableau.beta(1, 2.0), Some(-1.0 / 3.0));
        assert_eq!(tableau.defect_weight(1, 2.0), Some(-3.0));
        assert_eq!(tableau.defect_scale(1.0), Some(1.0 / 3.0));
        assert_eq!(tableau.startup_alpha(), &[1.0, -1.0]);
        assert_eq!(tableau.startup_beta(), &[1.0, 0.0]);
        assert_eq!(tableau.alpha(0, 0.0), None);
    }

    #[test]
    fn rejects_malformed_or_inconsistent_variable_formulas() {
        assert!(parse_variable_multistep_tableau(SOURCE, "Other").is_err());
        for (from, to) in [
            ("\"order\":2", "\"order\":0"),
            ("[-1,0,\"-1/3\"]", "[-1,0,\"-1/2\"]"),
            ("[\"2/3\"]", "[\"3/4\"]"),
            ("\"order\":1,\"alpha\"", "\"order\":2,\"alpha\""),
            ("\"alpha\":[1,-1]", "\"alpha\":[1,1]"),
            ("\"denominator\":[0,6]", "\"denominator\":[0,0]"),
            ("[0,1]],\"scale\"", "[0,2]],\"scale\""),
            ("\"numerator\":[1,1]", "\"numerator\":[-1,-1]"),
        ] {
            let invalid = SOURCE.replace(from, to);
            assert_ne!(invalid, SOURCE, "test pattern did not match: {from}");
            assert!(
                parse_variable_multistep_tableau(&invalid, "ABDF2").is_err(),
                "accepted {invalid}"
            );
        }
    }
}
