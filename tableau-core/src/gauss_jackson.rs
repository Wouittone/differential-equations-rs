use serde::Deserialize;

use crate::{Scalar, TableauError};

/// Coefficients for one position or velocity component across the method phases.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussJacksonPhaseCoefficients {
    center: [f64; 9],
    corrector: [f64; 9],
    predictor: [f64; 9],
}

impl GaussJacksonPhaseCoefficients {
    /// Centered weights.
    pub fn center(&self) -> &[f64; 9] {
        &self.center
    }
    /// Corrector weights.
    pub fn corrector(&self) -> &[f64; 9] {
        &self.corrector
    }
    /// Predictor weights.
    pub fn predictor(&self) -> &[f64; 9] {
        &self.predictor
    }
}

/// Coefficients for the fixed-step Gauss-Jackson second-order method.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussJacksonTableau {
    name: String,
    description: String,
    position: GaussJacksonPhaseCoefficients,
    velocity: GaussJacksonPhaseCoefficients,
}

impl GaussJacksonTableau {
    /// Resource method name.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Human-readable method description.
    pub fn description(&self) -> &str {
        &self.description
    }
    /// Position weights across the centered, corrector, and predictor phases.
    pub fn position(&self) -> &GaussJacksonPhaseCoefficients {
        &self.position
    }
    /// Velocity weights across the centered, corrector, and predictor phases.
    pub fn velocity(&self) -> &GaussJacksonPhaseCoefficients {
        &self.velocity
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTableau {
    #[serde(rename = "$schema", default)]
    _schema: Option<String>,
    name: String,
    description: String,
    kind: String,
    position: RawGaussJacksonPhaseCoefficients,
    velocity: RawGaussJacksonPhaseCoefficients,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawGaussJacksonPhaseCoefficients {
    center: Vec<Scalar>,
    corrector: Vec<Scalar>,
    predictor: Vec<Scalar>,
}

fn materialize_array(values: Vec<Scalar>, field: &str) -> Result<[f64; 9], TableauError> {
    if values.len() != 9 {
        return Err(TableauError::new(format!(
            "{field} must contain exactly nine coefficients"
        )));
    }
    let mut result = [0.0; 9];
    for (index, value) in values.into_iter().enumerate() {
        result[index] = value.materialize()?;
    }
    Ok(result)
}

fn materialize_coefficients(
    values: RawGaussJacksonPhaseCoefficients,
    component: &str,
) -> Result<GaussJacksonPhaseCoefficients, TableauError> {
    Ok(GaussJacksonPhaseCoefficients {
        center: materialize_array(values.center, &format!("{component}.center"))?,
        corrector: materialize_array(values.corrector, &format!("{component}.corrector"))?,
        predictor: materialize_array(values.predictor, &format!("{component}.predictor"))?,
    })
}

/// Parses and validates a Gauss-Jackson tableau resource.
pub fn parse_gauss_jackson_tableau(
    source: &str,
    requested_name: &str,
) -> Result<GaussJacksonTableau, TableauError> {
    let raw: RawTableau = serde_json::from_str(source)
        .map_err(|error| TableauError::json("Gauss-Jackson tableau", error))?;
    if raw.name != requested_name {
        return Err(TableauError::name_mismatch(format!(
            "resource method `{}` does not match requested method `{requested_name}`",
            raw.name
        )));
    }
    if raw.description.trim().is_empty() || raw.kind != "gauss-jackson" {
        return Err(TableauError::new(
            "Gauss-Jackson tableau requires a description and kind `gauss-jackson`",
        ));
    }
    Ok(GaussJacksonTableau {
        name: raw.name,
        description: raw.description,
        position: materialize_coefficients(raw.position, "position")?,
        velocity: materialize_coefficients(raw.velocity, "velocity")?,
    })
}
