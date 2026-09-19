use serde::Deserialize;

use crate::{Scalar, TableauError, materialize_matrix, materialize_vector};

use super::validation::{validate_order_two, validate_rock4_orders, validate_serk2_order};
use super::{Rock2Tableau, Rock4Tableau, RockRecurrence, RockRecurrenceStage, Serk2Tableau};

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

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRock4Tableau {
    /// Editor-only JSON Schema hint, deliberately discarded after parsing.
    #[serde(rename = "$schema", default)]
    _schema: Option<String>,
    name: String,
    description: String,
    #[serde(rename = "kind")]
    _kind: RawRock4Kind,
    order: usize,
    embedded_order: usize,
    degree: usize,
    recurrence: RawRockRecurrence,
    finishing: RawRock4Finish,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum RawRock4Kind {
    Rock4,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRock4Finish {
    #[serde(rename = "A")]
    a: Vec<Vec<Scalar>>,
    b: Vec<Scalar>,
    b_hat: Vec<Scalar>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSerk2Tableau {
    /// Editor-only JSON Schema hint, deliberately discarded after parsing.
    #[serde(rename = "$schema", default)]
    _schema: Option<String>,
    name: String,
    description: String,
    #[serde(rename = "kind")]
    _kind: RawSerk2Kind,
    order: usize,
    degree: usize,
    alpha: Scalar,
    subdivisions: usize,
    weights: Vec<Scalar>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum RawSerk2Kind {
    Serk2,
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
    let raw: RawRock2Tableau =
        serde_json::from_str(source).map_err(|error| TableauError::json("ROCK2 tableau", error))?;

    if raw.name != requested_name {
        return Err(TableauError::name_mismatch(format!(
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

    let recurrence = materialize_rock_recurrence(raw.recurrence, raw.degree, "ROCK2")?;
    let finish_first = raw
        .finishing
        .first
        .materialize()
        .map_err(|error| error.with_context("finishing.first"))?;
    let finish_second = raw
        .finishing
        .second
        .materialize()
        .map_err(|error| error.with_context("finishing.second"))?;

    validate_order_two(
        recurrence.first_stage,
        &recurrence.stages,
        finish_first,
        finish_second,
    )?;

    Ok(Rock2Tableau {
        name: raw.name,
        description: raw.description,
        order: raw.order,
        degree: raw.degree,
        recurrence,
        finish_first,
        finish_second,
    })
}

fn materialize_rock_recurrence(
    raw: RawRockRecurrence,
    degree: usize,
    family: &str,
) -> Result<RockRecurrence, TableauError> {
    let expected_stages = degree.saturating_sub(1);
    if raw.stages.len() != expected_stages {
        return Err(TableauError::new(format!(
            "{family} degree {degree} requires {expected_stages} recurrence stages; found {}",
            raw.stages.len()
        )));
    }
    let first_stage = raw
        .first
        .materialize()
        .map_err(|error| error.with_context("recurrence.first"))?;
    let stages = raw
        .stages
        .into_iter()
        .enumerate()
        .map(|(index, stage)| {
            Ok(RockRecurrenceStage {
                mu: stage[0].materialize().map_err(|error| {
                    error.with_context(format_args!("recurrence.stages[{index}].mu"))
                })?,
                kappa: stage[1].materialize().map_err(|error| {
                    error.with_context(format_args!("recurrence.stages[{index}].kappa"))
                })?,
            })
        })
        .collect::<Result<Vec<_>, TableauError>>()?;
    Ok(RockRecurrence {
        first_stage,
        stages,
    })
}

/// Parses and validates one degree-specific ROCK4 JSON resource.
///
/// The polynomial recurrence and finishing tableau are reconstructed into one
/// explicit RK method. Its primary fourth-order and embedded third-order
/// conditions are verified before the value can be used.
pub fn parse_rock4_tableau(
    source: &str,
    requested_name: &str,
    requested_degree: usize,
) -> Result<Rock4Tableau, TableauError> {
    let raw: RawRock4Tableau =
        serde_json::from_str(source).map_err(|error| TableauError::json("ROCK4 tableau", error))?;
    if raw.name != requested_name {
        return Err(TableauError::name_mismatch(format!(
            "resource method `{}` does not match requested method `{requested_name}`",
            raw.name
        )));
    }
    if raw.name.trim().is_empty() {
        return Err(TableauError::new("ROCK4 tableau name must not be empty"));
    }
    if raw.degree != requested_degree {
        return Err(TableauError::new(format!(
            "resource degree {} does not match requested degree {requested_degree}",
            raw.degree
        )));
    }
    if raw.description.trim().is_empty() {
        return Err(TableauError::new(
            "ROCK4 tableau description must not be empty",
        ));
    }
    if raw.degree == 0 {
        return Err(TableauError::new("ROCK4 degree must be positive"));
    }
    if raw.order != 4 || raw.embedded_order != 3 {
        return Err(TableauError::new(
            "ROCK4 resources must declare primary order 4 and embedded order 3",
        ));
    }

    let recurrence = materialize_rock_recurrence(raw.recurrence, raw.degree, "ROCK4")?;
    let finishing_a = materialize_matrix(&raw.finishing.a, "finishing.A")?;
    if finishing_a.len() != 4
        || finishing_a
            .iter()
            .enumerate()
            .any(|(stage, row)| row.len() != stage)
    {
        return Err(TableauError::new(
            "ROCK4 finishing A must contain rows of lengths 0, 1, 2, and 3",
        ));
    }
    let b = materialize_vector(&raw.finishing.b, "finishing.b")?;
    if b.len() != 4 {
        return Err(TableauError::new(
            "ROCK4 finishing b must contain four weights",
        ));
    }
    let b_hat = materialize_vector(&raw.finishing.b_hat, "finishing.b_hat")?;
    if b_hat.len() != 5 {
        return Err(TableauError::new(
            "ROCK4 finishing b_hat must contain five weights",
        ));
    }

    validate_rock4_orders(&recurrence, &finishing_a, &b, &b_hat)?;
    Ok(Rock4Tableau {
        name: raw.name,
        description: raw.description,
        order: raw.order,
        embedded_order: raw.embedded_order,
        degree: raw.degree,
        recurrence,
        finishing_a,
        b,
        b_hat,
    })
}

/// Parses and validates one degree-specific SERK2 JSON resource.
///
/// Validation reconstructs the state recurrence and its equivalent autonomous
/// explicit Runge--Kutta weights, including both second-order conditions.
pub fn parse_serk2_tableau(
    source: &str,
    requested_name: &str,
    requested_degree: usize,
) -> Result<Serk2Tableau, TableauError> {
    let raw: RawSerk2Tableau =
        serde_json::from_str(source).map_err(|error| TableauError::json("SERK2 tableau", error))?;
    if raw.name != requested_name {
        return Err(TableauError::name_mismatch(format!(
            "resource method `{}` does not match requested method `{requested_name}`",
            raw.name
        )));
    }
    if raw.name.trim().is_empty() {
        return Err(TableauError::new("SERK2 tableau name must not be empty"));
    }
    if raw.degree != requested_degree {
        return Err(TableauError::new(format!(
            "resource degree {} does not match requested degree {requested_degree}",
            raw.degree
        )));
    }
    if raw.description.trim().is_empty() {
        return Err(TableauError::new(
            "SERK2 tableau description must not be empty",
        ));
    }
    if raw.order != 2 {
        return Err(TableauError::new("SERK2 resources must declare order 2"));
    }
    if raw.degree == 0 || raw.degree > 250 {
        return Err(TableauError::new("SERK2 degree must be between 1 and 250"));
    }
    if raw.subdivisions == 0 || raw.degree % raw.subdivisions != 0 {
        return Err(TableauError::new(
            "SERK2 subdivisions must be positive and divide the degree",
        ));
    }

    let alpha = raw
        .alpha
        .materialize()
        .map_err(|error| error.with_context("alpha"))?;
    if alpha <= 0.0 {
        return Err(TableauError::new("SERK2 alpha must be positive"));
    }
    let weights = materialize_vector(&raw.weights, "weights")?;
    if weights.len() != raw.degree + 1 {
        return Err(TableauError::new(format!(
            "SERK2 degree {} requires {} weights; found {}",
            raw.degree,
            raw.degree + 1,
            weights.len()
        )));
    }
    validate_serk2_order(raw.degree, raw.subdivisions, alpha, &weights)?;

    Ok(Serk2Tableau {
        name: raw.name,
        description: raw.description,
        order: raw.order,
        degree: raw.degree,
        alpha,
        subdivisions: raw.subdivisions,
        weights,
    })
}
