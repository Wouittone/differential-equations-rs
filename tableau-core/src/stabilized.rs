use serde::Deserialize;

use super::{Scalar, TableauError, approximately_equal, materialize_matrix, materialize_vector};

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

/// One validated, degree-specific ROCK4 tableau.
///
/// The polynomial recurrence advances to a stable base state. The finishing
/// tableau then supplies the fourth-order solution and a third-order embedded
/// companion used for adaptive error estimation.
#[derive(Clone, Debug, PartialEq)]
pub struct Rock4Tableau {
    name: String,
    description: String,
    order: usize,
    embedded_order: usize,
    degree: usize,
    recurrence: RockRecurrence,
    finishing_a: Vec<Vec<f64>>,
    b: Vec<f64>,
    b_hat: Vec<f64>,
}

impl Rock4Tableau {
    /// Resource method name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Human-readable method description and coefficient provenance.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Classical order verified from the reconstructed full RK method.
    pub fn order(&self) -> usize {
        self.order
    }

    /// Order of the verified embedded companion.
    pub fn embedded_order(&self) -> usize {
        self.embedded_order
    }

    /// Polynomial degree represented by this resource.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Validated ROCK recurrence coefficients.
    pub fn recurrence(&self) -> &RockRecurrence {
        &self.recurrence
    }

    /// Strictly lower-triangular rows of the four-stage finishing tableau.
    pub fn finishing_a(&self) -> &[Vec<f64>] {
        &self.finishing_a
    }

    /// Primary finishing weights.
    pub fn b(&self) -> &[f64] {
        &self.b
    }

    /// Embedded finishing weights, including the endpoint derivative weight.
    pub fn b_hat(&self) -> &[f64] {
        &self.b_hat
    }
}

/// One validated, degree-specific SERK2 tableau.
///
/// The method combines states from a second-order stabilized recurrence. The
/// subdivision count controls the recurrence restarts; the output weights
/// contain the initial-state coefficient followed by one coefficient for
/// every generated stage.
#[derive(Clone, Debug, PartialEq)]
pub struct Serk2Tableau {
    name: String,
    description: String,
    order: usize,
    degree: usize,
    subdivisions: usize,
    weights: Vec<f64>,
}

impl Serk2Tableau {
    /// Resource method name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Human-readable method description and coefficient provenance.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Classical order verified from the reconstructed recurrence.
    pub fn order(&self) -> usize {
        self.order
    }

    /// Total polynomial degree represented by this resource.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Number of equal recurrence subdivisions.
    pub fn subdivisions(&self) -> usize {
        self.subdivisions
    }

    /// Polynomial degree within each recurrence subdivision.
    pub fn internal_degree(&self) -> usize {
        self.degree / self.subdivisions
    }

    /// Initial-state and generated-stage combination weights.
    pub fn weights(&self) -> &[f64] {
        &self.weights
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

    let recurrence = materialize_rock_recurrence(raw.recurrence, raw.degree, "ROCK2")?;
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
        .map_err(|error| TableauError::new(format!("recurrence.first: {error}")))?;
    let stages = raw
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
    let raw: RawRock4Tableau = serde_json::from_str(source)
        .map_err(|error| TableauError::new(format!("invalid ROCK4 tableau JSON: {error}")))?;
    if raw.name != requested_name {
        return Err(TableauError::new(format!(
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
    let raw: RawSerk2Tableau = serde_json::from_str(source)
        .map_err(|error| TableauError::new(format!("invalid SERK2 tableau JSON: {error}")))?;
    if raw.name != requested_name {
        return Err(TableauError::new(format!(
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

    let weights = materialize_vector(&raw.weights, "weights")?;
    if weights.len() != raw.degree + 1 {
        return Err(TableauError::new(format!(
            "SERK2 degree {} requires {} weights; found {}",
            raw.degree,
            raw.degree + 1,
            weights.len()
        )));
    }
    validate_serk2_order(raw.degree, raw.subdivisions, &weights)?;

    Ok(Serk2Tableau {
        name: raw.name,
        description: raw.description,
        order: raw.order,
        degree: raw.degree,
        subdivisions: raw.subdivisions,
        weights,
    })
}

fn validate_serk2_order(
    degree: usize,
    subdivisions: usize,
    weights: &[f64],
) -> Result<(), TableauError> {
    let weight_sum = weights.iter().sum::<f64>();
    if !approximately_equal(weight_sum, 1.0) {
        return Err(TableauError::new(format!(
            "SERK2 output weights must sum to one; found {weight_sum}"
        )));
    }

    let alpha = 2.5 / (degree * degree) as f64;
    let internal_degree = degree / subdivisions;
    let mut stage_rows = Vec::with_capacity(degree + 1);
    stage_rows.push(vec![0.0; degree]);
    let mut previous_one = vec![0.0; degree];
    let mut derivative = 0;
    for _ in 0..subdivisions {
        let mut next = previous_one.clone();
        next[derivative] += alpha;
        derivative += 1;
        stage_rows.push(next.clone());
        let mut previous_two = std::mem::replace(&mut previous_one, next);

        for _ in 2..=internal_degree {
            let mut next = previous_one
                .iter()
                .zip(&previous_two)
                .map(|(previous, previous_two)| 2.0 * previous - previous_two)
                .collect::<Vec<_>>();
            next[derivative] += 2.0 * alpha;
            derivative += 1;
            stage_rows.push(next.clone());
            previous_two = previous_one;
            previous_one = next;
        }
    }
    debug_assert_eq!(derivative, degree);
    debug_assert_eq!(stage_rows.len(), degree + 1);

    let mut equivalent_weights = vec![0.0; degree];
    for (weight, row) in weights.iter().skip(1).zip(stage_rows.iter().skip(1)) {
        for (equivalent, coefficient) in equivalent_weights.iter_mut().zip(row) {
            *equivalent += weight * coefficient;
        }
    }
    validate_explicit_rk_order(&stage_rows[..degree], &equivalent_weights, 2, "SERK2")
}

fn reconstruct_rock_rows(recurrence: &RockRecurrence, stage_count: usize) -> Vec<Vec<f64>> {
    let degree = recurrence.stages.len() + 1;
    let mut a = vec![vec![0.0; stage_count]; stage_count];
    a[1][0] = recurrence.first_stage;
    for (offset, recurrence_stage) in recurrence.stages.iter().enumerate() {
        let stage = offset + 2;
        let (previous_rows, current_rows) = a.split_at_mut(stage);
        let current = &mut current_rows[0];
        for ((value, previous), previous_two) in current
            .iter_mut()
            .zip(&previous_rows[stage - 1])
            .zip(&previous_rows[stage - 2])
        {
            *value =
                (1.0 + recurrence_stage.kappa) * previous - recurrence_stage.kappa * previous_two;
        }
        current[stage - 1] += recurrence_stage.mu;
    }
    debug_assert!(degree < stage_count);
    a
}

fn validate_rock4_orders(
    recurrence: &RockRecurrence,
    finishing_a: &[Vec<f64>],
    b: &[f64],
    b_hat: &[f64],
) -> Result<(), TableauError> {
    let degree = recurrence.stages.len() + 1;
    let stage_count = degree + 5;
    let mut a = reconstruct_rock_rows(recurrence, stage_count);
    let base = a[degree].clone();
    for (finishing_stage, coefficients) in finishing_a.iter().enumerate().take(4).skip(1) {
        let stage = degree + finishing_stage;
        a[stage].copy_from_slice(&base);
        for (offset, coefficient) in coefficients.iter().enumerate() {
            a[stage][degree + offset] += coefficient;
        }
    }

    let mut primary = base.clone();
    for (offset, weight) in b.iter().enumerate() {
        primary[degree + offset] += weight;
    }
    a[degree + 4].copy_from_slice(&primary);

    let mut embedded = base;
    for (offset, weight) in b_hat.iter().enumerate() {
        embedded[degree + offset] += weight;
    }
    validate_explicit_rk_order(&a, &primary, 4, "ROCK4 primary")?;
    validate_explicit_rk_order(&a, &embedded, 3, "ROCK4 embedded")
}

fn validate_explicit_rk_order(
    a: &[Vec<f64>],
    b: &[f64],
    order: usize,
    label: &str,
) -> Result<(), TableauError> {
    let c = a
        .iter()
        .map(|row| row.iter().sum::<f64>())
        .collect::<Vec<_>>();
    let condition =
        |value: f64, expected: f64, name: &str| {
            approximately_equal(value, expected).then_some(()).ok_or_else(|| {
            TableauError::new(format!(
                "{label} violates order condition {name}: found {value}, expected {expected}"
            ))
        })
        };
    condition(b.iter().sum(), 1.0, "b·1")?;
    if order < 2 {
        return Ok(());
    }
    condition(dot(b, &c), 0.5, "b·c")?;
    if order < 3 {
        return Ok(());
    }
    let c_squared = c.iter().map(|value| value * value).collect::<Vec<_>>();
    let a_c = matrix_vector(a, &c);
    condition(dot(b, &c_squared), 1.0 / 3.0, "b·c²")?;
    condition(dot(b, &a_c), 1.0 / 6.0, "b·A·c")?;
    if order < 4 {
        return Ok(());
    }
    let c_cubed = c_squared
        .iter()
        .zip(&c)
        .map(|(squared, value)| squared * value)
        .collect::<Vec<_>>();
    let c_a_c = c.iter().zip(&a_c).map(|(c, ac)| c * ac).collect::<Vec<_>>();
    let a_c_squared = matrix_vector(a, &c_squared);
    let a_a_c = matrix_vector(a, &a_c);
    condition(dot(b, &c_cubed), 0.25, "b·c³")?;
    condition(dot(b, &c_a_c), 0.125, "b·C·A·c")?;
    condition(dot(b, &a_c_squared), 1.0 / 12.0, "b·A·c²")?;
    condition(dot(b, &a_a_c), 1.0 / 24.0, "b·A·A·c")
}

fn matrix_vector(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    matrix.iter().map(|row| dot(row, vector)).collect()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
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
}
