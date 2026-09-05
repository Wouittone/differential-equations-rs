use serde::Deserialize;

use super::{Scalar, TableauError, approximately_equal, materialize_matrix, materialize_vector};

/// Stepping policy encoded by a Runge--Kutta--Nyström tableau.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RungeKuttaNystromKind {
    /// A formula without an embedded local-error estimator.
    Fixed,
    /// A formula with an embedded local-error estimator.
    Adaptive,
}

/// A validated explicit Runge--Kutta--Nyström tableau.
///
/// `A`, `b`, and `c` advance position. `A_velocity` and `b_velocity`
/// describe the corresponding velocity stages and update. Omitting
/// `A_velocity` declares an acceleration function that is independent of
/// velocity. Adaptive methods additionally carry direct error weights and may
/// provide dense-output polynomials in ascending powers of the interpolation
/// fraction.
#[derive(Clone, Debug, PartialEq)]
pub struct RungeKuttaNystromTableau {
    name: String,
    description: String,
    kind: RungeKuttaNystromKind,
    order: usize,
    a: Vec<Vec<f64>>,
    a_velocity: Option<Vec<Vec<f64>>>,
    b: Vec<f64>,
    b_velocity: Vec<f64>,
    c: Vec<f64>,
    error: Option<Vec<f64>>,
    velocity_error: Option<Vec<f64>>,
    position_only_error: bool,
    dense: Option<Vec<Vec<f64>>>,
    velocity_dense: Option<Vec<Vec<f64>>>,
}

impl RungeKuttaNystromTableau {
    /// Resource method name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Human-readable method description and provenance.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Whether this is a fixed or adaptive formula.
    pub fn kind(&self) -> RungeKuttaNystromKind {
        self.kind
    }

    /// Classical order of the primary formula.
    pub fn order(&self) -> usize {
        self.order
    }

    /// Number of acceleration stages.
    pub fn stages(&self) -> usize {
        self.c.len()
    }

    /// Position-stage coefficient matrix `A`.
    pub fn a(&self) -> &[Vec<f64>] {
        &self.a
    }

    /// Velocity-stage coefficient matrix, when acceleration depends on velocity.
    pub fn a_velocity(&self) -> Option<&[Vec<f64>]> {
        self.a_velocity.as_deref()
    }

    /// Position-update weights `b`.
    pub fn b(&self) -> &[f64] {
        &self.b
    }

    /// Velocity-update weights.
    pub fn b_velocity(&self) -> &[f64] {
        &self.b_velocity
    }

    /// Stage nodes `c`.
    pub fn c(&self) -> &[f64] {
        &self.c
    }

    /// Direct position-error weights for an adaptive formula.
    pub fn error(&self) -> Option<&[f64]> {
        self.error.as_deref()
    }

    /// Direct velocity-error weights, unless position alone controls error.
    pub fn velocity_error(&self) -> Option<&[f64]> {
        self.velocity_error.as_deref()
    }

    /// Whether an adaptive formula intentionally estimates only position error.
    pub fn position_only_error(&self) -> bool {
        self.position_only_error
    }

    /// Position dense-output polynomials, one row per stage.
    pub fn dense(&self) -> Option<&[Vec<f64>]> {
        self.dense.as_deref()
    }

    /// Velocity dense-output polynomials, one row per stage.
    pub fn velocity_dense(&self) -> Option<&[Vec<f64>]> {
        self.velocity_dense.as_deref()
    }
}

/// Endpoint acceleration used to seed an improved RKN history.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IrknBootstrapSeed {
    /// Seed from the endpoint at the start of the bootstrap step.
    PreviousEndpoint,
    /// Seed from the newly computed endpoint after the bootstrap step.
    NewEndpoint,
}

/// A validated fixed-step improved Runge--Kutta--Nyström tableau.
#[derive(Clone, Debug, PartialEq)]
pub struct IrknTableau {
    name: String,
    description: String,
    order: usize,
    bootstrap_order: usize,
    bootstrap_seed: IrknBootstrapSeed,
    velocity_history: Vec<f64>,
    c: Vec<f64>,
    a: Vec<f64>,
    velocity_weights: Vec<f64>,
    history_weights: Vec<f64>,
}

impl IrknTableau {
    /// Resource method name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Human-readable method description and provenance.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Classical order of the history formula.
    pub fn order(&self) -> usize {
        self.order
    }

    /// Order required from the one-step startup method.
    pub fn bootstrap_order(&self) -> usize {
        self.bootstrap_order
    }

    /// Endpoint acceleration used to seed the first retained history.
    pub fn bootstrap_seed(&self) -> IrknBootstrapSeed {
        self.bootstrap_seed
    }

    /// Number of internal stages retained between steps.
    pub fn stages(&self) -> usize {
        self.c.len()
    }

    /// Current- and previous-step velocity-history weights.
    pub fn velocity_history(&self) -> &[f64] {
        &self.velocity_history
    }

    /// Internal stage nodes.
    pub fn c(&self) -> &[f64] {
        &self.c
    }

    /// Chained internal-stage acceleration coefficients.
    pub fn a(&self) -> &[f64] {
        &self.a
    }

    /// Endpoint and internal-difference weights for the velocity update.
    pub fn velocity_weights(&self) -> &[f64] {
        &self.velocity_weights
    }

    /// Endpoint-history and internal-difference weights for the position update.
    pub fn history_weights(&self) -> &[f64] {
        &self.history_weights
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum RawRknKind {
    FixedRungeKuttaNystrom,
    AdaptiveRungeKuttaNystrom,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRknTableau {
    #[serde(rename = "$schema", default)]
    _schema: Option<String>,
    name: String,
    description: String,
    kind: RawRknKind,
    order: usize,
    #[serde(rename = "A")]
    a: Vec<Vec<Scalar>>,
    #[serde(rename = "A_velocity")]
    a_velocity: Option<Vec<Vec<Scalar>>>,
    b: Vec<Scalar>,
    b_velocity: Vec<Scalar>,
    c: Vec<Scalar>,
    error: Option<Vec<Scalar>>,
    velocity_error: Option<Vec<Scalar>>,
    #[serde(default)]
    position_only_error: bool,
    dense: Option<Vec<Vec<Scalar>>>,
    velocity_dense: Option<Vec<Vec<Scalar>>>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum RawBootstrapSeed {
    PreviousEndpoint,
    NewEndpoint,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawIrknTableau {
    #[serde(rename = "$schema", default)]
    _schema: Option<String>,
    name: String,
    description: String,
    #[serde(rename = "kind")]
    _kind: RawIrknKind,
    order: usize,
    bootstrap_order: usize,
    bootstrap_seed: RawBootstrapSeed,
    velocity_history: Vec<Scalar>,
    c: Vec<Scalar>,
    #[serde(rename = "A")]
    a: Vec<Scalar>,
    velocity_weights: Vec<Scalar>,
    history_weights: Vec<Scalar>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum RawIrknKind {
    ImprovedRungeKuttaNystrom,
}

/// Parses and validates a canonical explicit RKN JSON resource.
///
/// Validation covers metadata, complete square matrices, explicit causality,
/// stage consistency, endpoint consistency, adaptive estimator shape, and
/// paired dense-output polynomials. The same function is run by the procedural
/// macro at compile time and by the per-method lazy value on first use.
pub fn parse_rkn_tableau(
    source: &str,
    requested_name: &str,
) -> Result<RungeKuttaNystromTableau, TableauError> {
    let raw: RawRknTableau = serde_json::from_str(source)
        .map_err(|error| TableauError::new(format!("invalid RKN tableau JSON: {error}")))?;
    validate_metadata(
        &raw.name,
        &raw.description,
        raw.order,
        requested_name,
        "RKN",
    )?;

    let a = materialize_matrix(&raw.a, "A")?;
    let a_velocity = raw
        .a_velocity
        .as_deref()
        .map(|matrix| materialize_matrix(matrix, "A_velocity"))
        .transpose()?;
    let b = materialize_vector(&raw.b, "b")?;
    let b_velocity = materialize_vector(&raw.b_velocity, "b_velocity")?;
    let c = materialize_vector(&raw.c, "c")?;
    let stages = c.len();
    if stages == 0 || a.len() != stages || b.len() != stages || b_velocity.len() != stages {
        return Err(TableauError::new(
            "A, b, b_velocity, and c must have the same non-zero stage count",
        ));
    }
    validate_explicit_matrix(&a, stages, "A")?;
    if let Some(matrix) = &a_velocity {
        if matrix.len() != stages {
            return Err(TableauError::new(
                "A_velocity must have the same stage count as c",
            ));
        }
        validate_explicit_matrix(matrix, stages, "A_velocity")?;
    }
    for (stage, (row, node)) in a.iter().zip(&c).enumerate() {
        let expected = node * node / 2.0;
        if !approximately_equal(row.iter().sum(), expected) {
            return Err(TableauError::new(format!(
                "A row {stage} must sum to c[{stage}]^2 / 2"
            )));
        }
    }
    if let Some(matrix) = &a_velocity {
        for (stage, (row, node)) in matrix.iter().zip(&c).enumerate() {
            if !approximately_equal(row.iter().sum(), *node) {
                return Err(TableauError::new(format!(
                    "A_velocity row {stage} must sum to c[{stage}]"
                )));
            }
        }
    }
    validate_sum(&b, 0.5, "position weights b")?;
    validate_sum(&b_velocity, 1.0, "velocity weights b_velocity")?;

    let error = raw
        .error
        .as_deref()
        .map(|values| materialize_sized_vector(values, "error", stages))
        .transpose()?;
    let velocity_error = raw
        .velocity_error
        .as_deref()
        .map(|values| materialize_sized_vector(values, "velocity_error", stages))
        .transpose()?;
    let dense = raw
        .dense
        .as_deref()
        .map(|rows| materialize_matrix(rows, "dense"))
        .transpose()?;
    let velocity_dense = raw
        .velocity_dense
        .as_deref()
        .map(|rows| materialize_matrix(rows, "velocity_dense"))
        .transpose()?;

    let kind = match raw.kind {
        RawRknKind::FixedRungeKuttaNystrom => {
            if error.is_some()
                || velocity_error.is_some()
                || raw.position_only_error
                || dense.is_some()
                || velocity_dense.is_some()
            {
                return Err(TableauError::new(
                    "fixed RKN tableaus cannot define adaptive-error or dense-output fields",
                ));
            }
            RungeKuttaNystromKind::Fixed
        }
        RawRknKind::AdaptiveRungeKuttaNystrom => {
            let error = error.as_ref().ok_or_else(|| {
                TableauError::new("adaptive RKN tableaus require position error weights")
            })?;
            validate_sum(error, 0.0, "position error weights")?;
            match (raw.position_only_error, velocity_error.as_ref()) {
                (true, None) => {}
                (false, Some(weights)) => validate_sum(weights, 0.0, "velocity error weights")?,
                (true, Some(_)) => {
                    return Err(TableauError::new(
                        "position-only error control must omit velocity_error",
                    ));
                }
                (false, None) => {
                    return Err(TableauError::new(
                        "adaptive RKN tableaus require velocity_error unless position_only_error is true",
                    ));
                }
            }
            if error.iter().all(|weight| *weight == 0.0)
                && velocity_error
                    .as_ref()
                    .is_none_or(|weights| weights.iter().all(|weight| *weight == 0.0))
            {
                return Err(TableauError::new(
                    "adaptive RKN tableaus require a non-zero error estimator",
                ));
            }
            validate_dense_pair(
                dense.as_deref(),
                velocity_dense.as_deref(),
                &b,
                &b_velocity,
                stages,
            )?;
            RungeKuttaNystromKind::Adaptive
        }
    };

    Ok(RungeKuttaNystromTableau {
        name: raw.name,
        description: raw.description,
        kind,
        order: raw.order,
        a,
        a_velocity,
        b,
        b_velocity,
        c,
        error,
        velocity_error,
        position_only_error: raw.position_only_error,
        dense,
        velocity_dense,
    })
}

/// Parses and validates a canonical improved RKN history tableau.
pub fn parse_irkn_tableau(source: &str, requested_name: &str) -> Result<IrknTableau, TableauError> {
    let raw: RawIrknTableau = serde_json::from_str(source)
        .map_err(|error| TableauError::new(format!("invalid IRKN tableau JSON: {error}")))?;
    validate_metadata(
        &raw.name,
        &raw.description,
        raw.order,
        requested_name,
        "IRKN",
    )?;
    if raw.bootstrap_order == 0 {
        return Err(TableauError::new("IRKN bootstrap order must be positive"));
    }
    let velocity_history = materialize_sized_vector(&raw.velocity_history, "velocity_history", 2)?;
    let c = materialize_vector(&raw.c, "c")?;
    let a = materialize_vector(&raw.a, "A")?;
    let velocity_weights = materialize_vector(&raw.velocity_weights, "velocity_weights")?;
    let history_weights = materialize_vector(&raw.history_weights, "history_weights")?;
    let stages = c.len();
    if stages == 0
        || a.len() != stages
        || velocity_weights.len() != stages + 1
        || history_weights.len() != stages + 1
    {
        return Err(TableauError::new(
            "IRKN A and c need one entry per stage and update weights need stages + 1 entries",
        ));
    }
    if c.iter().any(|node| !(0.0..1.0).contains(node))
        || c.windows(2).any(|nodes| nodes[0] >= nodes[1])
    {
        return Err(TableauError::new(
            "IRKN internal nodes c must be strictly increasing inside (0, 1)",
        ));
    }
    validate_sum(&velocity_history, 1.0, "velocity history weights")?;
    if !approximately_equal(velocity_history[1], -0.5) {
        return Err(TableauError::new(
            "IRKN previous-step velocity history weight must equal -1/2",
        ));
    }
    for (stage, (coefficient, node)) in a.iter().zip(&c).enumerate() {
        if !approximately_equal(*coefficient, node * node / 2.0) {
            return Err(TableauError::new(format!(
                "IRKN A[{stage}] must equal c[{stage}]^2 / 2"
            )));
        }
    }
    if !approximately_equal(velocity_weights[0] + history_weights[0], 1.0) {
        return Err(TableauError::new(
            "IRKN endpoint acceleration weights must sum to one",
        ));
    }

    Ok(IrknTableau {
        name: raw.name,
        description: raw.description,
        order: raw.order,
        bootstrap_order: raw.bootstrap_order,
        bootstrap_seed: match raw.bootstrap_seed {
            RawBootstrapSeed::PreviousEndpoint => IrknBootstrapSeed::PreviousEndpoint,
            RawBootstrapSeed::NewEndpoint => IrknBootstrapSeed::NewEndpoint,
        },
        velocity_history,
        c,
        a,
        velocity_weights,
        history_weights,
    })
}

fn validate_metadata(
    name: &str,
    description: &str,
    order: usize,
    requested_name: &str,
    family: &str,
) -> Result<(), TableauError> {
    if name.trim().is_empty() || name != requested_name {
        return Err(TableauError::new(format!(
            "resource method `{name}` does not match requested method `{requested_name}`"
        )));
    }
    if description.trim().is_empty() || order == 0 {
        return Err(TableauError::new(format!(
            "{family} tableau requires a description and positive order"
        )));
    }
    Ok(())
}

fn validate_explicit_matrix(
    matrix: &[Vec<f64>],
    stages: usize,
    label: &str,
) -> Result<(), TableauError> {
    if matrix.iter().any(|row| row.len() != stages) {
        return Err(TableauError::new(format!(
            "{label} must be a square {stages}-stage matrix"
        )));
    }
    if matrix
        .iter()
        .enumerate()
        .any(|(stage, row)| row[stage..].iter().any(|coefficient| *coefficient != 0.0))
    {
        return Err(TableauError::new(format!(
            "{label} must be strictly lower triangular"
        )));
    }
    Ok(())
}

fn materialize_sized_vector(
    values: &[Scalar],
    label: &str,
    expected: usize,
) -> Result<Vec<f64>, TableauError> {
    if values.len() != expected {
        return Err(TableauError::new(format!(
            "{label} must contain {expected} entries"
        )));
    }
    materialize_vector(values, label)
}

fn validate_sum(values: &[f64], expected: f64, label: &str) -> Result<(), TableauError> {
    let sum = values.iter().sum::<f64>();
    if !approximately_equal(sum, expected) {
        return Err(TableauError::new(format!(
            "{label} must sum to {expected}; found {sum}"
        )));
    }
    Ok(())
}

fn validate_dense_pair(
    dense: Option<&[Vec<f64>]>,
    velocity_dense: Option<&[Vec<f64>]>,
    b: &[f64],
    b_velocity: &[f64],
    stages: usize,
) -> Result<(), TableauError> {
    match (dense, velocity_dense) {
        (None, None) => Ok(()),
        (Some(position), Some(velocity)) => {
            if position.len() != stages
                || velocity.len() != stages
                || position
                    .iter()
                    .zip(velocity)
                    .any(|(position_row, velocity_row)| {
                        position_row.is_empty() || position_row.len() != velocity_row.len()
                    })
            {
                return Err(TableauError::new(
                    "dense and velocity_dense require equally sized non-empty rows for every stage",
                ));
            }
            for stage in 0..stages {
                if !approximately_equal(position[stage].iter().sum(), b[stage])
                    || !approximately_equal(velocity[stage].iter().sum(), b_velocity[stage])
                {
                    return Err(TableauError::new(format!(
                        "dense-output row {stage} must reproduce its endpoint weights"
                    )));
                }
            }
            Ok(())
        }
        _ => Err(TableauError::new(
            "dense and velocity_dense must be provided together",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXED: &str = r#"{"name":"Rkn","description":"fixed test","kind":"fixed-runge-kutta-nystrom","order":2,"A":[[0,0],["1/8",0]],"A_velocity":null,"b":["1/2",0],"b_velocity":[0,1],"c":[0,"1/2"],"error":null,"velocity_error":null,"dense":null,"velocity_dense":null}"#;
    const ADAPTIVE: &str = r#"{"name":"Rkn","description":"adaptive test","kind":"adaptive-runge-kutta-nystrom","order":2,"A":[[0,0],["1/8",0]],"A_velocity":null,"b":["1/2",0],"b_velocity":[0,1],"c":[0,"1/2"],"error":["1/2","-1/2"],"velocity_error":[-1,1],"dense":null,"velocity_dense":null}"#;
    const IRKN: &str = r#"{"name":"Irkn","description":"history test","kind":"improved-runge-kutta-nystrom","order":3,"bootstrap_order":4,"bootstrap_seed":"previous-endpoint","velocity_history":["3/2","-1/2"],"c":["1/2"],"A":["1/8"],"velocity_weights":["2/3","5/6"],"history_weights":["1/3","5/12"]}"#;

    #[test]
    fn parses_fixed_adaptive_and_history_tableaus() {
        let fixed = parse_rkn_tableau(FIXED, "Rkn").unwrap();
        assert_eq!(fixed.kind(), RungeKuttaNystromKind::Fixed);
        assert_eq!(fixed.a()[1], [0.125, 0.0]);
        assert!(fixed.error().is_none());

        let adaptive = parse_rkn_tableau(ADAPTIVE, "Rkn").unwrap();
        assert_eq!(adaptive.kind(), RungeKuttaNystromKind::Adaptive);
        assert_eq!(adaptive.velocity_error().unwrap(), [-1.0, 1.0]);

        let history = parse_irkn_tableau(IRKN, "Irkn").unwrap();
        assert_eq!(history.stages(), 1);
        assert_eq!(
            history.bootstrap_seed(),
            IrknBootstrapSeed::PreviousEndpoint
        );
    }

    #[test]
    fn rejects_invalid_shapes_causality_and_policy_fields() {
        for invalid in [
            FIXED.replace("\"name\":\"Rkn\"", "\"name\":\"Other\""),
            FIXED.replace("\"description\":\"fixed test\"", "\"description\":\" \""),
            FIXED.replace("\"order\":2", "\"order\":0"),
            FIXED.replace("[\"1/8\",0]", "[\"1/8\"]"),
            FIXED.replace("[\"1/8\",0]", "[\"1/8\",1]"),
            FIXED.replace("[\"1/2\",0]", "[1,0]"),
            FIXED.replace("\"b_velocity\":[0,1]", "\"b_velocity\":[0,2]"),
            FIXED.replace("\"A_velocity\":null", "\"A_velocity\":[[0,0],[1,0]]"),
            FIXED.replace("\"error\":null", "\"error\":[0,0]"),
            ADAPTIVE.replace("\"velocity_error\":[-1,1]", "\"velocity_error\":null"),
            ADAPTIVE.replace(
                "\"velocity_error\":[-1,1]",
                "\"velocity_error\":[-1,1],\"position_only_error\":true",
            ),
            ADAPTIVE.replace("\"error\":[\"1/2\",\"-1/2\"]", "\"error\":[1,0]"),
            ADAPTIVE
                .replace("\"error\":[\"1/2\",\"-1/2\"]", "\"error\":[0,0]")
                .replace("\"velocity_error\":[-1,1]", "\"velocity_error\":[0,0]"),
            ADAPTIVE
                .replace("\"error\":[\"1/2\",\"-1/2\"]", "\"error\":[0,0]")
                .replace(
                    "\"velocity_error\":[-1,1]",
                    "\"velocity_error\":null,\"position_only_error\":true",
                ),
            ADAPTIVE.replace(
                "\"dense\":null,\"velocity_dense\":null",
                "\"dense\":[[\"1/2\"],[0]],\"velocity_dense\":null",
            ),
            ADAPTIVE.replace(
                "\"dense\":null,\"velocity_dense\":null",
                "\"dense\":[[1],[0]],\"velocity_dense\":[[0],[1]]",
            ),
            FIXED.replace("\"order\":2", "\"order\":2,\"extra\":0"),
            FIXED.replace("\"1/8\"", "\"1/0\""),
        ] {
            assert!(
                parse_rkn_tableau(&invalid, "Rkn").is_err(),
                "accepted {invalid}"
            );
        }
        for invalid in [
            IRKN.replace("\"name\":\"Irkn\"", "\"name\":\"Other\""),
            IRKN.replace("improved-runge-kutta-nystrom", "fixed-runge-kutta-nystrom"),
            IRKN.replace("\"order\":3", "\"order\":0"),
            IRKN.replace("\"c\":[\"1/2\"]", "\"c\":[]"),
            IRKN.replace("\"c\":[\"1/2\"]", "\"c\":[1]"),
            IRKN.replace(
                "\"velocity_history\":[\"3/2\",\"-1/2\"]",
                "\"velocity_history\":[1]",
            ),
            IRKN.replace(
                "\"velocity_history\":[\"3/2\",\"-1/2\"]",
                "\"velocity_history\":[1,1]",
            ),
            IRKN.replace(
                "\"velocity_history\":[\"3/2\",\"-1/2\"]",
                "\"velocity_history\":[1,0]",
            ),
            IRKN.replace("\"bootstrap_order\":4", "\"bootstrap_order\":0"),
            IRKN.replace("\"A\":[\"1/8\"]", "\"A\":[]"),
            IRKN.replace("\"A\":[\"1/8\"]", "\"A\":[\"1/4\"]"),
            IRKN.replace(
                "\"velocity_weights\":[\"2/3\",\"5/6\"]",
                "\"velocity_weights\":[1]",
            ),
            IRKN.replace(
                "\"velocity_weights\":[\"2/3\",\"5/6\"]",
                "\"velocity_weights\":[\"1/2\",\"5/6\"]",
            ),
            IRKN.replace(
                "\"history_weights\":[\"1/3\",\"5/12\"]",
                "\"history_weights\":[1]",
            ),
            IRKN.replace("\"order\":3", "\"order\":3,\"extra\":0"),
            IRKN.replace("\"1/8\"", "\"sqrt(-1)\""),
        ] {
            assert!(
                parse_irkn_tableau(&invalid, "Irkn").is_err(),
                "accepted {invalid}"
            );
        }
    }
}
