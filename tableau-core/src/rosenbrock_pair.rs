use serde::Deserialize;

use super::{Scalar, TableauError, approximately_equal, materialize_matrix, materialize_vector};

/// Validated coefficients for the shared Rosenbrock 2/3 W-method stage scheme.
///
/// The three stage derivatives `k` use one factorization `W` per step. For
/// stage `i`, `state`, `derivative`, and `stage` combine the initial state,
/// previously evaluated derivatives, and prior `k` values before the solve;
/// `post_solve` is added afterwards. This representation keeps every numeric
/// part of the specialized low-storage scheme in its resource.
#[derive(Clone, Debug, PartialEq)]
pub struct RosenbrockPairTableau {
    name: String,
    description: String,
    orders: [usize; 2],
    gamma: f64,
    nodes: Vec<f64>,
    state: Vec<Vec<f64>>,
    derivative: Vec<Vec<f64>>,
    stage: Vec<Vec<f64>>,
    post_solve: Vec<Vec<f64>>,
    time_derivative: Vec<f64>,
    second_order: Vec<f64>,
    third_order: Vec<f64>,
    error: Vec<f64>,
    dense: Vec<Vec<f64>>,
}

impl RosenbrockPairTableau {
    /// Resource name.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Resource provenance and stage-convention description.
    pub fn description(&self) -> &str {
        &self.description
    }
    /// Orders of the lower- and higher-order formulas.
    pub fn orders(&self) -> [usize; 2] {
        self.orders
    }
    /// Diagonal coefficient in `W = I - gamma*h*J`.
    pub fn gamma(&self) -> f64 {
        self.gamma
    }
    /// Stage time nodes.
    pub fn nodes(&self) -> &[f64] {
        &self.nodes
    }
    /// State weights applied to solved stage derivatives before RHS evaluation.
    pub fn state(&self) -> &[Vec<f64>] {
        &self.state
    }
    /// Weights applied to evaluated derivatives in each linear-solve RHS.
    pub fn derivative(&self) -> &[Vec<f64>] {
        &self.derivative
    }
    /// Weights applied to prior solved stages in each linear-solve RHS.
    pub fn stage(&self) -> &[Vec<f64>] {
        &self.stage
    }
    /// Prior-stage weights added after each linear solve.
    pub fn post_solve(&self) -> &[Vec<f64>] {
        &self.post_solve
    }
    /// Per-stage weights multiplying `h*f_t` in a linear-solve RHS.
    pub fn time_derivative(&self) -> &[f64] {
        &self.time_derivative
    }
    /// State-update weights for the second-order formula.
    pub fn second_order(&self) -> &[f64] {
        &self.second_order
    }
    /// State-update weights for the third-order formula.
    pub fn third_order(&self) -> &[f64] {
        &self.third_order
    }
    /// Direct local-error weights for the third-order stages.
    pub fn error(&self) -> &[f64] {
        &self.error
    }
    /// Polynomial dense-output rows for the first two solved stages.
    pub fn dense(&self) -> &[Vec<f64>] {
        &self.dense
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRosenbrockPairTableau {
    #[serde(rename = "$schema", default)]
    _schema: Option<String>,
    name: String,
    description: String,
    orders: [usize; 2],
    gamma: Scalar,
    nodes: Vec<Scalar>,
    state: Vec<Vec<Scalar>>,
    derivative: Vec<Vec<Scalar>>,
    stage: Vec<Vec<Scalar>>,
    post_solve: Vec<Vec<Scalar>>,
    time_derivative: Vec<Scalar>,
    second_order: Vec<Scalar>,
    third_order: Vec<Scalar>,
    error: Vec<Scalar>,
    dense: Vec<Vec<Scalar>>,
}

/// Parses the canonical resource for the specialized Rosenbrock 2/3 pair.
///
/// The parser requires exactly three causal stages and two dense-output rows.
/// It checks all dimensions and basic structural invariants, but does not claim
/// to prove the methods' order or stability properties.
pub fn parse_rosenbrock_pair_tableau(
    source: &str,
    requested_name: &str,
) -> Result<RosenbrockPairTableau, TableauError> {
    let raw: RawRosenbrockPairTableau = serde_json::from_str(source).map_err(|error| {
        TableauError::new(format!("invalid Rosenbrock pair tableau JSON: {error}"))
    })?;
    if raw.name.trim().is_empty() || raw.name != requested_name {
        return Err(TableauError::new(format!(
            "resource method `{}` does not match requested method `{requested_name}`",
            raw.name
        )));
    }
    if raw.description.trim().is_empty() || raw.orders != [2, 3] {
        return Err(TableauError::new(
            "Rosenbrock pair requires a description and orders [2, 3]",
        ));
    }
    let gamma = raw.gamma.materialize()?;
    if gamma == 0.0 {
        return Err(TableauError::new("gamma must be nonzero"));
    }
    let nodes = materialize_vector(&raw.nodes, "nodes")?;
    let state = materialize_matrix(&raw.state, "state")?;
    let derivative = materialize_matrix(&raw.derivative, "derivative")?;
    let stage = materialize_matrix(&raw.stage, "stage")?;
    let post_solve = materialize_matrix(&raw.post_solve, "post_solve")?;
    let time_derivative = materialize_vector(&raw.time_derivative, "time_derivative")?;
    let second_order = materialize_vector(&raw.second_order, "second_order")?;
    let third_order = materialize_vector(&raw.third_order, "third_order")?;
    let error = materialize_vector(&raw.error, "error")?;
    let dense = materialize_matrix(&raw.dense, "dense")?;
    const STAGES: usize = 3;
    for (label, values) in [
        ("nodes", &nodes),
        ("time_derivative", &time_derivative),
        ("second_order", &second_order),
        ("third_order", &third_order),
        ("error", &error),
    ] {
        if values.len() != STAGES {
            return Err(TableauError::new(format!(
                "{label} must have three entries"
            )));
        }
    }
    for (label, matrix) in [
        ("state", &state),
        ("derivative", &derivative),
        ("stage", &stage),
        ("post_solve", &post_solve),
    ] {
        if matrix.len() != STAGES || matrix.iter().any(|row| row.len() != STAGES) {
            return Err(TableauError::new(format!(
                "{label} must be a 3 by 3 matrix"
            )));
        }
        if matrix.iter().enumerate().any(|(i, row)| {
            let first_forbidden = i + usize::from(label == "derivative");
            row[first_forbidden..].iter().any(|value| *value != 0.0)
        }) {
            return Err(TableauError::new(format!("{label} is not causal")));
        }
    }
    if nodes[0] != 0.0 || nodes[1] <= 0.0 || nodes[1] >= nodes[2] || nodes[2] != 1.0 {
        return Err(TableauError::new(
            "nodes must be ordered as [0, interior, 1]",
        ));
    }
    if derivative.iter().enumerate().any(|(i, row)| row[i] != 1.0) {
        return Err(TableauError::new(
            "each stage must include its newly evaluated derivative exactly once",
        ));
    }
    if time_derivative[0] != gamma {
        return Err(TableauError::new(
            "the first time-derivative weight must equal gamma",
        ));
    }
    if dense.len() != 2 || dense.iter().any(|row| row.len() != 2) {
        return Err(TableauError::new(
            "dense must have two stage rows with two polynomial coefficients",
        ));
    }
    for (label, row) in [
        ("second_order", &second_order),
        ("third_order", &third_order),
        ("error", &error),
    ] {
        if row.iter().all(|value| *value == 0.0) {
            return Err(TableauError::new(format!(
                "{label} must contain a nonzero weight"
            )));
        }
    }
    if !approximately_equal(second_order.iter().sum(), 1.0)
        || !approximately_equal(third_order.iter().sum(), 1.0)
        || !approximately_equal(error.iter().sum(), 0.0)
    {
        return Err(TableauError::new(
            "solution weights must sum to one and error weights must sum to zero",
        ));
    }
    Ok(RosenbrockPairTableau {
        name: raw.name,
        description: raw.description,
        orders: raw.orders,
        gamma,
        nodes,
        state,
        derivative,
        stage,
        post_solve,
        time_derivative,
        second_order,
        third_order,
        error,
        dense,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = r#"{
      "name":"Pair","description":"test pair","orders":[2,3],"gamma":"1/(2+sqrt(2))",
      "nodes":[0,"1/2",1],"state":[[0,0,0],["1/2",0,0],[0,1,0]],
      "derivative":[[1,0,0],[0,1,0],[2,"6+sqrt(2)",1]],
      "stage":[[0,0,0],[-1,0,0],[-2,"-(6+sqrt(2))",0]],
      "post_solve":[[0,0,0],[1,0,0],[0,0,0]],
      "time_derivative":["1/(2+sqrt(2))",0,1],
      "second_order":[0,1,0],"third_order":["1/6","4/6","1/6"],
      "error":["1/6","-2/6","1/6"],"dense":[[1,-1],[-1,1]]
    }"#;

    #[test]
    fn parses_operational_pair_coefficients() {
        let tableau = parse_rosenbrock_pair_tableau(SOURCE, "Pair").unwrap();
        assert_eq!(tableau.orders(), [2, 3]);
        assert_eq!(tableau.nodes(), &[0.0, 0.5, 1.0]);
        assert_eq!(tableau.derivative()[2][1], 6.0 + 2.0_f64.sqrt());
        assert_eq!(tableau.stage()[2][1], -(6.0 + 2.0_f64.sqrt()));
        assert_eq!(tableau.third_order(), &[1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0]);
    }

    #[test]
    fn rejects_wrong_shapes_metadata_and_causality() {
        assert!(parse_rosenbrock_pair_tableau(SOURCE, "Other").is_err());
        for (from, to) in [
            ("\"orders\":[2,3]", "\"orders\":[3,2]"),
            ("\"nodes\":[0,\"1/2\",1]", "\"nodes\":[0,1]"),
            ("[\"1/2\",0,0]", "[\"1/2\",1,0]"),
            (
                "\"state\":[[0,0,0],[\"1/2\",0,0],[0,1,0]]",
                "\"state\":[[0,0,0],[\"1/2\",0,0],[0,1]]",
            ),
            ("[1,0,0],[0,1,0]", "[1,1,0],[0,1,0]"),
            ("\"gamma\":\"1/(2+sqrt(2))\"", "\"gamma\":0"),
            (
                "\"time_derivative\":[\"1/(2+sqrt(2))\",0,1]",
                "\"time_derivative\":[1,0,1]",
            ),
            ("\"second_order\":[0,1,0]", "\"second_order\":[0,2,0]"),
            ("\"error\":[\"1/6\",\"-2/6\",\"1/6\"]", "\"error\":[0,0,0]"),
            ("\"dense\":[[1,-1],[-1,1]]", "\"dense\":[[1,-1]]"),
        ] {
            let invalid = SOURCE.replace(from, to);
            assert_ne!(invalid, SOURCE, "test pattern did not match: {from}");
            assert!(
                parse_rosenbrock_pair_tableau(&invalid, "Pair").is_err(),
                "accepted {invalid}"
            );
        }
    }
}
