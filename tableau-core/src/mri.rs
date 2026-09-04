use serde::Deserialize;

use super::{Scalar, TableauError, approximately_equal, materialize_matrix, materialize_vector};

/// A validated MRI-GARK coupling tableau.
///
/// `W0` and `W1` contain the constant and linear slow-forcing polynomials for
/// each fast subinterval. `gamma` selects implicit slow endpoints. Optional
/// embedded rows define a second traversal used for local error estimation.
#[derive(Clone, Debug, PartialEq)]
pub struct MriTableau {
    name: String,
    description: String,
    order: usize,
    inner_order: usize,
    dc: Vec<f64>,
    w0: Vec<Vec<f64>>,
    w1: Vec<Vec<f64>>,
    embedded0: Option<Vec<f64>>,
    embedded1: Option<Vec<f64>>,
    gamma: Vec<f64>,
}

/// A validated multirate infinitesimal-step (MIS) coupling tableau.
#[derive(Clone, Debug, PartialEq)]
pub struct MisTableau {
    name: String,
    description: String,
    order: usize,
    alpha: Vec<Vec<f64>>,
    beta: Vec<Vec<f64>>,
    gamma: Vec<Vec<f64>>,
    d: Vec<f64>,
    c: Vec<f64>,
    c_tilde: Vec<f64>,
}

impl MisTableau {
    /// Resource method name.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Human-readable method description and provenance.
    pub fn description(&self) -> &str {
        &self.description
    }
    /// Classical method order.
    pub fn order(&self) -> usize {
        self.order
    }
    /// Prior-stage state coupling matrix.
    pub fn alpha(&self) -> &[Vec<f64>] {
        &self.alpha
    }
    /// Prior slow-derivative coupling matrix.
    pub fn beta(&self) -> &[Vec<f64>] {
        &self.beta
    }
    /// Prior stage-increment coupling matrix.
    pub fn gamma(&self) -> &[Vec<f64>] {
        &self.gamma
    }
    /// Normalized fast-subinterval lengths.
    pub fn d(&self) -> &[f64] {
        &self.d
    }
    /// Slow-stage time nodes.
    pub fn c(&self) -> &[f64] {
        &self.c
    }
    /// Fast-subinterval starting nodes.
    pub fn c_tilde(&self) -> &[f64] {
        &self.c_tilde
    }
}

impl MriTableau {
    /// Resource method name.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Human-readable method description and provenance.
    pub fn description(&self) -> &str {
        &self.description
    }
    /// Classical outer-method order.
    pub fn order(&self) -> usize {
        self.order
    }
    /// Order used for explicit fast subinterval integration.
    pub fn inner_order(&self) -> usize {
        self.inner_order
    }
    /// Normalized lengths of consecutive fast subintervals.
    pub fn dc(&self) -> &[f64] {
        &self.dc
    }
    /// Constant slow-forcing polynomial coefficients.
    pub fn w0(&self) -> &[Vec<f64>] {
        &self.w0
    }
    /// Linear slow-forcing polynomial coefficients.
    pub fn w1(&self) -> &[Vec<f64>] {
        &self.w1
    }
    /// Constant coefficients for an embedded final traversal.
    pub fn embedded0(&self) -> Option<&[f64]> {
        self.embedded0.as_deref()
    }
    /// Linear coefficients for an embedded final traversal.
    pub fn embedded1(&self) -> Option<&[f64]> {
        self.embedded1.as_deref()
    }
    /// Diagonal implicit-slow coefficients for each endpoint.
    pub fn gamma(&self) -> &[f64] {
        &self.gamma
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawMriTableau {
    #[serde(rename = "$schema", default)]
    _schema: Option<String>,
    name: String,
    description: String,
    order: usize,
    inner_order: usize,
    dc: Vec<Scalar>,
    #[serde(rename = "W0")]
    w0: Vec<Vec<Scalar>>,
    #[serde(rename = "W1")]
    w1: Vec<Vec<Scalar>>,
    embedded0: Option<Vec<Scalar>>,
    embedded1: Option<Vec<Scalar>>,
    gamma: Vec<Scalar>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawMisTableau {
    #[serde(rename = "$schema", default)]
    _schema: Option<String>,
    name: String,
    description: String,
    order: usize,
    alpha: Vec<Vec<Scalar>>,
    beta: Vec<Vec<Scalar>>,
    gamma: Vec<Vec<Scalar>>,
    d: Vec<Scalar>,
    c: Vec<Scalar>,
    c_tilde: Vec<Scalar>,
}

/// Parses and structurally validates a canonical MIS JSON resource.
pub fn parse_mis_tableau(source: &str, requested_name: &str) -> Result<MisTableau, TableauError> {
    let raw: RawMisTableau = serde_json::from_str(source)
        .map_err(|error| TableauError::new(format!("invalid MIS tableau JSON: {error}")))?;
    if raw.name.trim().is_empty() || raw.name != requested_name {
        return Err(TableauError::new(format!(
            "resource method `{}` does not match requested method `{requested_name}`",
            raw.name
        )));
    }
    if raw.description.trim().is_empty() || raw.order == 0 {
        return Err(TableauError::new(
            "MIS tableau requires a description and positive order",
        ));
    }
    let alpha = materialize_matrix(&raw.alpha, "alpha")?;
    let beta = materialize_matrix(&raw.beta, "beta")?;
    let gamma = materialize_matrix(&raw.gamma, "gamma")?;
    let d = materialize_vector(&raw.d, "d")?;
    let c = materialize_vector(&raw.c, "c")?;
    let c_tilde = materialize_vector(&raw.c_tilde, "c_tilde")?;
    let stages = d.len();
    if stages < 2 || c.len() != stages || c_tilde.len() != stages {
        return Err(TableauError::new(
            "d, c, and c_tilde must contain the same number of at least two stages",
        ));
    }
    for (label, matrix) in [("alpha", &alpha), ("beta", &beta), ("gamma", &gamma)] {
        if matrix.len() != stages || matrix.iter().any(|row| row.len() != stages) {
            return Err(TableauError::new(format!(
                "{label} must be square with {stages} stages"
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
    }
    if d[0] != 0.0
        || c[0] != 0.0
        || c_tilde[0] != 0.0
        || d[1..].iter().any(|value| *value <= 0.0)
        || c.last() != Some(&1.0)
    {
        return Err(TableauError::new(
            "MIS nodes require zero initial entries, positive later d, and final c equal to one",
        ));
    }
    Ok(MisTableau {
        name: raw.name,
        description: raw.description,
        order: raw.order,
        alpha,
        beta,
        gamma,
        d,
        c,
        c_tilde,
    })
}

/// Parses and structurally validates one canonical MRI-GARK JSON resource.
///
/// Validation covers metadata, dimensions, causal coupling rows, normalized
/// subinterval lengths, paired embedded rows, and finite coefficients. It does
/// not prove the declared order or stability properties.
pub fn parse_mri_tableau(source: &str, requested_name: &str) -> Result<MriTableau, TableauError> {
    let raw: RawMriTableau = serde_json::from_str(source)
        .map_err(|error| TableauError::new(format!("invalid MRI tableau JSON: {error}")))?;
    if raw.name.trim().is_empty() || raw.name != requested_name {
        return Err(TableauError::new(format!(
            "resource method `{}` does not match requested method `{requested_name}`",
            raw.name
        )));
    }
    if raw.description.trim().is_empty() || raw.order == 0 || !(2..=4).contains(&raw.inner_order) {
        return Err(TableauError::new(
            "MRI tableau requires a description, positive order, and inner order 2 through 4",
        ));
    }
    let dc = materialize_vector(&raw.dc, "dc")?;
    let w0 = materialize_matrix(&raw.w0, "W0")?;
    let w1 = materialize_matrix(&raw.w1, "W1")?;
    let gamma = materialize_vector(&raw.gamma, "gamma")?;
    let embedded0 = raw
        .embedded0
        .as_ref()
        .map(|row| materialize_vector(row, "embedded0"))
        .transpose()?;
    let embedded1 = raw
        .embedded1
        .as_ref()
        .map(|row| materialize_vector(row, "embedded1"))
        .transpose()?;
    let stages = dc.len();
    if stages == 0 || gamma.len() != stages {
        return Err(TableauError::new(
            "dc and gamma must contain the same nonzero number of stages",
        ));
    }
    if !approximately_equal(dc.iter().sum(), 1.0) {
        return Err(TableauError::new(
            "MRI subinterval lengths dc must sum to one",
        ));
    }
    for (label, matrix) in [("W0", &w0), ("W1", &w1)] {
        if matrix.len() != stages || matrix.iter().any(|row| row.len() != stages) {
            return Err(TableauError::new(format!(
                "{label} must be square with {stages} stages"
            )));
        }
        if matrix.iter().enumerate().any(|(stage, row)| {
            row[stage + 1..]
                .iter()
                .any(|coefficient| *coefficient != 0.0)
        }) {
            return Err(TableauError::new(format!(
                "{label} may only couple the current and prior slow stages"
            )));
        }
    }
    match (&embedded0, &embedded1) {
        (None, None) => {}
        (Some(first), Some(second))
            if first.len() == stages
                && second.len() == stages
                && first.iter().chain(second).any(|value| *value != 0.0) => {}
        _ => {
            return Err(TableauError::new(
                "embedded0 and embedded1 must both be omitted or both be nonzero stage rows",
            ));
        }
    }
    Ok(MriTableau {
        name: raw.name,
        description: raw.description,
        order: raw.order,
        inner_order: raw.inner_order,
        dc,
        w0,
        w1,
        embedded0,
        embedded1,
        gamma,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = r#"{"name":"MRI","description":"test MRI","order":2,"inner_order":2,"dc":["1/2","1/2"],"W0":[["1/2",0],["-1/2",1]],"W1":[[0,0],[0,0]],"embedded0":null,"embedded1":null,"gamma":[0,0]}"#;
    const MIS_SOURCE: &str = r#"{"name":"MIS","description":"test MIS","order":3,"alpha":[[0,0],[1,0]],"beta":[[0,0],[1,0]],"gamma":[[0,0],[0,0]],"d":[0,1],"c":[0,1],"c_tilde":[0,0]}"#;

    #[test]
    fn parses_shared_expressions_and_causal_rows() {
        let tableau = parse_mri_tableau(SOURCE, "MRI").unwrap();
        assert_eq!(tableau.name(), "MRI");
        assert_eq!(tableau.order(), 2);
        assert_eq!(tableau.inner_order(), 2);
        assert_eq!(tableau.dc(), &[0.5, 0.5]);
        assert_eq!(tableau.w0()[1], [-0.5, 1.0]);
        assert!(tableau.embedded0().is_none());
    }

    #[test]
    fn rejects_invalid_metadata_dimensions_and_embedded_rows() {
        assert!(parse_mri_tableau(SOURCE, "Other").is_err());
        for (from, to) in [
            ("\"order\":2", "\"order\":0"),
            ("\"inner_order\":2", "\"inner_order\":1"),
            ("[\"1/2\",\"1/2\"]", "[\"1/2\",\"1/3\"]"),
            ("[\"-1/2\",1]", "[\"-1/2\"]"),
            ("[\"1/2\",0]", "[\"1/2\",1]"),
            ("\"gamma\":[0,0]", "\"gamma\":[0]"),
            ("\"embedded0\":null", "\"embedded0\":[1,0]"),
            ("\"W1\":[[0,0],[0,0]]", "\"W1\":[[0,0],[0,\"1/0\"]]"),
        ] {
            let invalid = SOURCE.replace(from, to);
            assert_ne!(invalid, SOURCE, "test pattern did not match: {from}");
            assert!(
                parse_mri_tableau(&invalid, "MRI").is_err(),
                "accepted {invalid}"
            );
        }
    }

    #[test]
    fn parses_and_rejects_structurally_invalid_mis_tableaus() {
        let tableau = parse_mis_tableau(MIS_SOURCE, "MIS").unwrap();
        assert_eq!(tableau.order(), 3);
        assert_eq!(tableau.alpha()[1], [1.0, 0.0]);
        assert_eq!(tableau.d(), &[0.0, 1.0]);
        for (from, to) in [
            ("\"order\":3", "\"order\":0"),
            ("\"alpha\":[[0,0],[1,0]]", "\"alpha\":[[0,1],[1,0]]"),
            ("\"beta\":[[0,0],[1,0]]", "\"beta\":[[0,0],[1]]"),
            ("\"d\":[0,1]", "\"d\":[0,0]"),
            ("\"c\":[0,1]", "\"c\":[0,0]"),
            ("\"c_tilde\":[0,0]", "\"c_tilde\":[1,0]"),
            ("\"gamma\":[[0,0],[0,0]]", "\"gamma\":[[0,0],[0,\"1/0\"]]"),
        ] {
            let invalid = MIS_SOURCE.replace(from, to);
            assert_ne!(invalid, MIS_SOURCE, "test pattern did not match: {from}");
            assert!(
                parse_mis_tableau(&invalid, "MIS").is_err(),
                "accepted {invalid}"
            );
        }
    }
}
