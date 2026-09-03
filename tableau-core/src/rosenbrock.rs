use serde::Deserialize;

use super::{Scalar, TableauError, approximately_equal, materialize_matrix, materialize_vector};

/// Stage-equation convention for a Rosenbrock-family tableau.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum RosenbrockKind {
    /// SciML's transformed `RodasTableau` with strictly lower-triangular `C`.
    Rosenbrock,
    /// Explicit differential stages and raw lower-triangular `Gamma` in `C`.
    ///
    /// Its diagonal is `gamma`. The implicit coefficients describe the upstream
    /// algebraic specialization; an ordinary ODE uses only `A`, `b`, and `c`.
    HybridExplicitImplicit,
}

/// A Rosenbrock-family tableau using a declared SciML stage convention.
///
/// For [`RosenbrockKind::Rosenbrock`], the stage equations are:
///
/// ```text
/// (I - h*gamma*J) k[i] = h*gamma*(f(u + A[i]*k, t + c[i]*h)
///                                + h*d[i]*f_t + C[i]*k/h)
/// ```
///
/// Only prior stages enter the matrix products.
/// The solution is `u + sum(b[i]*k[i])`; `btilde` directly weights its
/// embedded error. These are not ordinary Runge--Kutta weights.
///
/// For [`RosenbrockKind::HybridExplicitImplicit`], ordinary ODE stages instead
/// use `k[i] = h*f(u + A[i]*k, t + c[i]*h)`. The resource also retains the raw
/// `Gamma` matrix in `C` and time-derivative weights `d` for the upstream algebraic
/// specialization. Parsed hybrid metadata does not imply DAE solver support.
///
/// Parsing validates the representation, not the claimed order or stability.
#[derive(Clone, Debug, PartialEq)]
pub struct RosenbrockTableau {
    kind: RosenbrockKind,
    name: String,
    description: String,
    order: usize,
    gamma: f64,
    a: Vec<Vec<f64>>,
    coupling: Vec<Vec<f64>>,
    c: Vec<f64>,
    d: Vec<f64>,
    b: Vec<f64>,
    btilde: Option<Vec<f64>>,
    h: Vec<Vec<f64>>,
}

impl RosenbrockTableau {
    /// Stage-equation and coupling-matrix convention.
    pub fn kind(&self) -> RosenbrockKind {
        self.kind
    }
    /// Resource method name.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Method description and provenance.
    pub fn description(&self) -> &str {
        &self.description
    }
    /// Declared classical order, not a proof of the order conditions.
    pub fn order(&self) -> usize {
        self.order
    }
    /// Number of stages.
    pub fn stages(&self) -> usize {
        self.b.len()
    }
    /// Common diagonal Jacobian coefficient.
    pub fn gamma(&self) -> f64 {
        self.gamma
    }
    /// Strictly lower-triangular state-increment matrix `A`.
    pub fn a(&self) -> &[Vec<f64>] {
        &self.a
    }
    /// Increment-coupling matrix `C`, interpreted according to [`Self::kind`].
    ///
    /// Strictly lower triangular for Rosenbrock methods; a raw lower-triangular
    /// `Gamma` with diagonal [`Self::gamma`] for hybrid methods.
    pub fn coupling(&self) -> &[Vec<f64>] {
        &self.coupling
    }
    /// Stage time nodes.
    pub fn c(&self) -> &[f64] {
        &self.c
    }
    /// Explicit time-derivative weights.
    pub fn d(&self) -> &[f64] {
        &self.d
    }
    /// Solution increment weights.
    pub fn b(&self) -> &[f64] {
        &self.b
    }
    /// Direct embedded-error increment weights, absent for fixed-step methods.
    pub fn btilde(&self) -> Option<&[f64]> {
        self.btilde.as_deref()
    }
    /// Stiff dense-output correction rows `H`; empty selects Hermite fallback.
    ///
    /// The correction to linear endpoint interpolation is
    /// `theta*(1-theta)*(H[0]*k + theta*(H[1]*k + ...))`.
    pub fn h(&self) -> &[Vec<f64>] {
        &self.h
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRosenbrockTableau {
    #[serde(rename = "$schema", default)]
    _schema: Option<String>,
    name: String,
    description: String,
    kind: RosenbrockKind,
    order: usize,
    gamma: Scalar,
    #[serde(rename = "A")]
    a: Vec<Vec<Scalar>>,
    #[serde(rename = "C")]
    coupling: Vec<Vec<Scalar>>,
    c: Vec<Scalar>,
    d: Vec<Scalar>,
    b: Vec<Scalar>,
    btilde: Option<Vec<Scalar>>,
    #[serde(rename = "H", default)]
    h: Vec<Vec<Scalar>>,
}

/// Parses a canonical JSON Rosenbrock tableau with the shared scalar parser.
///
/// Checks metadata, nonzero finite gamma, finite coefficients, square causal
/// stage matrices, matching vectors, an initial zero time node, nonzero
/// solution/error weights, and rectangular dense-output rows. The stiff
/// interpolant supports two through four correction rows (or none).
/// Hybrid tableaus additionally require matching Gamma diagonals and consistent
/// solution/error weight sums. Nodes `c` and time-derivative weights `d` are
/// independent input vectors, checked for dimension and finiteness only.
/// Neither classical order conditions nor stability are proved here.
/// Called identically during macro expansion and lazy runtime initialization.
pub fn parse_rosenbrock_tableau(
    source: &str,
    requested_name: &str,
) -> Result<RosenbrockTableau, TableauError> {
    let raw: RawRosenbrockTableau = serde_json::from_str(source)
        .map_err(|error| TableauError::new(format!("invalid Rosenbrock tableau JSON: {error}")))?;
    if raw.name.trim().is_empty() || raw.name != requested_name {
        return Err(TableauError::new(format!(
            "resource method `{}` does not match requested method `{requested_name}`",
            raw.name
        )));
    }
    if raw.description.trim().is_empty() || raw.order == 0 {
        return Err(TableauError::new(
            "Rosenbrock tableau requires a description and positive order",
        ));
    }
    let gamma = raw.gamma.materialize()?;
    if gamma == 0.0 {
        return Err(TableauError::new("gamma must be nonzero"));
    }
    let a = materialize_matrix(&raw.a, "A")?;
    let coupling = materialize_matrix(&raw.coupling, "C")?;
    let c = materialize_vector(&raw.c, "c")?;
    let d = materialize_vector(&raw.d, "d")?;
    let b = materialize_vector(&raw.b, "b")?;
    let btilde = raw
        .btilde
        .as_ref()
        .map(|row| materialize_vector(row, "btilde"))
        .transpose()?;
    let h = materialize_matrix(&raw.h, "H")?;
    let stages = b.len();
    if stages == 0 || b.iter().all(|value| *value == 0.0) {
        return Err(TableauError::new("b must contain nonzero solution weights"));
    }
    for (label, matrix) in [("A", &a), ("C", &coupling)] {
        if matrix.len() != stages || matrix.iter().any(|row| row.len() != stages) {
            return Err(TableauError::new(format!(
                "{label} must be square with {stages} stages"
            )));
        }
        let diagonal_allowed = label == "C" && raw.kind == RosenbrockKind::HybridExplicitImplicit;
        if matrix.iter().enumerate().any(|(i, row)| {
            row[i + usize::from(diagonal_allowed)..]
                .iter()
                .any(|value| *value != 0.0)
        }) {
            return Err(TableauError::new(format!(
                "{label} must be {}lower triangular",
                if diagonal_allowed { "" } else { "strictly " }
            )));
        }
    }
    for (label, row) in [("c", &c), ("d", &d)] {
        if row.len() != stages {
            return Err(TableauError::new(format!(
                "{label} must have {stages} entries"
            )));
        }
    }
    if c[0] != 0.0 {
        return Err(TableauError::new("the first stage node c[0] must be zero"));
    }
    if raw.kind == RosenbrockKind::HybridExplicitImplicit {
        for (stage, row) in coupling.iter().enumerate() {
            if row[stage] != gamma {
                return Err(TableauError::new(format!(
                    "hybrid C[{stage}][{stage}] must equal gamma"
                )));
            }
        }
        if !approximately_equal(b.iter().sum::<f64>(), 1.0)
            || btilde
                .as_ref()
                .is_some_and(|error| !approximately_equal(error.iter().sum::<f64>(), 0.0))
        {
            return Err(TableauError::new(
                "hybrid b must sum to one and btilde must sum to zero",
            ));
        }
    }
    if let Some(error) = &btilde {
        if error.len() != stages || error.iter().all(|value| *value == 0.0) {
            return Err(TableauError::new(
                "btilde must match the stages and contain a nonzero weight; omit it for fixed-step methods",
            ));
        }
    }
    if (!h.is_empty() && !(2..=4).contains(&h.len())) || h.iter().any(|row| row.len() != stages) {
        return Err(TableauError::new(
            "H must have zero or two through four rows, each with one entry per stage",
        ));
    }
    Ok(RosenbrockTableau {
        kind: raw.kind,
        name: raw.name,
        description: raw.description,
        order: raw.order,
        gamma,
        a,
        coupling,
        c,
        d,
        b,
        btilde,
        h,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: &str = r#"{"name":"Test","description":"Test Rosenbrock formula","kind":"rosenbrock","order":2,"gamma":"1/2","A":[[0,0],[2,0]],"C":[[0,0],[-4,0]],"c":[0,1],"d":[0.5,-0.5],"b":[3,1],"btilde":[1,1],"H":[[1,0],[0,1]]}"#;

    #[test]
    fn parses_sci_ml_convention_with_shared_expressions() {
        let t = parse_rosenbrock_tableau(SOURCE, "Test").unwrap();
        assert_eq!(t.name(), "Test");
        assert_eq!(t.kind(), RosenbrockKind::Rosenbrock);
        assert_eq!(t.description(), "Test Rosenbrock formula");
        assert_eq!(t.order(), 2);
        assert_eq!(t.stages(), 2);
        assert_eq!(t.gamma(), 0.5);
        assert_eq!(t.a()[1], [2.0, 0.0]);
        assert_eq!(t.coupling()[1], [-4.0, 0.0]);
        assert_eq!(t.d(), &[0.5, -0.5]);
        assert_eq!(t.b(), &[3.0, 1.0]); // Rosenbrock weights need not sum to one.
        assert_eq!(t.btilde(), Some([1.0, 1.0].as_slice()));
        assert_eq!(t.h().len(), 2);
        let fixed = SOURCE
            .replace(",\"btilde\":[1,1]", "")
            .replace(",\"H\":[[1,0],[0,1]]", "");
        let t = parse_rosenbrock_tableau(&fixed, "Test").unwrap();
        assert!(t.btilde().is_none());
        assert!(t.h().is_empty());
        let explicit_null = SOURCE.replace("\"btilde\":[1,1]", "\"btilde\":null");
        assert!(
            parse_rosenbrock_tableau(&explicit_null, "Test")
                .unwrap()
                .btilde()
                .is_none()
        );
    }

    #[test]
    fn hybrid_convention_preserves_independent_nodes_and_time_weights() {
        let source = SOURCE
            .replace("\"rosenbrock\"", "\"hybrid-explicit-implicit\"")
            .replace("\"C\":[[0,0],[-4,0]]", "\"C\":[[0.5,0],[0,0.5]]")
            .replace("\"b\":[3,1]", "\"b\":[0.5,0.5]")
            .replace("\"btilde\":[1,1]", "\"btilde\":[1,-1]");
        let t = parse_rosenbrock_tableau(&source, "Test").unwrap();
        assert_eq!(t.kind(), RosenbrockKind::HybridExplicitImplicit);
        assert_eq!(t.coupling()[1], [0.0, 0.5]);
        // c and d are supplied independently, not silently recomputed from A/C.
        assert_eq!(t.a()[1].iter().sum::<f64>(), 2.0);
        assert_eq!(t.c()[1], 1.0);
        assert_eq!(t.d()[1], -0.5);
        for (from, to) in [
            ("\"C\":[[0.5,0],[0,0.5]]", "\"C\":[[0.5,1],[0,0.5]]"),
            ("\"C\":[[0.5,0],[0,0.5]]", "\"C\":[[0.5,0],[0,0]]"),
            ("\"C\":[[0.5,0],[0,0.5]]", "\"C\":[[0.5,0],[0,\"1/0\"]]"),
            ("\"b\":[0.5,0.5]", "\"b\":[0.5,1]"),
            ("\"btilde\":[1,-1]", "\"btilde\":[1,1]"),
            ("\"btilde\":[1,-1]", "\"btilde\":[1]"),
            ("\"btilde\":[1,-1]", "\"btilde\":[0,0]"),
            ("\"d\":[0.5,-0.5]", "\"d\":[0.5,\"1/0\"]"),
        ] {
            let invalid = source.replace(from, to);
            assert!(
                parse_rosenbrock_tableau(&invalid, "Test").is_err(),
                "accepted {invalid}"
            );
        }
        assert!(
            parse_rosenbrock_tableau(
                &source.replace("hybrid-explicit-implicit", "rosenbrock"),
                "Test"
            )
            .is_err()
        );
    }

    #[test]
    fn numeric_coefficients_match_rust_decimal_rounding() {
        for decimal in [
            "1.7071067811865475",
            "0.8786796564403574",
            "0.2928932188134525",
            "-1.7071067811865475",
            "1.2345678901234567e-300",
        ] {
            let numeric: Scalar = serde_json::from_str(decimal).unwrap();
            let text = Scalar::Text(decimal.to_owned());
            assert_eq!(
                numeric.materialize().unwrap().to_bits(),
                text.materialize().unwrap().to_bits(),
                "{decimal}"
            );
        }
    }

    #[test]
    fn rejects_malformed_metadata_coefficients_and_dimensions() {
        assert!(parse_rosenbrock_tableau(SOURCE, "Other").is_err());
        for (from, to) in [
            ("\"name\":\"Test\"", "\"name\":\"\""),
            ("Test Rosenbrock formula", " "),
            ("\"order\":2", "\"order\":0"),
            ("\"kind\":\"rosenbrock\"", "\"kind\":\"implicit\""),
            ("\"order\":2", "\"order\":2,\"schema_version\":1"),
            ("\"order\":2", "\"order\":2,\"$schema\":3"),
            ("\"1/2\"", "0"),
            ("\"1/2\"", "\"1/0\""),
            ("\"1/2\"", "\"sqrt(-1)\""),
            ("\"1/2\"", "\"x + 1\""),
            ("\"A\":[[0,0],[2,0]]", "\"A\":[[0,0]]"),
            ("\"A\":[[0,0],[2,0]]", "\"A\":[[0,1],[2,0]]"),
            ("\"C\":[[0,0],[-4,0]]", "\"C\":[[0,0],[-4,1]]"),
            ("\"C\":[[0,0],[-4,0]]", "\"C\":[[0,0],[-4]]"),
            ("\"c\":[0,1]", "\"c\":[1,1]"),
            ("\"c\":[0,1]", "\"c\":[0]"),
            ("\"d\":[0.5,-0.5]", "\"d\":[]"),
            ("\"b\":[3,1]", "\"b\":[0,0]"),
            ("\"b\":[3,1]", "\"b\":[]"),
            ("\"btilde\":[1,1]", "\"btilde\":[0,0]"),
            ("\"btilde\":[1,1]", "\"btilde\":[1]"),
            ("\"H\":[[1,0],[0,1]]", "\"H\":[[1,0]]"),
            ("\"H\":[[1,0],[0,1]]", "\"H\":[[1,0],[0]]"),
            ("\"H\":[[1,0],[0,1]]", "\"H\":[[1,0],[0,\"1/0\"]]"),
        ] {
            let bad = SOURCE.replace(from, to);
            assert!(
                parse_rosenbrock_tableau(&bad, "Test").is_err(),
                "accepted {bad}"
            );
        }
    }
}
