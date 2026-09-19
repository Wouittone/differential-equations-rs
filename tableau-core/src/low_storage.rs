use serde::Deserialize;

use super::{Scalar, TableauError, materialize_matrix, materialize_vector};

const DEFAULT_CONSISTENCY_TOLERANCE: f64 = 1.0e-10;
const MAX_CONSISTENCY_TOLERANCE: f64 = 1.0e-6;

/// Coefficients shared by the 2N and 2C low-storage recurrences.
///
/// Both layouts evaluate an initial stage at zero. Consequently, `b` has one
/// entry per stage while `A` and `c` omit that initial stage.
#[derive(Clone, Debug, PartialEq)]
pub struct LowStorageAbcTableau {
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
}

impl LowStorageAbcTableau {
    /// Number of derivative stages evaluated by the recurrence.
    pub fn stages(&self) -> usize {
        self.b.len()
    }

    /// Recurrence multipliers, one for every stage after the initial stage.
    pub fn a(&self) -> &[f64] {
        &self.a
    }

    /// Solution-update weights, including the initial-stage weight.
    pub fn b(&self) -> &[f64] {
        &self.b
    }

    /// Evaluation nodes for every stage after the initial stage.
    pub fn c(&self) -> &[f64] {
        &self.c
    }
}

/// Coefficients for a 3S or 3S-plus low-storage recurrence.
#[derive(Clone, Debug, PartialEq)]
pub struct ThreeSTableau {
    gamma1: Vec<f64>,
    gamma2: Vec<f64>,
    gamma3: Vec<f64>,
    delta: Vec<f64>,
    beta1: f64,
    beta2: Vec<f64>,
    c: Vec<f64>,
    endpoint_evaluation: LowStorageEndpointEvaluation,
}

impl ThreeSTableau {
    /// Number of derivative stages used to form the step solution.
    pub fn stages(&self) -> usize {
        self.gamma1.len() + 1
    }

    /// First register-combination weights.
    pub fn gamma1(&self) -> &[f64] {
        &self.gamma1
    }

    /// Second register-combination weights.
    pub fn gamma2(&self) -> &[f64] {
        &self.gamma2
    }

    /// Initial-state register-combination weights.
    pub fn gamma3(&self) -> &[f64] {
        &self.gamma3
    }

    /// Accumulation weights for the temporary register.
    pub fn delta(&self) -> &[f64] {
        &self.delta
    }

    /// Initial derivative weight.
    pub fn beta1(&self) -> f64 {
        self.beta1
    }

    /// Derivative weights for the remaining stages.
    pub fn beta2(&self) -> &[f64] {
        &self.beta2
    }

    /// Evaluation nodes for every stage after the initial stage.
    pub fn c(&self) -> &[f64] {
        &self.c
    }

    /// Policy for evaluating an additional endpoint derivative.
    pub fn endpoint_evaluation(&self) -> LowStorageEndpointEvaluation {
        self.endpoint_evaluation
    }
}

/// The two 2N recurrences alternated between accepted steps.
#[derive(Clone, Debug, PartialEq)]
pub struct AlternatingTwoNTableau {
    first: LowStorageAbcTableau,
    second: LowStorageAbcTableau,
}

impl AlternatingTwoNTableau {
    /// Recurrence used on the first and every other accepted step.
    pub fn first(&self) -> &LowStorageAbcTableau {
        &self.first
    }

    /// Recurrence used on alternating accepted steps.
    pub fn second(&self) -> &LowStorageAbcTableau {
        &self.second
    }
}

/// Coefficients for a register-pipeline low-storage recurrence.
#[derive(Clone, Debug, PartialEq)]
pub struct RegisterPipelineTableau {
    history_states: usize,
    a: Vec<Vec<f64>>,
    b: Vec<f64>,
    b_final: f64,
    c: Vec<f64>,
}

impl RegisterPipelineTableau {
    /// Number of derivative stages used to form the step solution.
    pub fn stages(&self) -> usize {
        self.c.len() + 1
    }

    /// Number of rolling state registers retained by the recurrence.
    pub fn history_states(&self) -> usize {
        self.history_states
    }

    /// Register-pipeline derivative coefficients.
    ///
    /// Rows correspond to the current derivative followed by rolling
    /// derivative registers. Columns correspond to non-initial stages.
    pub fn a(&self) -> &[Vec<f64>] {
        &self.a
    }

    /// Solution weights applied before the final derivative.
    pub fn b(&self) -> &[f64] {
        &self.b
    }

    /// Solution weight applied to the final derivative.
    pub fn b_final(&self) -> f64 {
        self.b_final
    }

    /// Evaluation nodes for every stage after the initial stage.
    pub fn c(&self) -> &[f64] {
        &self.c
    }
}

/// Validated recurrence layout for a low-storage Runge--Kutta method.
#[derive(Clone, Debug, PartialEq)]
pub enum LowStorageRungeKuttaLayout {
    /// Williamson-style two-register recurrence.
    TwoN(LowStorageAbcTableau),
    /// Two-register recurrence with a copied stage state.
    TwoC(LowStorageAbcTableau),
    /// Three-register 3S or 3S-plus recurrence.
    ThreeS(ThreeSTableau),
    /// A pair of 2N recurrences alternated between accepted steps.
    AlternatingTwoN(AlternatingTwoNTableau),
    /// Rolling register-pipeline recurrence.
    RegisterPipeline(RegisterPipelineTableau),
}

/// Relationship between declared stage times and reconstructed stage states.
///
/// Classical recurrences derive every node from the corresponding effective
/// Runge--Kutta row. A few published low-storage formulas provide independent
/// nodes and are therefore primarily intended for autonomous equations.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LowStorageNodePolicy {
    /// Validate each node against the reconstructed stage-row sum.
    #[default]
    Derived,
    /// Preserve independently supplied stage times without a row-sum check.
    Independent,
}

/// Endpoint-derivative work encoded by a 3S recurrence resource.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LowStorageEndpointEvaluation {
    /// Finish after the derivative stages used to update the solution.
    Omit,
    /// Evaluate one additional derivative at the completed step endpoint.
    Evaluate,
}

impl LowStorageRungeKuttaLayout {
    /// Number of derivative stages in the primary recurrence.
    pub fn stages(&self) -> usize {
        match self {
            Self::TwoN(tableau) | Self::TwoC(tableau) => tableau.stages(),
            Self::ThreeS(tableau) => tableau.stages(),
            Self::AlternatingTwoN(tableau) => tableau.first().stages(),
            Self::RegisterPipeline(tableau) => tableau.stages(),
        }
    }

    /// Number of stages in the alternate recurrence, when one is present.
    pub fn alternate_stages(&self) -> Option<usize> {
        match self {
            Self::AlternatingTwoN(tableau) => Some(tableau.second().stages()),
            _ => None,
        }
    }
}

/// A validated low-storage Runge--Kutta resource.
#[derive(Clone, Debug, PartialEq)]
pub struct LowStorageRungeKuttaTableau {
    name: String,
    description: String,
    order: usize,
    consistency_tolerance: f64,
    node_policy: LowStorageNodePolicy,
    embedded: Option<LowStorageEmbeddedTableau>,
    layout: LowStorageRungeKuttaLayout,
}

/// Embedded local-error formula attached to a low-storage recurrence.
///
/// The weights multiply derivative stages in evaluation order and already
/// represent the difference used by the estimator. An endpoint derivative,
/// when present, is the final entry.
#[derive(Clone, Debug, PartialEq)]
pub struct LowStorageEmbeddedTableau {
    order: usize,
    error: Vec<f64>,
    controller: LowStorageAdaptiveController,
}

impl LowStorageEmbeddedTableau {
    /// Order of the embedded companion formula.
    pub fn order(&self) -> usize {
        self.order
    }

    /// Error-difference weights in derivative-evaluation order.
    pub fn error(&self) -> &[f64] {
        &self.error
    }

    /// Step-size controller policy supplied by the resource.
    pub fn controller(&self) -> LowStorageAdaptiveController {
        self.controller
    }
}

/// Adaptive controller policy for a low-storage embedded pair.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum LowStorageAdaptiveController {
    /// Use the library's standard PI policy for the method order.
    #[default]
    StandardPi,
    /// Use the resource's optimized PID policy.
    Pid(LowStoragePidController),
}

/// Coefficients for a method-specific PID step-size controller.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LowStoragePidController {
    beta: [f64; 3],
    acceptance_safety: f64,
}

impl LowStoragePidController {
    /// PID filter coefficients before division by the method order.
    pub fn beta(self) -> [f64; 3] {
        self.beta
    }

    /// Minimum filtered step ratio for accepting an attempted step.
    pub fn acceptance_safety(self) -> f64 {
        self.acceptance_safety
    }
}

impl LowStorageRungeKuttaTableau {
    /// Resource method name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Human-readable method description and provenance.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Classical order claimed by the resource.
    pub fn order(&self) -> usize {
        self.order
    }

    /// Relative tolerance used to validate reconstructed recurrence identities.
    ///
    /// Full-precision resources use `1e-10`. A resource may explicitly request
    /// a value no larger than `1e-6` when its published coefficients are
    /// precision-limited decimals.
    pub fn consistency_tolerance(&self) -> f64 {
        self.consistency_tolerance
    }

    /// How declared stage-time nodes relate to the recurrence coefficients.
    pub fn node_policy(&self) -> LowStorageNodePolicy {
        self.node_policy
    }

    /// Returns the embedded local-error formula, when the method is adaptive.
    pub fn embedded(&self) -> Option<&LowStorageEmbeddedTableau> {
        self.embedded.as_ref()
    }

    /// Returns whether an accepted endpoint derivative can seed the next step.
    pub fn fsal(&self) -> bool {
        match &self.layout {
            LowStorageRungeKuttaLayout::ThreeS(tableau) => {
                tableau.endpoint_evaluation() == LowStorageEndpointEvaluation::Evaluate
            }
            LowStorageRungeKuttaLayout::AlternatingTwoN(_)
            | LowStorageRungeKuttaLayout::RegisterPipeline(_) => true,
            LowStorageRungeKuttaLayout::TwoN(_) | LowStorageRungeKuttaLayout::TwoC(_) => false,
        }
    }

    /// Validated recurrence coefficients.
    pub fn layout(&self) -> &LowStorageRungeKuttaLayout {
        &self.layout
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum RawKind {
    LowStorageRungeKutta,
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum RawNodePolicy {
    #[default]
    Derived,
    Independent,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum RawEndpointEvaluation {
    Omit,
    Evaluate,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawEmbedded {
    order: usize,
    error: Option<Vec<Scalar>>,
    b_hat: Option<Vec<Scalar>>,
    b_hat_final: Option<Scalar>,
    controller: Option<RawAdaptiveController>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum RawAdaptiveController {
    Pid {
        beta: Vec<Scalar>,
        acceptance_safety: Scalar,
    },
}

impl From<RawEndpointEvaluation> for LowStorageEndpointEvaluation {
    fn from(value: RawEndpointEvaluation) -> Self {
        match value {
            RawEndpointEvaluation::Omit => Self::Omit,
            RawEndpointEvaluation::Evaluate => Self::Evaluate,
        }
    }
}

impl From<RawNodePolicy> for LowStorageNodePolicy {
    fn from(value: RawNodePolicy) -> Self {
        match value {
            RawNodePolicy::Derived => Self::Derived,
            RawNodePolicy::Independent => Self::Independent,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum RawLayout {
    TwoN,
    TwoC,
    ThreeS,
    AlternatingTwoN,
    RegisterPipeline,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RawCoefficientArray {
    Vector(Vec<Scalar>),
    Matrix(Vec<Vec<Scalar>>),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTableau {
    #[serde(rename = "$schema", default)]
    _schema: Option<String>,
    name: String,
    description: String,
    #[serde(rename = "kind")]
    _kind: RawKind,
    layout: RawLayout,
    order: usize,
    consistency_tolerance: Option<Scalar>,
    #[serde(default)]
    node_policy: RawNodePolicy,
    #[serde(rename = "A")]
    a: Option<RawCoefficientArray>,
    b: Option<Vec<Scalar>>,
    c: Option<Vec<Scalar>>,
    gamma1: Option<Vec<Scalar>>,
    gamma2: Option<Vec<Scalar>>,
    gamma3: Option<Vec<Scalar>>,
    delta: Option<Vec<Scalar>>,
    beta1: Option<Scalar>,
    beta2: Option<Vec<Scalar>>,
    endpoint_evaluation: Option<RawEndpointEvaluation>,
    #[serde(rename = "A1")]
    a1: Option<Vec<Scalar>>,
    b1: Option<Vec<Scalar>>,
    c1: Option<Vec<Scalar>>,
    #[serde(rename = "A2")]
    a2: Option<Vec<Scalar>>,
    b2: Option<Vec<Scalar>>,
    c2: Option<Vec<Scalar>>,
    history_states: Option<usize>,
    b_final: Option<Scalar>,
    embedded: Option<RawEmbedded>,
}

/// Parses and validates a canonical low-storage Runge--Kutta JSON resource.
///
/// Besides rejecting malformed or non-finite coefficient data, validation
/// symbolically reconstructs every stage state. It verifies that each state is
/// affine in the initial value, that each declared node is the row sum of its
/// equivalent explicit Runge--Kutta stage, and that the final update is
/// first-order consistent. These checks cover every recurrence consumed by the
/// solver without embedding method-specific coefficients in Rust.
pub fn parse_low_storage_tableau(
    source: &str,
    requested_name: &str,
) -> Result<LowStorageRungeKuttaTableau, TableauError> {
    let raw: RawTableau = serde_json::from_str(source)
        .map_err(|error| TableauError::json("low-storage Runge--Kutta tableau", error))?;
    validate_metadata(&raw, requested_name)?;
    let consistency_tolerance = raw
        .consistency_tolerance
        .as_ref()
        .map(Scalar::materialize)
        .transpose()?
        .unwrap_or(DEFAULT_CONSISTENCY_TOLERANCE);
    if consistency_tolerance <= 0.0 || consistency_tolerance > MAX_CONSISTENCY_TOLERANCE {
        return Err(TableauError::name_mismatch(format!(
            "low-storage consistency_tolerance must be positive and no greater than {MAX_CONSISTENCY_TOLERANCE}"
        )));
    }
    let node_policy = raw.node_policy.into();

    validate_layout_fields(&raw)?;
    let layout = match raw.layout {
        RawLayout::TwoN => {
            let tableau = materialize_abc(
                required_flattened_vector(raw.a, "A")?,
                required(raw.b, "b")?,
                required(raw.c, "c")?,
                "2N",
            )?;
            validate_two_n(&tableau, "2N", node_policy, consistency_tolerance)?;
            LowStorageRungeKuttaLayout::TwoN(tableau)
        }
        RawLayout::TwoC => {
            let tableau = materialize_abc(
                required_flattened_vector(raw.a, "A")?,
                required(raw.b, "b")?,
                required(raw.c, "c")?,
                "2C",
            )?;
            validate_two_c(&tableau, "2C", node_policy, consistency_tolerance)?;
            LowStorageRungeKuttaLayout::TwoC(tableau)
        }
        RawLayout::ThreeS => {
            let tableau = ThreeSTableau {
                gamma1: materialize_vector(&required(raw.gamma1, "gamma1")?, "gamma1")?,
                gamma2: materialize_vector(&required(raw.gamma2, "gamma2")?, "gamma2")?,
                gamma3: materialize_vector(&required(raw.gamma3, "gamma3")?, "gamma3")?,
                delta: materialize_vector(&required(raw.delta, "delta")?, "delta")?,
                beta1: required(raw.beta1, "beta1")?.materialize()?,
                beta2: materialize_vector(&required(raw.beta2, "beta2")?, "beta2")?,
                c: materialize_vector(&required(raw.c, "c")?, "c")?,
                endpoint_evaluation: required(raw.endpoint_evaluation, "endpoint_evaluation")?
                    .into(),
            };
            validate_three_s(&tableau, node_policy, consistency_tolerance)?;
            LowStorageRungeKuttaLayout::ThreeS(tableau)
        }
        RawLayout::AlternatingTwoN => {
            let first = materialize_abc(
                required(raw.a1, "A1")?,
                required(raw.b1, "b1")?,
                required(raw.c1, "c1")?,
                "alternating 2N first",
            )?;
            let second = materialize_abc(
                required(raw.a2, "A2")?,
                required(raw.b2, "b2")?,
                required(raw.c2, "c2")?,
                "alternating 2N second",
            )?;
            validate_two_n(
                &first,
                "alternating 2N first",
                node_policy,
                consistency_tolerance,
            )?;
            validate_two_n(
                &second,
                "alternating 2N second",
                node_policy,
                consistency_tolerance,
            )?;
            LowStorageRungeKuttaLayout::AlternatingTwoN(AlternatingTwoNTableau { first, second })
        }
        RawLayout::RegisterPipeline => {
            let tableau = RegisterPipelineTableau {
                history_states: required(raw.history_states, "history_states")?,
                a: materialize_matrix(&required_matrix(raw.a, "A")?, "A")?,
                b: materialize_vector(&required(raw.b, "b")?, "b")?,
                b_final: required(raw.b_final, "b_final")?.materialize()?,
                c: materialize_vector(&required(raw.c, "c")?, "c")?,
            };
            validate_register_pipeline(&tableau, node_policy, consistency_tolerance)?;
            LowStorageRungeKuttaLayout::RegisterPipeline(tableau)
        }
    };

    let stages = layout.stages();
    if raw.order > stages
        || layout
            .alternate_stages()
            .is_some_and(|alternate| raw.order > alternate)
    {
        return Err(TableauError::new(
            "low-storage Runge--Kutta order cannot exceed either recurrence's stage count",
        ));
    }

    let embedded = raw
        .embedded
        .map(|embedded| materialize_embedded(embedded, raw.order, &layout, consistency_tolerance))
        .transpose()?;

    Ok(LowStorageRungeKuttaTableau {
        name: raw.name,
        description: raw.description,
        order: raw.order,
        consistency_tolerance,
        node_policy,
        embedded,
        layout,
    })
}

fn materialize_embedded(
    raw: RawEmbedded,
    primary_order: usize,
    layout: &LowStorageRungeKuttaLayout,
    consistency_tolerance: f64,
) -> Result<LowStorageEmbeddedTableau, TableauError> {
    if raw.order == 0 || raw.order >= primary_order {
        return Err(TableauError::new(
            "low-storage embedded order must be positive and lower than the primary order",
        ));
    }
    if !matches!(
        layout,
        LowStorageRungeKuttaLayout::ThreeS(_) | LowStorageRungeKuttaLayout::RegisterPipeline(_)
    ) {
        return Err(TableauError::new(
            "embedded error formulas are only supported by 3S-plus and register-pipeline layouts",
        ));
    }

    let error = match layout {
        LowStorageRungeKuttaLayout::ThreeS(_) => {
            if raw.b_hat.is_some() || raw.b_hat_final.is_some() {
                return Err(TableauError::new(
                    "3S-plus embedded formulas require direct `error` weights",
                ));
            }
            materialize_vector(
                &raw.error.ok_or_else(|| {
                    TableauError::new("3S-plus embedded formulas require `error` weights")
                })?,
                "embedded error",
            )?
        }
        LowStorageRungeKuttaLayout::RegisterPipeline(tableau) => {
            match (raw.error, raw.b_hat, raw.b_hat_final) {
                (Some(error), None, None) => materialize_vector(&error, "embedded error")?,
                (None, Some(b_hat), Some(b_hat_final)) => {
                    let b_hat = materialize_vector(&b_hat, "embedded b_hat")?;
                    if b_hat.len() != tableau.b().len() {
                        return Err(TableauError::new(format!(
                            "embedded b_hat needs {} weights",
                            tableau.b().len()
                        )));
                    }
                    let mut error = tableau
                        .b()
                        .iter()
                        .zip(b_hat)
                        .map(|(b, b_hat)| b - b_hat)
                        .collect::<Vec<_>>();
                    error.push(tableau.b_final() - b_hat_final.materialize()?);
                    error
                }
                _ => {
                    return Err(TableauError::new(
                        "register-pipeline embedded formulas require either `error`, or both `b_hat` and `b_hat_final`",
                    ));
                }
            }
        }
        LowStorageRungeKuttaLayout::TwoN(_)
        | LowStorageRungeKuttaLayout::TwoC(_)
        | LowStorageRungeKuttaLayout::AlternatingTwoN(_) => unreachable!(),
    };
    let stages = layout.stages();
    let endpoint_available = matches!(
        layout,
        LowStorageRungeKuttaLayout::ThreeS(tableau)
            if tableau.endpoint_evaluation() == LowStorageEndpointEvaluation::Evaluate
    ) || matches!(layout, LowStorageRungeKuttaLayout::RegisterPipeline(_));
    if error.len() != stages && !(endpoint_available && error.len() == stages + 1) {
        return Err(TableauError::new(format!(
            "embedded error needs {stages} stage weights{}",
            if endpoint_available {
                " or one additional endpoint weight"
            } else {
                ""
            }
        )));
    }
    let sum: f64 = error.iter().sum();
    if !approximately_equal_recurrence(sum, 0.0, consistency_tolerance) {
        return Err(TableauError::new(format!(
            "embedded error weights must sum to zero; found {sum}"
        )));
    }

    let controller = match raw.controller {
        None => LowStorageAdaptiveController::StandardPi,
        Some(RawAdaptiveController::Pid {
            beta,
            acceptance_safety,
        }) => {
            let beta = materialize_vector(&beta, "PID beta")?;
            let beta: [f64; 3] = beta.try_into().map_err(|_| {
                TableauError::new("low-storage PID controller requires exactly three beta values")
            })?;
            let acceptance_safety = acceptance_safety.materialize()?;
            if !(acceptance_safety.is_finite()
                && acceptance_safety > 0.0
                && acceptance_safety <= 1.0)
            {
                return Err(TableauError::new(
                    "low-storage PID acceptance_safety must be in (0, 1]",
                ));
            }
            LowStorageAdaptiveController::Pid(LowStoragePidController {
                beta,
                acceptance_safety,
            })
        }
    };

    Ok(LowStorageEmbeddedTableau {
        order: raw.order,
        error,
        controller,
    })
}

fn required<T>(value: Option<T>, field: &str) -> Result<T, TableauError> {
    value.ok_or_else(|| TableauError::new(format!("layout requires field `{field}`")))
}

fn required_flattened_vector(
    value: Option<RawCoefficientArray>,
    field: &str,
) -> Result<Vec<Scalar>, TableauError> {
    match required(value, field)? {
        RawCoefficientArray::Vector(values) => Ok(values),
        RawCoefficientArray::Matrix(_) => Err(TableauError::new(format!(
            "{field} must be a coefficient vector for this layout"
        ))),
    }
}

fn required_matrix(
    value: Option<RawCoefficientArray>,
    field: &str,
) -> Result<Vec<Vec<Scalar>>, TableauError> {
    match required(value, field)? {
        RawCoefficientArray::Matrix(rows) => Ok(rows),
        RawCoefficientArray::Vector(_) => Err(TableauError::new(format!(
            "{field} must be a coefficient matrix for this layout"
        ))),
    }
}

fn validate_layout_fields(raw: &RawTableau) -> Result<(), TableauError> {
    let fields = [
        ("A", raw.a.is_some()),
        ("b", raw.b.is_some()),
        ("c", raw.c.is_some()),
        ("gamma1", raw.gamma1.is_some()),
        ("gamma2", raw.gamma2.is_some()),
        ("gamma3", raw.gamma3.is_some()),
        ("delta", raw.delta.is_some()),
        ("beta1", raw.beta1.is_some()),
        ("beta2", raw.beta2.is_some()),
        ("endpoint_evaluation", raw.endpoint_evaluation.is_some()),
        ("A1", raw.a1.is_some()),
        ("b1", raw.b1.is_some()),
        ("c1", raw.c1.is_some()),
        ("A2", raw.a2.is_some()),
        ("b2", raw.b2.is_some()),
        ("c2", raw.c2.is_some()),
        ("history_states", raw.history_states.is_some()),
        ("b_final", raw.b_final.is_some()),
    ];
    for (field, present) in fields {
        let allowed = match raw.layout {
            RawLayout::TwoN | RawLayout::TwoC => matches!(field, "A" | "b" | "c"),
            RawLayout::ThreeS => matches!(
                field,
                "gamma1"
                    | "gamma2"
                    | "gamma3"
                    | "delta"
                    | "beta1"
                    | "beta2"
                    | "c"
                    | "endpoint_evaluation"
            ),
            RawLayout::AlternatingTwoN => {
                matches!(field, "A1" | "b1" | "c1" | "A2" | "b2" | "c2")
            }
            RawLayout::RegisterPipeline => {
                matches!(field, "A" | "b" | "c" | "history_states" | "b_final")
            }
        };
        if present && !allowed {
            return Err(TableauError::new(format!(
                "field `{field}` is not valid for this low-storage layout"
            )));
        }
    }
    Ok(())
}

fn validate_metadata(raw: &RawTableau, requested_name: &str) -> Result<(), TableauError> {
    if raw.name.trim().is_empty() || raw.name != requested_name {
        return Err(TableauError::name_mismatch(format!(
            "resource method `{}` does not match requested method `{requested_name}`",
            raw.name
        )));
    }
    if raw.description.trim().is_empty() || raw.order == 0 {
        return Err(TableauError::new(
            "low-storage Runge--Kutta tableau requires a description and positive order",
        ));
    }
    Ok(())
}

fn materialize_abc(
    a: Vec<Scalar>,
    b: Vec<Scalar>,
    c: Vec<Scalar>,
    label: &str,
) -> Result<LowStorageAbcTableau, TableauError> {
    Ok(LowStorageAbcTableau {
        a: materialize_vector(&a, &format!("{label} A"))?,
        b: materialize_vector(&b, &format!("{label} b"))?,
        c: materialize_vector(&c, &format!("{label} c"))?,
    })
}

#[derive(Clone)]
struct AffineState {
    base: f64,
    derivatives: Vec<f64>,
}

impl AffineState {
    fn initial() -> Self {
        Self {
            base: 1.0,
            derivatives: Vec::new(),
        }
    }

    fn zero() -> Self {
        Self {
            base: 0.0,
            derivatives: Vec::new(),
        }
    }

    fn coefficient_sum(&self) -> f64 {
        self.derivatives.iter().sum()
    }

    fn is_finite(&self) -> bool {
        self.base.is_finite() && self.derivatives.iter().all(|value| value.is_finite())
    }

    fn add_scaled(&mut self, other: &Self, scale: f64) {
        self.base += scale * other.base;
        if self.derivatives.len() < other.derivatives.len() {
            self.derivatives.resize(other.derivatives.len(), 0.0);
        }
        for (target, value) in self.derivatives.iter_mut().zip(&other.derivatives) {
            *target += scale * value;
        }
    }

    fn scale(&mut self, scale: f64) {
        self.base *= scale;
        for coefficient in &mut self.derivatives {
            *coefficient *= scale;
        }
    }

    fn add_derivative(&mut self, stage: usize, coefficient: f64) {
        if self.derivatives.len() <= stage {
            self.derivatives.resize(stage + 1, 0.0);
        }
        self.derivatives[stage] += coefficient;
    }
}

fn validate_abc_shape(tableau: &LowStorageAbcTableau, label: &str) -> Result<(), TableauError> {
    if tableau.b.is_empty()
        || tableau.a.len() + 1 != tableau.b.len()
        || tableau.c.len() != tableau.a.len()
    {
        return Err(TableauError::new(format!(
            "{label} requires non-empty b and equally sized A and c with b.len() = A.len() + 1"
        )));
    }
    Ok(())
}

fn validate_two_n(
    tableau: &LowStorageAbcTableau,
    label: &str,
    node_policy: LowStorageNodePolicy,
    consistency_tolerance: f64,
) -> Result<(), TableauError> {
    validate_abc_shape(tableau, label)?;
    let mut candidate = AffineState::initial();
    let mut residual = AffineState::zero();
    residual.add_derivative(0, 1.0);
    candidate.add_scaled(&residual, tableau.b[0]);
    for stage in 0..tableau.a.len() {
        validate_stage(
            &candidate,
            tableau.c[stage],
            stage + 1,
            label,
            node_policy,
            consistency_tolerance,
        )?;
        residual.scale(tableau.a[stage]);
        residual.add_derivative(stage + 1, 1.0);
        candidate.add_scaled(&residual, tableau.b[stage + 1]);
    }
    validate_solution(&candidate, label, consistency_tolerance)
}

fn validate_two_c(
    tableau: &LowStorageAbcTableau,
    label: &str,
    node_policy: LowStorageNodePolicy,
    consistency_tolerance: f64,
) -> Result<(), TableauError> {
    validate_abc_shape(tableau, label)?;
    let mut candidate = AffineState::initial();
    candidate.add_derivative(0, tableau.b[0]);
    for stage in 0..tableau.a.len() {
        let mut temporary = candidate.clone();
        temporary.add_derivative(stage, tableau.a[stage]);
        validate_stage(
            &temporary,
            tableau.c[stage],
            stage + 1,
            label,
            node_policy,
            consistency_tolerance,
        )?;
        candidate.add_derivative(stage + 1, tableau.b[stage + 1]);
    }
    validate_solution(&candidate, label, consistency_tolerance)
}

fn validate_three_s(
    tableau: &ThreeSTableau,
    node_policy: LowStorageNodePolicy,
    consistency_tolerance: f64,
) -> Result<(), TableauError> {
    let stages_after_initial = tableau.gamma1.len();
    if stages_after_initial == 0
        || tableau.gamma2.len() != stages_after_initial
        || tableau.gamma3.len() != stages_after_initial
        || tableau.delta.len() != stages_after_initial
        || tableau.beta2.len() != stages_after_initial
        || tableau.c.len() != stages_after_initial
    {
        return Err(TableauError::new(
            "3S coefficient vectors must have the same non-zero length",
        ));
    }

    let initial = AffineState::initial();
    let mut candidate = initial.clone();
    let mut temporary = initial.clone();
    candidate.add_derivative(0, tableau.beta1);
    for stage in 0..stages_after_initial {
        validate_stage(
            &candidate,
            tableau.c[stage],
            stage + 1,
            "3S",
            node_policy,
            consistency_tolerance,
        )?;
        temporary.add_scaled(&candidate, tableau.delta[stage]);
        candidate.scale(tableau.gamma1[stage]);
        candidate.add_scaled(&temporary, tableau.gamma2[stage]);
        candidate.add_scaled(&initial, tableau.gamma3[stage]);
        candidate.add_derivative(stage + 1, tableau.beta2[stage]);
    }
    validate_solution(&candidate, "3S", consistency_tolerance)
}

fn validate_register_pipeline(
    tableau: &RegisterPipelineTableau,
    node_policy: LowStorageNodePolicy,
    consistency_tolerance: f64,
) -> Result<(), TableauError> {
    let columns = tableau.c.len();
    if columns == 0
        || tableau.history_states == 0
        || tableau.history_states > columns
        || tableau.a.len() != tableau.history_states
        || tableau.a.iter().any(|row| row.len() != columns)
        || tableau.b.len() != columns
    {
        return Err(TableauError::new(
            "register-pipeline A needs history_states non-empty rows, b and c need one entry per column, and history_states cannot exceed the column count",
        ));
    }

    let initial = AffineState::initial();
    let mut candidate = initial.clone();
    let mut history_states = vec![initial; tableau.history_states];
    let mut history_derivatives = vec![AffineState::zero(); tableau.history_states - 1];
    for stage in 0..columns {
        let mut stage_state = history_states[tableau.history_states - 1].clone();
        stage_state.add_derivative(stage, tableau.a[0][stage]);
        for (register, coefficients) in history_derivatives.iter().zip(&tableau.a[1..]) {
            stage_state.add_scaled(register, coefficients[stage]);
        }
        validate_stage(
            &stage_state,
            tableau.c[stage],
            stage + 1,
            "register-pipeline",
            node_policy,
            consistency_tolerance,
        )?;

        candidate.add_derivative(stage, tableau.b[stage]);
        if !history_derivatives.is_empty() {
            history_derivatives.rotate_right(1);
        }
        if let Some(current) = history_derivatives.first_mut() {
            *current = AffineState::zero();
            current.add_derivative(stage, 1.0);
        }
        history_states.rotate_right(1);
        history_states[0] = candidate.clone();
    }
    candidate.add_derivative(columns, tableau.b_final);
    validate_solution(&candidate, "register-pipeline", consistency_tolerance)
}

fn validate_stage(
    stage: &AffineState,
    expected_node: f64,
    stage_number: usize,
    label: &str,
    node_policy: LowStorageNodePolicy,
    consistency_tolerance: f64,
) -> Result<(), TableauError> {
    if !stage.is_finite() {
        return Err(TableauError::new(format!(
            "{label} stage {stage_number} reconstruction is not finite"
        )));
    }
    if !approximately_equal_recurrence(stage.base, 1.0, consistency_tolerance) {
        return Err(TableauError::new(format!(
            "{label} stage {stage_number} is not affine in the initial state"
        )));
    }
    let actual_node = stage.coefficient_sum();
    if node_policy == LowStorageNodePolicy::Derived
        && !approximately_equal_recurrence(actual_node, expected_node, consistency_tolerance)
    {
        return Err(TableauError::new(format!(
            "{label} c[{}] must equal the equivalent Runge--Kutta stage-row sum; found {expected_node}, expected {actual_node}",
            stage_number - 1
        )));
    }
    Ok(())
}

fn approximately_equal_recurrence(left: f64, right: f64, tolerance: f64) -> bool {
    let scale = left.abs().max(right.abs()).max(1.0);
    left.is_finite() && right.is_finite() && (left - right).abs() <= tolerance * scale
}

fn validate_solution(
    solution: &AffineState,
    label: &str,
    consistency_tolerance: f64,
) -> Result<(), TableauError> {
    if !solution.is_finite() {
        return Err(TableauError::new(format!(
            "{label} final recurrence reconstruction is not finite"
        )));
    }
    if !approximately_equal_recurrence(solution.base, 1.0, consistency_tolerance) {
        return Err(TableauError::new(format!(
            "{label} final update is not affine in the initial state"
        )));
    }
    let weight_sum = solution.coefficient_sum();
    if !approximately_equal_recurrence(weight_sum, 1.0, consistency_tolerance) {
        return Err(TableauError::new(format!(
            "{label} equivalent Runge--Kutta weights must sum to one; found {weight_sum}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TWO_N: &str = r#"{"name":"Williamson2","description":"two-stage 2N method","kind":"low-storage-runge-kutta","layout":"two-n","order":2,"A":["-1/2"],"b":["1/2",1],"c":["1/2"]}"#;
    const EULER_2N: &str = r#"{"name":"Euler2N","description":"one-stage 2N method","kind":"low-storage-runge-kutta","layout":"two-n","order":1,"A":[],"b":[1],"c":[]}"#;
    const TWO_C: &str = r#"{"name":"Midpoint2C","description":"two-stage 2C method","kind":"low-storage-runge-kutta","layout":"two-c","order":2,"A":["1/2"],"b":[0,1],"c":["1/2"]}"#;
    const THREE_S: &str = r#"{"name":"Euler3S","description":"one-stage 3S recurrence","kind":"low-storage-runge-kutta","layout":"three-s","order":1,"gamma1":[0],"gamma2":[0],"gamma3":[1],"delta":[0],"beta1":0,"beta2":[1],"c":[0],"endpoint_evaluation":"omit"}"#;
    const ALTERNATING: &str = r#"{"name":"Alternating","description":"alternating midpoint recurrences","kind":"low-storage-runge-kutta","layout":"alternating-two-n","order":2,"A1":["-1/2"],"b1":["1/2",1],"c1":["1/2"],"A2":["-1/2"],"b2":["1/2",1],"c2":["1/2"]}"#;
    const PIPELINE: &str = r#"{"name":"Pipeline","description":"two-stage register pipeline","kind":"low-storage-runge-kutta","layout":"register-pipeline","order":2,"history_states":1,"A":[["1/2"]],"b":[0],"b_final":1,"c":["1/2"]}"#;

    #[test]
    fn parses_every_recurrence_layout() {
        let euler = parse_low_storage_tableau(EULER_2N, "Euler2N").unwrap();
        assert_eq!(euler.layout().stages(), 1);

        let two_n = parse_low_storage_tableau(TWO_N, "Williamson2").unwrap();
        assert_eq!(two_n.order(), 2);
        let LowStorageRungeKuttaLayout::TwoN(two_n) = two_n.layout() else {
            panic!("expected 2N layout")
        };
        assert_eq!(two_n.stages(), 2);
        assert_eq!(two_n.a(), [-0.5]);

        let two_c = parse_low_storage_tableau(TWO_C, "Midpoint2C").unwrap();
        assert!(matches!(
            two_c.layout(),
            LowStorageRungeKuttaLayout::TwoC(_)
        ));

        let three_s = parse_low_storage_tableau(THREE_S, "Euler3S").unwrap();
        let LowStorageRungeKuttaLayout::ThreeS(three_s) = three_s.layout() else {
            panic!("expected 3S layout")
        };
        assert_eq!(three_s.stages(), 2);
        assert_eq!(
            three_s.endpoint_evaluation(),
            LowStorageEndpointEvaluation::Omit
        );

        let alternating = parse_low_storage_tableau(ALTERNATING, "Alternating").unwrap();
        assert_eq!(alternating.layout().stages(), 2);
        assert_eq!(alternating.layout().alternate_stages(), Some(2));

        let pipeline = parse_low_storage_tableau(PIPELINE, "Pipeline").unwrap();
        let LowStorageRungeKuttaLayout::RegisterPipeline(pipeline) = pipeline.layout() else {
            panic!("expected register-pipeline layout")
        };
        assert_eq!(pipeline.history_states(), 1);
        assert_eq!(pipeline.b_final(), 1.0);
    }

    #[test]
    fn requires_three_s_endpoint_evaluation_policy() {
        let source = THREE_S.replace(",\"endpoint_evaluation\":\"omit\"", "");
        let error = parse_low_storage_tableau(&source, "Euler3S").unwrap_err();
        assert!(error.to_string().contains("endpoint_evaluation"), "{error}");
    }

    #[test]
    fn independent_node_policy_is_explicit_and_typed() {
        let mismatched = PIPELINE.replace("\"c\":[\"1/2\"]", "\"c\":[\"3/4\"]");
        assert!(parse_low_storage_tableau(&mismatched, "Pipeline").is_err());

        let independent =
            mismatched.replace("\"order\":2", "\"order\":2,\"node_policy\":\"independent\"");
        let tableau = parse_low_storage_tableau(&independent, "Pipeline").unwrap();
        assert_eq!(tableau.node_policy(), LowStorageNodePolicy::Independent);
    }

    #[test]
    fn relaxed_consistency_tolerance_is_explicit_and_bounded() {
        let rounded = TWO_N.replace("\"c\":[\"1/2\"]", "\"c\":[\"0.5000005\"]");
        assert!(parse_low_storage_tableau(&rounded, "Williamson2").is_err());

        let declared = rounded.replace(
            "\"order\":2",
            "\"order\":2,\"consistency_tolerance\":\"1e-6\"",
        );
        let tableau = parse_low_storage_tableau(&declared, "Williamson2").unwrap();
        assert_eq!(tableau.consistency_tolerance(), 1.0e-6);

        for invalid in ["0", "1e-5"] {
            let source = TWO_N.replace(
                "\"order\":2",
                &format!("\"order\":2,\"consistency_tolerance\":\"{invalid}\""),
            );
            assert!(parse_low_storage_tableau(&source, "Williamson2").is_err());
        }
    }

    #[test]
    fn independent_nodes_still_require_finite_reconstructed_stages() {
        let source = r#"{"name":"Overflow","description":"overflowing 3S recurrence","kind":"low-storage-runge-kutta","layout":"three-s","order":1,"node_policy":"independent","gamma1":["1e308",0],"gamma2":["1e308",0],"gamma3":[0,1],"delta":[0,0],"beta1":0,"beta2":[0,1],"c":[0,0],"endpoint_evaluation":"omit"}"#;
        let error = parse_low_storage_tableau(source, "Overflow").unwrap_err();
        assert!(error.to_string().contains("not finite"), "{error}");
    }

    #[test]
    fn rejects_invalid_metadata_and_unknown_fields() {
        for invalid in [
            TWO_N.replace("\"name\":\"Williamson2\"", "\"name\":\"Other\""),
            TWO_N.replace("two-stage 2N method", " "),
            TWO_N.replace("\"order\":2", "\"order\":0"),
            TWO_N.replace("\"order\":2", "\"order\":3"),
            TWO_N.replace("\"order\":2", "\"order\":2,\"typo\":0"),
            TWO_N.replace("low-storage-runge-kutta", "explicit-runge-kutta"),
            TWO_N.replace("\"1/2\"", "\"1/0\""),
        ] {
            assert!(
                parse_low_storage_tableau(&invalid, "Williamson2").is_err(),
                "accepted {invalid}"
            );
        }
    }

    #[test]
    fn rejects_malformed_abc_and_inconsistent_nodes_or_weights() {
        for (source, name) in [
            (TWO_N.replace("\"A\":[\"-1/2\"]", "\"A\":[]"), "Williamson2"),
            (
                TWO_N.replace("\"b\":[\"1/2\",1]", "\"b\":[1]"),
                "Williamson2",
            ),
            (TWO_N.replace("\"c\":[\"1/2\"]", "\"c\":[0]"), "Williamson2"),
            (
                TWO_N.replace("\"b\":[\"1/2\",1]", "\"b\":[0,0]"),
                "Williamson2",
            ),
            (TWO_C.replace("\"A\":[\"1/2\"]", "\"A\":[0]"), "Midpoint2C"),
            (TWO_C.replace("\"b\":[0,1]", "\"b\":[0,2]"), "Midpoint2C"),
        ] {
            assert!(
                parse_low_storage_tableau(&source, name).is_err(),
                "accepted {source}"
            );
        }
    }

    #[test]
    fn rejects_malformed_three_s_recurrences() {
        for invalid in [
            THREE_S.replace("\"gamma2\":[0]", "\"gamma2\":[]"),
            THREE_S.replace("\"c\":[0]", "\"c\":[1]"),
            THREE_S.replace("\"gamma3\":[1]", "\"gamma3\":[0]"),
            THREE_S.replace("\"beta2\":[1]", "\"beta2\":[2]"),
        ] {
            assert!(
                parse_low_storage_tableau(&invalid, "Euler3S").is_err(),
                "accepted {invalid}"
            );
        }
    }

    #[test]
    fn rejects_malformed_alternating_and_pipeline_recurrences() {
        for (source, name) in [
            (
                ALTERNATING.replace("\"c2\":[\"1/2\"]", "\"c2\":[0]"),
                "Alternating",
            ),
            (
                PIPELINE.replace("\"history_states\":1", "\"history_states\":0"),
                "Pipeline",
            ),
            (
                PIPELINE.replace("\"history_states\":1", "\"history_states\":2"),
                "Pipeline",
            ),
            (
                PIPELINE.replace("\"A\":[[\"1/2\"]]", "\"A\":[[]]"),
                "Pipeline",
            ),
            (PIPELINE.replace("\"c\":[\"1/2\"]", "\"c\":[0]"), "Pipeline"),
            (
                PIPELINE.replace("\"b_final\":1", "\"b_final\":2"),
                "Pipeline",
            ),
        ] {
            assert!(
                parse_low_storage_tableau(&source, name).is_err(),
                "accepted {source}"
            );
        }
    }

    #[test]
    fn parses_direct_and_embedded_weight_formulas() {
        let pipeline = PIPELINE.replace(
            "\"c\":[\"1/2\"]",
            "\"c\":[\"1/2\"],\"embedded\":{\"order\":1,\"b_hat\":[1],\"b_hat_final\":0}",
        );
        let tableau = parse_low_storage_tableau(&pipeline, "Pipeline").unwrap();
        let embedded = tableau.embedded().unwrap();
        assert_eq!(embedded.order(), 1);
        assert_eq!(embedded.error(), [-1.0, 1.0]);
        assert_eq!(
            embedded.controller(),
            LowStorageAdaptiveController::StandardPi
        );

        let three_s = THREE_S
            .replace("\"order\":1", "\"order\":2")
            .replace(
                "\"endpoint_evaluation\":\"omit\"",
                "\"endpoint_evaluation\":\"omit\",\"embedded\":{\"order\":1,\"error\":[-1,1],\"controller\":{\"kind\":\"pid\",\"beta\":[\"0.7\",\"-0.2\",0],\"acceptance_safety\":\"0.81\"}}",
            );
        let tableau = parse_low_storage_tableau(&three_s, "Euler3S").unwrap();
        let embedded = tableau.embedded().unwrap();
        let LowStorageAdaptiveController::Pid(controller) = embedded.controller() else {
            panic!("expected a PID controller")
        };
        assert_eq!(controller.beta(), [0.7, -0.2, 0.0]);
        assert_eq!(controller.acceptance_safety(), 0.81);
    }

    #[test]
    fn rejects_malformed_embedded_formulas_and_controllers() {
        let embedded_pipeline = PIPELINE.replace(
            "\"c\":[\"1/2\"]",
            "\"c\":[\"1/2\"],\"embedded\":{\"order\":1,\"b_hat\":[1],\"b_hat_final\":0}",
        );
        let invalid = [
            embedded_pipeline.replace("\"order\":1,\"b_hat\"", "\"order\":2,\"b_hat\""),
            embedded_pipeline.replace(
                "\"b_hat\":[1],\"b_hat_final\":0",
                "\"b_hat\":[1]",
            ),
            embedded_pipeline.replace("\"b_hat\":[1]", "\"b_hat\":[1,0]"),
            embedded_pipeline.replace(
                "\"b_hat\":[1],\"b_hat_final\":0",
                "\"error\":[-1,1],\"b_hat\":[1],\"b_hat_final\":0",
            ),
            embedded_pipeline.replace(
                "\"b_hat\":[1],\"b_hat_final\":0",
                "\"error\":[1,1]",
            ),
            embedded_pipeline.replace(
                "\"b_hat\":[1],\"b_hat_final\":0",
                "\"error\":[-1,1],\"controller\":{\"kind\":\"pid\",\"beta\":[1,0],\"acceptance_safety\":\"0.81\"}",
            ),
            embedded_pipeline.replace(
                "\"b_hat\":[1],\"b_hat_final\":0",
                "\"error\":[-1,1],\"controller\":{\"kind\":\"pid\",\"beta\":[1,0,0],\"acceptance_safety\":2}",
            ),
        ];
        for source in invalid {
            assert!(
                parse_low_storage_tableau(&source, "Pipeline").is_err(),
                "accepted {source}"
            );
        }

        let fixed_layout = TWO_N.replace(
            "\"c\":[\"1/2\"]",
            "\"c\":[\"1/2\"],\"embedded\":{\"order\":1,\"error\":[-1,1]}",
        );
        assert!(parse_low_storage_tableau(&fixed_layout, "Williamson2").is_err());
    }
}
