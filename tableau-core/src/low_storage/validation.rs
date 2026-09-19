use super::{
    LowStorageAbcTableau, LowStorageNodePolicy, RegisterPipelineTableau, TableauError,
    ThreeSTableau,
};

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

pub(super) fn validate_two_n(
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

pub(super) fn validate_two_c(
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

pub(super) fn validate_three_s(
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

pub(super) fn validate_register_pipeline(
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

pub(super) fn approximately_equal_recurrence(left: f64, right: f64, tolerance: f64) -> bool {
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
