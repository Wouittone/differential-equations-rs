use crate::tableau::{RosenbrockKind, RosenbrockPairTableau, RosenbrockTableau};

pub(crate) struct Workspace {
    pub(super) tableau: Option<&'static RosenbrockTableau>,
    pub(super) pair_tableau: Option<&'static RosenbrockPairTableau>,
    pub(super) current_derivative: Vec<f64>,
    pub(super) perturbed_state: Vec<f64>,
    pub(super) perturbed_derivative: Vec<f64>,
    pub(super) time_derivative: Vec<f64>,
    pub(super) stage_state: Vec<f64>,
    pub(super) stage_derivative: Vec<f64>,
    pub(super) right_hand_side: Vec<f64>,
    pub(super) error: Vec<f64>,
    pub(super) stages: Vec<f64>,
    pub(super) dense_endpoint_state: Vec<f64>,
    pub(super) dense_endpoint_derivative: Vec<f64>,
    pub(super) dense_corrections: Vec<f64>,
    pub(super) jacobian: Vec<f64>,
    pub(super) factorization: Vec<f64>,
    pub(super) pivots: Vec<usize>,
    pub(super) differentiation_valid: bool,
}

impl Workspace {
    pub(super) fn new(
        dimension: usize,
        tableau: Option<&'static RosenbrockTableau>,
        pair_tableau: Option<&'static RosenbrockPairTableau>,
    ) -> Self {
        let stages = tableau.map_or(3, RosenbrockTableau::stages);
        let dense_order = tableau.map_or(0, |tableau| tableau.h().len());
        // The implemented hybrid ODE specialization is explicit. It never
        // differentiates or solves a linear system, so n*n storage would be
        // both unused and prohibitively expensive for large array states.
        let linear_dimension = if tableau
            .is_some_and(|tableau| tableau.kind() == RosenbrockKind::HybridExplicitImplicit)
        {
            0
        } else {
            dimension
        };
        Self {
            tableau,
            pair_tableau,
            current_derivative: vec![0.0; dimension],
            perturbed_state: vec![0.0; linear_dimension],
            perturbed_derivative: vec![0.0; linear_dimension],
            time_derivative: vec![0.0; linear_dimension],
            stage_state: vec![0.0; dimension],
            stage_derivative: vec![0.0; dimension],
            right_hand_side: vec![0.0; linear_dimension],
            error: vec![0.0; dimension],
            stages: vec![0.0; stages * dimension],
            dense_endpoint_state: vec![0.0; dimension],
            dense_endpoint_derivative: vec![0.0; dimension],
            dense_corrections: vec![0.0; dense_order * dimension],
            jacobian: vec![0.0; linear_dimension * linear_dimension],
            factorization: vec![0.0; linear_dimension * linear_dimension],
            pivots: vec![0; linear_dimension],
            differentiation_valid: false,
        }
    }
}
