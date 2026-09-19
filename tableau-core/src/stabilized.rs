mod eserk;
mod parser;
mod validation;

pub use eserk::{EserkTableau, parse_eserk_tableau};
pub use parser::{parse_rock2_tableau, parse_rock4_tableau, parse_serk2_tableau};

/// One two-term stage in a ROCK polynomial recurrence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RockRecurrenceStage {
    mu: f64,
    kappa: f64,
}

impl RockRecurrenceStage {
    /// Multiplier applied to the current derivative.
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Multiplier applied to the stage from two recurrence steps ago.
    pub fn kappa(&self) -> f64 {
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
    alpha: f64,
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

    /// Recurrence step coefficient applied to each derivative.
    pub fn alpha(&self) -> f64 {
        self.alpha
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

#[cfg(test)]
mod tests;
