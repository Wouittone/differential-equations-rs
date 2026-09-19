//! Divided-difference history and variable-step coefficient updates.

pub(super) struct Workspace {
    pub(super) predicted: Vec<f64>,
    pub(super) temporary: Vec<f64>,
    pub(super) derivative: Vec<f64>,
    pub(super) next_derivative: Vec<f64>,
    pub(super) error: Vec<f64>,
    pub(super) stages: Vec<Vec<f64>>,
    pub(super) phi_previous: Vec<Vec<f64>>,
    pub(super) phi: Vec<Vec<f64>>,
    pub(super) phi_unscaled: Vec<Vec<f64>>,
    pub(super) phi_endpoint: Vec<Vec<f64>>,
    pub(super) accepted_steps: Vec<f64>,
    pub(super) trial_steps: Vec<f64>,
    pub(super) coefficients: Vec<Vec<f64>>,
    pub(super) g: Vec<f64>,
}

impl Workspace {
    pub(super) fn new(dimension: usize, order: usize) -> Self {
        let vectors = || (0..order).map(|_| vec![0.0; dimension]).collect();
        Self {
            predicted: vec![0.0; dimension],
            temporary: vec![0.0; dimension],
            derivative: vec![0.0; dimension],
            next_derivative: vec![0.0; dimension],
            error: vec![0.0; dimension],
            stages: (0..4).map(|_| vec![0.0; dimension]).collect(),
            phi_previous: vectors(),
            phi: vectors(),
            phi_unscaled: vectors(),
            phi_endpoint: (0..=order).map(|_| vec![0.0; dimension]).collect(),
            accepted_steps: vec![0.0; order],
            trial_steps: vec![0.0; order],
            coefficients: (0..=order).map(|_| vec![0.0; order + 1]).collect(),
            g: vec![0.0; order + 1],
        }
    }

    pub(super) fn reset_history(&mut self) {
        self.accepted_steps.fill(0.0);
        for difference in &mut self.phi_previous {
            difference.fill(0.0);
        }
    }
}

pub(super) fn prepare_trial_steps(
    workspace: &mut Workspace,
    step: f64,
    step_number: usize,
    order: usize,
) {
    workspace
        .trial_steps
        .copy_from_slice(&workspace.accepted_steps);
    for index in (1..step_number.min(order)).rev() {
        workspace.trial_steps[index] = workspace.accepted_steps[index - 1];
    }
    workspace.trial_steps[0] = step;
}

// Hairer, Norsett and Wanner III.5 (5.9). This is the same scaled
// divided-difference update used by OrdinaryDiffEq's pinned VCAB caches.
pub(super) fn update_differences(workspace: &mut Workspace, count: usize) {
    workspace.phi_unscaled[0].copy_from_slice(&workspace.derivative);
    workspace.phi[0].copy_from_slice(&workspace.derivative);
    let mut xi = workspace.trial_steps[0];
    let mut xi_zero = 0.0;
    let mut beta = 1.0;
    for index in 1..count {
        xi_zero += workspace.trial_steps[index];
        beta *= xi / xi_zero;
        xi += workspace.trial_steps[index];
        let (before, after) = workspace.phi_unscaled.split_at_mut(index);
        for component in 0..after[0].len() {
            after[0][component] =
                before[index - 1][component] - workspace.phi_previous[index - 1][component];
            workspace.phi[index][component] = beta * after[0][component];
        }
    }
}

// Hairer, Norsett and Wanner III.5 (5.9--5.10). `g` includes the step,
// exactly as in OrdinaryDiffEq's `g_coefs!`.
pub(super) fn update_g(workspace: &mut Workspace, count: usize) {
    let step = workspace.trial_steps[0];
    let mut xi = step;
    for index in 0..count {
        if index > 1 {
            xi += workspace.trial_steps[index - 1];
        }
        for q in 0..(count - index) {
            workspace.coefficients[index][q] = match index {
                0 => 1.0 / (q + 1) as f64,
                1 => 1.0 / ((q + 1) * (q + 2)) as f64,
                _ => {
                    workspace.coefficients[index - 1][q]
                        - (step / xi) * workspace.coefficients[index - 1][q + 1]
                }
            };
        }
        workspace.g[index] = workspace.coefficients[index][0] * step;
    }
}
