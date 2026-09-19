//! Adaptive-order Adams--Moulton kernel and Lagrange integration weights.

use super::common::{ensure_finite, evaluate, scaled_error_norm};
use super::method::{MAX_FACTOR, MIN_FACTOR, SAFETY};
use crate::integrator::{ControllerConfig, KernelCapabilities, StepEstimate, StepKernel};
use crate::{OdeProblem, SolveError, SolveOptions, SolverStats};

const VCABM_MAX_ORDER: usize = 12;

pub(super) struct VariableOrderAdamsKernel {
    derivatives: Vec<Vec<f64>>,
    times: [f64; VCABM_MAX_ORDER],
    history_len: usize,
    predicted: Vec<f64>,
    trial_derivative: Vec<f64>,
    error: Vec<f64>,
    order: usize,
    accepted_at_order: usize,
    attempted_error: f64,
}

impl VariableOrderAdamsKernel {
    pub(super) const fn new() -> Self {
        Self {
            derivatives: Vec::new(),
            times: [0.0; VCABM_MAX_ORDER],
            history_len: 0,
            predicted: Vec::new(),
            trial_derivative: Vec::new(),
            error: Vec::new(),
            order: 1,
            accepted_at_order: 0,
            attempted_error: 0.0,
        }
    }

    fn reset<F, P>(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        if self.derivatives.len() != VCABM_MAX_ORDER
            || self
                .derivatives
                .first()
                .is_none_or(|row| row.len() != state.len())
        {
            self.derivatives = (0..VCABM_MAX_ORDER)
                .map(|_| vec![0.0; state.len()])
                .collect();
        }
        evaluate(problem, &mut self.derivatives[0], state, time, stats)?;
        ensure_finite(&self.derivatives[0])?;
        self.times[0] = time;
        self.history_len = 1;
        self.predicted.resize(state.len(), 0.0);
        self.trial_derivative.resize(state.len(), 0.0);
        self.error.resize(state.len(), 0.0);
        self.order = 1;
        self.accepted_at_order = 0;
        Ok(())
    }
}

impl<F, P> StepKernel<F, P> for VariableOrderAdamsKernel
where
    F: crate::OdeFunction<P>,
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::with_controller(
            true,
            ControllerConfig::proportional(6, SAFETY, MIN_FACTOR, MAX_FACTOR, MIN_FACTOR),
        )
    }

    fn initialize(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        self.reset(problem, state, time, stats)
    }

    fn estimate_initial_step(
        &mut self,
        _: &OdeProblem<F, P>,
        state: &[f64],
        _: f64,
        _: f64,
        maximum_step: f64,
        _: &mut [f64],
        options: &SolveOptions,
        _: &mut SolverStats,
    ) -> Result<f64, SolveError> {
        let scale = state
            .iter()
            .zip(&self.derivatives[0])
            .map(|(&state, &derivative)| {
                derivative.abs()
                    / (options.absolute_tolerance + options.relative_tolerance * state.abs())
            })
            .fold(0.0, f64::max);
        Ok(if scale > 0.0 {
            (0.01 / scale).sqrt().min(maximum_step)
        } else {
            1.0e-3_f64.min(maximum_step)
        }
        .max(f64::EPSILON))
    }

    fn attempt_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        options: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError> {
        let order = self.order.min(self.history_len).max(1);
        let mut nodes = [0.0; VCABM_MAX_ORDER + 1];
        let mut weights = [0.0; VCABM_MAX_ORDER + 1];
        for (node, &history_time) in nodes.iter_mut().zip(self.times.iter()).take(order) {
            *node = (history_time - time) / step;
        }
        integrated_lagrange_weights(&nodes[..order], &mut weights[..order]);
        self.predicted.copy_from_slice(state);
        for (weight, derivative) in weights.iter().zip(&self.derivatives).take(order) {
            for (value, &slope) in self.predicted.iter_mut().zip(derivative) {
                *value += step * weight * slope;
            }
        }
        ensure_finite(&self.predicted)?;
        evaluate(
            problem,
            &mut self.trial_derivative,
            &self.predicted,
            time + step,
            stats,
        )?;
        ensure_finite(&self.trial_derivative)?;

        nodes[0] = 1.0;
        for (index, &history_time) in self.times.iter().take(order).enumerate() {
            nodes[index + 1] = (history_time - time) / step;
        }
        integrated_lagrange_weights(&nodes[..=order], &mut weights[..=order]);
        candidate.copy_from_slice(state);
        for (value, &slope) in candidate.iter_mut().zip(&self.trial_derivative) {
            *value += step * weights[0] * slope;
        }
        for (weight, derivative) in weights[1..].iter().zip(&self.derivatives).take(order) {
            for (value, &slope) in candidate.iter_mut().zip(derivative) {
                *value += step * weight * slope;
            }
        }
        ensure_finite(candidate)?;
        for ((error, &corrected), &predicted) in self
            .error
            .iter_mut()
            .zip(candidate.iter())
            .zip(&self.predicted)
        {
            *error = (corrected - predicted) / (order + 1) as f64;
        }
        self.attempted_error = if options.adaptive {
            scaled_error_norm(&self.error, state, candidate, options)
        } else {
            0.0
        };
        Ok(StepEstimate::new(self.attempted_error))
    }

    fn accept_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        _: &[f64],
        state: &[f64],
        time: f64,
        _: f64,
        callback_applied: bool,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        if callback_applied {
            return self.reset(problem, state, time, stats);
        }
        self.derivatives.rotate_right(1);
        self.times.rotate_right(1);
        evaluate(problem, &mut self.derivatives[0], state, time, stats)?;
        ensure_finite(&self.derivatives[0])?;
        self.times[0] = time;
        self.history_len = (self.history_len + 1).min(VCABM_MAX_ORDER);
        self.accepted_at_order += 1;
        if self.attempted_error < 0.2
            && self.order < VCABM_MAX_ORDER
            && self.history_len > self.order
            && self.accepted_at_order >= self.order
        {
            self.order += 1;
            self.accepted_at_order = 0;
        }
        Ok(())
    }

    fn reject_step(&mut self) {
        if self.order > 1 {
            self.order -= 1;
        }
        self.accepted_at_order = 0;
    }
}

fn integrated_lagrange_weights(nodes: &[f64], weights: &mut [f64]) {
    debug_assert_eq!(nodes.len(), weights.len());
    weights.fill(0.0);
    for (basis, &node) in nodes.iter().enumerate() {
        let mut polynomial = [0.0; VCABM_MAX_ORDER + 1];
        let mut degree = 0;
        let mut denominator = 1.0;
        polynomial[0] = 1.0;
        for (other, &other_node) in nodes.iter().enumerate() {
            if other == basis {
                continue;
            }
            denominator *= node - other_node;
            for power in (0..=degree).rev() {
                polynomial[power + 1] += polynomial[power];
                polynomial[power] *= -other_node;
            }
            degree += 1;
        }
        weights[basis] = polynomial[..=degree]
            .iter()
            .enumerate()
            .map(|(power, coefficient)| coefficient / (power + 1) as f64)
            .sum::<f64>()
            / denominator;
    }
}
