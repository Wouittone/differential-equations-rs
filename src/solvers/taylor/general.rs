//! Explicit Taylor-series integration for ordinary `f64` ODE problems.
//!
//! OrdinaryDiffEq obtains jets through Julia's TaylorDiff scalar type. Rust's
//! public RHS deliberately accepts plain `f64` slices, so this port recovers
//! the same time-series coefficients by polynomial continuation: known state
//! coefficients are evaluated at Chebyshev nodes and the corresponding RHS
//! coefficient is extracted from a precomputed interpolation inverse. The
//! resulting step is a genuine Taylor polynomial, not an RK substitution.

use std::f64::consts::PI;

use crate::integrator::{
    ControllerConfig, KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::linear::{factorize, solve_factorized};
use crate::solution::{BorrowedTaylorSegment, DenseSegment, TaylorSegment, TrajectoryRecorder};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

const MAX_ORDER: usize = 12;

/// Second-order explicit Taylor method.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExplicitTaylor2;

/// Fixed-order Taylor method with optional adaptive step control.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExplicitTaylor {
    order: usize,
}

impl Default for ExplicitTaylor {
    fn default() -> Self {
        Self { order: 1 }
    }
}

impl ExplicitTaylor {
    /// Constructs a Taylor method, clamping `order` to the supported range.
    pub const fn new(order: usize) -> Self {
        Self {
            order: if order < 1 {
                1
            } else if order > MAX_ORDER {
                MAX_ORDER
            } else {
                order
            },
        }
    }

    /// Returns the configured Taylor order.
    pub const fn order(self) -> usize {
        self.order
    }
}

/// Taylor method that chooses an order from a configured work window.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExplicitTaylorAdaptiveOrder {
    min_order: usize,
    max_order: usize,
}

impl Default for ExplicitTaylorAdaptiveOrder {
    fn default() -> Self {
        Self {
            min_order: 1,
            max_order: 10,
        }
    }
}

impl ExplicitTaylorAdaptiveOrder {
    /// Constructs an adaptive-order Taylor method with a clamped order window.
    pub const fn new(min_order: usize, max_order: usize) -> Self {
        let min_order = if min_order < 1 {
            1
        } else if min_order > MAX_ORDER - 1 {
            MAX_ORDER - 1
        } else {
            min_order
        };
        let max_order = if max_order < min_order + 1 {
            min_order + 1
        } else if max_order > MAX_ORDER {
            MAX_ORDER
        } else {
            max_order
        };
        Self {
            min_order,
            max_order,
        }
    }

    /// Returns the minimum candidate order.
    pub const fn min_order(self) -> usize {
        self.min_order
    }

    /// Returns the maximum candidate order.
    pub const fn max_order(self) -> usize {
        self.max_order
    }
}

impl OdeAlgorithm for ExplicitTaylor2 {
    fn solve_validated<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        drive_integration(
            problem,
            options,
            TaylorKernel::new(problem.initial_state().len(), TaylorMode::SecondOrder)?,
        )
    }
}

impl OdeAlgorithm for ExplicitTaylor {
    fn solve_validated<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        drive_integration(
            problem,
            options,
            TaylorKernel::new(problem.initial_state().len(), TaylorMode::Fixed(self.order))?,
        )
    }
}

impl OdeAlgorithm for ExplicitTaylorAdaptiveOrder {
    fn solve_validated<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        drive_integration(
            problem,
            options,
            TaylorKernel::new(
                problem.initial_state().len(),
                TaylorMode::AdaptiveOrder {
                    min: self.min_order,
                    max: self.max_order,
                },
            )?,
        )
    }
}

#[derive(Clone, Copy)]
enum TaylorMode {
    SecondOrder,
    Fixed(usize),
    AdaptiveOrder { min: usize, max: usize },
}

struct TaylorKernel {
    dimension: usize,
    mode: TaylorMode,
    current_order: usize,
    nodes: Vec<f64>,
    inverse: Vec<f64>,
    coefficients: Vec<f64>,
    samples: Vec<f64>,
    stage_state: Vec<f64>,
    start_derivative: Vec<f64>,
    endpoint_derivative: Vec<f64>,
    endpoint_state: Vec<f64>,
    order_errors: Vec<f64>,
    endpoint_valid: bool,
}

impl TaylorKernel {
    fn new(dimension: usize, mode: TaylorMode) -> Result<Self, SolveError> {
        let maximum = match mode {
            TaylorMode::SecondOrder => 2,
            TaylorMode::Fixed(order) => order + 1,
            TaylorMode::AdaptiveOrder { max, .. } => max,
        }
        .min(MAX_ORDER);
        let current_order = match mode {
            TaylorMode::SecondOrder => 2,
            TaylorMode::Fixed(order) => order,
            TaylorMode::AdaptiveOrder { min, max } => max.saturating_sub(1).max(min),
        };
        let node_count = maximum + 1;
        let nodes = (0..node_count)
            .map(|index| (PI * index as f64 / maximum as f64).cos())
            .collect::<Vec<_>>();
        let inverse = interpolation_inverse(&nodes)?;
        Ok(Self {
            dimension,
            mode,
            current_order,
            nodes,
            inverse,
            coefficients: vec![0.0; (maximum + 1) * dimension],
            samples: vec![0.0; node_count * dimension],
            stage_state: vec![0.0; dimension],
            start_derivative: vec![0.0; dimension],
            endpoint_derivative: vec![0.0; dimension],
            endpoint_state: vec![0.0; dimension],
            order_errors: vec![f64::INFINITY; maximum + 1],
            endpoint_valid: false,
        })
    }

    fn adaptive(&self) -> bool {
        !matches!(self.mode, TaylorMode::SecondOrder)
    }

    fn evaluate<F, P>(
        problem: &OdeProblem<F, P>,
        output: &mut [f64],
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        (problem.rhs)(output, state, problem.parameters(), time);
        stats.rhs_evaluations += 1;
        output
            .iter()
            .all(|value| value.is_finite())
            .then_some(())
            .ok_or(SolveError::NonFiniteDerivative)
    }

    fn build_coefficients<F, P>(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        degree: usize,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        self.coefficients.fill(0.0);
        self.order_errors.fill(f64::INFINITY);
        for component in 0..self.dimension {
            self.coefficients[self.dimension + component] = step * self.start_derivative[component];
        }
        let node_count = self.nodes.len();
        for power in 1..degree {
            for (node_index, &node) in self.nodes.iter().enumerate() {
                self.stage_state.copy_from_slice(state);
                let mut node_power = node;
                for coefficient_power in 1..=power {
                    let offset = coefficient_power * self.dimension;
                    for component in 0..self.dimension {
                        self.stage_state[component] +=
                            node_power * self.coefficients[offset + component];
                    }
                    node_power *= node;
                }
                let sample_start = node_index * self.dimension;
                Self::evaluate(
                    problem,
                    &mut self.samples[sample_start..sample_start + self.dimension],
                    &self.stage_state,
                    time + step * node,
                    stats,
                )?;
            }
            let offset = (power + 1) * self.dimension;
            for component in 0..self.dimension {
                let rhs_coefficient = (0..node_count)
                    .map(|node| {
                        self.inverse[power * node_count + node]
                            * self.samples[node * self.dimension + component]
                    })
                    .sum::<f64>();
                self.coefficients[offset + component] = step * rhs_coefficient / (power + 1) as f64;
            }
        }
        Ok(())
    }
}

impl<F, P> StepKernel<F, P> for TaylorKernel
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    fn has_custom_dense_output(&self) -> bool {
        true
    }

    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::with_controller(
            self.adaptive(),
            ControllerConfig::proportional(self.current_order.max(1), 0.8, 0.1, 4.0, 0.2),
        )
    }

    fn initialize(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        Self::evaluate(problem, &mut self.start_derivative, state, time, stats)
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
            .zip(&self.start_derivative)
            .map(|(state, derivative)| {
                derivative.abs()
                    / (options.absolute_tolerance + options.relative_tolerance * state.abs())
            })
            .fold(0.0_f64, f64::max);
        Ok((if scale == 0.0 { 1.0e-3 } else { 0.01 / scale }).clamp(f64::EPSILON, maximum_step))
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
        let candidate_order = self.current_order;
        let computed_order = if options.adaptive {
            (candidate_order + 1).min(self.coefficients.len() / self.dimension - 1)
        } else {
            candidate_order
        };
        self.build_coefficients(problem, state, time, step, computed_order, stats)?;
        candidate.copy_from_slice(state);
        for power in 1..=candidate_order {
            let offset = power * self.dimension;
            for (component, candidate) in candidate.iter_mut().enumerate() {
                *candidate += self.coefficients[offset + component];
            }
        }
        self.endpoint_state.copy_from_slice(candidate);
        Self::evaluate(
            problem,
            &mut self.endpoint_derivative,
            candidate,
            time + step,
            stats,
        )?;
        self.endpoint_valid = true;
        if !options.adaptive {
            return Ok(StepEstimate::new(0.0));
        }
        let error_offset = (candidate_order + 1) * self.dimension;
        let error = &self.coefficients[error_offset..error_offset + self.dimension];
        let error_norm = scaled_error(error, state, candidate, options);
        if let TaylorMode::AdaptiveOrder { min, max } = self.mode {
            let lower = candidate_order.saturating_sub(1).max(min);
            let upper = (candidate_order + 1).min(max - 1);
            for order in lower..=upper {
                let offset = (order + 1) * self.dimension;
                self.order_errors[order] = scaled_error(
                    &self.coefficients[offset..offset + self.dimension],
                    state,
                    candidate,
                    options,
                );
            }
        }
        Ok(StepEstimate::new(error_norm))
    }

    fn apply_step_callbacks(
        &mut self,
        problem: &OdeProblem<F, P>,
        previous_state: &[f64],
        previous_time: f64,
        state: &mut [f64],
        time: &mut f64,
        state_before_effect: &mut [f64],
        event_tolerance: f64,
        _: &mut SolverStats,
    ) -> Result<crate::callback::CallbackOutcome, SolveError> {
        let endpoint_time = *time;
        let segment = BorrowedTaylorSegment::new(
            previous_time,
            endpoint_time,
            previous_state,
            &self.endpoint_state,
            &self.coefficients,
            self.current_order,
        )
        .map_err(|_| SolveError::NonFiniteDerivative)?;
        let mut interpolate = |query: f64, output: &mut [f64]| {
            segment
                .interpolate(query, output)
                .map_err(|_| SolveError::NonFiniteDerivative)
        };
        problem.apply_step_callbacks(
            previous_state,
            previous_time,
            state,
            time,
            state_before_effect,
            event_tolerance,
            Some(&mut interpolate),
        )
    }

    fn record_dense_step(
        &mut self,
        _: &OdeProblem<F, P>,
        previous_state: &[f64],
        state: &[f64],
        previous_time: f64,
        attempted_time: f64,
        time: f64,
        final_time: bool,
        recorder: &mut TrajectoryRecorder<'_>,
        _: &mut SolverStats,
    ) -> Result<bool, SolveError> {
        let segment = BorrowedTaylorSegment::new(
            previous_time,
            attempted_time,
            previous_state,
            &self.endpoint_state,
            &self.coefficients,
            self.current_order,
        )
        .map_err(|_| SolveError::NonFiniteDerivative)?;
        recorder
            .record_step_dense(
                previous_state,
                previous_time,
                state,
                time,
                final_time,
                &segment,
            )
            .map_err(|_| SolveError::NonFiniteDerivative)?;
        if recorder.retains_dense_output() {
            recorder.retain_taylor_segment(
                TaylorSegment::new_bounded(
                    previous_time,
                    attempted_time,
                    time,
                    previous_state,
                    state,
                    &self.coefficients,
                    self.current_order,
                )
                .map_err(|_| SolveError::NonFiniteDerivative)?,
            );
        }
        Ok(true)
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
        if let TaylorMode::AdaptiveOrder { min, max } = self.mode {
            let lower = self.current_order.saturating_sub(1).max(min);
            let upper = (self.current_order + 1).min(max - 1);
            self.current_order = (lower..=upper)
                .min_by(|&left, &right| {
                    work_score(left, self.order_errors[left])
                        .total_cmp(&work_score(right, self.order_errors[right]))
                })
                .unwrap_or(self.current_order);
        }
        if self.endpoint_valid && !callback_applied {
            std::mem::swap(&mut self.start_derivative, &mut self.endpoint_derivative);
            Ok(())
        } else {
            Self::evaluate(problem, &mut self.start_derivative, state, time, stats)
        }
    }

    fn reject_step(&mut self) {
        self.endpoint_valid = false;
    }
}

fn interpolation_inverse(nodes: &[f64]) -> Result<Vec<f64>, SolveError> {
    let size = nodes.len();
    let mut vandermonde = vec![0.0; size * size];
    for (row, &node) in nodes.iter().enumerate() {
        let mut power = 1.0;
        for column in 0..size {
            vandermonde[row * size + column] = power;
            power *= node;
        }
    }
    let mut pivots = vec![0; size];
    factorize(&mut vandermonde, &mut pivots, size).map_err(|_| SolveError::SingularLinearSystem)?;
    let mut inverse = vec![0.0; size * size];
    for node in 0..size {
        let mut basis = vec![0.0; size];
        basis[node] = 1.0;
        solve_factorized(&vandermonde, &pivots, &mut basis, size);
        for coefficient in 0..size {
            inverse[coefficient * size + node] = basis[coefficient];
        }
    }
    Ok(inverse)
}

fn scaled_error(error: &[f64], old: &[f64], new: &[f64], options: &SolveOptions) -> f64 {
    (error
        .iter()
        .zip(old)
        .zip(new)
        .map(|((error, old), new)| {
            let scale =
                options.absolute_tolerance + options.relative_tolerance * old.abs().max(new.abs());
            (error / scale).powi(2)
        })
        .sum::<f64>()
        / error.len() as f64)
        .sqrt()
}

fn work_score(order: usize, error: f64) -> f64 {
    let factor = if error == 0.0 {
        4.0
    } else if error.is_finite() {
        (0.8 * error.powf(-1.0 / (order + 1) as f64)).clamp(0.1, 4.0)
    } else {
        0.1
    };
    (order * order) as f64 / factor
}
