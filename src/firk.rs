//! Fully implicit collocation Runge--Kutta methods.
//!
//! Tableaus are generated from the Radau-right and Gauss--Legendre nodes in
//! the same way as the pinned `OrdinaryDiffEqFIRK` implementation.  The stage
//! equations are solved as one coupled dense Newton system.  This is less
//! specialized than OrdinaryDiffEq's real/complex basis transform, but it is
//! the same collocation method and is deliberately kept independent of the
//! SDIRK kernels.

use std::f64::consts::PI;

use crate::integrator::{
    ControllerConfig, KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::linear::{factorize, solve_factorized};
use crate::solution::{DenseSegment, TrajectoryRecorder};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

const MAX_NEWTON_ITERATIONS: usize = 12;
const NEWTON_TOLERANCE: f64 = 2.0e-11;

/// Third-order, two-stage Radau IIA collocation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RadauIIA3;

/// Fifth-order, three-stage Radau IIA collocation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RadauIIA5;

/// Ninth-order, five-stage Radau IIA collocation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RadauIIA9;

/// Variable-order Radau IIA collocation.
///
/// Odd orders are clamped to the upstream default range 5 through 13.  The
/// kernel starts at the minimum order and raises or lowers the number of
/// stages after accepted steps according to the embedded step-doubling error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdaptiveRadau {
    min_order: usize,
    max_order: usize,
}

impl Default for AdaptiveRadau {
    fn default() -> Self {
        Self {
            min_order: 5,
            max_order: 13,
        }
    }
}

impl AdaptiveRadau {
    /// Creates a variable-order Radau method. Even bounds are rounded up to
    /// the next valid odd Radau order and the supported range is 3 through 13.
    pub fn new(min_order: usize, max_order: usize) -> Self {
        let min_order = normalize_radau_order(min_order);
        let max_order = normalize_radau_order(max_order).max(min_order);
        Self {
            min_order,
            max_order,
        }
    }

    pub fn min_order(self) -> usize {
        self.min_order
    }

    pub fn max_order(self) -> usize {
        self.max_order
    }
}

/// Gauss--Legendre collocation with configurable stage count.
///
/// The default two-stage method is fourth order and symplectic. Adaptive
/// stepping uses Richardson step doubling, matching the pinned upstream
/// algorithm's documented controller foundation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GaussLegendre {
    num_stages: usize,
}

impl Default for GaussLegendre {
    fn default() -> Self {
        Self { num_stages: 2 }
    }
}

impl GaussLegendre {
    pub fn new(num_stages: usize) -> Self {
        Self {
            num_stages: num_stages.clamp(2, 8),
        }
    }

    pub fn num_stages(self) -> usize {
        self.num_stages
    }
}

#[derive(Clone, Copy)]
enum Family {
    Radau,
    Gauss,
}

macro_rules! impl_fixed_radau {
    ($name:ty, $stages:expr) => {
        impl OdeAlgorithm for $name {
            fn solve<F, P>(
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
                    FirkKernel::new(
                        problem.initial_state().len(),
                        Family::Radau,
                        $stages,
                        $stages,
                    ),
                )
            }
        }
    };
}

impl_fixed_radau!(RadauIIA3, 2);
impl_fixed_radau!(RadauIIA5, 3);
impl_fixed_radau!(RadauIIA9, 5);

impl OdeAlgorithm for AdaptiveRadau {
    fn solve<F, P>(
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
            FirkKernel::new(
                problem.initial_state().len(),
                Family::Radau,
                self.min_order.div_ceil(2),
                self.max_order.div_ceil(2),
            ),
        )
    }
}

impl OdeAlgorithm for GaussLegendre {
    fn solve<F, P>(
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
            FirkKernel::new(
                problem.initial_state().len(),
                Family::Gauss,
                self.num_stages,
                self.num_stages,
            ),
        )
    }
}

fn normalize_radau_order(order: usize) -> usize {
    let order = order.clamp(3, 13);
    if order % 2 == 0 { order + 1 } else { order }
}

#[derive(Clone)]
struct Tableau {
    stages: usize,
    order: usize,
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    // Ascending coefficients of each Lagrange cardinal polynomial.
    lagrange: Vec<f64>,
}

impl Tableau {
    fn generate(family: Family, stages: usize) -> Self {
        let c = match family {
            Family::Radau => radau_nodes(stages),
            Family::Gauss => gauss_nodes(stages),
        };
        let mut lagrange = vec![0.0; stages * stages];
        for j in 0..stages {
            let mut coefficients = vec![1.0];
            let mut denominator = 1.0;
            for k in 0..stages {
                if k == j {
                    continue;
                }
                denominator *= c[j] - c[k];
                let mut next = vec![0.0; coefficients.len() + 1];
                for (power, &value) in coefficients.iter().enumerate() {
                    next[power] -= c[k] * value;
                    next[power + 1] += value;
                }
                coefficients = next;
            }
            for power in 0..stages {
                lagrange[j * stages + power] = coefficients[power] / denominator;
            }
        }
        let mut a = vec![0.0; stages * stages];
        let mut b = vec![0.0; stages];
        for j in 0..stages {
            b[j] = integrated_cardinal(&lagrange, stages, j, 1.0);
            for i in 0..stages {
                a[i * stages + j] = integrated_cardinal(&lagrange, stages, j, c[i]);
            }
        }
        Self {
            stages,
            order: match family {
                Family::Radau => 2 * stages - 1,
                Family::Gauss => 2 * stages,
            },
            a,
            b,
            c,
            lagrange,
        }
    }

    fn weights_at(&self, theta: f64, output: &mut [f64]) {
        for (j, weight) in output.iter_mut().enumerate().take(self.stages) {
            *weight = integrated_cardinal(&self.lagrange, self.stages, j, theta);
        }
    }
}

fn integrated_cardinal(coefficients: &[f64], stages: usize, row: usize, x: f64) -> f64 {
    let mut power = x;
    let mut value = 0.0;
    for degree in 0..stages {
        value += coefficients[row * stages + degree] * power / (degree + 1) as f64;
        power *= x;
    }
    value
}

fn legendre(n: usize, x: f64) -> (f64, f64) {
    if n == 0 {
        return (1.0, 0.0);
    }
    let mut previous = 1.0;
    let mut current = x;
    for degree in 2..=n {
        let next = ((2 * degree - 1) as f64 * x * current - (degree - 1) as f64 * previous)
            / degree as f64;
        previous = current;
        current = next;
    }
    let derivative = n as f64 * (x * current - previous) / (x * x - 1.0);
    (current, derivative)
}

fn gauss_nodes(stages: usize) -> Vec<f64> {
    let mut nodes = Vec::with_capacity(stages);
    for k in 1..=stages {
        let mut x = (PI * (4 * k - 1) as f64 / (4 * stages + 2) as f64).cos();
        for _ in 0..30 {
            let (value, derivative) = legendre(stages, x);
            let next = x - value / derivative;
            if (next - x).abs() <= 4.0 * f64::EPSILON {
                x = next;
                break;
            }
            x = next;
        }
        nodes.push(0.5 * (x + 1.0));
    }
    nodes.sort_by(f64::total_cmp);
    nodes
}

fn radau_nodes(stages: usize) -> Vec<f64> {
    if stages == 1 {
        return vec![1.0];
    }
    let mut nodes = Vec::with_capacity(stages);
    for k in 0..stages {
        if k == 0 {
            nodes.push(1.0);
            continue;
        }
        let mut x = (2.0 * PI * k as f64 / (2 * stages - 1) as f64).cos();
        for _ in 0..40 {
            let (pn, dpn) = legendre(stages, x);
            let (pm, dpm) = legendre(stages - 1, x);
            let next = x - (pn - pm) / (dpn - dpm);
            if (next - x).abs() <= 8.0 * f64::EPSILON {
                x = next;
                break;
            }
            x = next.clamp(-1.0 + 1.0e-14, 1.0 - 1.0e-14);
        }
        nodes.push(0.5 * (x + 1.0));
    }
    nodes.sort_by(f64::total_cmp);
    nodes
}

struct FirkKernel {
    dimension: usize,
    family: Family,
    current_stages: usize,
    minimum_stages: usize,
    maximum_stages: usize,
    tableau: Tableau,
    stage_derivatives: Vec<f64>,
    stage_states: Vec<f64>,
    stage_jacobians: Vec<f64>,
    residual: Vec<f64>,
    correction: Vec<f64>,
    matrix: Vec<f64>,
    pivots: Vec<usize>,
    perturbed_state: Vec<f64>,
    perturbed_derivative: Vec<f64>,
    full_state: Vec<f64>,
    midpoint_state: Vec<f64>,
    first_half_stages: Vec<f64>,
    second_half_stages: Vec<f64>,
    interpolation_weights: Vec<f64>,
    last_error: f64,
    adaptive_attempt: bool,
    segment_start_time: f64,
    segment_step: f64,
}

impl FirkKernel {
    fn new(dimension: usize, family: Family, minimum_stages: usize, maximum_stages: usize) -> Self {
        let tableau = Tableau::generate(family, minimum_stages);
        let mut kernel = Self {
            dimension,
            family,
            current_stages: minimum_stages,
            minimum_stages,
            maximum_stages,
            tableau,
            stage_derivatives: Vec::new(),
            stage_states: Vec::new(),
            stage_jacobians: Vec::new(),
            residual: Vec::new(),
            correction: Vec::new(),
            matrix: Vec::new(),
            pivots: Vec::new(),
            perturbed_state: vec![0.0; dimension],
            perturbed_derivative: vec![0.0; dimension],
            full_state: vec![0.0; dimension],
            midpoint_state: vec![0.0; dimension],
            first_half_stages: Vec::new(),
            second_half_stages: Vec::new(),
            interpolation_weights: Vec::new(),
            last_error: 0.0,
            adaptive_attempt: false,
            segment_start_time: 0.0,
            segment_step: 0.0,
        };
        kernel.resize_stage_workspace();
        kernel
    }

    fn resize_stage_workspace(&mut self) {
        if self.tableau.stages != self.current_stages {
            self.tableau = Tableau::generate(self.family, self.current_stages);
        }
        let coupled = self.dimension * self.current_stages;
        self.stage_derivatives.resize(coupled, 0.0);
        self.stage_states.resize(coupled, 0.0);
        self.stage_jacobians
            .resize(self.current_stages * self.dimension * self.dimension, 0.0);
        self.residual.resize(coupled, 0.0);
        self.correction.resize(coupled, 0.0);
        self.matrix.resize(coupled * coupled, 0.0);
        self.pivots.resize(coupled, 0);
        self.first_half_stages.resize(coupled, 0.0);
        self.second_half_stages.resize(coupled, 0.0);
        self.interpolation_weights.resize(self.current_stages, 0.0);
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

    fn collocation_step<F, P>(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        output: &mut [f64],
        stats: &mut SolverStats,
    ) -> Result<(), SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        let n = self.dimension;
        let s = self.current_stages;
        let mut initial = vec![0.0; n];
        Self::evaluate(problem, &mut initial, state, time, stats)?;
        for stage in self.stage_derivatives.chunks_exact_mut(n) {
            stage.copy_from_slice(&initial);
        }

        for _ in 0..MAX_NEWTON_ITERATIONS {
            for i in 0..s {
                let stage_state = &mut self.stage_states[i * n..(i + 1) * n];
                stage_state.copy_from_slice(state);
                for j in 0..s {
                    let coefficient = step * self.tableau.a[i * s + j];
                    let derivative = &self.stage_derivatives[j * n..(j + 1) * n];
                    for component in 0..n {
                        stage_state[component] += coefficient * derivative[component];
                    }
                }
                let residual = &mut self.residual[i * n..(i + 1) * n];
                Self::evaluate(
                    problem,
                    residual,
                    stage_state,
                    time + self.tableau.c[i] * step,
                    stats,
                )?;
                for (component, residual_value) in residual.iter_mut().enumerate() {
                    *residual_value = self.stage_derivatives[i * n + component] - *residual_value;
                }
            }
            let residual_norm = self
                .residual
                .iter()
                .map(|value| value.abs())
                .fold(0.0, f64::max);
            let scale = self
                .stage_derivatives
                .iter()
                .map(|value| value.abs())
                .fold(1.0, f64::max);
            if residual_norm <= NEWTON_TOLERANCE * scale {
                output.copy_from_slice(state);
                for j in 0..s {
                    let coefficient = step * self.tableau.b[j];
                    for (component, output_value) in output.iter_mut().enumerate() {
                        *output_value += coefficient * self.stage_derivatives[j * n + component];
                    }
                }
                return output
                    .iter()
                    .all(|value| value.is_finite())
                    .then_some(())
                    .ok_or(SolveError::NonFiniteDerivative);
            }

            for i in 0..s {
                let stage_state = &self.stage_states[i * n..(i + 1) * n];
                let jacobian = &mut self.stage_jacobians[i * n * n..(i + 1) * n * n];
                if problem.evaluate_jacobian(jacobian, stage_state, time + self.tableau.c[i] * step)
                {
                    if jacobian.iter().any(|value| !value.is_finite()) {
                        return Err(SolveError::NonFiniteDerivative);
                    }
                } else {
                    for column in 0..n {
                        self.perturbed_state.copy_from_slice(stage_state);
                        let perturbation = f64::EPSILON.sqrt() * stage_state[column].abs().max(1.0);
                        self.perturbed_state[column] += perturbation;
                        Self::evaluate(
                            problem,
                            &mut self.perturbed_derivative,
                            &self.perturbed_state,
                            time + self.tableau.c[i] * step,
                            stats,
                        )?;
                        for row in 0..n {
                            let base_derivative =
                                self.stage_derivatives[i * n + row] - self.residual[i * n + row];
                            jacobian[row * n + column] =
                                (self.perturbed_derivative[row] - base_derivative) / perturbation;
                        }
                    }
                }
                stats.jacobian_evaluations += 1;
            }

            let coupled = n * s;
            self.matrix.fill(0.0);
            for i in 0..s {
                let jacobian = &self.stage_jacobians[i * n * n..(i + 1) * n * n];
                for j in 0..s {
                    let a = step * self.tableau.a[i * s + j];
                    for row in 0..n {
                        for column in 0..n {
                            let matrix_row = i * n + row;
                            let matrix_column = j * n + column;
                            self.matrix[matrix_row * coupled + matrix_column] =
                                f64::from(i == j && row == column) - a * jacobian[row * n + column];
                        }
                    }
                }
            }
            for (correction, residual) in self.correction.iter_mut().zip(&self.residual) {
                *correction = -*residual;
            }
            factorize(&mut self.matrix, &mut self.pivots, coupled)?;
            stats.linear_factorizations += 1;
            solve_factorized(&self.matrix, &self.pivots, &mut self.correction, coupled);
            stats.linear_solves += 1;
            stats.nonlinear_iterations += 1;
            for (stage, correction) in self.stage_derivatives.iter_mut().zip(&self.correction) {
                *stage += correction;
            }
        }
        Err(SolveError::NonlinearSolveFailed)
    }

    #[allow(clippy::too_many_arguments)]
    fn interpolate_segment(
        tableau: &Tableau,
        dimension: usize,
        stages: &[f64],
        start_state: &[f64],
        start_time: f64,
        step: f64,
        time: f64,
        output: &mut [f64],
        weights: &mut [f64],
    ) -> Result<(), SolveError> {
        let theta = ((time - start_time) / step).clamp(0.0, 1.0);
        tableau.weights_at(theta, weights);
        output.copy_from_slice(start_state);
        for j in 0..tableau.stages {
            for component in 0..dimension {
                output[component] += step * weights[j] * stages[j * dimension + component];
            }
        }
        output
            .iter()
            .all(|value| value.is_finite())
            .then_some(())
            .ok_or(SolveError::NonFiniteDerivative)
    }

    fn interpolate_attempt(
        &mut self,
        start_state: &[f64],
        time: f64,
        output: &mut [f64],
    ) -> Result<(), SolveError> {
        if self.adaptive_attempt {
            let half = 0.5 * self.segment_step;
            let direction = self.segment_step.signum();
            if direction * (time - (self.segment_start_time + half)) <= 0.0 {
                Self::interpolate_segment(
                    &self.tableau,
                    self.dimension,
                    &self.first_half_stages,
                    start_state,
                    self.segment_start_time,
                    half,
                    time,
                    output,
                    &mut self.interpolation_weights,
                )
            } else {
                Self::interpolate_segment(
                    &self.tableau,
                    self.dimension,
                    &self.second_half_stages,
                    &self.midpoint_state,
                    self.segment_start_time + half,
                    half,
                    time,
                    output,
                    &mut self.interpolation_weights,
                )
            }
        } else {
            Self::interpolate_segment(
                &self.tableau,
                self.dimension,
                &self.stage_derivatives,
                start_state,
                self.segment_start_time,
                self.segment_step,
                time,
                output,
                &mut self.interpolation_weights,
            )
        }
    }
}

impl<F, P> StepKernel<F, P> for FirkKernel
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::with_controller(
            true,
            ControllerConfig::proportional(self.tableau.order + 1, 0.9, 0.2, 5.0, 0.25),
        )
        .recover_nonlinear_and_singular_failures()
    }

    fn initialize(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: f64,
        _: &mut SolverStats,
    ) -> Result<(), SolveError> {
        Ok(())
    }

    fn estimate_initial_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        _: f64,
        maximum_step: f64,
        candidate: &mut [f64],
        options: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<f64, SolveError> {
        Self::evaluate(problem, candidate, state, time, stats)?;
        let mut state_norm = 0.0_f64;
        let mut derivative_norm = 0.0_f64;
        for (&value, &derivative) in state.iter().zip(candidate.iter()) {
            let scale = options.absolute_tolerance + options.relative_tolerance * value.abs();
            state_norm = state_norm.max(value.abs() / scale);
            derivative_norm = derivative_norm.max(derivative.abs() / scale);
        }
        let estimate = if derivative_norm <= 1.0e-14 {
            1.0e-3
        } else {
            0.01 * state_norm.max(1.0) / derivative_norm
        };
        Ok(estimate.clamp(1.0e-8, maximum_step))
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
        self.segment_start_time = time;
        self.segment_step = step;
        self.adaptive_attempt = options.adaptive;
        if !options.adaptive {
            // Move the stage storage out to avoid borrowing the whole kernel
            // through the output argument.
            let mut output = vec![0.0; self.dimension];
            self.collocation_step(problem, state, time, step, &mut output, stats)?;
            candidate.copy_from_slice(&output);
            self.last_error = 0.0;
            return Ok(StepEstimate::new(0.0));
        }

        let mut full = vec![0.0; self.dimension];
        self.collocation_step(problem, state, time, step, &mut full, stats)?;
        self.full_state.copy_from_slice(&full);

        let half = 0.5 * step;
        let mut midpoint = vec![0.0; self.dimension];
        self.collocation_step(problem, state, time, half, &mut midpoint, stats)?;
        self.midpoint_state.copy_from_slice(&midpoint);
        self.first_half_stages
            .copy_from_slice(&self.stage_derivatives);

        let mut endpoint = vec![0.0; self.dimension];
        self.collocation_step(problem, &midpoint, time + half, half, &mut endpoint, stats)?;
        candidate.copy_from_slice(&endpoint);
        self.second_half_stages
            .copy_from_slice(&self.stage_derivatives);

        let divisor = (2.0_f64).powi(self.tableau.order as i32) - 1.0;
        let mut error_norm = 0.0_f64;
        for component in 0..self.dimension {
            let scale = options.absolute_tolerance
                + options.relative_tolerance
                    * state[component].abs().max(candidate[component].abs());
            error_norm =
                error_norm.max(((candidate[component] - full[component]) / divisor / scale).abs());
        }
        self.last_error = error_norm;
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
        let mut scratch = vec![0.0; self.dimension];
        let mut interpolator = |query: f64, output: &mut [f64]| {
            self.interpolate_attempt(previous_state, query, &mut scratch)?;
            output.copy_from_slice(&scratch);
            Ok(())
        };
        problem.apply_step_callbacks(
            previous_state,
            previous_time,
            state,
            time,
            state_before_effect,
            event_tolerance,
            Some(&mut interpolator),
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
        let segment = CollocationAttemptSegment {
            tableau: &self.tableau,
            dimension: self.dimension,
            start_state: previous_state,
            midpoint_state: &self.midpoint_state,
            endpoint_state: state,
            full_stages: &self.stage_derivatives,
            first_half_stages: &self.first_half_stages,
            second_half_stages: &self.second_half_stages,
            start_time: previous_time,
            attempted_time,
            bound_time: time,
            adaptive: self.adaptive_attempt,
        };
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
        Ok(true)
    }

    fn accept_step(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: &[f64],
        _: f64,
        _: f64,
        callback_applied: bool,
        _: &mut SolverStats,
    ) -> Result<(), SolveError> {
        if self.maximum_stages > self.minimum_stages && !callback_applied {
            let next = if self.last_error < 0.03 && self.current_stages < self.maximum_stages {
                self.current_stages + 1
            } else if self.last_error > 0.8 && self.current_stages > self.minimum_stages {
                self.current_stages - 1
            } else {
                self.current_stages
            };
            if next != self.current_stages {
                self.current_stages = next;
                self.resize_stage_workspace();
            }
        }
        Ok(())
    }

    fn reject_step(&mut self) {
        if self.current_stages < self.maximum_stages {
            self.current_stages += 1;
            self.resize_stage_workspace();
        }
    }
}

struct CollocationAttemptSegment<'a> {
    tableau: &'a Tableau,
    dimension: usize,
    start_state: &'a [f64],
    midpoint_state: &'a [f64],
    endpoint_state: &'a [f64],
    full_stages: &'a [f64],
    first_half_stages: &'a [f64],
    second_half_stages: &'a [f64],
    start_time: f64,
    attempted_time: f64,
    bound_time: f64,
    adaptive: bool,
}

impl DenseSegment for CollocationAttemptSegment<'_> {
    fn interpolate(&self, time: f64, output: &mut [f64]) -> Result<(), &'static str> {
        if output.len() != self.dimension {
            return Err("dense output dimension mismatch");
        }
        if time == self.bound_time {
            output.copy_from_slice(self.endpoint_state);
            return Ok(());
        }
        let mut weights = vec![0.0; self.tableau.stages];
        let step = self.attempted_time - self.start_time;
        let result = if self.adaptive {
            let half = 0.5 * step;
            if step.signum() * (time - (self.start_time + half)) <= 0.0 {
                FirkKernel::interpolate_segment(
                    self.tableau,
                    self.dimension,
                    self.first_half_stages,
                    self.start_state,
                    self.start_time,
                    half,
                    time,
                    output,
                    &mut weights,
                )
            } else {
                FirkKernel::interpolate_segment(
                    self.tableau,
                    self.dimension,
                    self.second_half_stages,
                    self.midpoint_state,
                    self.start_time + half,
                    half,
                    time,
                    output,
                    &mut weights,
                )
            }
        } else {
            FirkKernel::interpolate_segment(
                self.tableau,
                self.dimension,
                self.full_stages,
                self.start_state,
                self.start_time,
                step,
                time,
                output,
                &mut weights,
            )
        };
        result.map_err(|_| "non-finite collocation interpolation")
    }
}

#[cfg(test)]
mod tests {
    use super::{AdaptiveRadau, Family, GaussLegendre, RadauIIA3, RadauIIA5, RadauIIA9, Tableau};
    use crate::{OdeProblem, SaveMode, SolveOptions, solve};

    #[test]
    fn generated_tableaus_have_collocation_moments() {
        for (family, stages) in [
            (Family::Radau, 2),
            (Family::Radau, 3),
            (Family::Radau, 5),
            (Family::Radau, 7),
            (Family::Gauss, 2),
        ] {
            let tableau = Tableau::generate(family, stages);
            assert!((tableau.b.iter().sum::<f64>() - 1.0).abs() < 2.0e-12);
            for i in 0..stages {
                assert!(
                    (tableau.a[i * stages..(i + 1) * stages].iter().sum::<f64>() - tableau.c[i])
                        .abs()
                        < 2.0e-12
                );
            }
        }
    }

    #[test]
    fn fixed_radau_methods_integrate_stiff_decay() {
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -40.0 * u[0],
            vec![1.0],
            (0.0, 0.2),
            (),
        );
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.01),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };
        for value in [
            solve(&problem, RadauIIA3, &options).unwrap().last_state()[0],
            solve(&problem, RadauIIA5, &options).unwrap().last_state()[0],
            solve(&problem, RadauIIA9, &options).unwrap().last_state()[0],
        ] {
            assert!((value - (-8.0_f64).exp()).abs() < 3.0e-6, "{value}");
        }
    }

    #[test]
    fn adaptive_firk_variants_honor_tolerances() {
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
            vec![1.0],
            (0.0, 1.0),
            (),
        );
        let options = SolveOptions {
            absolute_tolerance: 1.0e-8,
            relative_tolerance: 1.0e-8,
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };
        for value in [
            solve(&problem, AdaptiveRadau::default(), &options)
                .unwrap()
                .last_state()[0],
            solve(&problem, GaussLegendre::default(), &options)
                .unwrap()
                .last_state()[0],
        ] {
            assert!((value - std::f64::consts::E).abs() < 2.0e-7, "{value}");
        }
    }
}
