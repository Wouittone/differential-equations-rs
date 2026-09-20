use crate::integrator::{ControllerConfig, KernelCapabilities, StepEstimate, StepKernel};
use crate::linear::{factorize, solve_factorized};
use crate::solution::{BorrowedCollocationSegment, CollocationSegment, TrajectoryRecorder};
use crate::{OdeProblem, SolveError, SolveOptions, SolverStats};

use super::dense::interpolate_segment;
use super::tableau::{Family, Tableau};

const MAX_NEWTON_ITERATIONS: usize = 12;
const NEWTON_TOLERANCE: f64 = 2.0e-11;

pub(super) struct FirkKernel {
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
    pub(super) fn new(
        dimension: usize,
        family: Family,
        minimum_stages: usize,
        maximum_stages: usize,
    ) -> Self {
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
        F: crate::OdeFunction<P>,
    {
        problem
            .rhs
            .evaluate(output, state, problem.parameters(), time)?;
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
        F: crate::OdeFunction<P>,
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
                interpolate_segment(
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
                interpolate_segment(
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
            interpolate_segment(
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
    F: crate::OdeFunction<P>,
{
    fn has_custom_dense_output(&self) -> bool {
        true
    }

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
        let segment = BorrowedCollocationSegment::new(
            previous_time,
            attempted_time,
            previous_state,
            &self.midpoint_state,
            state,
            &self.stage_derivatives,
            &self.first_half_stages,
            &self.second_half_stages,
            &self.tableau.lagrange,
            self.tableau.stages,
            self.adaptive_attempt,
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
            recorder.retain_collocation_segment(
                CollocationSegment::new(
                    previous_time,
                    attempted_time,
                    time,
                    previous_state,
                    &self.midpoint_state,
                    state,
                    &self.stage_derivatives,
                    &self.first_half_stages,
                    &self.second_half_stages,
                    &self.tableau.lagrange,
                    self.tableau.stages,
                    self.adaptive_attempt,
                )
                .map_err(|_| SolveError::NonFiniteDerivative)?,
            );
        }
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
