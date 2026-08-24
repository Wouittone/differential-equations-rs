//! Variable-order Adams and BDF methods in Nordsieck form.

use std::marker::PhantomData;

use crate::explicit_rk::ButcherTableau;
use crate::integrator::{
    ControllerConfig, KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::linear::{factorize, solve_factorized};
use crate::solution::{BorrowedHermiteSegment, DenseSegment, HermiteSegment, TrajectoryRecorder};
use crate::tsit5::Tsit5;
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

const MAX_NEWTON_ITERATIONS: usize = 12;
const NEWTON_TOLERANCE: f64 = 1.0e-11;

/// Adaptive fifth-order fixed-leading-coefficient Adams method in Nordsieck form.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AN5;

/// Equation family used by [`JVODE`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum JvodeMethod {
    /// Variable-order Adams corrector (orders 1 through 12).
    #[default]
    Adams,
    /// Variable-order BDF corrector (orders 1 through 5).
    Bdf,
}

/// Variable-order, variable-step Nordsieck integrator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JVODE {
    method: JvodeMethod,
    bias1: f64,
    bias2: f64,
    bias3: f64,
    addon: f64,
    minimum_factor: f64,
    maximum_factor: f64,
    steady_maximum: f64,
}

impl JVODE {
    pub const fn new(method: JvodeMethod) -> Self {
        Self {
            method,
            bias1: 6.0,
            bias2: 6.0,
            bias3: 10.0,
            addon: 1.0e-6,
            minimum_factor: 0.2,
            maximum_factor: 10.0,
            steady_maximum: 1.5,
        }
    }

    pub const fn adams() -> Self {
        Self::new(JvodeMethod::Adams)
    }

    pub const fn bdf() -> Self {
        Self::new(JvodeMethod::Bdf)
    }

    pub const fn method(&self) -> JvodeMethod {
        self.method
    }

    #[must_use]
    pub const fn with_biases(mut self, lower: f64, current: f64, higher: f64) -> Self {
        self.bias1 = lower;
        self.bias2 = current;
        self.bias3 = higher;
        self
    }

    #[must_use]
    pub const fn with_step_factors(mut self, minimum: f64, maximum: f64) -> Self {
        self.minimum_factor = minimum;
        self.maximum_factor = maximum;
        self
    }
}

impl Default for JVODE {
    fn default() -> Self {
        Self::adams()
    }
}

/// Configured Adams alias for [`JVODE`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct JvodeAdams;

/// Configured BDF alias for [`JVODE`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct JvodeBdf;

#[allow(non_camel_case_types)]
pub type JVODE_Adams = JvodeAdams;
#[allow(non_camel_case_types)]
pub type JVODE_BDF = JvodeBdf;

impl OdeAlgorithm for AN5 {
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
            NordsieckKernel::an5(problem.initial_state().len()),
        )
    }
}

impl OdeAlgorithm for JVODE {
    fn solve<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        if !self.bias1.is_finite()
            || !self.bias2.is_finite()
            || !self.bias3.is_finite()
            || self.bias1 <= 0.0
            || self.bias2 <= 0.0
            || self.bias3 <= 0.0
            || !self.minimum_factor.is_finite()
            || !self.maximum_factor.is_finite()
            || self.minimum_factor <= 0.0
            || self.maximum_factor < self.minimum_factor
        {
            return Err(SolveError::InvalidMultistepOrder);
        }
        drive_integration(
            problem,
            options,
            NordsieckKernel::jvode(problem.initial_state().len(), *self),
        )
    }
}

impl OdeAlgorithm for JvodeAdams {
    fn solve<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        JVODE::adams().solve(problem, options)
    }
}

impl OdeAlgorithm for JvodeBdf {
    fn solve<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        JVODE::bdf().solve(problem, options)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Family {
    An5,
    Jvode(JvodeMethod),
}

#[derive(Clone)]
struct Snapshot {
    z: Vec<Vec<f64>>,
    dts: Vec<f64>,
    order: usize,
    next_order: usize,
    n_wait: usize,
    previous_step: f64,
    previous_d: f64,
    current_d: f64,
    started: bool,
}

struct NordsieckKernel<F, P> {
    family: Family,
    config: JVODE,
    dimension: usize,
    maximum_order: usize,
    z: Vec<Vec<f64>>,
    l: Vec<f64>,
    m: Vec<f64>,
    dts: Vec<f64>,
    delta: Vec<f64>,
    start_derivative: Vec<f64>,
    end_derivative: Vec<f64>,
    dense_endpoint: Vec<f64>,
    order: usize,
    next_order: usize,
    n_wait: usize,
    previous_step: f64,
    previous_d: f64,
    current_d: f64,
    c_lte: f64,
    c_lte_down: f64,
    c_lte_up: f64,
    c_conv: f64,
    proposed_factor: f64,
    last_error: f64,
    started: bool,
    backup: Option<Snapshot>,
    _marker: PhantomData<fn(F, P)>,
}

impl<F, P> NordsieckKernel<F, P> {
    fn an5(dimension: usize) -> Self {
        Self::new(Family::An5, JVODE::adams(), dimension, 5)
    }

    fn jvode(dimension: usize, config: JVODE) -> Self {
        let maximum_order = match config.method {
            JvodeMethod::Adams => 12,
            JvodeMethod::Bdf => 5,
        };
        Self::new(
            Family::Jvode(config.method),
            config,
            dimension,
            maximum_order,
        )
    }

    fn new(family: Family, config: JVODE, dimension: usize, maximum_order: usize) -> Self {
        Self {
            family,
            config,
            dimension,
            maximum_order,
            z: vec![vec![0.0; dimension]; maximum_order + 1],
            l: vec![0.0; maximum_order + 1],
            m: vec![0.0; maximum_order + 1],
            dts: vec![0.0; maximum_order + 1],
            delta: vec![0.0; dimension],
            start_derivative: vec![0.0; dimension],
            end_derivative: vec![0.0; dimension],
            dense_endpoint: vec![0.0; dimension],
            order: 1,
            next_order: 1,
            n_wait: 2,
            previous_step: 0.0,
            previous_d: 0.0,
            current_d: 0.0,
            c_lte: 0.5,
            c_lte_down: 1.0,
            c_lte_up: 1.0 / 12.0,
            c_conv: 0.2,
            proposed_factor: 1.0,
            last_error: 0.0,
            started: false,
            backup: None,
            _marker: PhantomData,
        }
    }

    fn snapshot(&self) -> Snapshot {
        Snapshot {
            z: self.z.clone(),
            dts: self.dts.clone(),
            order: self.order,
            next_order: self.next_order,
            n_wait: self.n_wait,
            previous_step: self.previous_step,
            previous_d: self.previous_d,
            current_d: self.current_d,
            started: self.started,
        }
    }

    fn restore(&mut self, snapshot: Snapshot) {
        self.z = snapshot.z;
        self.dts = snapshot.dts;
        self.order = snapshot.order;
        self.next_order = snapshot.next_order;
        self.n_wait = snapshot.n_wait;
        self.previous_step = snapshot.previous_step;
        self.previous_d = snapshot.previous_d;
        self.current_d = snapshot.current_d;
        self.started = snapshot.started;
    }

    fn evaluate<FN, PP>(
        problem: &OdeProblem<FN, PP>,
        output: &mut [f64],
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError>
    where
        FN: Fn(&mut [f64], &[f64], &PP, f64),
    {
        (problem.rhs)(output, state, problem.parameters(), time);
        stats.rhs_evaluations += 1;
        output
            .iter()
            .all(|value| value.is_finite())
            .then_some(())
            .ok_or(SolveError::NonFiniteDerivative)
    }

    fn scaled_norm(&self, values: &[f64], old: &[f64], new: &[f64], options: &SolveOptions) -> f64 {
        values
            .iter()
            .zip(old)
            .zip(new)
            .fold(0.0_f64, |norm, ((value, old), new)| {
                let scale = options.absolute_tolerance
                    + options.relative_tolerance * old.abs().max(new.abs());
                norm.max(value.abs() / scale)
            })
    }

    fn predict(&mut self, rewind: bool) {
        for i in 1..=self.order {
            for j in (i..=self.order).rev() {
                for component in 0..self.dimension {
                    if rewind {
                        self.z[j - 1][component] -= self.z[j][component];
                    } else {
                        self.z[j - 1][component] += self.z[j][component];
                    }
                }
            }
        }
    }

    fn rescale(&mut self, ratio: f64) {
        let mut factor = ratio;
        for derivative in 1..=self.order {
            for value in &mut self.z[derivative] {
                *value *= factor;
            }
            factor *= ratio;
        }
    }

    fn adjust_order(&mut self, new_order: usize) {
        if new_order == self.order {
            return;
        }
        if new_order > self.order {
            self.z[self.order + 1].fill(0.0);
            if self.family == Family::Jvode(JvodeMethod::Bdf) {
                self.z[self.order + 1].copy_from_slice(&self.delta);
            }
        } else if self.order > 1 {
            self.l.fill(0.0);
            self.l[1] = 1.0;
            let mut sum = 0.0;
            for j in 1..self.order.saturating_sub(2) {
                sum += self.dts[j];
                let xi = sum / self.dts[0];
                for i in (0..=j).rev() {
                    self.l[i + 1] = self.l[i + 1] * xi + self.l[i];
                }
            }
            for j in 1..self.order.saturating_sub(1) {
                self.l[j + 1] = self.order as f64 * self.l[j] / j as f64;
            }
            for j in 2..self.order {
                for component in 0..self.dimension {
                    self.z[j][component] -= self.l[j] * self.z[self.order][component];
                }
            }
        }
        self.order = new_order;
        self.next_order = new_order;
        self.n_wait = new_order + 1;
    }

    fn integral_minus_one(coefficients: &[f64], degree: usize, offset: usize) -> f64 {
        let mut integral = 0.0;
        let mut sign = 1.0;
        for (index, coefficient) in coefficients.iter().take(degree + 1).enumerate() {
            integral += sign * coefficient / (index + offset) as f64;
            sign = -sign;
        }
        integral
    }

    fn calculate_coefficients(&mut self) {
        if self.order == 1 {
            self.l[0] = 1.0;
            self.l[1] = 1.0;
            self.c_lte_down = 1.0;
            self.current_d = 1.0;
            self.c_lte = 0.5;
            self.c_lte_up = 1.0 / 12.0;
            self.c_conv = 0.1 / self.c_lte;
            return;
        }
        self.m.fill(0.0);
        self.m[0] = 1.0;
        let dt = self.dts[0];
        let mut dt_sum = dt;
        let changing_order = self.n_wait == 1;
        let mut xi_inverse;
        for j in 0..(self.order - 1) {
            if changing_order && j == self.order - 2 {
                let down = Self::integral_minus_one(&self.m, self.order - 2, 2);
                self.c_lte_down = self.order as f64 * down / self.m[self.order - 2];
            }
            xi_inverse = dt / dt_sum;
            for i in (0..=j).rev() {
                self.m[i + 1] += self.m[i] * xi_inverse;
            }
            dt_sum += self.dts[j + 1];
        }
        xi_inverse = dt / dt_sum;
        let m0 = Self::integral_minus_one(&self.m, self.order - 1, 1);
        let m1 = Self::integral_minus_one(&self.m, self.order - 1, 2);
        let inverse_m0 = 1.0 / m0;
        self.l[0] = 1.0;
        for i in 1..=self.order {
            self.l[i] = inverse_m0 * self.m[i - 1] / i as f64;
        }
        self.c_lte = m1 * inverse_m0 * xi_inverse;
        self.current_d = 1.0 / xi_inverse / self.l[self.order];
        if changing_order {
            for i in (0..self.order).rev() {
                self.m[i + 1] += xi_inverse * self.m[i];
            }
            let m2 = Self::integral_minus_one(&self.m, self.order, 2);
            self.c_lte_up = m2 * inverse_m0 / (self.order + 1) as f64;
        }
        self.c_conv = 0.1 / self.c_lte.abs();
    }

    fn update_z(&mut self) {
        for derivative in 0..=self.order {
            for component in 0..self.dimension {
                self.z[derivative][component] += self.l[derivative] * self.delta[component];
            }
        }
    }

    fn choose_factor(&mut self, error: f64, old: &[f64], new: &[f64], options: &SolveOptions) {
        let length = self.order + 1;
        if error > 1.0 {
            self.n_wait = self.n_wait.max(2);
            self.next_order = self.order;
            self.proposed_factor =
                1.0 / ((self.config.bias2 * error).powf(1.0 / length as f64) + self.config.addon);
            return;
        }
        self.n_wait = self.n_wait.saturating_sub(1);
        let eta_current = if error == 0.0 {
            self.config.maximum_factor
        } else {
            1.0 / ((self.config.bias2 * error).powf(1.0 / length as f64) + self.config.addon)
        };
        let mut chosen = eta_current;
        self.next_order = self.order;
        if self.n_wait == 1 && self.order < self.maximum_order {
            self.z[self.maximum_order].copy_from_slice(&self.delta);
            self.previous_d = self.current_d;
        }
        if self.n_wait == 0 {
            self.n_wait = 2;
            if self.order > 1 {
                let down_error = self.scaled_norm(&self.z[self.order], old, new, options)
                    * self.c_lte_down.abs();
                let eta_down = if down_error == 0.0 {
                    self.config.maximum_factor
                } else {
                    1.0 / ((self.config.bias1 * down_error).powf(1.0 / self.order as f64)
                        + self.config.addon)
                };
                if eta_down > chosen {
                    chosen = eta_down;
                    self.next_order = self.order - 1;
                }
            }
            if self.order < self.maximum_order && self.previous_d != 0.0 && self.dts[1] != 0.0 {
                let quotient = (self.current_d / self.previous_d)
                    * (self.dts[0] / self.dts[1]).powi(length as i32);
                let difference: Vec<f64> = self
                    .delta
                    .iter()
                    .zip(&self.z[self.maximum_order])
                    .map(|(delta, stored)| delta - quotient * stored)
                    .collect();
                let up_error =
                    self.scaled_norm(&difference, old, new, options) * self.c_lte_up.abs();
                let eta_up = if up_error == 0.0 {
                    self.config.maximum_factor
                } else {
                    1.0 / ((self.config.bias3 * up_error).powf(1.0 / (length + 1) as f64)
                        + self.config.addon)
                };
                if eta_up > chosen {
                    chosen = eta_up;
                    self.next_order = self.order + 1;
                }
            }
        }
        if chosen < self.config.steady_maximum {
            chosen = 1.0;
            self.next_order = self.order;
        } else {
            chosen = chosen.clamp(self.config.minimum_factor, self.config.maximum_factor);
        }
        self.proposed_factor = chosen;
    }

    fn functional_correct<FN, PP>(
        &mut self,
        problem: &OdeProblem<FN, PP>,
        time: f64,
        step: f64,
        candidate: &mut [f64],
        stats: &mut SolverStats,
    ) -> Result<(), SolveError>
    where
        FN: Fn(&mut [f64], &[f64], &PP, f64),
    {
        let mut derivative = vec![0.0; self.dimension];
        let mut previous_delta = vec![0.0; self.dimension];
        candidate.copy_from_slice(&self.z[0]);
        for iteration in 0..3 {
            Self::evaluate(problem, &mut derivative, candidate, time + step, stats)?;
            let mut correction_norm = 0.0_f64;
            for component in 0..self.dimension {
                let next_delta = (step * derivative[component] - self.z[1][component]) / self.l[1];
                correction_norm =
                    correction_norm.max((next_delta - previous_delta[component]).abs());
                self.delta[component] = next_delta;
                candidate[component] = self.z[0][component] + next_delta;
            }
            stats.nonlinear_iterations += 1;
            if correction_norm
                <= self.c_conv * (1.0 + candidate.iter().fold(0.0_f64, |n, x| n.max(x.abs())))
            {
                return Ok(());
            }
            previous_delta.copy_from_slice(&self.delta);
            if iteration == 2 {
                return Err(SolveError::NonlinearSolveFailed);
            }
        }
        Err(SolveError::NonlinearSolveFailed)
    }

    #[allow(clippy::needless_range_loop)]
    fn newton_correct<FN, PP>(
        &mut self,
        problem: &OdeProblem<FN, PP>,
        time: f64,
        step: f64,
        candidate: &mut [f64],
        stats: &mut SolverStats,
    ) -> Result<(), SolveError>
    where
        FN: Fn(&mut [f64], &[f64], &PP, f64),
    {
        let n = self.dimension;
        let scale = step / self.l[1];
        let mut derivative = vec![0.0; n];
        let mut perturbed_derivative = vec![0.0; n];
        let mut perturbed = vec![0.0; n];
        let mut residual = vec![0.0; n];
        let mut matrix = vec![0.0; n * n];
        let mut pivots = vec![0usize; n];
        candidate.copy_from_slice(&self.z[0]);
        for _ in 0..MAX_NEWTON_ITERATIONS {
            Self::evaluate(problem, &mut derivative, candidate, time + step, stats)?;
            let mut norm = 0.0_f64;
            for component in 0..n {
                residual[component] = candidate[component]
                    - self.z[0][component]
                    - (step * derivative[component] - self.z[1][component]) / self.l[1];
                norm = norm.max(residual[component].abs());
            }
            if norm
                <= NEWTON_TOLERANCE * (1.0 + candidate.iter().fold(0.0_f64, |n, x| n.max(x.abs())))
            {
                for component in 0..n {
                    self.delta[component] = candidate[component] - self.z[0][component];
                }
                return Ok(());
            }
            if !problem.evaluate_jacobian(&mut matrix, candidate, time + step) {
                for column in 0..n {
                    perturbed.copy_from_slice(candidate);
                    let epsilon = f64::EPSILON.sqrt() * (1.0 + candidate[column].abs());
                    perturbed[column] += epsilon;
                    Self::evaluate(
                        problem,
                        &mut perturbed_derivative,
                        &perturbed,
                        time + step,
                        stats,
                    )?;
                    for row in 0..n {
                        matrix[row * n + column] =
                            (perturbed_derivative[row] - derivative[row]) / epsilon;
                    }
                }
            }
            stats.jacobian_evaluations += 1;
            for row in 0..n {
                for column in 0..n {
                    matrix[row * n + column] *= -scale;
                }
                matrix[row * n + row] += 1.0;
                residual[row] = -residual[row];
            }
            factorize(&mut matrix, &mut pivots, n)?;
            stats.linear_factorizations += 1;
            solve_factorized(&matrix, &pivots, &mut residual, n);
            stats.linear_solves += 1;
            stats.nonlinear_iterations += 1;
            for (value, correction) in candidate.iter_mut().zip(&residual) {
                *value += correction;
            }
        }
        Err(SolveError::NonlinearSolveFailed)
    }

    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn tsit5_start<FN, PP>(
        &mut self,
        problem: &OdeProblem<FN, PP>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        options: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<f64, SolveError>
    where
        FN: Fn(&mut [f64], &[f64], &PP, f64),
    {
        let stages_count = <Tsit5 as ButcherTableau>::NODES.len();
        let mut stages = vec![vec![0.0; self.dimension]; stages_count];
        stages[0].copy_from_slice(&self.start_derivative);
        let mut temporary = vec![0.0; self.dimension];
        for stage in 1..stages_count {
            temporary.copy_from_slice(state);
            for previous in 0..stage {
                let coefficient = <Tsit5 as ButcherTableau>::COEFFICIENTS[stage][previous];
                for component in 0..self.dimension {
                    temporary[component] += step * coefficient * stages[previous][component];
                }
            }
            Self::evaluate(
                problem,
                &mut stages[stage],
                &temporary,
                time + <Tsit5 as ButcherTableau>::NODES[stage] * step,
                stats,
            )?;
        }
        candidate.copy_from_slice(state);
        let mut error = vec![0.0; self.dimension];
        for stage in 0..stages_count {
            for component in 0..self.dimension {
                candidate[component] +=
                    step * <Tsit5 as ButcherTableau>::WEIGHTS[stage] * stages[stage][component];
                error[component] += step
                    * <Tsit5 as ButcherTableau>::ERROR_WEIGHTS.expect("Tsit5 embedded pair")[stage]
                    * stages[stage][component];
            }
        }
        let dense =
            <Tsit5 as ButcherTableau>::DENSE_COEFFICIENTS.expect("Tsit5 continuous extension");
        self.z[0].copy_from_slice(state);
        for derivative in 1..=4 {
            self.z[derivative].fill(0.0);
            for stage in 0..stages_count {
                for component in 0..self.dimension {
                    self.z[derivative][component] +=
                        step * dense[stage][derivative - 1] * stages[stage][component];
                }
            }
        }
        self.order = 4;
        self.next_order = 4;
        self.dts.fill(step);
        self.predict(false);
        self.previous_step = step;
        self.started = true;
        self.proposed_factor = 1.0;
        Ok(self.scaled_norm(&error, state, candidate, options))
    }

    fn hermite_interpolate(
        &self,
        previous_state: &[f64],
        state: &[f64],
        previous_time: f64,
        time: f64,
        query: f64,
        output: &mut [f64],
    ) -> Result<(), SolveError> {
        let segment = BorrowedHermiteSegment::new(
            previous_time,
            time,
            previous_state,
            state,
            &self.start_derivative,
            &self.end_derivative,
        )
        .map_err(|_| SolveError::NonFiniteDerivative)?;
        segment
            .interpolate(query, output)
            .map_err(|_| SolveError::NonFiniteDerivative)
    }
}

impl<F, P> StepKernel<F, P> for NordsieckKernel<F, P>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::with_controller(
            true,
            ControllerConfig::proportional(6, 0.9, 0.2, 10.0, 0.25),
        )
        .recover_nonlinear_and_singular_failures()
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
        let mut state_norm = 0.0_f64;
        let mut derivative_norm = 0.0_f64;
        for (&value, &derivative) in state.iter().zip(&self.start_derivative) {
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
        self.backup = Some(self.snapshot());
        if self.family == Family::An5 && !self.started {
            let error = self.tsit5_start(problem, state, time, step, candidate, options, stats)?;
            Self::evaluate(
                problem,
                &mut self.end_derivative,
                candidate,
                time + step,
                stats,
            )?;
            self.last_error = error;
            let factor = if error == 0.0 {
                10.0
            } else {
                (0.9 * error.powf(-1.0 / 5.0)).clamp(0.2, 10.0)
            };
            self.proposed_factor = factor;
            return Ok(StepEstimate::with_factor(error, factor));
        }
        if !self.started {
            self.order = 1;
            self.next_order = 1;
            self.n_wait = 2;
            self.z.iter_mut().for_each(|entry| entry.fill(0.0));
            self.z[0].copy_from_slice(state);
            for component in 0..self.dimension {
                self.z[1][component] = step * self.start_derivative[component];
            }
            self.dts.fill(step);
            self.previous_step = step;
            self.started = true;
        } else {
            self.adjust_order(self.next_order);
            if self.previous_step != step {
                self.rescale(step / self.previous_step);
            }
            for index in (1..self.dts.len()).rev() {
                self.dts[index] = self.dts[index - 1];
            }
            self.dts[0] = step;
        }
        self.predict(false);
        self.calculate_coefficients();
        let correction = match self.family {
            Family::Jvode(JvodeMethod::Bdf) => {
                self.newton_correct(problem, time, step, candidate, stats)
            }
            _ => self.functional_correct(problem, time, step, candidate, stats),
        };
        if let Err(error) = correction {
            if let Some(snapshot) = self.backup.take() {
                self.restore(snapshot);
            }
            return Err(error);
        }
        let error = if options.adaptive {
            self.scaled_norm(&self.delta, state, candidate, options) * self.c_lte.abs()
        } else {
            0.0
        };
        self.update_z();
        if self.family == Family::An5 {
            self.order = 5;
            self.next_order = 5;
            self.proposed_factor = if error == 0.0 {
                10.0
            } else {
                (0.9 * error.powf(-1.0 / 6.0)).clamp(0.2, 10.0)
            };
        } else {
            self.choose_factor(error, state, candidate, options);
        }
        Self::evaluate(
            problem,
            &mut self.end_derivative,
            candidate,
            time + step,
            stats,
        )?;
        self.last_error = error;
        Ok(StepEstimate::with_factor(error, self.proposed_factor))
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
        self.dense_endpoint.copy_from_slice(state);
        let dense_endpoint = self.dense_endpoint.clone();
        let mut interpolator = |query: f64, output: &mut [f64]| {
            self.hermite_interpolate(
                previous_state,
                &dense_endpoint,
                previous_time,
                endpoint_time,
                query,
                output,
            )
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
        problem: &OdeProblem<F, P>,
        previous_state: &[f64],
        state: &[f64],
        previous_time: f64,
        attempted_time: f64,
        time: f64,
        final_time: bool,
        recorder: &mut TrajectoryRecorder<'_>,
        stats: &mut SolverStats,
    ) -> Result<bool, SolveError> {
        if self.dense_endpoint.iter().all(|value| *value == 0.0) {
            self.dense_endpoint.copy_from_slice(state);
        }
        if time != attempted_time {
            Self::evaluate(problem, &mut self.end_derivative, state, time, stats)?;
        }
        let segment = BorrowedHermiteSegment::new(
            previous_time,
            attempted_time,
            previous_state,
            &self.dense_endpoint,
            &self.start_derivative,
            &self.end_derivative,
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
            let owned = HermiteSegment::new_bounded(
                previous_time,
                attempted_time,
                time,
                previous_state.to_vec(),
                self.dense_endpoint.clone(),
                self.start_derivative.clone(),
                self.end_derivative.clone(),
            )
            .map_err(|_| SolveError::NonFiniteDerivative)?;
            recorder.retain_hermite_segment(owned);
        }
        self.dense_endpoint.fill(0.0);
        Ok(true)
    }

    fn accept_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        _: &[f64],
        state: &[f64],
        time: f64,
        accepted_step: f64,
        callback_applied: bool,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        self.backup = None;
        self.previous_step = accepted_step;
        if callback_applied {
            self.started = false;
            self.order = 1;
            self.next_order = 1;
            Self::evaluate(problem, &mut self.start_derivative, state, time, stats)?;
        } else {
            self.start_derivative.copy_from_slice(&self.end_derivative);
        }
        Ok(())
    }

    fn reject_step(&mut self) {
        if let Some(snapshot) = self.backup.take() {
            self.restore(snapshot);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Family, JVODE, JvodeMethod, NordsieckKernel};

    #[test]
    fn jvode_cache_switches_order_after_waiting() {
        let mut kernel = NordsieckKernel::<fn(), ()>::jvode(1, JVODE::adams());
        kernel.started = true;
        kernel.order = 1;
        kernel.n_wait = 0;
        kernel.previous_d = 1.0;
        kernel.current_d = 1.0;
        kernel.dts.fill(0.1);
        kernel.delta[0] = 1.0e-8;
        kernel.z[kernel.maximum_order][0] = 0.0;
        let options = crate::SolveOptions::default();
        kernel.choose_factor(1.0e-8, &[1.0], &[1.0], &options);
        assert!(kernel.next_order >= kernel.order);
        assert_eq!(kernel.family, Family::Jvode(JvodeMethod::Adams));
    }
}
