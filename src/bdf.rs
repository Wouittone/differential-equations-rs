//! Variable-order regular-ODE backward differentiation formulas.
//!
//! The implementations in this module follow the identity-mass, regular-ODE
//! paths in OrdinaryDiffEqBDF at commit
//! `211142263781255a9aa2f910f6760b9f18ec29c8`.  Residual DAEs and singular
//! mass matrices are deliberately outside the crate's current problem model.

use crate::integrator::{
    ControllerConfig, KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::linear::{DenseLu, LinearError, StateLayout, factorize, solve_factorized};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

const MAX_ORDER: usize = 5;
const DIFFERENCE_COUNT: usize = MAX_ORDER + 2;
const MAX_NEWTON_ITERATIONS: usize = 12;
const NEWTON_TOLERANCE: f64 = 1.0e-12;
const KAPPA_NDF: [f64; MAX_ORDER] = [
    -37.0 / 200.0,
    -1.0 / 9.0,
    -823.0 / 10_000.0,
    -83.0 / 2_000.0,
    0.0,
];
const KAPPA_BDF: [f64; MAX_ORDER] = [0.0; MAX_ORDER];
const GAMMA: [f64; MAX_ORDER] = [1.0, 3.0 / 2.0, 11.0 / 6.0, 25.0 / 12.0, 137.0 / 60.0];
const BDF_COEFFICIENTS: [[f64; MAX_ORDER + 1]; MAX_ORDER] = [
    [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 2.0, -2.0, 1.0 / 2.0, 0.0, 0.0, 0.0],
    [11.0 / 6.0, -3.0, 3.0 / 2.0, -1.0 / 3.0, 0.0, 0.0],
    [25.0 / 12.0, -4.0, 3.0, -4.0 / 3.0, 1.0 / 4.0, 0.0],
    [137.0 / 60.0, -5.0, 5.0, -10.0 / 3.0, 5.0 / 4.0, -1.0 / 5.0],
];

// A sixth-order controller exponent is converted back to each BDF order in
// `reported_error`, so the shared driver uses the pinned 1/(k+1) exponent.
const CONTROLLER: ControllerConfig = ControllerConfig::proportional(6, 1.0 / 1.2, 0.1, 10.0, 0.2);

/// Adaptive-order quasi-constant-step NDF, orders one through five.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qndf;

/// Adaptive-order quasi-constant-step BDF (`QNDF` with all kappa values zero).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qbdf;

/// Adaptive-order fixed-leading-coefficient BDF, orders one through five.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Fbdf;

/// Exact Julia-compatible spelling alias for [`Qndf`].
pub type QNDF = Qndf;
/// Exact Julia-compatible spelling alias for [`Qbdf`].
pub type QBDF = Qbdf;
/// Exact Julia-compatible spelling alias for [`Fbdf`].
pub type FBDF = Fbdf;

#[allow(non_upper_case_globals)]
pub const QNDF: Qndf = Qndf;
#[allow(non_upper_case_globals)]
pub const QBDF: Qbdf = Qbdf;
#[allow(non_upper_case_globals)]
pub const FBDF: Fbdf = Fbdf;

impl OdeAlgorithm for Qndf {
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
            QndfKernel::new(problem.initial_state().len(), KAPPA_NDF),
        )
    }
}

impl OdeAlgorithm for Qbdf {
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
            QndfKernel::new(problem.initial_state().len(), KAPPA_BDF),
        )
    }
}

impl OdeAlgorithm for Fbdf {
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
            FbdfKernel::new(problem.initial_state().len()),
        )
    }
}

struct NewtonWorkspace {
    layout: StateLayout,
    evaluation_derivative: Vec<f64>,
    perturbed_state: Vec<f64>,
    perturbed_derivative: Vec<f64>,
    residual: Vec<f64>,
    correction: Vec<f64>,
    matrix: Vec<f64>,
    pivots: Vec<usize>,
    factorization: Option<DenseLu>,
    dense_active: bool,
    factorization_ready: bool,
}

impl NewtonWorkspace {
    fn new(dimension: usize) -> Self {
        Self {
            layout: StateLayout::for_validated_state(dimension),
            evaluation_derivative: vec![0.0; dimension],
            perturbed_state: vec![0.0; dimension],
            perturbed_derivative: vec![0.0; dimension],
            residual: vec![0.0; dimension],
            correction: vec![0.0; dimension],
            matrix: vec![0.0; dimension * dimension],
            pivots: vec![0; dimension],
            factorization: None,
            dense_active: false,
            factorization_ready: false,
        }
    }
}

struct QndfKernel {
    newton: NewtonWorkspace,
    differences: Vec<Vec<f64>>,
    attempted_differences: Vec<Vec<f64>>,
    predictor: Vec<f64>,
    forcing: Vec<f64>,
    current_derivative: Vec<f64>,
    kappa: [f64; MAX_ORDER],
    order: usize,
    attempted_order: usize,
    previous_order: usize,
    accepted_steps: usize,
    constant_steps: usize,
    last_step: Option<f64>,
    attempted_lower_error: f64,
    attempted_error: f64,
    attempted_upper_error: f64,
    attempted_adaptive: bool,
    consecutive_rejections: usize,
}

impl QndfKernel {
    fn new(dimension: usize, kappa: [f64; MAX_ORDER]) -> Self {
        Self {
            newton: NewtonWorkspace::new(dimension),
            differences: vec![vec![0.0; dimension]; DIFFERENCE_COUNT],
            attempted_differences: vec![vec![0.0; dimension]; DIFFERENCE_COUNT],
            predictor: vec![0.0; dimension],
            forcing: vec![0.0; dimension],
            current_derivative: vec![0.0; dimension],
            kappa,
            order: 1,
            attempted_order: 1,
            previous_order: 1,
            accepted_steps: 0,
            constant_steps: 0,
            last_step: None,
            attempted_lower_error: f64::INFINITY,
            attempted_error: f64::INFINITY,
            attempted_upper_error: f64::INFINITY,
            attempted_adaptive: true,
            consecutive_rejections: 0,
        }
    }

    fn reset_history(&mut self) {
        self.differences
            .iter_mut()
            .for_each(|difference| difference.fill(0.0));
        self.order = 1;
        self.previous_order = 1;
        self.accepted_steps = 0;
        self.constant_steps = 0;
        self.last_step = None;
        self.consecutive_rejections = 0;
    }
}

impl<F, P> StepKernel<F, P> for QndfKernel
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::with_controller(true, CONTROLLER)
            .recover_nonlinear_and_singular_failures()
    }

    fn initialize(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        self.reset_history();
        evaluate_checked(problem, &mut self.current_derivative, state, time, stats)
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
        Ok(estimate_initial_step(
            state,
            &self.current_derivative,
            options,
            maximum_step,
        ))
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
        let order = self.order.min(self.accepted_steps + 1).clamp(1, MAX_ORDER);
        self.attempted_order = order;
        self.attempted_adaptive = options.adaptive;
        for (target, source) in self.attempted_differences.iter_mut().zip(&self.differences) {
            target.copy_from_slice(source);
        }
        if let Some(previous_step) = self.last_step {
            if step != previous_step || self.previous_order != order {
                reinterpolate_differences(
                    &mut self.attempted_differences,
                    order,
                    step / previous_step,
                );
                self.constant_steps = 0;
            }
        }

        self.predictor.copy_from_slice(state);
        for difference in &self.attempted_differences[..order] {
            for (value, &delta) in self.predictor.iter_mut().zip(difference) {
                *value += delta;
            }
        }
        let beta = 1.0 / ((1.0 - self.kappa[order - 1]) * GAMMA[order - 1]);
        self.forcing.copy_from_slice(&self.predictor);
        for (index, difference) in self.attempted_differences[..order].iter().enumerate() {
            for (value, &delta) in self.forcing.iter_mut().zip(difference) {
                *value -= beta * GAMMA[index] * delta;
            }
        }
        candidate.copy_from_slice(&self.predictor);
        newton_solve(
            problem,
            candidate,
            time + step,
            beta * step,
            &self.forcing,
            &mut self.newton,
            stats,
        )?;

        let mut correction = vec![0.0; candidate.len()];
        for ((dd, &next), &predicted) in correction
            .iter_mut()
            .zip(candidate.iter())
            .zip(&self.predictor)
        {
            *dd = next - predicted;
        }
        update_differences(&mut self.attempted_differences, &correction, order);

        let error = error_constant(self.kappa, order)
            * rms_scaled(correction.iter().copied(), candidate, state, options);
        self.attempted_error = error;
        self.attempted_lower_error = if order > 1 {
            error_constant(self.kappa, order - 1)
                * rms_scaled(
                    self.attempted_differences[order - 1].iter().copied(),
                    candidate,
                    state,
                    options,
                )
        } else {
            f64::INFINITY
        };
        self.attempted_upper_error = if order < MAX_ORDER {
            error_constant(self.kappa, order + 1)
                * rms_scaled(
                    self.attempted_differences[order + 1].iter().copied(),
                    candidate,
                    state,
                    options,
                )
        } else {
            f64::INFINITY
        };
        let reported = if options.adaptive {
            reported_error(error, order)
        } else {
            0.0
        };
        Ok(StepEstimate::new(reported))
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
        if callback_applied {
            self.reset_history();
        } else {
            for (target, source) in self.differences.iter_mut().zip(&self.attempted_differences) {
                target.copy_from_slice(source);
            }
            self.previous_order = self.attempted_order;
            self.consecutive_rejections = 0;
            self.accepted_steps += 1;
            if self
                .last_step
                .is_some_and(|previous| relative_step_change(previous, accepted_step) <= 1.0e-12)
            {
                self.constant_steps += 1;
            } else {
                self.constant_steps = 1;
            }
            self.last_step = Some(accepted_step);
            self.order = if !self.attempted_adaptive {
                self.attempted_order
            } else {
                select_qndf_order(
                    self.attempted_order,
                    self.attempted_error,
                    self.attempted_lower_error,
                    self.attempted_upper_error,
                )
            };
        }
        evaluate_checked(problem, &mut self.current_derivative, state, time, stats)
    }

    fn reject_step(&mut self) {
        self.consecutive_rejections += 1;
        let current_score = if self.attempted_error > 0.0 {
            self.attempted_error
                .powf(-1.0 / (self.attempted_order + 1) as f64)
                / 1.2
        } else {
            10.0
        };
        let lower_score = if self.attempted_lower_error > 0.0 {
            self.attempted_lower_error
                .powf(-1.0 / self.attempted_order as f64)
                / 1.3
        } else {
            0.0
        };
        if self.attempted_order > 1
            && self.attempted_lower_error.is_finite()
            && (self.consecutive_rejections > 2 || lower_score > current_score)
        {
            self.order = self.attempted_order - 1;
        }
        self.constant_steps = 0;
        self.newton.factorization_ready = false;
    }
}

struct FbdfKernel {
    newton: NewtonWorkspace,
    states: Vec<Vec<f64>>,
    times: Vec<f64>,
    predictor: Vec<f64>,
    forcing: Vec<f64>,
    current_derivative: Vec<f64>,
    order: usize,
    attempted_order: usize,
    accepted_steps: usize,
    constant_steps: usize,
    last_step: Option<f64>,
    attempted_adaptive: bool,
}

impl FbdfKernel {
    fn new(dimension: usize) -> Self {
        Self {
            newton: NewtonWorkspace::new(dimension),
            states: Vec::with_capacity(DIFFERENCE_COUNT),
            times: Vec::with_capacity(DIFFERENCE_COUNT),
            predictor: vec![0.0; dimension],
            forcing: vec![0.0; dimension],
            current_derivative: vec![0.0; dimension],
            order: 1,
            attempted_order: 1,
            accepted_steps: 0,
            constant_steps: 0,
            last_step: None,
            attempted_adaptive: true,
        }
    }

    fn reset_history(&mut self, state: &[f64], time: f64) {
        self.states.clear();
        self.times.clear();
        self.states.push(state.to_vec());
        self.times.push(time);
        self.order = 1;
        self.accepted_steps = 0;
        self.constant_steps = 0;
        self.last_step = None;
    }
}

impl<F, P> StepKernel<F, P> for FbdfKernel
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::with_controller(true, CONTROLLER)
            .recover_nonlinear_and_singular_failures()
    }

    fn initialize(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        self.reset_history(state, time);
        evaluate_checked(problem, &mut self.current_derivative, state, time, stats)
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
        Ok(estimate_initial_step(
            state,
            &self.current_derivative,
            options,
            maximum_step,
        ))
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
        let order = self.order.min(self.states.len()).clamp(1, MAX_ORDER);
        self.attempted_order = order;
        self.attempted_adaptive = options.adaptive;
        let evaluation_time = time + step;
        lagrange_state(
            &mut self.predictor,
            evaluation_time,
            &self.times,
            &self.states,
            (order + 1).min(self.states.len()),
        );

        let coefficients = &BDF_COEFFICIENTS[order - 1];
        let beta = 1.0 / coefficients[0];
        for (value, &now) in self.forcing.iter_mut().zip(state) {
            *value = -beta * coefficients[1] * now;
        }
        let mut interpolated = vec![0.0; state.len()];
        for history_index in 1..order {
            lagrange_state(
                &mut interpolated,
                time - history_index as f64 * step,
                &self.times,
                &self.states,
                (order + 1).min(self.states.len()),
            );
            for (value, &history) in self.forcing.iter_mut().zip(&interpolated) {
                *value -= beta * coefficients[history_index + 1] * history;
            }
        }
        candidate.copy_from_slice(&self.predictor);
        newton_solve(
            problem,
            candidate,
            evaluation_time,
            beta * step,
            &self.forcing,
            &mut self.newton,
            stats,
        )?;

        let correction = candidate
            .iter()
            .zip(&self.predictor)
            .map(|(&next, &predicted)| next - predicted);
        let raw_error = rms_scaled(correction, candidate, state, options);
        let lte_scale = fbdf_lte_scale(order, evaluation_time, step, &self.times, coefficients);
        let error = lte_scale.abs() * raw_error;
        Ok(StepEstimate::new(if options.adaptive {
            reported_error(error, order)
        } else {
            0.0
        }))
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
        if callback_applied {
            self.reset_history(state, time);
        } else {
            self.states.insert(0, state.to_vec());
            self.times.insert(0, time);
            self.states.truncate(DIFFERENCE_COUNT);
            self.times.truncate(DIFFERENCE_COUNT);
            self.accepted_steps += 1;
            if self
                .last_step
                .is_some_and(|previous| relative_step_change(previous, accepted_step) <= 1.0e-12)
            {
                self.constant_steps += 1;
            } else {
                self.constant_steps = 1;
            }
            self.last_step = Some(accepted_step);
            if self.attempted_adaptive
                && self.attempted_order < MAX_ORDER
                && self.states.len() > self.attempted_order
                && self.accepted_steps >= fixed_order_threshold(self.attempted_order + 1)
            {
                self.order = self.attempted_order + 1;
            } else {
                self.order = self.attempted_order;
            }
        }
        evaluate_checked(problem, &mut self.current_derivative, state, time, stats)
    }

    fn reject_step(&mut self) {
        self.order = self.attempted_order.saturating_sub(1).max(1);
        self.constant_steps = 0;
        self.newton.factorization_ready = false;
    }
}

fn error_constant(kappa: [f64; MAX_ORDER], order: usize) -> f64 {
    kappa[order - 1] * GAMMA[order - 1] + 1.0 / (order + 1) as f64
}

fn reported_error(error: f64, order: usize) -> f64 {
    if error == 0.0 {
        0.0
    } else {
        error.powf(6.0 / (order + 1) as f64)
    }
}

fn select_qndf_order(order: usize, error: f64, lower_error: f64, upper_error: f64) -> usize {
    let current_score = if error > 0.0 {
        error.powf(-1.0 / (order + 1) as f64) / 1.2
    } else {
        10.0
    };
    let lower_score = if order > 1 && lower_error > 0.0 {
        lower_error.powf(-1.0 / order as f64) / 1.3
    } else {
        0.0
    };
    let upper_score = if order < MAX_ORDER && upper_error > 0.0 {
        upper_error.powf(-1.0 / (order + 2) as f64) / 1.4
    } else {
        0.0
    };
    if upper_score > current_score && upper_score >= lower_score {
        order + 1
    } else if lower_score > current_score {
        order - 1
    } else {
        order
    }
}

fn fixed_order_threshold(order: usize) -> usize {
    match order {
        1 => 0,
        2 => 2,
        3 => 5,
        4 => 9,
        _ => 14,
    }
}

fn reinterpolate_differences(differences: &mut [Vec<f64>], order: usize, ratio: f64) {
    let mut u = [[0.0; MAX_ORDER]; MAX_ORDER];
    let mut r = [[0.0; MAX_ORDER]; MAX_ORDER];
    for column in 0..order {
        u[0][column] = -(column as f64 + 1.0);
        r[0][column] = -(column as f64 + 1.0) * ratio;
        for row in 1..order {
            u[row][column] =
                u[row - 1][column] * (row as f64 - (column as f64 + 1.0)) / (row as f64 + 1.0);
            r[row][column] = r[row - 1][column] * (row as f64 - (column as f64 + 1.0) * ratio)
                / (row as f64 + 1.0);
        }
    }
    let mut ru = [[0.0; MAX_ORDER]; MAX_ORDER];
    for row in 0..order {
        for column in 0..order {
            ru[row][column] = (0..order)
                .map(|middle| r[row][middle] * u[middle][column])
                .sum();
        }
    }
    let old = differences[..order].to_vec();
    for (column, difference) in differences.iter_mut().enumerate().take(order) {
        difference.fill(0.0);
        for row in 0..order {
            for (value, &delta) in difference.iter_mut().zip(&old[row]) {
                *value += delta * ru[row][column];
            }
        }
    }
}

fn update_differences(differences: &mut [Vec<f64>], correction: &[f64], order: usize) {
    let (through_order, higher) = differences.split_at_mut(order + 1);
    for ((higher, &dd), &previous) in higher[0]
        .iter_mut()
        .zip(correction)
        .zip(&through_order[order])
    {
        *higher = dd - previous;
    }
    through_order[order].copy_from_slice(correction);
    for index in (0..order).rev() {
        let (lower, upper) = differences.split_at_mut(index + 1);
        for (value, &next) in lower[index].iter_mut().zip(&upper[0]) {
            *value += next;
        }
    }
}

fn lagrange_state(
    output: &mut [f64],
    target: f64,
    times: &[f64],
    states: &[Vec<f64>],
    count: usize,
) {
    output.fill(0.0);
    if count == 0 {
        return;
    }
    for j in 0..count {
        let mut weight = 1.0;
        for m in 0..count {
            if m != j {
                weight *= (target - times[m]) / (times[j] - times[m]);
            }
        }
        for (value, &state) in output.iter_mut().zip(&states[j]) {
            *value += weight * state;
        }
    }
}

fn fbdf_lte_scale(
    order: usize,
    evaluation_time: f64,
    step: f64,
    times: &[f64],
    coefficients: &[f64; MAX_ORDER + 1],
) -> f64 {
    if times.len() < order + 1 {
        return 1.0 / (order + 1) as f64;
    }
    let mut lte = -1.0 / (order + 1) as f64;
    for j in 2..=order {
        let mut r = 1.0 - j as f64;
        for i in 2..=order + 1 {
            r *= ((evaluation_time - j as f64 * step) - times[i - 1]) / (i as f64 * step);
        }
        lte -= coefficients[j - 1] * r;
    }
    let product = (1..=order + 1)
        .map(|j| j as f64 * step / (evaluation_time - times[j - 1]))
        .product::<f64>();
    lte * product
}

#[allow(clippy::too_many_arguments)]
fn newton_solve<F, P>(
    problem: &OdeProblem<F, P>,
    candidate: &mut [f64],
    time: f64,
    derivative_scale: f64,
    forcing: &[f64],
    workspace: &mut NewtonWorkspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    workspace.factorization_ready = false;
    for _ in 0..MAX_NEWTON_ITERATIONS {
        stats.nonlinear_iterations += 1;
        evaluate_checked(
            problem,
            &mut workspace.evaluation_derivative,
            candidate,
            time,
            stats,
        )?;
        let mut residual_norm: f64 = 0.0;
        for index in 0..candidate.len() {
            workspace.residual[index] = candidate[index]
                - derivative_scale * workspace.evaluation_derivative[index]
                - forcing[index];
            residual_norm = residual_norm.max(workspace.residual[index].abs());
        }
        if residual_norm <= NEWTON_TOLERANCE * (1.0 + infinity_norm(candidate)) {
            return Ok(());
        }
        if !workspace.factorization_ready {
            build_factorization(problem, candidate, time, derivative_scale, workspace, stats)?;
        }
        for (correction, &residual) in workspace.correction.iter_mut().zip(&workspace.residual) {
            *correction = -residual;
        }
        if workspace.dense_active {
            workspace
                .factorization
                .as_ref()
                .ok_or(SolveError::SingularLinearSystem)?
                .solve(&mut workspace.correction)
                .map_err(map_linear_error)?;
            workspace.dense_active = false;
        } else {
            solve_factorized(
                &workspace.matrix,
                &workspace.pivots,
                &mut workspace.correction,
                candidate.len(),
            );
        }
        stats.linear_solves += 1;
        for (value, &correction) in candidate.iter_mut().zip(&workspace.correction) {
            *value += correction;
        }
    }
    Err(SolveError::NonlinearSolveFailed)
}

fn build_factorization<F, P>(
    problem: &OdeProblem<F, P>,
    state: &[f64],
    time: f64,
    derivative_scale: f64,
    workspace: &mut NewtonWorkspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let dimension = workspace.layout.dimension();
    if problem.evaluate_jacobian(&mut workspace.matrix, state, time) {
        for row in 0..dimension {
            for column in 0..dimension {
                let index = row * dimension + column;
                workspace.matrix[index] =
                    f64::from(row == column) - derivative_scale * workspace.matrix[index];
            }
        }
    } else {
        for column in 0..dimension {
            workspace.perturbed_state.copy_from_slice(state);
            let perturbation = f64::EPSILON.sqrt() * state[column].abs().max(1.0);
            workspace.perturbed_state[column] += perturbation;
            evaluate_checked(
                problem,
                &mut workspace.perturbed_derivative,
                &workspace.perturbed_state,
                time,
                stats,
            )?;
            for row in 0..dimension {
                let derivative = (workspace.perturbed_derivative[row]
                    - workspace.evaluation_derivative[row])
                    / perturbation;
                workspace.matrix[row * dimension + column] =
                    f64::from(row == column) - derivative_scale * derivative;
            }
        }
    }
    stats.jacobian_evaluations += 1;
    stats.linear_factorizations += 1;
    let factorization = if workspace.factorization.is_none() {
        let matrix = workspace
            .layout
            .matrix(&workspace.matrix)
            .map_err(map_linear_error)?;
        let dense = DenseLu::factorize(
            workspace.layout,
            matrix.as_slice(),
            stats.jacobian_evaluations as u64,
        )
        .map_err(map_linear_error)?;
        factorize(&mut workspace.matrix, &mut workspace.pivots, dimension)
            .map_err(|_| SolveError::SingularLinearSystem)?;
        workspace.dense_active = true;
        dense
    } else {
        factorize(&mut workspace.matrix, &mut workspace.pivots, dimension)
            .map_err(|_| SolveError::SingularLinearSystem)?;
        workspace.dense_active = false;
        workspace
            .factorization
            .take()
            .ok_or(SolveError::NonlinearSolveFailed)?
    };
    workspace.factorization = Some(factorization);
    workspace.factorization_ready = true;
    Ok(())
}

fn rms_scaled<I>(values: I, candidate: &[f64], previous: &[f64], options: &SolveOptions) -> f64
where
    I: Iterator<Item = f64>,
{
    let mut squared = 0.0;
    for ((defect, &next), &old) in values.zip(candidate).zip(previous) {
        let scale =
            options.absolute_tolerance + options.relative_tolerance * next.abs().max(old.abs());
        squared += (defect / scale).powi(2);
    }
    (squared / candidate.len() as f64).sqrt()
}

fn estimate_initial_step(
    state: &[f64],
    derivative: &[f64],
    options: &SolveOptions,
    maximum_step: f64,
) -> f64 {
    let scale = state
        .iter()
        .zip(derivative)
        .map(|(&state, &derivative)| {
            derivative.abs()
                / (options.absolute_tolerance + options.relative_tolerance * state.abs())
        })
        .fold(0.0, f64::max);
    let estimate = if scale > 0.0 {
        (0.01 / scale).sqrt()
    } else {
        maximum_step.min(0.01)
    };
    estimate.max(f64::EPSILON).min(maximum_step)
}

fn relative_step_change(previous: f64, next: f64) -> f64 {
    (previous - next).abs() / previous.abs().max(next.abs()).max(f64::MIN_POSITIVE)
}

fn evaluate_checked<F, P>(
    problem: &OdeProblem<F, P>,
    derivative: &mut [f64],
    state: &[f64],
    time: f64,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    (problem.rhs)(derivative, state, problem.parameters(), time);
    stats.rhs_evaluations += 1;
    derivative
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteDerivative)
}

fn infinity_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

fn map_linear_error(error: LinearError) -> SolveError {
    match error {
        LinearError::Singular => SolveError::SingularLinearSystem,
        _ => SolveError::NonlinearSolveFailed,
    }
}
