use super::super::tableaux::{
    backward_differentiation, error_constant, map_tableau_access_error, ndf_kappa,
};
use super::helpers::{
    NewtonWorkspace, estimate_initial_step, evaluate_checked, fbdf_lte_scale,
    fixed_order_threshold, lagrange_state, newton_solve, reinterpolate_differences,
    relative_step_change, reported_error, rms_scaled, select_qndf_order, update_differences,
};
use super::{CONTROLLER, DIFFERENCE_COUNT, MAX_ORDER};
use crate::integrator::{KernelCapabilities, StepEstimate, StepKernel};
use crate::{OdeProblem, SolveError, SolveOptions, SolverStats};

pub(super) struct QndfKernel {
    newton: NewtonWorkspace,
    differences: Vec<Vec<f64>>,
    attempted_differences: Vec<Vec<f64>>,
    predictor: Vec<f64>,
    forcing: Vec<f64>,
    current_derivative: Vec<f64>,
    ndf: bool,
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
    pub(super) fn new(dimension: usize, ndf: bool) -> Self {
        Self {
            newton: NewtonWorkspace::new(dimension),
            differences: vec![vec![0.0; dimension]; DIFFERENCE_COUNT],
            attempted_differences: vec![vec![0.0; dimension]; DIFFERENCE_COUNT],
            predictor: vec![0.0; dimension],
            forcing: vec![0.0; dimension],
            current_derivative: vec![0.0; dimension],
            ndf,
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
    F: crate::OdeFunction<P>,
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
        let tableau = backward_differentiation(order).map_err(map_tableau_access_error)?;
        let beta = 1.0
            / ((1.0 - ndf_kappa(tableau, self.ndf).map_err(map_tableau_access_error)?)
                * tableau.alpha()[0]);
        self.forcing.copy_from_slice(&self.predictor);
        for (index, difference) in self.attempted_differences[..order].iter().enumerate() {
            let gamma = backward_differentiation(index + 1)
                .map_err(map_tableau_access_error)?
                .alpha()[0];
            for (value, &delta) in self.forcing.iter_mut().zip(difference) {
                *value -= beta * gamma * delta;
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

        let error = error_constant(tableau, self.ndf).map_err(map_tableau_access_error)?
            * rms_scaled(correction.iter().copied(), candidate, state, options);
        self.attempted_error = error;
        self.attempted_lower_error = if order > 1 {
            error_constant(
                backward_differentiation(order - 1).map_err(map_tableau_access_error)?,
                self.ndf,
            )
            .map_err(map_tableau_access_error)?
                * rms_scaled(
                    self.attempted_differences[order - 1].iter().copied(),
                    candidate,
                    state,
                    options,
                )
        } else {
            f64::INFINITY
        };
        self.attempted_upper_error = if options.adaptive && order < MAX_ORDER {
            error_constant(
                backward_differentiation(order + 1).map_err(map_tableau_access_error)?,
                self.ndf,
            )
            .map_err(map_tableau_access_error)?
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
        self.newton.invalidate_factorization();
    }
}

pub(super) struct FbdfKernel {
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
    pub(super) fn new(dimension: usize) -> Self {
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
    F: crate::OdeFunction<P>,
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

        let coefficients = backward_differentiation(order)
            .map_err(map_tableau_access_error)?
            .alpha();
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
        self.newton.invalidate_factorization();
    }
}
