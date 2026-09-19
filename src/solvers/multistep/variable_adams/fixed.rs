//! Fixed-order variable-coefficient Adams step kernels.

use super::common::{ensure_finite, estimate_initial_step, evaluate, scaled_error_norm};
use super::history::{Workspace, prepare_trial_steps, update_differences, update_g};
use super::method::{MAX_FACTOR, MIN_FACTOR, SAFETY, VariableAdamsMethod};
use crate::integrator::{ControllerConfig, KernelCapabilities, StepEstimate, StepKernel};
use crate::{OdeProblem, SolveError, SolveOptions, SolverStats};

pub(super) struct VariableAdamsKernel {
    pub(super) method: &'static VariableAdamsMethod,
    pub(super) workspace: Option<Workspace>,
    pub(super) step_number: usize,
}

impl VariableAdamsKernel {
    pub(super) const fn new(method: &'static VariableAdamsMethod) -> Self {
        Self {
            method,
            workspace: None,
            step_number: 1,
        }
    }

    fn workspace(&mut self) -> Result<&mut Workspace, SolveError> {
        self.workspace
            .as_mut()
            .ok_or(SolveError::InvalidMultistepHistory)
    }
}

impl<F, P> StepKernel<F, P> for VariableAdamsKernel
where
    F: crate::OdeFunction<P>,
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::with_controller(
            true,
            ControllerConfig::proportional(self.method.order, SAFETY, MIN_FACTOR, MAX_FACTOR, 0.2),
        )
    }

    fn initialize(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        let mut workspace = Workspace::new(state.len(), self.method.order);
        evaluate(problem, &mut workspace.derivative, state, time, stats)?;
        ensure_finite(&workspace.derivative)?;
        self.workspace = Some(workspace);
        self.step_number = 1;
        Ok(())
    }

    fn estimate_initial_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        direction: f64,
        maximum_step: f64,
        _: &mut [f64],
        options: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<f64, SolveError> {
        let order = self.method.order;
        estimate_initial_step(
            problem,
            options,
            state,
            time,
            direction,
            maximum_step,
            order,
            self.workspace()?,
            stats,
        )
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
        let method = self.method;
        let step_number = self.step_number;
        let workspace = self.workspace()?;
        prepare_trial_steps(workspace, step, step_number, method.order);
        update_differences(workspace, step_number);

        let startup = step_number < method.order;
        let error = if startup {
            startup_step(
                problem,
                options,
                state,
                time,
                step,
                method.order,
                candidate,
                workspace,
                stats,
            )?
        } else {
            variable_adams_step(
                problem, options, state, time, step, method, candidate, workspace, stats,
            )?
        };
        ensure_finite(candidate)?;
        if error <= 1.0 && !startup {
            evaluate(
                problem,
                &mut workspace.next_derivative,
                candidate,
                time + step,
                stats,
            )?;
            ensure_finite(&workspace.next_derivative)?;
        }
        Ok(StepEstimate::new(error))
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
        let method = self.method;
        let step_number = self.step_number;
        let workspace = self.workspace()?;

        if callback_applied {
            workspace.reset_history();
            evaluate(problem, &mut workspace.derivative, state, time, stats)?;
            ensure_finite(&workspace.derivative)?;
            self.step_number = 1;
        } else {
            workspace
                .accepted_steps
                .copy_from_slice(&workspace.trial_steps);
            for (previous, current) in workspace
                .phi_previous
                .iter_mut()
                .zip(&workspace.phi)
                .take(step_number)
            {
                previous.copy_from_slice(current);
            }
            std::mem::swap(&mut workspace.derivative, &mut workspace.next_derivative);
            self.step_number = (step_number + 1).min(method.order);
        }
        Ok(())
    }

    fn reject_step(&mut self) {}
}

#[allow(clippy::too_many_arguments)]
fn variable_adams_step<F, P>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
    state: &[f64],
    time: f64,
    step: f64,
    method: &VariableAdamsMethod,
    candidate: &mut [f64],
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<f64, SolveError>
where
    F: crate::OdeFunction<P>,
{
    let order = method.order;
    if !method.corrector {
        update_g(workspace, order);
        candidate.copy_from_slice(state);
        for index in 0..order {
            for (value, difference) in candidate.iter_mut().zip(&workspace.phi[index]) {
                *value += workspace.g[index] * difference;
            }
        }
        for (error, difference) in workspace.error.iter_mut().zip(&workspace.phi[order - 1]) {
            *error = workspace.g[order - 1] * difference;
        }
    } else {
        update_g(workspace, order + 1);
        workspace.predicted.copy_from_slice(state);
        for index in 0..(order - 1) {
            for (value, difference) in workspace.predicted.iter_mut().zip(&workspace.phi[index]) {
                *value += workspace.g[index] * difference;
            }
        }
        ensure_finite(&workspace.predicted)?;
        evaluate(
            problem,
            &mut workspace.next_derivative,
            &workspace.predicted,
            time + step,
            stats,
        )?;
        ensure_finite(&workspace.next_derivative)?;
        workspace.phi_endpoint[0].copy_from_slice(&workspace.next_derivative);
        for index in 1..=order {
            let (before, after) = workspace.phi_endpoint.split_at_mut(index);
            for component in 0..after[0].len() {
                after[0][component] =
                    before[index - 1][component] - workspace.phi[index - 1][component];
            }
        }
        candidate.copy_from_slice(&workspace.predicted);
        for (value, difference) in candidate.iter_mut().zip(&workspace.phi_endpoint[order - 1]) {
            *value += workspace.g[order - 1] * difference;
        }
        let error_coefficient = workspace.g[order] - workspace.g[order - 1];
        for (error, difference) in workspace
            .error
            .iter_mut()
            .zip(&workspace.phi_endpoint[order])
        {
            *error = error_coefficient * difference;
        }
    }

    Ok(if options.adaptive {
        scaled_error_norm(&workspace.error, state, candidate, options)
    } else {
        0.0
    })
}

#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
fn startup_step<F, P>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
    state: &[f64],
    time: f64,
    step: f64,
    order: usize,
    candidate: &mut [f64],
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<f64, SolveError>
where
    F: crate::OdeFunction<P>,
{
    if order == 3 {
        bogacki_shampine_step(problem, state, time, step, candidate, workspace, stats)?;
        ensure_finite(&workspace.next_derivative)?;
        return Ok(if options.adaptive {
            scaled_error_norm(&workspace.error, state, candidate, options)
        } else {
            0.0
        });
    }

    rk4_step(problem, state, time, step, candidate, workspace, stats)?;
    ensure_finite(&workspace.next_derivative)?;
    if !options.adaptive {
        return Ok(0.0);
    }

    let sigma = [0.5 - 3.0_f64.sqrt() / 6.0, 0.5 + 3.0_f64.sqrt() / 6.0];
    let mut largest: f64 = 0.0;
    for fraction in sigma {
        for component in 0..state.len() {
            let delta = candidate[component] - state[component];
            workspace.temporary[component] = (1.0 - fraction) * state[component]
                + fraction * candidate[component]
                + fraction
                    * (fraction - 1.0)
                    * ((1.0 - 2.0 * fraction) * delta
                        + (fraction - 1.0) * step * workspace.derivative[component]
                        + fraction * step * workspace.next_derivative[component]);
            workspace.error[component] = workspace.derivative[component]
                + fraction
                    * (-4.0 * step * workspace.derivative[component]
                        - 2.0 * step * workspace.next_derivative[component]
                        - 6.0 * state[component]
                        + fraction
                            * (3.0 * step * workspace.derivative[component]
                                + 3.0 * step * workspace.next_derivative[component]
                                + 6.0 * state[component]
                                - 6.0 * candidate[component])
                        + 6.0 * candidate[component])
                    / step;
        }
        evaluate(
            problem,
            &mut workspace.predicted,
            &workspace.temporary,
            time + fraction * step,
            stats,
        )?;
        ensure_finite(&workspace.predicted)?;
        for (error, derivative) in workspace.error.iter_mut().zip(&workspace.predicted) {
            *error = step * (derivative - *error);
        }
        largest = largest.max(scaled_error_norm(
            &workspace.error,
            state,
            candidate,
            options,
        ));
    }
    Ok(2.1342 * largest)
}

#[allow(clippy::needless_range_loop)]
fn bogacki_shampine_step<F, P>(
    problem: &OdeProblem<F, P>,
    state: &[f64],
    time: f64,
    step: f64,
    candidate: &mut [f64],
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: crate::OdeFunction<P>,
{
    for component in 0..state.len() {
        workspace.temporary[component] =
            state[component] + 0.5 * step * workspace.derivative[component];
    }
    evaluate(
        problem,
        &mut workspace.stages[0],
        &workspace.temporary,
        time + 0.5 * step,
        stats,
    )?;
    for component in 0..state.len() {
        workspace.temporary[component] =
            state[component] + 0.75 * step * workspace.stages[0][component];
    }
    evaluate(
        problem,
        &mut workspace.stages[1],
        &workspace.temporary,
        time + 0.75 * step,
        stats,
    )?;
    for component in 0..state.len() {
        candidate[component] = state[component]
            + step
                * (2.0 / 9.0 * workspace.derivative[component]
                    + 1.0 / 3.0 * workspace.stages[0][component]
                    + 4.0 / 9.0 * workspace.stages[1][component]);
    }
    evaluate(
        problem,
        &mut workspace.next_derivative,
        candidate,
        time + step,
        stats,
    )?;
    for component in 0..state.len() {
        workspace.error[component] = step
            * (5.0 / 72.0 * workspace.derivative[component]
                - 1.0 / 12.0 * workspace.stages[0][component]
                - 1.0 / 9.0 * workspace.stages[1][component]
                + 1.0 / 8.0 * workspace.next_derivative[component]);
    }
    Ok(())
}

#[allow(clippy::needless_range_loop)]
fn rk4_step<F, P>(
    problem: &OdeProblem<F, P>,
    state: &[f64],
    time: f64,
    step: f64,
    candidate: &mut [f64],
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: crate::OdeFunction<P>,
{
    for component in 0..state.len() {
        workspace.temporary[component] =
            state[component] + 0.5 * step * workspace.derivative[component];
    }
    evaluate(
        problem,
        &mut workspace.stages[0],
        &workspace.temporary,
        time + 0.5 * step,
        stats,
    )?;
    for component in 0..state.len() {
        workspace.temporary[component] =
            state[component] + 0.5 * step * workspace.stages[0][component];
    }
    evaluate(
        problem,
        &mut workspace.stages[1],
        &workspace.temporary,
        time + 0.5 * step,
        stats,
    )?;
    for component in 0..state.len() {
        workspace.temporary[component] = state[component] + step * workspace.stages[1][component];
    }
    evaluate(
        problem,
        &mut workspace.stages[2],
        &workspace.temporary,
        time + step,
        stats,
    )?;
    for component in 0..state.len() {
        candidate[component] = state[component]
            + step / 6.0
                * (workspace.derivative[component]
                    + 2.0 * workspace.stages[0][component]
                    + 2.0 * workspace.stages[1][component]
                    + workspace.stages[2][component]);
    }
    evaluate(
        problem,
        &mut workspace.next_derivative,
        candidate,
        time + step,
        stats,
    )?;
    Ok(())
}
