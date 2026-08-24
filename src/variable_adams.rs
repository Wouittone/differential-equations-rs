use crate::integrator::{
    ControllerConfig, KernelCapabilities, StepEstimate, StepKernel, integrate,
};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 5.0;
const VCABM_MAX_ORDER: usize = 12;

#[derive(Clone, Copy)]
struct VariableAdamsMethod {
    order: usize,
    corrector: bool,
}

const VCAB3_METHOD: VariableAdamsMethod = VariableAdamsMethod {
    order: 3,
    corrector: false,
};
const VCAB4_METHOD: VariableAdamsMethod = VariableAdamsMethod {
    order: 4,
    corrector: false,
};
const VCAB5_METHOD: VariableAdamsMethod = VariableAdamsMethod {
    order: 5,
    corrector: false,
};
const VCABM3_METHOD: VariableAdamsMethod = VariableAdamsMethod {
    order: 3,
    corrector: true,
};
const VCABM4_METHOD: VariableAdamsMethod = VariableAdamsMethod {
    order: 4,
    corrector: true,
};
const VCABM5_METHOD: VariableAdamsMethod = VariableAdamsMethod {
    order: 5,
    corrector: true,
};

macro_rules! algorithm {
    ($name:ident, $documentation:literal, $method:ident) => {
        #[doc = $documentation]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        pub struct $name;

        impl OdeAlgorithm for $name {
            fn solve<F, P>(
                &self,
                problem: &OdeProblem<F, P>,
                options: &SolveOptions,
            ) -> Result<Solution, SolveError>
            where
                F: Fn(&mut [f64], &[f64], &P, f64),
            {
                integrate(problem, options, VariableAdamsKernel::new(&$method))
            }
        }
    };
}

algorithm!(
    Vcab3,
    "The adaptive third-order variable-coefficient Adams--Bashforth method.",
    VCAB3_METHOD
);
algorithm!(
    Vcab4,
    "The adaptive fourth-order variable-coefficient Adams--Bashforth method.",
    VCAB4_METHOD
);
algorithm!(
    Vcab5,
    "The adaptive fifth-order variable-coefficient Adams--Bashforth method.",
    VCAB5_METHOD
);
algorithm!(
    Vcabm3,
    "The adaptive third-order variable-coefficient Adams--Moulton method.",
    VCABM3_METHOD
);
algorithm!(
    Vcabm4,
    "The adaptive fourth-order variable-coefficient Adams--Moulton method.",
    VCABM4_METHOD
);
algorithm!(
    Vcabm5,
    "The adaptive fifth-order variable-coefficient Adams--Moulton method.",
    VCABM5_METHOD
);

/// Adaptive-order, adaptive-step Adams--Moulton predictor/corrector method.
///
/// The order starts at one and rises through order twelve as accepted divided-
/// difference history becomes available. Rejections lower the active order.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Vcabm;

/// Exact Julia-compatible spelling alias for [`Vcabm`].
pub type VCABM = Vcabm;

#[allow(non_upper_case_globals)]
pub const VCABM: Vcabm = Vcabm;

impl OdeAlgorithm for Vcabm {
    fn solve<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        integrate(problem, options, VariableOrderAdamsKernel::new())
    }
}

struct Workspace {
    predicted: Vec<f64>,
    temporary: Vec<f64>,
    derivative: Vec<f64>,
    next_derivative: Vec<f64>,
    error: Vec<f64>,
    stages: Vec<Vec<f64>>,
    phi_previous: Vec<Vec<f64>>,
    phi: Vec<Vec<f64>>,
    phi_unscaled: Vec<Vec<f64>>,
    phi_endpoint: Vec<Vec<f64>>,
    accepted_steps: Vec<f64>,
    trial_steps: Vec<f64>,
    coefficients: Vec<Vec<f64>>,
    g: Vec<f64>,
}

impl Workspace {
    fn new(dimension: usize, order: usize) -> Self {
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

    fn reset_history(&mut self) {
        self.accepted_steps.fill(0.0);
        for difference in &mut self.phi_previous {
            difference.fill(0.0);
        }
    }
}

struct VariableAdamsKernel {
    method: &'static VariableAdamsMethod,
    workspace: Option<Workspace>,
    step_number: usize,
}

impl VariableAdamsKernel {
    const fn new(method: &'static VariableAdamsMethod) -> Self {
        Self {
            method,
            workspace: None,
            step_number: 1,
        }
    }

    fn workspace(&mut self) -> &mut Workspace {
        self.workspace
            .as_mut()
            .expect("variable Adams kernel is initialized before stepping")
    }
}

impl<F, P> StepKernel<F, P> for VariableAdamsKernel
where
    F: Fn(&mut [f64], &[f64], &P, f64),
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
        evaluate(problem, &mut workspace.derivative, state, time, stats);
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
            self.workspace(),
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
        let workspace = self.workspace();
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
            );
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
        let workspace = self.workspace();

        if callback_applied {
            workspace.reset_history();
            evaluate(problem, &mut workspace.derivative, state, time, stats);
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

fn prepare_trial_steps(workspace: &mut Workspace, step: f64, step_number: usize, order: usize) {
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
fn update_differences(workspace: &mut Workspace, count: usize) {
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
fn update_g(workspace: &mut Workspace, count: usize) {
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
    F: Fn(&mut [f64], &[f64], &P, f64),
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
        );
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
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    if order == 3 {
        bogacki_shampine_step(problem, state, time, step, candidate, workspace, stats);
        ensure_finite(&workspace.next_derivative)?;
        return Ok(if options.adaptive {
            scaled_error_norm(&workspace.error, state, candidate, options)
        } else {
            0.0
        });
    }

    rk4_step(problem, state, time, step, candidate, workspace, stats);
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
        );
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
) where
    F: Fn(&mut [f64], &[f64], &P, f64),
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
    );
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
    );
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
    );
    for component in 0..state.len() {
        workspace.error[component] = step
            * (5.0 / 72.0 * workspace.derivative[component]
                - 1.0 / 12.0 * workspace.stages[0][component]
                - 1.0 / 9.0 * workspace.stages[1][component]
                + 1.0 / 8.0 * workspace.next_derivative[component]);
    }
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
) where
    F: Fn(&mut [f64], &[f64], &P, f64),
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
    );
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
    );
    for component in 0..state.len() {
        workspace.temporary[component] = state[component] + step * workspace.stages[1][component];
    }
    evaluate(
        problem,
        &mut workspace.stages[2],
        &workspace.temporary,
        time + step,
        stats,
    );
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
    );
}

fn scaled_error_norm(
    error: &[f64],
    state: &[f64],
    candidate: &[f64],
    options: &SolveOptions,
) -> f64 {
    let mut squared = 0.0;
    for ((error, state), candidate) in error.iter().zip(state).zip(candidate) {
        let scale = options.absolute_tolerance
            + options.relative_tolerance * state.abs().max(candidate.abs());
        squared += (error / scale).powi(2);
    }
    (squared / state.len() as f64).sqrt()
}

#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
fn estimate_initial_step<F, P>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
    state: &[f64],
    time: f64,
    direction: f64,
    maximum_step: f64,
    order: usize,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<f64, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let dimension = state.len() as f64;
    let mut state_norm = 0.0;
    let mut derivative_norm = 0.0;
    for (value, derivative) in state.iter().zip(&workspace.derivative) {
        let scale = options.absolute_tolerance + options.relative_tolerance * value.abs();
        state_norm += (value / scale).powi(2);
        derivative_norm += (derivative / scale).powi(2);
    }
    state_norm = (state_norm / dimension).sqrt();
    derivative_norm = (derivative_norm / dimension).sqrt();
    let trial_step = if state_norm < 1.0e-5 || derivative_norm < 1.0e-5 {
        1.0e-6
    } else {
        0.01 * state_norm / derivative_norm
    }
    .min(maximum_step);
    for component in 0..state.len() {
        workspace.temporary[component] =
            state[component] + direction * trial_step * workspace.derivative[component];
    }
    evaluate(
        problem,
        &mut workspace.predicted,
        &workspace.temporary,
        time + direction * trial_step,
        stats,
    );
    ensure_finite(&workspace.predicted)?;
    let mut curvature_norm = 0.0;
    for component in 0..state.len() {
        let scale =
            options.absolute_tolerance + options.relative_tolerance * state[component].abs();
        curvature_norm +=
            ((workspace.predicted[component] - workspace.derivative[component]) / scale).powi(2);
    }
    curvature_norm = (curvature_norm / dimension).sqrt() / trial_step;
    let largest = derivative_norm.max(curvature_norm);
    let accuracy_step = if largest <= 1.0e-15 {
        (trial_step * 1.0e-3).max(1.0e-6)
    } else {
        (0.01 / largest).powf(1.0 / order as f64)
    };
    Ok((100.0 * trial_step).min(accuracy_step).min(maximum_step))
}

fn evaluate<F, P>(
    problem: &OdeProblem<F, P>,
    derivative: &mut [f64],
    state: &[f64],
    time: f64,
    stats: &mut SolverStats,
) where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    (problem.rhs)(derivative, state, problem.parameters(), time);
    stats.rhs_evaluations += 1;
}

fn ensure_finite(values: &[f64]) -> Result<(), SolveError> {
    values
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteDerivative)
}

struct VariableOrderAdamsKernel {
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
    const fn new() -> Self {
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
        F: Fn(&mut [f64], &[f64], &P, f64),
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
        evaluate(problem, &mut self.derivatives[0], state, time, stats);
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
    F: Fn(&mut [f64], &[f64], &P, f64),
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
        );
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
        evaluate(problem, &mut self.derivatives[0], state, time, stats);
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

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::f64::consts::E;
    use std::rc::Rc;

    use super::{
        VCAB5_METHOD, VariableAdamsKernel, Vcab3, Vcab4, Vcab5, Vcabm3, Vcabm4, Vcabm5, update_g,
    };
    use crate::integrator::StepKernel;
    use crate::{CallbackAction, OdeProblem, SaveMode, SolveOptions, SolverStats, solve};

    type TestRhs = fn(&mut [f64], &[f64], &(), f64);

    fn exponential(time_span: (f64, f64)) -> OdeProblem<TestRhs, ()> {
        fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
            du[0] = u[0];
        }
        OdeProblem::new(rhs as TestRhs, vec![1.0], time_span, ())
    }

    fn options() -> SolveOptions {
        SolveOptions {
            absolute_tolerance: 1.0e-10,
            relative_tolerance: 1.0e-10,
            initial_step: Some(0.013),
            max_step: 0.17,
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        }
    }

    #[test]
    fn coefficients_cover_equal_and_unequal_step_histories() {
        let mut workspace = super::Workspace::new(1, 3);
        workspace.trial_steps.copy_from_slice(&[0.2, 0.2, 0.2]);
        update_g(&mut workspace, 3);

        // `g` multiplies backward differences. Expanding these three values
        // into raw derivatives produces the classical 23/-16/5 weights.
        assert!((workspace.g[0] - 0.2).abs() < 1.0e-15);
        assert!((workspace.g[1] - 0.5 * 0.2).abs() < 1.0e-15);
        assert!((workspace.g[2] - 5.0 / 12.0 * 0.2).abs() < 1.0e-15);

        workspace.trial_steps.copy_from_slice(&[0.2, 0.1, 0.3]);
        update_g(&mut workspace, 3);
        assert!((workspace.g[2] - 7.0 / 90.0).abs() < 1.0e-15);
    }

    #[test]
    fn all_variable_adams_methods_solve_forward_and_backward() {
        macro_rules! check {
            ($algorithm:expr) => {{
                let forward = solve(&exponential((0.0, 1.0)), $algorithm, &options()).unwrap();
                let backward_problem = OdeProblem::new(
                    |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
                    vec![E],
                    (1.0, 0.0),
                    (),
                );
                let backward = solve(&backward_problem, $algorithm, &options()).unwrap();
                assert!(
                    (forward.last_state()[0] - E).abs() < 2.0e-6,
                    "forward endpoint: {}",
                    forward.last_state()[0]
                );
                assert!(
                    (backward.last_state()[0] - 1.0).abs() < 2.0e-6,
                    "backward endpoint: {}",
                    backward.last_state()[0]
                );
                assert!(forward.stats().accepted_steps > 5);
            }};
        }
        check!(Vcab3);
        check!(Vcab4);
        check!(Vcab5);
        check!(Vcabm3);
        check!(Vcabm4);
        check!(Vcabm5);
    }

    #[test]
    fn callback_resets_multistep_history_and_save_at_is_honored() {
        let problem = exponential((0.0, 1.0)).with_continuous_callback(
            |_: &[f64], _: &(), time| time - 0.5,
            |state: &mut [f64], _: &(), _: f64| {
                state[0] *= 0.5;
                CallbackAction::Continue
            },
        );
        let options = SolveOptions {
            save_at: vec![0.0, 0.25, 0.5, 0.75, 1.0],
            ..options()
        };
        let solution = solve(&problem, Vcab5, &options).unwrap();

        assert_eq!(solution.times(), &[0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(solution.stats().callback_invocations, 1);
        assert!(
            (solution.last_state()[0] - 0.5 * E).abs() < 5.0e-6,
            "callback endpoint: {}",
            solution.last_state()[0]
        );
    }

    #[test]
    fn tighter_tolerances_reduce_error() {
        let loose = SolveOptions {
            absolute_tolerance: 1.0e-5,
            relative_tolerance: 1.0e-5,
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };
        let tight = SolveOptions {
            absolute_tolerance: 1.0e-10,
            relative_tolerance: 1.0e-10,
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };
        let loose_solution = solve(&exponential((0.0, 4.0)), Vcab4, &loose).unwrap();
        let tight_solution = solve(&exponential((0.0, 4.0)), Vcab4, &tight).unwrap();
        let exact = 4.0_f64.exp();

        assert!(
            (tight_solution.last_state()[0] - exact).abs()
                < (loose_solution.last_state()[0] - exact).abs()
        );
        assert!(tight_solution.stats().accepted_steps > loose_solution.stats().accepted_steps);
    }

    #[test]
    fn fixed_step_runs_recover_each_methods_design_order() {
        macro_rules! ratio {
            ($algorithm:expr, $order:expr) => {{
                let run = |step| {
                    let options = SolveOptions {
                        adaptive: false,
                        initial_step: Some(step),
                        save: SaveMode::Endpoints,
                        ..SolveOptions::default()
                    };
                    (solve(&exponential((0.0, 1.0)), $algorithm, &options)
                        .unwrap()
                        .last_state()[0]
                        - E)
                        .abs()
                };
                let observed = run(0.05) / run(0.025);
                let minimum = 2.0_f64.powi($order - 1);
                assert!(observed > minimum, "order {} ratio: {}", $order, observed);
            }};
        }

        ratio!(Vcab3, 3);
        ratio!(Vcab4, 4);
        ratio!(Vcab5, 5);
        ratio!(Vcabm3, 3);
        ratio!(Vcabm4, 4);
        ratio!(Vcabm5, 5);
    }

    #[test]
    fn rejected_attempt_does_not_advance_or_contaminate_history() {
        let problem = exponential((0.0, 1.0));
        let options = SolveOptions {
            absolute_tolerance: 1.0e-12,
            relative_tolerance: 1.0e-12,
            ..SolveOptions::default()
        };
        let mut rejected_kernel = VariableAdamsKernel::new(&VCAB5_METHOD);
        let mut fresh_kernel = VariableAdamsKernel::new(&VCAB5_METHOD);
        let mut rejected_stats = SolverStats::default();
        let mut fresh_stats = SolverStats::default();
        let state = [1.0];
        let mut rejected_candidate = [0.0];
        let mut fresh_candidate = [0.0];

        <VariableAdamsKernel as StepKernel<TestRhs, ()>>::initialize(
            &mut rejected_kernel,
            &problem,
            &state,
            0.0,
            &mut rejected_stats,
        )
        .unwrap();
        let rejected = <VariableAdamsKernel as StepKernel<TestRhs, ()>>::attempt_step(
            &mut rejected_kernel,
            &problem,
            &state,
            0.0,
            1.0,
            &mut rejected_candidate,
            &options,
            &mut rejected_stats,
        )
        .unwrap();
        assert!(rejected.error_norm > 1.0);
        <VariableAdamsKernel as StepKernel<TestRhs, ()>>::reject_step(&mut rejected_kernel);

        let accepted_workspace = rejected_kernel.workspace.as_ref().unwrap();
        assert_eq!(rejected_kernel.step_number, 1);
        assert!(
            accepted_workspace
                .accepted_steps
                .iter()
                .all(|step| *step == 0.0)
        );
        assert!(
            accepted_workspace
                .phi_previous
                .iter()
                .flatten()
                .all(|value| *value == 0.0)
        );
        assert_eq!(accepted_workspace.derivative, [1.0]);

        <VariableAdamsKernel as StepKernel<TestRhs, ()>>::initialize(
            &mut fresh_kernel,
            &problem,
            &state,
            0.0,
            &mut fresh_stats,
        )
        .unwrap();
        let after_rejection = <VariableAdamsKernel as StepKernel<TestRhs, ()>>::attempt_step(
            &mut rejected_kernel,
            &problem,
            &state,
            0.0,
            1.0e-4,
            &mut rejected_candidate,
            &options,
            &mut rejected_stats,
        )
        .unwrap();
        let fresh = <VariableAdamsKernel as StepKernel<TestRhs, ()>>::attempt_step(
            &mut fresh_kernel,
            &problem,
            &state,
            0.0,
            1.0e-4,
            &mut fresh_candidate,
            &options,
            &mut fresh_stats,
        )
        .unwrap();

        assert_eq!(after_rejection, fresh);
        assert_eq!(rejected_candidate, fresh_candidate);
    }

    #[test]
    fn terminating_callback_skips_accepted_history_derivative() {
        let calls = Rc::new(Cell::new(0));
        let terminated = Rc::new(Cell::new(false));
        let calls_after_termination = Rc::new(Cell::new(0));
        let rhs_calls = Rc::clone(&calls);
        let rhs_terminated = Rc::clone(&terminated);
        let rhs_calls_after_termination = Rc::clone(&calls_after_termination);
        let effect_terminated = Rc::clone(&terminated);
        let problem = OdeProblem::new(
            move |du: &mut [f64], u: &[f64], _: &(), _: f64| {
                rhs_calls.set(rhs_calls.get() + 1);
                if rhs_terminated.get() {
                    rhs_calls_after_termination.set(rhs_calls_after_termination.get() + 1);
                }
                du[0] = u[0];
            },
            vec![1.0],
            (0.0, 1.0),
            (),
        )
        .with_continuous_callback(
            |_: &[f64], _: &(), time| time - 0.625,
            move |_: &mut [f64], _: &(), _: f64| {
                effect_terminated.set(true);
                CallbackAction::Terminate
            },
        );
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.25),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        let solution = solve(&problem, Vcab3, &options).unwrap();

        assert_eq!(solution.stats().accepted_steps, 3);
        assert_eq!(solution.stats().rhs_evaluations, 8);
        assert_eq!(calls.get(), 8);
        assert_eq!(calls_after_termination.get(), 0);
    }
}
