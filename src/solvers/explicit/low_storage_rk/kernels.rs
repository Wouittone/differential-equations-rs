use crate::integrator::{
    ControllerConfig, KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::tableau::{
    AlternatingTwoNTableau, LazyLowStorageRungeKuttaTableau, LowStorageAbcTableau,
    LowStorageAdaptiveController, LowStorageEmbeddedTableau, LowStorageEndpointEvaluation,
    LowStorageRungeKuttaLayout, LowStorageRungeKuttaTableau, RegisterPipelineTableau, TableauError,
    ThreeSTableau, load_tableau,
};
use crate::{
    OdeAlgorithm, OdeFunction, OdeProblem, Solution, SolveError, SolveOptions, SolverStats,
};

/// A solver backed by one lazily parsed low-storage Runge--Kutta resource.
///
/// Constructing this value does not parse or allocate coefficient storage.
/// The selected resource is materialized once, when it is first inspected or
/// solved, and its recurrence layout chooses a layout-specific workspace.
#[derive(Clone, Copy)]
pub struct ResourceLowStorageRungeKutta {
    resource: &'static LazyLowStorageRungeKuttaTableau,
}

impl ResourceLowStorageRungeKutta {
    /// Creates a solver referring to a compile-time-validated lazy resource.
    pub const fn new(resource: &'static LazyLowStorageRungeKuttaTableau) -> Self {
        Self { resource }
    }

    /// Loads and returns the method tableau.
    pub fn tableau(self) -> Result<&'static LowStorageRungeKuttaTableau, TableauError> {
        load_tableau(self.resource)
    }
}

impl std::fmt::Debug for ResourceLowStorageRungeKutta {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("ResourceLowStorageRungeKutta { .. }")
    }
}

impl OdeAlgorithm for ResourceLowStorageRungeKutta {
    fn solve_validated<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: OdeFunction<P>,
    {
        let tableau = self.tableau().map_err(SolveError::from)?;
        integrate_resource(problem, options, tableau)
    }
}

fn integrate_resource<F, P>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
    tableau: &'static LowStorageRungeKuttaTableau,
) -> Result<Solution, SolveError>
where
    F: OdeFunction<P>,
{
    let dimension = problem.initial_state().len();
    match tableau.layout() {
        LowStorageRungeKuttaLayout::TwoN(coefficients) => drive_integration(
            problem,
            options,
            LowStorageKernel::new(tableau, TwoNKernel::new(coefficients, dimension), dimension),
        ),
        LowStorageRungeKuttaLayout::TwoC(coefficients) => drive_integration(
            problem,
            options,
            LowStorageKernel::new(tableau, TwoCKernel::new(coefficients, dimension), dimension),
        ),
        LowStorageRungeKuttaLayout::ThreeS(coefficients) => drive_integration(
            problem,
            options,
            LowStorageKernel::new(
                tableau,
                ThreeSKernel::new(coefficients, dimension),
                dimension,
            ),
        ),
        LowStorageRungeKuttaLayout::AlternatingTwoN(coefficients) => drive_integration(
            problem,
            options,
            LowStorageKernel::new(
                tableau,
                AlternatingTwoNKernel::new(coefficients, dimension),
                dimension,
            ),
        ),
        LowStorageRungeKuttaLayout::RegisterPipeline(coefficients) => drive_integration(
            problem,
            options,
            LowStorageKernel::new(
                tableau,
                RegisterPipelineKernel::new(coefficients, dimension),
                dimension,
            ),
        ),
    }
}

struct LowStorageAttempt<'a, F, P> {
    problem: &'a OdeProblem<F, P>,
    start_derivative: &'a [f64],
    state: &'a [f64],
    time: f64,
    step: f64,
    candidate: &'a mut [f64],
    adaptive_scratch: &'a mut [f64],
    embedded: Option<&'a LowStorageEmbeddedTableau>,
    options: &'a SolveOptions,
    stats: &'a mut SolverStats,
}

trait LowStorageRecurrence<F, P>
where
    F: OdeFunction<P>,
{
    fn attempt(&mut self, context: LowStorageAttempt<'_, F, P>)
    -> Result<StepEstimate, SolveError>;

    fn accepted(&mut self) {}

    fn endpoint_derivative(&self) -> Option<&[f64]> {
        None
    }
}

struct LowStorageKernel<K> {
    tableau: &'static LowStorageRungeKuttaTableau,
    recurrence: K,
    start_derivative: Vec<f64>,
    start_derivative_valid: bool,
    start_time: Option<f64>,
    endpoint_time: Option<f64>,
    adaptive_scratch: Vec<f64>,
}

impl<K> LowStorageKernel<K> {
    fn new(tableau: &'static LowStorageRungeKuttaTableau, recurrence: K, dimension: usize) -> Self {
        Self {
            tableau,
            recurrence,
            start_derivative: vec![0.0; dimension],
            start_derivative_valid: false,
            start_time: None,
            endpoint_time: None,
            adaptive_scratch: Vec::new(),
        }
    }

    fn controller(&self) -> ControllerConfig {
        match self
            .tableau
            .embedded()
            .map(LowStorageEmbeddedTableau::controller)
        {
            Some(LowStorageAdaptiveController::Pid(controller)) => {
                let embedded_order = self
                    .tableau
                    .embedded()
                    .expect("the matched embedded estimator exists")
                    .order();
                ControllerConfig::pid(
                    self.tableau.order().min(embedded_order) + 1,
                    controller.beta(),
                    controller.acceptance_safety(),
                )
            }
            Some(LowStorageAdaptiveController::StandardPi) => {
                ControllerConfig::standard_pi(self.tableau.order())
            }
            None => ControllerConfig::proportional(self.tableau.order(), 0.9, 0.2, 10.0, 0.2),
        }
    }
}

impl<F, P, K> StepKernel<F, P> for LowStorageKernel<K>
where
    F: OdeFunction<P>,
    K: LowStorageRecurrence<F, P>,
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::with_controller(self.tableau.embedded().is_some(), self.controller())
    }

    fn initialize(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        evaluate(problem, &mut self.start_derivative, state, time, stats)?;
        self.start_derivative_valid = true;
        self.start_time = Some(time);
        Ok(())
    }

    fn estimate_initial_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        direction: f64,
        maximum_step: f64,
        candidate: &mut [f64],
        options: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<f64, SolveError> {
        self.adaptive_scratch.resize(state.len(), 0.0);
        estimate_initial_step(
            problem,
            options,
            state,
            &self.start_derivative,
            time,
            direction,
            maximum_step,
            &mut self.adaptive_scratch,
            candidate,
            self.tableau.order(),
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
        if !self.start_derivative_valid || self.start_time != Some(time) {
            evaluate(problem, &mut self.start_derivative, state, time, stats)?;
            self.start_derivative_valid = true;
            self.start_time = Some(time);
        }
        if options.adaptive {
            self.adaptive_scratch.resize(state.len(), 0.0);
        }
        let estimate = self.recurrence.attempt(LowStorageAttempt {
            problem,
            start_derivative: &self.start_derivative,
            state,
            time,
            step,
            candidate,
            adaptive_scratch: &mut self.adaptive_scratch,
            embedded: self.tableau.embedded(),
            options,
            stats,
        })?;
        self.endpoint_time = self.recurrence.endpoint_derivative().map(|_| time + step);
        Ok(estimate)
    }

    fn evaluate_dense_derivative(
        &mut self,
        problem: &OdeProblem<F, P>,
        output: &mut [f64],
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        if self.start_derivative_valid && self.start_time == Some(time) {
            output.copy_from_slice(&self.start_derivative);
            return Ok(());
        }
        if self.endpoint_time == Some(time) {
            if let Some(endpoint) = self.recurrence.endpoint_derivative() {
                output.copy_from_slice(endpoint);
                return Ok(());
            }
        }
        evaluate(problem, output, state, time, stats)?;
        self.start_derivative.copy_from_slice(output);
        self.start_derivative_valid = true;
        self.start_time = Some(time);
        Ok(())
    }

    fn accept_step(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: &[f64],
        time: f64,
        _: f64,
        callback_applied: bool,
        _: &mut SolverStats,
    ) -> Result<(), SolveError> {
        self.recurrence.accepted();
        if callback_applied {
            self.start_derivative_valid = false;
            self.start_time = None;
        } else if self.tableau.fsal() {
            let endpoint = self
                .recurrence
                .endpoint_derivative()
                .ok_or(SolveError::InvalidTableau)?;
            self.start_derivative.copy_from_slice(endpoint);
            self.start_derivative_valid = true;
            self.start_time = Some(time);
        } else if self.start_time != Some(time) {
            self.start_derivative_valid = false;
            self.start_time = None;
        }
        self.endpoint_time = None;
        Ok(())
    }

    fn reject_step(&mut self) {
        self.endpoint_time = None;
    }
}

struct TwoNKernel {
    coefficients: &'static LowStorageAbcTableau,
    derivative: Vec<f64>,
    residual: Vec<f64>,
}

impl TwoNKernel {
    fn new(coefficients: &'static LowStorageAbcTableau, dimension: usize) -> Self {
        Self {
            coefficients,
            derivative: vec![0.0; dimension],
            residual: vec![0.0; dimension],
        }
    }
}

impl<F, P> LowStorageRecurrence<F, P> for TwoNKernel
where
    F: OdeFunction<P>,
{
    fn attempt(
        &mut self,
        context: LowStorageAttempt<'_, F, P>,
    ) -> Result<StepEstimate, SolveError> {
        let LowStorageAttempt {
            problem,
            start_derivative,
            state,
            time,
            step,
            candidate,
            stats,
            ..
        } = context;
        let a = self.coefficients.a();
        let b = self.coefficients.b();
        let c = self.coefficients.c();
        candidate.copy_from_slice(state);
        self.derivative.copy_from_slice(start_derivative);
        for ((residual, candidate), derivative) in self
            .residual
            .iter_mut()
            .zip(&mut *candidate)
            .zip(&self.derivative)
        {
            *residual = step * derivative;
            *candidate += b[0] * *residual;
        }
        for stage in 0..a.len() {
            evaluate(
                problem,
                &mut self.derivative,
                candidate,
                time + c[stage] * step,
                stats,
            )?;
            for ((residual, candidate), derivative) in self
                .residual
                .iter_mut()
                .zip(&mut *candidate)
                .zip(&self.derivative)
            {
                *residual = a[stage] * *residual + step * derivative;
                *candidate += b[stage + 1] * *residual;
            }
        }
        finish(candidate)
    }
}

struct TwoCKernel {
    coefficients: &'static LowStorageAbcTableau,
    derivative: Vec<f64>,
    temporary: Vec<f64>,
}

impl TwoCKernel {
    fn new(coefficients: &'static LowStorageAbcTableau, dimension: usize) -> Self {
        Self {
            coefficients,
            derivative: vec![0.0; dimension],
            temporary: vec![0.0; dimension],
        }
    }
}

impl<F, P> LowStorageRecurrence<F, P> for TwoCKernel
where
    F: OdeFunction<P>,
{
    fn attempt(
        &mut self,
        context: LowStorageAttempt<'_, F, P>,
    ) -> Result<StepEstimate, SolveError> {
        let LowStorageAttempt {
            problem,
            start_derivative,
            state,
            time,
            step,
            candidate,
            stats,
            ..
        } = context;
        let a = self.coefficients.a();
        let b = self.coefficients.b();
        let c = self.coefficients.c();
        candidate.copy_from_slice(state);
        self.derivative.copy_from_slice(start_derivative);
        for (candidate, derivative) in candidate.iter_mut().zip(&self.derivative) {
            *candidate += b[0] * step * *derivative;
        }
        for stage in 0..a.len() {
            self.temporary.copy_from_slice(candidate);
            for (temporary, derivative) in self.temporary.iter_mut().zip(&self.derivative) {
                *temporary += a[stage] * step * *derivative;
            }
            evaluate(
                problem,
                &mut self.derivative,
                &self.temporary,
                time + c[stage] * step,
                stats,
            )?;
            for (candidate, derivative) in candidate.iter_mut().zip(&self.derivative) {
                *candidate += b[stage + 1] * step * *derivative;
            }
        }
        finish(candidate)
    }
}

struct ThreeSKernel {
    coefficients: &'static ThreeSTableau,
    derivative: Vec<f64>,
    temporary: Vec<f64>,
}

impl ThreeSKernel {
    fn new(coefficients: &'static ThreeSTableau, dimension: usize) -> Self {
        Self {
            coefficients,
            derivative: vec![0.0; dimension],
            temporary: vec![0.0; dimension],
        }
    }
}

impl<F, P> LowStorageRecurrence<F, P> for ThreeSKernel
where
    F: OdeFunction<P>,
{
    fn attempt(
        &mut self,
        context: LowStorageAttempt<'_, F, P>,
    ) -> Result<StepEstimate, SolveError> {
        let LowStorageAttempt {
            problem,
            start_derivative,
            state,
            time,
            step,
            candidate,
            adaptive_scratch: error,
            embedded,
            options,
            stats,
        } = context;
        let coefficients = self.coefficients;
        candidate.copy_from_slice(state);
        self.temporary.copy_from_slice(state);
        self.derivative.copy_from_slice(start_derivative);
        if options.adaptive {
            let weights = embedded.ok_or(SolveError::AdaptiveStepUnsupported)?.error();
            for ((error, derivative), weight) in error
                .iter_mut()
                .zip(start_derivative)
                .zip(std::iter::repeat(weights[0]))
            {
                *error = step * weight * derivative;
            }
        }
        for (candidate, derivative) in candidate.iter_mut().zip(&self.derivative) {
            *candidate += coefficients.beta1() * step * *derivative;
        }
        for stage in 0..coefficients.gamma1().len() {
            evaluate(
                problem,
                &mut self.derivative,
                candidate,
                time + coefficients.c()[stage] * step,
                stats,
            )?;
            if options.adaptive {
                let weight =
                    embedded.ok_or(SolveError::AdaptiveStepUnsupported)?.error()[stage + 1];
                for (error, derivative) in error.iter_mut().zip(&self.derivative) {
                    *error += step * weight * derivative;
                }
            }
            for (((candidate, temporary), derivative), state_value) in candidate
                .iter_mut()
                .zip(&mut self.temporary)
                .zip(&self.derivative)
                .zip(state)
            {
                *temporary += coefficients.delta()[stage] * *candidate;
                *candidate = coefficients.gamma1()[stage] * *candidate
                    + coefficients.gamma2()[stage] * *temporary
                    + coefficients.gamma3()[stage] * *state_value
                    + coefficients.beta2()[stage] * step * *derivative;
            }
        }
        if coefficients.endpoint_evaluation() == LowStorageEndpointEvaluation::Evaluate {
            evaluate(problem, &mut self.derivative, candidate, time + step, stats)?;
            if options.adaptive {
                let weights = embedded.ok_or(SolveError::AdaptiveStepUnsupported)?.error();
                if let Some(weight) = weights.get(coefficients.stages()) {
                    for (error, derivative) in error.iter_mut().zip(&self.derivative) {
                        *error += step * weight * derivative;
                    }
                }
            }
        }
        finish_adaptive(candidate, state, error, embedded, options)
    }

    fn endpoint_derivative(&self) -> Option<&[f64]> {
        (self.coefficients.endpoint_evaluation() == LowStorageEndpointEvaluation::Evaluate)
            .then_some(self.derivative.as_slice())
    }
}

struct AlternatingTwoNKernel {
    coefficients: &'static AlternatingTwoNTableau,
    derivative: Vec<f64>,
    residual: Vec<f64>,
    use_second: bool,
}

impl AlternatingTwoNKernel {
    fn new(coefficients: &'static AlternatingTwoNTableau, dimension: usize) -> Self {
        Self {
            coefficients,
            derivative: vec![0.0; dimension],
            residual: vec![0.0; dimension],
            use_second: false,
        }
    }
}

impl<F, P> LowStorageRecurrence<F, P> for AlternatingTwoNKernel
where
    F: OdeFunction<P>,
{
    fn attempt(
        &mut self,
        context: LowStorageAttempt<'_, F, P>,
    ) -> Result<StepEstimate, SolveError> {
        let LowStorageAttempt {
            problem,
            start_derivative,
            state,
            time,
            step,
            candidate,
            stats,
            ..
        } = context;
        let coefficients = if self.use_second {
            self.coefficients.second()
        } else {
            self.coefficients.first()
        };
        let a = coefficients.a();
        let b = coefficients.b();
        let c = coefficients.c();
        candidate.copy_from_slice(state);
        self.derivative.copy_from_slice(start_derivative);
        for ((residual, candidate), derivative) in self
            .residual
            .iter_mut()
            .zip(&mut *candidate)
            .zip(&self.derivative)
        {
            *residual = step * derivative;
            *candidate += b[0] * *residual;
        }
        for stage in 0..a.len() {
            evaluate(
                problem,
                &mut self.derivative,
                candidate,
                time + c[stage] * step,
                stats,
            )?;
            for ((residual, candidate), derivative) in self
                .residual
                .iter_mut()
                .zip(&mut *candidate)
                .zip(&self.derivative)
            {
                *residual = a[stage] * *residual + step * derivative;
                *candidate += b[stage + 1] * *residual;
            }
        }
        evaluate(problem, &mut self.derivative, candidate, time + step, stats)?;
        finish(candidate)
    }

    fn accepted(&mut self) {
        self.use_second = !self.use_second;
    }

    fn endpoint_derivative(&self) -> Option<&[f64]> {
        Some(&self.derivative)
    }
}

struct RegisterPipelineKernel {
    coefficients: &'static RegisterPipelineTableau,
    dimension: usize,
    workspace: Vec<f64>,
}

impl RegisterPipelineKernel {
    fn new(coefficients: &'static RegisterPipelineTableau, dimension: usize) -> Self {
        Self {
            coefficients,
            dimension,
            workspace: vec![0.0; (2 * coefficients.history_states() + 1) * dimension],
        }
    }
}

impl<F, P> LowStorageRecurrence<F, P> for RegisterPipelineKernel
where
    F: OdeFunction<P>,
{
    fn attempt(
        &mut self,
        context: LowStorageAttempt<'_, F, P>,
    ) -> Result<StepEstimate, SolveError> {
        let LowStorageAttempt {
            problem,
            start_derivative,
            state,
            time,
            step,
            candidate,
            adaptive_scratch: error,
            embedded,
            options,
            stats,
        } = context;
        let coefficients = self.coefficients;
        let state_history_len = coefficients.history_states() * self.dimension;
        let (derivative, workspace) = self.workspace.split_at_mut(self.dimension);
        let (stage_state, history) = workspace.split_at_mut(self.dimension);
        let (history_states, history_derivatives) = history.split_at_mut(state_history_len);
        candidate.copy_from_slice(state);
        for history in history_states.chunks_exact_mut(self.dimension) {
            history.copy_from_slice(state);
        }
        history_derivatives.fill(0.0);
        derivative.copy_from_slice(start_derivative);
        if options.adaptive {
            error.fill(0.0);
        }
        for stage in 0..coefficients.c().len() {
            let last_state_start = state_history_len
                .checked_sub(self.dimension)
                .ok_or(SolveError::InvalidTableau)?;
            stage_state.copy_from_slice(&history_states[last_state_start..]);
            for (value, derivative) in stage_state.iter_mut().zip(&*derivative) {
                *value += coefficients.a()[0][stage] * step * *derivative;
            }
            for (register, row) in history_derivatives
                .chunks_exact(self.dimension)
                .zip(coefficients.a().iter().skip(1))
            {
                for (value, derivative) in stage_state.iter_mut().zip(register) {
                    *value += derivative * row[stage] * step;
                }
            }
            for (candidate, derivative) in candidate.iter_mut().zip(&*derivative) {
                *candidate += coefficients.b()[stage] * step * *derivative;
            }
            if options.adaptive {
                let weight = embedded.ok_or(SolveError::AdaptiveStepUnsupported)?.error()[stage];
                for (error, derivative) in error.iter_mut().zip(&*derivative) {
                    *error += step * weight * derivative;
                }
            }
            if history_derivatives.len() >= self.dimension {
                let shift_end = history_derivatives.len() - self.dimension;
                history_derivatives.copy_within(0..shift_end, self.dimension);
                history_derivatives[..self.dimension].copy_from_slice(derivative);
            }
            let shift_end = history_states.len() - self.dimension;
            history_states.copy_within(0..shift_end, self.dimension);
            history_states[..self.dimension].copy_from_slice(candidate);
            evaluate(
                problem,
                derivative,
                stage_state,
                time + coefficients.c()[stage] * step,
                stats,
            )?;
        }
        for (candidate, derivative) in candidate.iter_mut().zip(&*derivative) {
            *candidate += coefficients.b_final() * step * *derivative;
        }
        if options.adaptive {
            let weight = embedded.ok_or(SolveError::AdaptiveStepUnsupported)?.error()
                [coefficients.c().len()];
            for (error, derivative) in error.iter_mut().zip(&*derivative) {
                *error += step * weight * derivative;
            }
        }
        evaluate(problem, derivative, candidate, time + step, stats)?;
        if options.adaptive {
            let weights = embedded.ok_or(SolveError::AdaptiveStepUnsupported)?.error();
            if let Some(weight) = weights.get(coefficients.stages()) {
                for (error, derivative) in error.iter_mut().zip(&*derivative) {
                    *error += step * weight * derivative;
                }
            }
        }
        finish_adaptive(candidate, state, error, embedded, options)
    }

    fn endpoint_derivative(&self) -> Option<&[f64]> {
        Some(&self.workspace[..self.dimension])
    }
}

fn evaluate<F, P>(
    problem: &OdeProblem<F, P>,
    derivative: &mut [f64],
    state: &[f64],
    time: f64,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: OdeFunction<P>,
{
    problem
        .rhs
        .evaluate(derivative, state, problem.parameters(), time)?;
    stats.rhs_evaluations += 1;
    ensure_finite(derivative)
}

#[allow(clippy::too_many_arguments)]
fn estimate_initial_step<F, P>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
    state: &[f64],
    start_derivative: &[f64],
    time: f64,
    direction: f64,
    maximum_step: f64,
    trial_state: &mut [f64],
    next_derivative: &mut [f64],
    order: usize,
    stats: &mut SolverStats,
) -> Result<f64, SolveError>
where
    F: OdeFunction<P>,
{
    let dimension = state.len() as f64;
    let mut state_norm = 0.0;
    let mut derivative_norm = 0.0;
    for (value, derivative) in state.iter().zip(start_derivative) {
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

    for ((trial, value), derivative) in trial_state.iter_mut().zip(state).zip(start_derivative) {
        *trial = value + direction * trial_step * derivative;
    }
    evaluate(
        problem,
        next_derivative,
        trial_state,
        time + direction * trial_step,
        stats,
    )?;

    let mut curvature_norm = 0.0;
    for ((next, initial), value) in next_derivative.iter().zip(start_derivative).zip(state) {
        let scale = options.absolute_tolerance + options.relative_tolerance * value.abs();
        curvature_norm += ((next - initial) / scale).powi(2);
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

fn finish(candidate: &[f64]) -> Result<StepEstimate, SolveError> {
    ensure_finite(candidate)?;
    Ok(StepEstimate::new(0.0))
}

fn finish_adaptive(
    candidate: &[f64],
    state: &[f64],
    error: &[f64],
    embedded: Option<&LowStorageEmbeddedTableau>,
    options: &SolveOptions,
) -> Result<StepEstimate, SolveError> {
    ensure_finite(candidate)?;
    if !options.adaptive {
        return Ok(StepEstimate::new(0.0));
    }
    embedded.ok_or(SolveError::AdaptiveStepUnsupported)?;
    ensure_finite(error)?;
    let mut squared_norm = 0.0;
    for ((error, old), new) in error.iter().zip(state).zip(candidate) {
        let scale =
            options.absolute_tolerance + options.relative_tolerance * old.abs().max(new.abs());
        squared_norm += (error / scale).powi(2);
    }
    Ok(StepEstimate::new(
        (squared_norm / state.len() as f64).sqrt(),
    ))
}

fn ensure_finite(values: &[f64]) -> Result<(), SolveError> {
    values
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteDerivative)
}
