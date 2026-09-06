use crate::integrator::{
    KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::tableau::{
    AlternatingTwoNTableau, LazyLowStorageRungeKuttaTableau, LowStorageAbcTableau,
    LowStorageEndpointEvaluation, LowStorageRungeKuttaLayout, LowStorageRungeKuttaTableau,
    RegisterPipelineTableau, TableauError, ThreeSTableau, load_tableau,
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
        let tableau = self.tableau().map_err(|_| SolveError::InvalidTableau)?;
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
            FixedKernel(TwoNKernel::new(tableau.order(), coefficients, dimension)),
        ),
        LowStorageRungeKuttaLayout::TwoC(coefficients) => drive_integration(
            problem,
            options,
            FixedKernel(TwoCKernel::new(tableau.order(), coefficients, dimension)),
        ),
        LowStorageRungeKuttaLayout::ThreeS(coefficients) => drive_integration(
            problem,
            options,
            FixedKernel(ThreeSKernel::new(tableau.order(), coefficients, dimension)),
        ),
        LowStorageRungeKuttaLayout::AlternatingTwoN(coefficients) => drive_integration(
            problem,
            options,
            FixedKernel(AlternatingTwoNKernel::new(
                tableau.order(),
                coefficients,
                dimension,
            )),
        ),
        LowStorageRungeKuttaLayout::RegisterPipeline(coefficients) => drive_integration(
            problem,
            options,
            FixedKernel(RegisterPipelineKernel::new(
                tableau.order(),
                coefficients,
                dimension,
            )),
        ),
    }
}

trait FixedLowStorageKernel<F, P>
where
    F: OdeFunction<P>,
{
    fn order(&self) -> usize;

    fn attempt(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        stats: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError>;

    fn accepted(&mut self) {}
}

struct FixedKernel<K>(K);

impl<F, P, K> StepKernel<F, P> for FixedKernel<K>
where
    F: OdeFunction<P>,
    K: FixedLowStorageKernel<F, P>,
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::new(false, self.0.order())
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
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: f64,
        _: f64,
        _: f64,
        _: &mut [f64],
        _: &SolveOptions,
        _: &mut SolverStats,
    ) -> Result<f64, SolveError> {
        Err(SolveError::InitialStepRequired)
    }

    fn attempt_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        _: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError> {
        self.0.attempt(problem, state, time, step, candidate, stats)
    }

    fn accept_step(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: &[f64],
        _: f64,
        _: f64,
        _: bool,
        _: &mut SolverStats,
    ) -> Result<(), SolveError> {
        self.0.accepted();
        Ok(())
    }

    fn reject_step(&mut self) {}
}

struct TwoNKernel {
    order: usize,
    coefficients: &'static LowStorageAbcTableau,
    derivative: Vec<f64>,
    residual: Vec<f64>,
}

impl TwoNKernel {
    fn new(order: usize, coefficients: &'static LowStorageAbcTableau, dimension: usize) -> Self {
        Self {
            order,
            coefficients,
            derivative: vec![0.0; dimension],
            residual: vec![0.0; dimension],
        }
    }
}

impl<F, P> FixedLowStorageKernel<F, P> for TwoNKernel
where
    F: OdeFunction<P>,
{
    fn order(&self) -> usize {
        self.order
    }

    fn attempt(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        stats: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError> {
        let a = self.coefficients.a();
        let b = self.coefficients.b();
        let c = self.coefficients.c();
        candidate.copy_from_slice(state);
        evaluate(problem, &mut self.derivative, state, time, stats)?;
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
    order: usize,
    coefficients: &'static LowStorageAbcTableau,
    derivative: Vec<f64>,
    temporary: Vec<f64>,
}

impl TwoCKernel {
    fn new(order: usize, coefficients: &'static LowStorageAbcTableau, dimension: usize) -> Self {
        Self {
            order,
            coefficients,
            derivative: vec![0.0; dimension],
            temporary: vec![0.0; dimension],
        }
    }
}

impl<F, P> FixedLowStorageKernel<F, P> for TwoCKernel
where
    F: OdeFunction<P>,
{
    fn order(&self) -> usize {
        self.order
    }

    fn attempt(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        stats: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError> {
        let a = self.coefficients.a();
        let b = self.coefficients.b();
        let c = self.coefficients.c();
        candidate.copy_from_slice(state);
        evaluate(problem, &mut self.derivative, state, time, stats)?;
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
    order: usize,
    coefficients: &'static ThreeSTableau,
    derivative: Vec<f64>,
    temporary: Vec<f64>,
}

impl ThreeSKernel {
    fn new(order: usize, coefficients: &'static ThreeSTableau, dimension: usize) -> Self {
        Self {
            order,
            coefficients,
            derivative: vec![0.0; dimension],
            temporary: vec![0.0; dimension],
        }
    }
}

impl<F, P> FixedLowStorageKernel<F, P> for ThreeSKernel
where
    F: OdeFunction<P>,
{
    fn order(&self) -> usize {
        self.order
    }

    fn attempt(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        stats: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError> {
        let coefficients = self.coefficients;
        candidate.copy_from_slice(state);
        self.temporary.copy_from_slice(state);
        evaluate(problem, &mut self.derivative, state, time, stats)?;
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
        }
        finish(candidate)
    }
}

struct AlternatingTwoNKernel {
    order: usize,
    coefficients: &'static AlternatingTwoNTableau,
    derivative: Vec<f64>,
    residual: Vec<f64>,
    use_second: bool,
}

impl AlternatingTwoNKernel {
    fn new(order: usize, coefficients: &'static AlternatingTwoNTableau, dimension: usize) -> Self {
        Self {
            order,
            coefficients,
            derivative: vec![0.0; dimension],
            residual: vec![0.0; dimension],
            use_second: false,
        }
    }
}

impl<F, P> FixedLowStorageKernel<F, P> for AlternatingTwoNKernel
where
    F: OdeFunction<P>,
{
    fn order(&self) -> usize {
        self.order
    }

    fn attempt(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        stats: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError> {
        let coefficients = if self.use_second {
            self.coefficients.second()
        } else {
            self.coefficients.first()
        };
        let a = coefficients.a();
        let b = coefficients.b();
        let c = coefficients.c();
        candidate.copy_from_slice(state);
        evaluate(problem, &mut self.derivative, state, time, stats)?;
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
}

struct RegisterPipelineKernel {
    order: usize,
    coefficients: &'static RegisterPipelineTableau,
    derivative: Vec<f64>,
    stage_state: Vec<f64>,
    history_states: Vec<Vec<f64>>,
    history_derivatives: Vec<Vec<f64>>,
}

impl RegisterPipelineKernel {
    fn new(order: usize, coefficients: &'static RegisterPipelineTableau, dimension: usize) -> Self {
        Self {
            order,
            coefficients,
            derivative: vec![0.0; dimension],
            stage_state: vec![0.0; dimension],
            history_states: (0..coefficients.history_states())
                .map(|_| vec![0.0; dimension])
                .collect(),
            history_derivatives: (0..coefficients.history_states().saturating_sub(1))
                .map(|_| vec![0.0; dimension])
                .collect(),
        }
    }
}

impl<F, P> FixedLowStorageKernel<F, P> for RegisterPipelineKernel
where
    F: OdeFunction<P>,
{
    fn order(&self) -> usize {
        self.order
    }

    fn attempt(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        stats: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError> {
        let coefficients = self.coefficients;
        candidate.copy_from_slice(state);
        for history in &mut self.history_states {
            history.copy_from_slice(state);
        }
        for history in &mut self.history_derivatives {
            history.fill(0.0);
        }
        evaluate(problem, &mut self.derivative, state, time, stats)?;
        for stage in 0..coefficients.c().len() {
            self.stage_state.copy_from_slice(
                self.history_states
                    .last()
                    .ok_or(SolveError::InvalidTableau)?,
            );
            for (value, derivative) in self.stage_state.iter_mut().zip(&self.derivative) {
                *value += coefficients.a()[0][stage] * step * *derivative;
            }
            for (register, row) in self
                .history_derivatives
                .iter()
                .zip(coefficients.a().iter().skip(1))
            {
                for (value, derivative) in self.stage_state.iter_mut().zip(register) {
                    *value += derivative * row[stage] * step;
                }
            }
            for (candidate, derivative) in candidate.iter_mut().zip(&self.derivative) {
                *candidate += coefficients.b()[stage] * step * *derivative;
            }
            for index in (1..self.history_derivatives.len()).rev() {
                let (head, tail) = self.history_derivatives.split_at_mut(index);
                tail[0].copy_from_slice(&head[index - 1]);
            }
            if let Some(history) = self.history_derivatives.first_mut() {
                history.copy_from_slice(&self.derivative);
            }
            for index in (1..self.history_states.len()).rev() {
                let (head, tail) = self.history_states.split_at_mut(index);
                tail[0].copy_from_slice(&head[index - 1]);
            }
            self.history_states[0].copy_from_slice(candidate);
            evaluate(
                problem,
                &mut self.derivative,
                &self.stage_state,
                time + coefficients.c()[stage] * step,
                stats,
            )?;
        }
        for (candidate, derivative) in candidate.iter_mut().zip(&self.derivative) {
            *candidate += coefficients.b_final() * step * *derivative;
        }
        evaluate(problem, &mut self.derivative, candidate, time + step, stats)?;
        finish(candidate)
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

fn finish(candidate: &[f64]) -> Result<StepEstimate, SolveError> {
    ensure_finite(candidate)?;
    Ok(StepEstimate::new(0.0))
}

fn ensure_finite(values: &[f64]) -> Result<(), SolveError> {
    values
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteDerivative)
}
