use crate::integrator::{
    ControllerConfig, KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::linear::{factorize, solve_factorized};
use crate::operator_problem::{LieGroupProblem, LieRepresentation, LinearOperatorProblem};
use crate::solver::validate_state_time_options;
use crate::solvers::exponential::{identity, mat_mul, mat_vec};
use crate::{OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

use super::schemes::{perform_step, transpose};
use super::{
    CayleyEuler, LieGroupAlgorithm, MAX_FACTOR, MIN_FACTOR, OperatorResult, SAFETY, Scheme,
};

impl LieGroupAlgorithm for CayleyEuler {
    fn order(&self) -> usize {
        2
    }

    fn solve_group_validated<O, P>(
        &self,
        problem: &LieGroupProblem<O, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        O: Fn(&mut [f64], &[f64], &P, f64),
    {
        if problem.representation != LieRepresentation::Matrix {
            return Err(SolveError::UnsupportedProblemRepresentation);
        }
        let dummy = OdeProblem::new(
            noop_rhs as fn(&mut [f64], &[f64], &(), f64),
            problem.initial_state().to_vec(),
            problem.time_span(),
            (),
        );
        let evaluate = |output: &mut [f64], state: &[f64], time: f64, stats: &mut SolverStats| {
            problem.evaluate_operator(output, state, time)?;
            stats.rhs_evaluations += 1;
            Ok(())
        };
        drive_integration(
            &dummy,
            options,
            CayleyKernel::new(problem.group_dimension(), evaluate),
        )
    }
}

fn noop_rhs(_: &mut [f64], _: &[f64], _: &(), _: f64) {}

pub(super) fn solve_typed_operator<O, P>(
    problem: &LinearOperatorProblem<O, P>,
    options: &SolveOptions,
    scheme: Scheme,
) -> Result<Solution, SolveError>
where
    O: Fn(&mut [f64], &[f64], &P, f64),
{
    let dummy = OdeProblem::new(
        noop_rhs as fn(&mut [f64], &[f64], &(), f64),
        problem.initial_state().to_vec(),
        problem.time_span(),
        (),
    );
    let evaluate = |output: &mut [f64], state: &[f64], time: f64, stats: &mut SolverStats| {
        problem.evaluate_operator(output, state, time)?;
        stats.rhs_evaluations += 1;
        Ok(())
    };
    drive_integration(
        &dummy,
        options,
        LinearKernel::new(problem.dimension(), scheme, evaluate),
    )
}

pub(super) fn solve_typed_group<O, P>(
    problem: &LieGroupProblem<O, P>,
    options: &SolveOptions,
    scheme: Scheme,
) -> Result<Solution, SolveError>
where
    O: Fn(&mut [f64], &[f64], &P, f64),
{
    let dummy = OdeProblem::new(
        noop_rhs as fn(&mut [f64], &[f64], &(), f64),
        problem.initial_state().to_vec(),
        problem.time_span(),
        (),
    );
    let evaluate = |output: &mut [f64], state: &[f64], time: f64, stats: &mut SolverStats| {
        problem.evaluate_operator(output, state, time)?;
        stats.rhs_evaluations += 1;
        Ok(())
    };
    drive_integration(
        &dummy,
        options,
        LinearKernel::new(problem.group_dimension(), scheme, evaluate),
    )
}

pub(super) fn solve_ode<F, P>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
    scheme: Scheme,
) -> Result<Solution, SolveError>
where
    F: crate::OdeFunction<P>,
{
    let n = problem.initial_state().len();
    let evaluate = |output: &mut [f64], state: &[f64], time: f64, stats: &mut SolverStats| {
        if problem.evaluate_jacobian(output, state, time) {
            stats.jacobian_evaluations += 1;
            return finite_operator(output);
        }
        finite_difference_operator(problem, output, state, time, stats)
    };
    drive_integration(problem, options, LinearKernel::new(n, scheme, evaluate))
}

struct LinearKernel<E> {
    scheme: Scheme,
    dimension: usize,
    evaluate: E,
    operator: Vec<f64>,
    constant_operator: Option<Vec<f64>>,
    leapfrog_previous: Option<Vec<f64>>,
}

impl<E> LinearKernel<E> {
    fn new(dimension: usize, scheme: Scheme, evaluate: E) -> Self {
        Self {
            scheme,
            dimension,
            evaluate,
            operator: vec![0.0; dimension * dimension],
            constant_operator: None,
            leapfrog_previous: None,
        }
    }
}

impl<F, P, E> StepKernel<F, P> for LinearKernel<E>
where
    F: crate::OdeFunction<P>,
    E: FnMut(&mut [f64], &[f64], f64, &mut SolverStats) -> OperatorResult,
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::with_controller(
            self.scheme.adaptive(),
            ControllerConfig::proportional(
                self.scheme.order(),
                SAFETY,
                MIN_FACTOR,
                MAX_FACTOR,
                MIN_FACTOR,
            ),
        )
    }

    fn evaluate_dense_derivative(
        &mut self,
        _: &OdeProblem<F, P>,
        output: &mut [f64],
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> OperatorResult {
        (self.evaluate)(&mut self.operator, state, time, stats)?;
        output.copy_from_slice(&mat_vec(&self.operator, state));
        finite_operator(output)
    }

    fn initialize(
        &mut self,
        _: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> OperatorResult {
        if self.scheme == Scheme::LinearExponential {
            (self.evaluate)(&mut self.operator, state, time, stats)?;
            self.constant_operator = Some(self.operator.clone());
        }
        Ok(())
    }

    fn estimate_initial_step(
        &mut self,
        _: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        _: f64,
        maximum_step: f64,
        _: &mut [f64],
        _: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<f64, SolveError> {
        (self.evaluate)(&mut self.operator, state, time, stats)?;
        let norm = self
            .operator
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        Ok(if norm == 0.0 { 1.0e-3 } else { 0.01 / norm }.clamp(f64::EPSILON, maximum_step))
    }

    fn attempt_step(
        &mut self,
        _: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        options: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError> {
        let result = perform_step(
            self.scheme,
            self.dimension,
            &mut self.evaluate,
            self.constant_operator.as_deref(),
            self.leapfrog_previous.as_deref(),
            state,
            time,
            step,
            stats,
        )?;
        candidate.copy_from_slice(&result.state);
        if !candidate.iter().all(|value| value.is_finite()) {
            return Err(SolveError::NonFiniteDerivative);
        }
        let error_norm = if options.adaptive {
            scaled_error_norm(&result.error, state, candidate, options)
        } else {
            0.0
        };
        Ok(StepEstimate::new(error_norm))
    }

    fn accept_step(
        &mut self,
        _: &OdeProblem<F, P>,
        previous_state: &[f64],
        _: &[f64],
        _: f64,
        _: f64,
        callback_applied: bool,
        _: &mut SolverStats,
    ) -> OperatorResult {
        if self.scheme == Scheme::MagnusLeapfrog {
            self.leapfrog_previous = if callback_applied {
                None
            } else {
                Some(previous_state.to_vec())
            };
        }
        Ok(())
    }

    fn reject_step(&mut self) {}
}

struct CayleyKernel<E> {
    dimension: usize,
    evaluate: E,
    generator: Vec<f64>,
}

impl<E> CayleyKernel<E> {
    fn new(dimension: usize, evaluate: E) -> Self {
        Self {
            dimension,
            evaluate,
            generator: vec![0.0; dimension * dimension],
        }
    }
}

impl<F, P, E> StepKernel<F, P> for CayleyKernel<E>
where
    F: crate::OdeFunction<P>,
    E: FnMut(&mut [f64], &[f64], f64, &mut SolverStats) -> OperatorResult,
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::new(false, 2)
    }
    fn evaluate_dense_derivative(
        &mut self,
        _: &OdeProblem<F, P>,
        output: &mut [f64],
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> OperatorResult {
        (self.evaluate)(&mut self.generator, state, time, stats)?;
        let left = mat_mul(&self.generator, state, self.dimension);
        let right = mat_mul(state, &self.generator, self.dimension);
        for ((output, left), right) in output.iter_mut().zip(left).zip(right) {
            *output = left - right;
        }
        finite_operator(output)
    }
    fn initialize(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: f64,
        _: &mut SolverStats,
    ) -> OperatorResult {
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
        _: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        _: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError> {
        (self.evaluate)(&mut self.generator, state, time, stats)?;
        let n = self.dimension;
        let mut minus = identity(n);
        let mut plus = identity(n);
        for i in 0..n * n {
            minus[i] -= step * self.generator[i] / 2.0;
            plus[i] += step * self.generator[i] / 2.0;
        }
        let mut factors = minus;
        let mut pivots = vec![0; n];
        factorize(&mut factors, &mut pivots, n)?;
        let mut transform = vec![0.0; n * n];
        for column in 0..n {
            let mut rhs = (0..n).map(|row| plus[row * n + column]).collect::<Vec<_>>();
            solve_factorized(&factors, &pivots, &mut rhs, n);
            for row in 0..n {
                transform[row * n + column] = rhs[row];
            }
            stats.linear_solves += 1;
        }
        stats.linear_factorizations += 1;
        let transpose = transpose(&transform, n);
        candidate.copy_from_slice(&mat_mul(&mat_mul(&transform, state, n), &transpose, n));
        Ok(StepEstimate::new(0.0))
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
    ) -> OperatorResult {
        Ok(())
    }
    fn reject_step(&mut self) {}
}

fn finite_operator(operator: &[f64]) -> OperatorResult {
    if operator.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(SolveError::NonFiniteDerivative)
    }
}

fn finite_difference_operator<F, P>(
    problem: &OdeProblem<F, P>,
    output: &mut [f64],
    state: &[f64],
    time: f64,
    stats: &mut SolverStats,
) -> OperatorResult
where
    F: crate::OdeFunction<P>,
{
    let n = state.len();
    let mut base = vec![0.0; n];
    problem
        .rhs
        .evaluate(&mut base, state, problem.parameters(), time)?;
    stats.rhs_evaluations += 1;
    if !base.iter().all(|value| value.is_finite()) {
        return Err(SolveError::NonFiniteDerivative);
    }
    let mut perturbed_state = state.to_vec();
    let mut perturbed = vec![0.0; n];
    for column in 0..n {
        let delta = f64::EPSILON.sqrt() * state[column].abs().max(1.0);
        perturbed_state[column] += delta;
        problem
            .rhs
            .evaluate(&mut perturbed, &perturbed_state, problem.parameters(), time)?;
        stats.rhs_evaluations += 1;
        for row in 0..n {
            output[row * n + column] = (perturbed[row] - base[row]) / delta;
        }
        perturbed_state[column] = state[column];
    }
    stats.jacobian_evaluations += 1;
    finite_operator(output)
}

fn scaled_error_norm(error: &[f64], old: &[f64], new: &[f64], options: &SolveOptions) -> f64 {
    if error.is_empty() {
        return 0.0;
    }
    let sum = error
        .iter()
        .zip(old)
        .zip(new)
        .map(|((error, old), new)| {
            let scale =
                options.absolute_tolerance + options.relative_tolerance * old.abs().max(new.abs());
            (error / scale).powi(2)
        })
        .sum::<f64>();
    (sum / error.len() as f64).sqrt()
}

pub(super) fn validate_inputs(
    state: &[f64],
    time_span: (f64, f64),
    options: &SolveOptions,
) -> OperatorResult {
    validate_state_time_options(state, time_span, options)
}
