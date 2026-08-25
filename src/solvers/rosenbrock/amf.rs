use std::marker::PhantomData;

use super::general::Rosenbrock23;
use crate::integrator::{
    ControllerConfig, KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::linear::{factorize, solve_factorized};
use crate::{
    ConfigurationError, OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats,
};

const GAMMA: f64 = 1.0 / (2.0 + std::f64::consts::SQRT_2);
const C32: f64 = 6.0 + std::f64::consts::SQRT_2;

/// Ordered dense approximation of a Rosenbrock W operator.
///
/// For split Jacobians `J_i`, the factorization represents
/// `prod_i(I-gamma*h*J_i)`. Solves apply the factors in product order.
#[derive(Clone, Debug)]
pub struct AmfOperator {
    dimension: usize,
    jacobian_factors: Vec<Vec<f64>>,
    factorizations: Vec<Vec<f64>>,
    pivots: Vec<Vec<usize>>,
}

/// Exact upstream spelling for [`AmfOperator`].
pub type AMFOperator = AmfOperator;

impl AmfOperator {
    pub fn from_jacobian(
        dimension: usize,
        jacobian: impl Into<Vec<f64>>,
    ) -> Result<Self, ConfigurationError> {
        Self::from_split(dimension, vec![jacobian.into()])
    }

    pub fn from_split(
        dimension: usize,
        factors: Vec<Vec<f64>>,
    ) -> Result<Self, ConfigurationError> {
        if dimension == 0 || factors.is_empty() {
            return Err(ConfigurationError::EmptyData {
                context: "AMF factor collection",
            });
        }
        let matrix_len =
            dimension
                .checked_mul(dimension)
                .ok_or(ConfigurationError::DimensionOverflow {
                    context: "AMF operator",
                })?;
        if factors.iter().any(|factor| factor.len() != matrix_len) {
            return Err(ConfigurationError::DimensionMismatch {
                context: "AMF factor collection",
            });
        }
        if factors.iter().flatten().any(|value| !value.is_finite()) {
            return Err(ConfigurationError::NonFiniteData {
                context: "AMF factor collection",
            });
        }
        Ok(Self {
            dimension,
            factorizations: vec![vec![0.0; dimension * dimension]; factors.len()],
            pivots: vec![vec![0; dimension]; factors.len()],
            jacobian_factors: factors,
        })
    }

    pub fn factor_count(&self) -> usize {
        self.jacobian_factors.len()
    }
    pub fn factors(&self) -> &[Vec<f64>] {
        &self.jacobian_factors
    }
    pub fn factors_mut(&mut self) -> &mut [Vec<f64>] {
        &mut self.jacobian_factors
    }

    pub fn factorize(&mut self, scaled_gamma: f64) -> Result<(), SolveError> {
        for ((jacobian, matrix), pivots) in self
            .jacobian_factors
            .iter()
            .zip(&mut self.factorizations)
            .zip(&mut self.pivots)
        {
            for row in 0..self.dimension {
                for column in 0..self.dimension {
                    matrix[row * self.dimension + column] = f64::from(row == column)
                        - scaled_gamma * jacobian[row * self.dimension + column];
                }
            }
            factorize(matrix, pivots, self.dimension)?;
        }
        Ok(())
    }

    pub fn solve_ordered(&self, right_hand_side: &mut [f64]) {
        for (matrix, pivots) in self.factorizations.iter().zip(&self.pivots) {
            solve_factorized(matrix, pivots, right_hand_side, self.dimension);
        }
    }
}

/// RHS, exact Jacobian, and updateable structured AMF factor callbacks.
pub struct AmfFunction<F, J, S> {
    rhs: F,
    jacobian: J,
    factor_prototypes: Vec<Vec<f64>>,
    update_factors: S,
}

/// Builds the typed equivalent of upstream `build_amf_function`.
pub fn build_amf_function<F, J, S>(
    dimension: usize,
    rhs: F,
    jacobian: J,
    factor_prototypes: Vec<Vec<f64>>,
    update_factors: S,
) -> Result<AmfFunction<F, J, S>, ConfigurationError> {
    AmfOperator::from_split(dimension, factor_prototypes.clone())?;
    Ok(AmfFunction {
        rhs,
        jacobian,
        factor_prototypes,
        update_factors,
    })
}

/// Checked structured problem consumed by [`solve_amf`].
pub struct AmfProblem<F, J, S, P> {
    function: AmfFunction<F, J, S>,
    initial_state: Vec<f64>,
    time_span: (f64, f64),
    parameters: P,
}

impl<F, J, S, P> AmfProblem<F, J, S, P> {
    pub fn new(
        function: AmfFunction<F, J, S>,
        initial_state: impl Into<Vec<f64>>,
        time_span: (f64, f64),
        parameters: P,
    ) -> Result<Self, ConfigurationError> {
        let initial_state = initial_state.into();
        if initial_state.is_empty()
            || function
                .factor_prototypes
                .iter()
                .any(|factor| factor.len() != initial_state.len() * initial_state.len())
        {
            return Err(ConfigurationError::DimensionMismatch {
                context: "AMF problem",
            });
        }
        Ok(Self {
            function,
            initial_state,
            time_span,
            parameters,
        })
    }
    pub fn initial_state(&self) -> &[f64] {
        &self.initial_state
    }
    pub fn time_span(&self) -> (f64, f64) {
        self.time_span
    }
    pub fn factor_count(&self) -> usize {
        self.function.factor_prototypes.len()
    }
}

/// Approximate-matrix-factorization wrapper for a Rosenbrock-W algorithm.
///
/// The current exact Rust port supports the pinned `Rosenbrock23` W tableau;
/// its W solves are performed by the ordered factor operator, not delegated.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AMF<A = Rosenbrock23> {
    inner: A,
}

impl<A> AMF<A> {
    pub const fn new(inner: A) -> Self {
        Self { inner }
    }
    pub fn inner(&self) -> &A {
        &self.inner
    }
}

impl OdeAlgorithm for AMF<Rosenbrock23> {
    fn solve<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        let n = problem.initial_state().len();
        let evaluate = |derivative: &mut [f64],
                        jacobian: &mut [f64],
                        factors: &mut [Vec<f64>],
                        state: &[f64],
                        time: f64,
                        stats: &mut SolverStats| {
            (problem.rhs)(derivative, state, problem.parameters(), time);
            stats.rhs_evaluations += 1;
            checked(derivative)?;
            if problem.evaluate_jacobian(jacobian, state, time) {
                stats.jacobian_evaluations += 1;
                checked(jacobian)?;
            } else {
                finite_difference(problem, derivative, jacobian, state, time, stats)?;
            }
            factors[0].copy_from_slice(jacobian);
            Ok(())
        };
        drive_integration(problem, options, AmfKernel::new(n, 1, evaluate))
    }
}

/// Solves a structured AMF problem with the Rosenbrock23 W tableau.
pub fn solve_amf<F, J, S, P>(
    problem: &AmfProblem<F, J, S, P>,
    algorithm: AMF<Rosenbrock23>,
    options: &SolveOptions,
) -> Result<Solution, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
    J: Fn(&mut [f64], &[f64], &P, f64),
    S: Fn(&mut [Vec<f64>], &[f64], &P, f64),
{
    let _ = algorithm;
    let dummy = OdeProblem::new(
        noop as fn(&mut [f64], &[f64], &(), f64),
        problem.initial_state.clone(),
        problem.time_span,
        (),
    );
    let n = problem.initial_state.len();
    let evaluate = |derivative: &mut [f64],
                    jacobian: &mut [f64],
                    factors: &mut [Vec<f64>],
                    state: &[f64],
                    time: f64,
                    stats: &mut SolverStats| {
        (problem.function.rhs)(derivative, state, &problem.parameters, time);
        stats.rhs_evaluations += 1;
        checked(derivative)?;
        (problem.function.jacobian)(jacobian, state, &problem.parameters, time);
        stats.jacobian_evaluations += 1;
        checked(jacobian)?;
        (problem.function.update_factors)(factors, state, &problem.parameters, time);
        for factor in factors.iter() {
            checked(factor)?;
        }
        Ok(())
    };
    drive_integration(
        &dummy,
        options,
        AmfKernel::new(n, problem.factor_count(), evaluate),
    )
}

fn noop(_: &mut [f64], _: &[f64], _: &(), _: f64) {}

struct AmfKernel<E> {
    n: usize,
    evaluate: E,
    derivative: Vec<f64>,
    candidate_derivative: Vec<f64>,
    time_derivative: Vec<f64>,
    midpoint_state: Vec<f64>,
    midpoint_derivative: Vec<f64>,
    jacobian: Vec<f64>,
    operator: AmfOperator,
    k1: Vec<f64>,
    k2: Vec<f64>,
    k3: Vec<f64>,
    rhs: Vec<f64>,
    candidate_valid: bool,
    _marker: PhantomData<fn()>,
}

impl<E> AmfKernel<E> {
    fn new(n: usize, factor_count: usize, evaluate: E) -> Self {
        Self {
            n,
            evaluate,
            derivative: vec![0.0; n],
            candidate_derivative: vec![0.0; n],
            time_derivative: vec![0.0; n],
            midpoint_state: vec![0.0; n],
            midpoint_derivative: vec![0.0; n],
            jacobian: vec![0.0; n * n],
            operator: AmfOperator {
                dimension: n,
                jacobian_factors: vec![vec![0.0; n * n]; factor_count],
                factorizations: vec![vec![0.0; n * n]; factor_count],
                pivots: vec![vec![0; n]; factor_count],
            },
            k1: vec![0.0; n],
            k2: vec![0.0; n],
            k3: vec![0.0; n],
            rhs: vec![0.0; n],
            candidate_valid: false,
            _marker: PhantomData,
        }
    }
}

impl<F, P, E> StepKernel<F, P> for AmfKernel<E>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
    E: FnMut(
        &mut [f64],
        &mut [f64],
        &mut [Vec<f64>],
        &[f64],
        f64,
        &mut SolverStats,
    ) -> Result<(), SolveError>,
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::with_controller(
            true,
            ControllerConfig::proportional(3, 0.9, 0.2, 6.0, 0.2),
        )
    }
    fn evaluate_dense_derivative(
        &mut self,
        _: &OdeProblem<F, P>,
        output: &mut [f64],
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        let mut jacobian = vec![0.0; self.n * self.n];
        let mut factors = vec![vec![0.0; self.n * self.n]; self.operator.factor_count()];
        (self.evaluate)(output, &mut jacobian, &mut factors, state, time, stats)
    }
    fn initialize(
        &mut self,
        _: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        (self.evaluate)(
            &mut self.derivative,
            &mut self.jacobian,
            self.operator.factors_mut(),
            state,
            time,
            stats,
        )
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
            .zip(&self.derivative)
            .map(|(state, derivative)| {
                derivative.abs()
                    / (options.absolute_tolerance + options.relative_tolerance * state.abs())
            })
            .fold(0.0_f64, f64::max);
        Ok((if scale == 0.0 { 1e-3 } else { 0.01 / scale }).clamp(f64::EPSILON, maximum_step))
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
        (self.evaluate)(
            &mut self.derivative,
            &mut self.jacobian,
            self.operator.factors_mut(),
            state,
            time,
            stats,
        )?;
        let epsilon = f64::EPSILON.sqrt() * time.abs().max(1.0);
        let mut shifted_derivative = vec![0.0; self.n];
        let mut shifted_jac = vec![0.0; self.n * self.n];
        let mut shifted_factors = vec![vec![0.0; self.n * self.n]; self.operator.factor_count()];
        (self.evaluate)(
            &mut shifted_derivative,
            &mut shifted_jac,
            &mut shifted_factors,
            state,
            time + epsilon,
            stats,
        )?;
        for ((time_derivative, shifted), derivative) in self
            .time_derivative
            .iter_mut()
            .zip(&shifted_derivative)
            .zip(&self.derivative)
        {
            *time_derivative = (shifted - derivative) / epsilon;
        }
        self.operator.factorize(GAMMA * step)?;
        stats.linear_factorizations += self.operator.factor_count();
        for k in 0..self.n {
            self.rhs[k] = self.derivative[k] + GAMMA * step * self.time_derivative[k];
        }
        self.operator.solve_ordered(&mut self.rhs);
        stats.linear_solves += self.operator.factor_count();
        self.k1.copy_from_slice(&self.rhs);
        for ((midpoint, state), stage) in self.midpoint_state.iter_mut().zip(state).zip(&self.k1) {
            *midpoint = state + 0.5 * step * stage;
        }
        let mut throw_jac = vec![0.0; self.n * self.n];
        let mut throw_factors = vec![vec![0.0; self.n * self.n]; self.operator.factor_count()];
        (self.evaluate)(
            &mut self.midpoint_derivative,
            &mut throw_jac,
            &mut throw_factors,
            &self.midpoint_state,
            time + 0.5 * step,
            stats,
        )?;
        for k in 0..self.n {
            self.rhs[k] = self.midpoint_derivative[k] - self.k1[k];
        }
        self.operator.solve_ordered(&mut self.rhs);
        stats.linear_solves += self.operator.factor_count();
        for k in 0..self.n {
            self.k2[k] = self.rhs[k] + self.k1[k];
            candidate[k] = state[k] + step * self.k2[k];
        }
        if !options.adaptive {
            self.candidate_valid = false;
            return Ok(StepEstimate::new(0.0));
        }
        (self.evaluate)(
            &mut self.candidate_derivative,
            &mut throw_jac,
            &mut throw_factors,
            candidate,
            time + step,
            stats,
        )?;
        for k in 0..self.n {
            self.rhs[k] = self.candidate_derivative[k]
                - C32 * (self.k2[k] - self.midpoint_derivative[k])
                - 2.0 * (self.k1[k] - self.derivative[k])
                + step * self.time_derivative[k];
        }
        self.operator.solve_ordered(&mut self.rhs);
        stats.linear_solves += self.operator.factor_count();
        self.k3.copy_from_slice(&self.rhs);
        self.candidate_valid = true;
        let mut error = vec![0.0; self.n];
        for (((error, first), second), third) in
            error.iter_mut().zip(&self.k1).zip(&self.k2).zip(&self.k3)
        {
            *error = step / 6.0 * (first - 2.0 * second + third);
        }
        Ok(StepEstimate::new(scaled_error(
            &error, state, candidate, options,
        )))
    }
    fn accept_step(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        state: &[f64],
        time: f64,
        _: f64,
        callback_applied: bool,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        if self.candidate_valid && !callback_applied {
            std::mem::swap(&mut self.derivative, &mut self.candidate_derivative);
            Ok(())
        } else {
            (self.evaluate)(
                &mut self.derivative,
                &mut self.jacobian,
                self.operator.factors_mut(),
                state,
                time,
                stats,
            )
        }
    }
    fn reject_step(&mut self) {}
}

fn finite_difference<F, P>(
    problem: &OdeProblem<F, P>,
    base: &[f64],
    jacobian: &mut [f64],
    state: &[f64],
    time: f64,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let n = state.len();
    let mut shifted = state.to_vec();
    let mut derivative = vec![0.0; n];
    for column in 0..n {
        let delta = f64::EPSILON.sqrt() * state[column].abs().max(1.0);
        shifted[column] += delta;
        (problem.rhs)(&mut derivative, &shifted, problem.parameters(), time);
        stats.rhs_evaluations += 1;
        for row in 0..n {
            jacobian[row * n + column] = (derivative[row] - base[row]) / delta;
        }
        shifted[column] = state[column];
    }
    stats.jacobian_evaluations += 1;
    checked(jacobian)
}
fn checked(values: &[f64]) -> Result<(), SolveError> {
    values
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteDerivative)
}
fn scaled_error(error: &[f64], old: &[f64], new: &[f64], options: &SolveOptions) -> f64 {
    (error
        .iter()
        .zip(old)
        .zip(new)
        .map(|((error, old), new)| {
            let scale =
                options.absolute_tolerance + options.relative_tolerance * old.abs().max(new.abs());
            (error / scale).powi(2)
        })
        .sum::<f64>()
        / error.len() as f64)
        .sqrt()
}
