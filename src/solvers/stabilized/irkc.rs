use crate::callback::CallbackOutcome;
use crate::integrator::{
    ControllerConfig, KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::linear::{factorize, solve_factorized};
use crate::solver::{
    validate_preset_time_sequences, validate_state_time_options, validate_vector_callback_lengths,
};
use crate::solvers::explicit::split_euler::SplitOdeAlgorithm;
use crate::{OdeProblem, Solution, SolveError, SolveOptions, SolverStats, SplitOdeProblem};

const MINIMUM_DEGREE: usize = 50;
const MAXIMUM_DEGREE: usize = 50;
const NEWTON_ITERATIONS: usize = 12;
const NEWTON_TOLERANCE: f64 = 1.0e-11;

/// Stabilized implicit Runge--Kutta--Chebyshev split method.
///
/// The first (Rust `implicit`) split is solved by Newton iteration. The second
/// (Rust `explicit`) split determines the Chebyshev stability degree.
/// The pinned upstream policy fixes that degree at 50; supplying an eigenvalue
/// estimate skips the power-iteration setup work but does not change the stage
/// count.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IRKC {
    eigenvalue_override: Option<f64>,
}

impl IRKC {
    /// Constructs IRKC with an internally estimated explicit spectral radius.
    pub fn new() -> Self {
        Self::default()
    }
    /// Supplies an explicit spectral-radius upper bound and skips estimation.
    ///
    /// For parity with the pinned upstream revision, IRKC still uses exactly 50
    /// Chebyshev stages. The override reduces setup evaluations rather than
    /// selecting a different stage count.
    #[must_use]
    pub fn with_eigenvalue_estimate(mut self, upper_bound: f64) -> Self {
        self.eigenvalue_override = Some(upper_bound);
        self
    }
    /// Returns the classical order of the method.
    pub fn order(&self) -> usize {
        2
    }
}

impl SplitOdeAlgorithm for IRKC {
    fn solve_validated<FE, FI, P>(
        &self,
        problem: &SplitOdeProblem<FE, FI, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        FE: Fn(&mut [f64], &[f64], &P, f64),
        FI: Fn(&mut [f64], &[f64], &P, f64),
    {
        solve_irkc(problem, *self, options)
    }
}

/// Solves a split problem with IRKC while retaining the implicit/explicit roles.
pub fn solve_irkc<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    algorithm: IRKC,
    options: &SolveOptions,
) -> Result<Solution, SolveError>
where
    FE: Fn(&mut [f64], &[f64], &P, f64),
    FI: Fn(&mut [f64], &[f64], &P, f64),
{
    validate_state_time_options(problem.initial_state(), problem.time_span(), options)?;
    validate_preset_time_sequences(problem.preset_time_sequences(), problem.time_span())?;
    validate_vector_callback_lengths(problem.vector_callback_lengths())?;
    if algorithm
        .eigenvalue_override
        .is_some_and(|value| !value.is_finite() || value <= 0.0)
    {
        return Err(SolveError::InvalidTolerance);
    }
    let dummy = OdeProblem::new(
        noop as fn(&mut [f64], &[f64], &(), f64),
        problem.initial_state().to_vec(),
        problem.time_span(),
        (),
    );
    drive_integration(&dummy, options, IrkcKernel::new(problem, algorithm))
}

fn noop(_: &mut [f64], _: &[f64], _: &(), _: f64) {}

struct IrkcKernel<'a, FE, FI, P> {
    problem: &'a SplitOdeProblem<FE, FI, P>,
    algorithm: IRKC,
    implicit_start: Vec<f64>,
    explicit_start: Vec<f64>,
    total_start: Vec<f64>,
    eigenvector: Vec<f64>,
}

impl<'a, FE, FI, P> IrkcKernel<'a, FE, FI, P> {
    fn new(problem: &'a SplitOdeProblem<FE, FI, P>, algorithm: IRKC) -> Self {
        let n = problem.dimension();
        Self {
            problem,
            algorithm,
            implicit_start: vec![0.0; n],
            explicit_start: vec![0.0; n],
            total_start: vec![0.0; n],
            eigenvector: vec![1.0; n],
        }
    }
}

impl<FE, FI, P> StepKernel<fn(&mut [f64], &[f64], &(), f64), ()> for IrkcKernel<'_, FE, FI, P>
where
    FE: Fn(&mut [f64], &[f64], &P, f64),
    FI: Fn(&mut [f64], &[f64], &P, f64),
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::with_controller(
            true,
            ControllerConfig::proportional(2, 0.8, 0.2, 5.0, 0.2),
        )
        .recover_nonlinear_and_singular_failures()
    }

    fn has_callbacks(&self, _: &OdeProblem<fn(&mut [f64], &[f64], &(), f64), ()>) -> bool {
        self.problem.has_callbacks()
    }

    fn next_callback_time_stop(
        &self,
        _: &OdeProblem<fn(&mut [f64], &[f64], &(), f64), ()>,
        time: f64,
        direction: f64,
    ) -> Option<f64> {
        self.problem.next_preset_time(time, direction)
    }

    fn apply_initial_callbacks(
        &mut self,
        _: &OdeProblem<fn(&mut [f64], &[f64], &(), f64), ()>,
        state: &mut [f64],
        time: f64,
    ) -> Result<CallbackOutcome, SolveError> {
        self.problem.apply_initial_callbacks(state, time)
    }

    fn apply_finalize_callbacks(
        &mut self,
        _: &OdeProblem<fn(&mut [f64], &[f64], &(), f64), ()>,
        state: &mut [f64],
        time: f64,
    ) -> Result<bool, SolveError> {
        self.problem.apply_finalize_callbacks(state, time)
    }

    fn domain_rejection_factor(
        &self,
        _: &OdeProblem<fn(&mut [f64], &[f64], &(), f64), ()>,
        state: &[f64],
        time: f64,
    ) -> Option<f64> {
        self.problem.domain_rejection_factor(state, time)
    }

    fn has_custom_callback_handling(&self) -> bool {
        true
    }

    fn apply_step_callbacks(
        &mut self,
        _: &OdeProblem<fn(&mut [f64], &[f64], &(), f64), ()>,
        previous_state: &[f64],
        previous_time: f64,
        state: &mut [f64],
        time: &mut f64,
        state_before_effect: &mut [f64],
        event_tolerance: f64,
        _: &mut SolverStats,
    ) -> Result<CallbackOutcome, SolveError> {
        self.problem.apply_step_callbacks(
            previous_state,
            previous_time,
            state,
            time,
            state_before_effect,
            event_tolerance,
            None,
        )
    }
    fn evaluate_dense_derivative(
        &mut self,
        _: &OdeProblem<fn(&mut [f64], &[f64], &(), f64), ()>,
        output: &mut [f64],
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        let mut implicit = vec![0.0; state.len()];
        let mut explicit = vec![0.0; state.len()];
        evaluate_parts(
            self.problem,
            state,
            time,
            &mut implicit,
            &mut explicit,
            stats,
        )?;
        for ((output, implicit), explicit) in output.iter_mut().zip(implicit).zip(explicit) {
            *output = implicit + explicit;
        }
        Ok(())
    }
    fn initialize(
        &mut self,
        _: &OdeProblem<fn(&mut [f64], &[f64], &(), f64), ()>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        evaluate_parts(
            self.problem,
            state,
            time,
            &mut self.implicit_start,
            &mut self.explicit_start,
            stats,
        )?;
        for i in 0..state.len() {
            self.total_start[i] = self.implicit_start[i] + self.explicit_start[i];
        }
        Ok(())
    }
    fn estimate_initial_step(
        &mut self,
        _: &OdeProblem<fn(&mut [f64], &[f64], &(), f64), ()>,
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
            .zip(&self.total_start)
            .map(|(state, derivative)| {
                derivative.abs()
                    / (options.absolute_tolerance + options.relative_tolerance * state.abs())
            })
            .fold(0.0_f64, f64::max);
        Ok((if scale == 0.0 { 1.0e-3 } else { 0.01 / scale }).clamp(f64::EPSILON, maximum_step))
    }
    fn attempt_step(
        &mut self,
        _: &OdeProblem<fn(&mut [f64], &[f64], &(), f64), ()>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        options: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError> {
        let eigenvalue = match self.algorithm.eigenvalue_override {
            Some(value) => value,
            None => estimate_eigenvalue(
                self.problem,
                state,
                time,
                &self.explicit_start,
                &mut self.eigenvector,
                stats,
            )?,
        };
        let degree = (1 + (1.54 * step.abs() * eigenvalue + 1.0).sqrt().floor() as usize)
            .clamp(MINIMUM_DEGREE, MAXIMUM_DEGREE);
        let omega0 = 1.0 + 2.0 / (13.0 * (degree * degree) as f64);
        let temp1 = omega0 * omega0 - 1.0;
        let temp2 = temp1.sqrt();
        let theta = degree as f64 * (omega0 + temp2).ln();
        let omega1 =
            theta.sinh() * temp1 / (theta.cosh() * degree as f64 * temp2 - omega0 * theta.sinh());
        let mut bjm2 = 1.0 / (4.0 * omega0 * omega0);
        let mut bjm1 = 1.0 / omega0;
        let mu1 = omega1 * bjm1;

        let mut previous2 = state.to_vec();
        let mut tmp = state
            .iter()
            .zip(&self.explicit_start)
            .map(|(state, explicit)| state + step * mu1 * explicit)
            .collect::<Vec<_>>();
        let mut z = self
            .implicit_start
            .iter()
            .map(|value| step * value)
            .collect::<Vec<_>>();
        let mut previous =
            solve_implicit_stage(self.problem, &tmp, &z, mu1, time + mu1 * step, step, stats)?;
        z = previous
            .iter()
            .zip(&tmp)
            .map(|(state, tmp)| (state - tmp) / mu1)
            .collect();

        let mut cjm2 = 0.0;
        let mut cjm1 = mu1;
        let mut tjm1 = omega0;
        let mut tjm2 = 1.0;
        let mut tpjm1 = 1.0;
        let mut tpjm2 = 0.0;
        let mut tppjm1 = 0.0;
        let mut tppjm2 = 0.0;
        let mut implicit_previous2 = self.implicit_start.clone();
        let mut implicit_previous = vec![0.0; state.len()];
        let mut explicit_previous = vec![0.0; state.len()];
        for iteration in 2..=degree {
            let tj = 2.0 * omega0 * tjm1 - tjm2;
            let tpj = 2.0 * omega0 * tpjm1 + 2.0 * tjm1 - tpjm2;
            let tppj = 2.0 * omega0 * tppjm1 + 4.0 * tpjm1 - tppjm2;
            let bj = tppj / (tpj * tpj);
            let mu = 2.0 * omega0 * bj / bjm1;
            let nu = -bj / bjm2;
            let mus = mu * omega1 / omega0;
            let nus = -(1.0 - tjm1 * bjm1) * mus;
            let cj = mu * cjm1 + nu * cjm2 + mus + nus;
            evaluate_parts(
                self.problem,
                &previous,
                time + cjm1 * step,
                &mut implicit_previous,
                &mut explicit_previous,
                stats,
            )?;
            tmp.resize(state.len(), 0.0);
            for k in 0..state.len() {
                tmp[k] = (1.0 - mu - nu) * state[k]
                    + mu * previous[k]
                    + nu * previous2[k]
                    + step * mus * explicit_previous[k]
                    + step * nus * self.explicit_start[k]
                    + (nus - (1.0 - mu - nu) * mu1) * step * self.implicit_start[k]
                    - nu * mu1 * step * implicit_previous2[k];
                z[k] = step * implicit_previous[k];
            }
            let next =
                solve_implicit_stage(self.problem, &tmp, &z, mu1, time + cj * step, step, stats)?;
            if iteration < degree {
                implicit_previous2.copy_from_slice(&implicit_previous);
                previous2 = previous;
                previous = next;
                cjm2 = cjm1;
                cjm1 = cj;
                bjm2 = bjm1;
                bjm1 = bj;
                tjm2 = tjm1;
                tjm1 = tj;
                tpjm2 = tpjm1;
                tpjm1 = tpj;
                tppjm2 = tppjm1;
                tppjm1 = tppj;
            } else {
                candidate.copy_from_slice(&next);
            }
        }
        let mut implicit_end = vec![0.0; state.len()];
        let mut explicit_end = vec![0.0; state.len()];
        evaluate_parts(
            self.problem,
            candidate,
            time + step,
            &mut implicit_end,
            &mut explicit_end,
            stats,
        )?;
        let error = if options.adaptive {
            let mut raw = vec![0.0; state.len()];
            for k in 0..state.len() {
                raw[k] = step
                    * (0.5 * (explicit_end[k] - self.explicit_start[k])
                        + (0.5 - mu1) * (implicit_end[k] - self.implicit_start[k]));
            }
            // Pinned estimator applies the same first-stage Newton W solve.
            solve_error_system(
                self.problem,
                candidate,
                time + step,
                step * mu1,
                &mut raw,
                stats,
            )?;
            scaled_error(&raw, state, candidate, options)
        } else {
            0.0
        };
        Ok(StepEstimate::new(error))
    }
    fn accept_step(
        &mut self,
        _: &OdeProblem<fn(&mut [f64], &[f64], &(), f64), ()>,
        _: &[f64],
        state: &[f64],
        time: f64,
        _: f64,
        _: bool,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        evaluate_parts(
            self.problem,
            state,
            time,
            &mut self.implicit_start,
            &mut self.explicit_start,
            stats,
        )?;
        for i in 0..state.len() {
            self.total_start[i] = self.implicit_start[i] + self.explicit_start[i];
        }
        Ok(())
    }
    fn reject_step(&mut self) {}
}

fn evaluate_parts<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    state: &[f64],
    time: f64,
    implicit: &mut [f64],
    explicit: &mut [f64],
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    FE: Fn(&mut [f64], &[f64], &P, f64),
    FI: Fn(&mut [f64], &[f64], &P, f64),
{
    problem.evaluate_implicit(implicit, state, time);
    problem.evaluate_explicit(explicit, state, time);
    stats.rhs_evaluations += 2;
    checked(implicit)?;
    checked(explicit)
}

fn solve_implicit_stage<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    base: &[f64],
    initial_z: &[f64],
    gamma: f64,
    time: f64,
    step: f64,
    stats: &mut SolverStats,
) -> Result<Vec<f64>, SolveError>
where
    FE: Fn(&mut [f64], &[f64], &P, f64),
    FI: Fn(&mut [f64], &[f64], &P, f64),
{
    let n = base.len();
    let mut z = initial_z.to_vec();
    let mut state = vec![0.0; n];
    let mut derivative = vec![0.0; n];
    let mut residual = vec![0.0; n];
    for _ in 0..NEWTON_ITERATIONS {
        for k in 0..n {
            state[k] = base[k] + gamma * z[k];
        }
        problem.evaluate_implicit(&mut derivative, &state, time);
        stats.rhs_evaluations += 1;
        checked(&derivative)?;
        for k in 0..n {
            residual[k] = z[k] - step * derivative[k];
        }
        if norm(&residual) <= NEWTON_TOLERANCE * (1.0 + norm(&z)) {
            return Ok(state);
        }
        let jacobian = implicit_jacobian(problem, &state, time, &derivative, stats)?;
        let mut matrix = vec![0.0; n * n];
        for row in 0..n {
            for column in 0..n {
                matrix[row * n + column] =
                    f64::from(row == column) - step * gamma * jacobian[row * n + column];
            }
        }
        let mut pivots = vec![0; n];
        factorize(&mut matrix, &mut pivots, n)?;
        stats.linear_factorizations += 1;
        solve_factorized(&matrix, &pivots, &mut residual, n);
        stats.linear_solves += 1;
        stats.nonlinear_iterations += 1;
        for k in 0..n {
            z[k] -= residual[k];
        }
    }
    Err(SolveError::NonlinearSolveFailed)
}

fn solve_error_system<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    state: &[f64],
    time: f64,
    scale: f64,
    rhs: &mut [f64],
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    FE: Fn(&mut [f64], &[f64], &P, f64),
    FI: Fn(&mut [f64], &[f64], &P, f64),
{
    let mut derivative = vec![0.0; state.len()];
    problem.evaluate_implicit(&mut derivative, state, time);
    stats.rhs_evaluations += 1;
    let jacobian = implicit_jacobian(problem, state, time, &derivative, stats)?;
    let n = state.len();
    let mut matrix = vec![0.0; n * n];
    for row in 0..n {
        for column in 0..n {
            matrix[row * n + column] =
                f64::from(row == column) - scale * jacobian[row * n + column];
        }
    }
    let mut pivots = vec![0; n];
    factorize(&mut matrix, &mut pivots, n)?;
    solve_factorized(&matrix, &pivots, rhs, n);
    stats.linear_factorizations += 1;
    stats.linear_solves += 1;
    Ok(())
}

fn implicit_jacobian<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    state: &[f64],
    time: f64,
    base: &[f64],
    stats: &mut SolverStats,
) -> Result<Vec<f64>, SolveError>
where
    FE: Fn(&mut [f64], &[f64], &P, f64),
    FI: Fn(&mut [f64], &[f64], &P, f64),
{
    let n = state.len();
    let mut jacobian = vec![0.0; n * n];
    if !problem.evaluate_implicit_jacobian(&mut jacobian, state, time) {
        let mut shifted = state.to_vec();
        let mut derivative = vec![0.0; n];
        for column in 0..n {
            let delta = f64::EPSILON.sqrt() * state[column].abs().max(1.0);
            shifted[column] += delta;
            problem.evaluate_implicit(&mut derivative, &shifted, time);
            stats.rhs_evaluations += 1;
            for row in 0..n {
                jacobian[row * n + column] = (derivative[row] - base[row]) / delta;
            }
            shifted[column] = state[column];
        }
    }
    stats.jacobian_evaluations += 1;
    checked(&jacobian)?;
    Ok(jacobian)
}

fn estimate_eigenvalue<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    state: &[f64],
    time: f64,
    base: &[f64],
    vector: &mut [f64],
    stats: &mut SolverStats,
) -> Result<f64, SolveError>
where
    FE: Fn(&mut [f64], &[f64], &P, f64),
    FI: Fn(&mut [f64], &[f64], &P, f64),
{
    let n = state.len();
    let radius = norm(state).max(1.0) * f64::EPSILON.sqrt();
    let mut shifted = vec![0.0; n];
    let mut derivative = vec![0.0; n];
    let mut estimate = 0.0;
    let initial_norm = norm(vector);
    if initial_norm == 0.0 {
        vector.fill(1.0);
    }
    for _ in 0..50 {
        let vector_norm = norm(vector).max(f64::EPSILON);
        for k in 0..n {
            shifted[k] = state[k] + radius * vector[k] / vector_norm;
        }
        problem.evaluate_explicit(&mut derivative, &shifted, time);
        stats.rhs_evaluations += 1;
        for k in 0..n {
            vector[k] = derivative[k] - base[k];
        }
        let next = 1.2 * norm(vector) / radius;
        if (next - estimate).abs() < next.max(1.0) * 0.01 {
            return Ok(next);
        }
        estimate = next;
    }
    Ok(estimate.max(f64::EPSILON))
}

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
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
