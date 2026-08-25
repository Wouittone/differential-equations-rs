//! Fixed-step implicit-explicit linear multistep methods.
//!
//! The formulas follow the regular split-ODE paths in OrdinaryDiffEqBDF and
//! OrdinaryDiffEqIMEXMultistep at SciML/OrdinaryDiffEq.jl revision
//! `211142263781255a9aa2f910f6760b9f18ec29c8`. The first split component in
//! this crate is explicit and the second is implicit, matching
//! [`SplitOdeProblem`](crate::SplitOdeProblem). Residual DAEs, mass matrices,
//! custom nonlinear solvers, and custom linear solvers are not represented.

use crate::linear::{factorize, solve_factorized};
use crate::solution::{BorrowedHermiteSegment, HermiteSegment, TrajectoryRecorder};
use crate::{Solution, SolveError, SolveOptions, SolverStats, SplitOdeAlgorithm, SplitOdeProblem};

const MAX_NEWTON_ITERATIONS: usize = 12;
const NEWTON_TOLERANCE: f64 = 1.0e-12;

/// Configured semi-implicit BDF method of order one through four.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Sbdf {
    order: usize,
    ark: bool,
}

impl Sbdf {
    /// Creates an SBDF method of the requested order (one through four).
    ///
    /// An unsupported order is reported as [`SolveError::InvalidMultistepOrder`]
    /// when the method is used.
    pub const fn new(order: usize) -> Self {
        Self { order, ark: false }
    }

    const fn ark() -> Self {
        Self {
            order: 1,
            ark: true,
        }
    }

    /// Returns the configured BDF order.
    pub const fn order(self) -> usize {
        self.order
    }
}

/// Exact Julia-compatible spelling alias for [`Sbdf`].
pub type SBDF = Sbdf;

macro_rules! fixed_algorithm {
    ($rust:ident, $julia:ident, $documentation:literal, $method:expr) => {
        #[doc = $documentation]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        pub struct $rust;

        #[doc = concat!("Exact Julia-compatible spelling alias for [`", stringify!($rust), "`].")]
        pub type $julia = $rust;

        #[allow(non_upper_case_globals)]
        pub const $julia: $rust = $rust;

        impl SplitOdeAlgorithm for $rust {
            fn solve<FE, FI, P>(
                &self,
                problem: &SplitOdeProblem<FE, FI, P>,
                options: &SolveOptions,
            ) -> Result<Solution, SolveError>
            where
                FE: Fn(&mut [f64], &[f64], &P, f64),
                FI: Fn(&mut [f64], &[f64], &P, f64),
            {
                integrate_split(problem, options, $method)
            }
        }
    };
}

fixed_algorithm!(
    ImexEuler,
    IMEXEuler,
    "First-order implicit-explicit Euler (`SBDF(1)`).",
    SplitMethod::Sbdf(Sbdf::new(1))
);
fixed_algorithm!(
    ImexEulerArk,
    IMEXEulerARK,
    "First-order additive Runge--Kutta IMEX Euler (`SBDF(1)` with ARK staging).",
    SplitMethod::Sbdf(Sbdf::ark())
);
fixed_algorithm!(
    Sbdf2,
    SBDF2,
    "The second-order semi-implicit BDF method.",
    SplitMethod::Sbdf(Sbdf::new(2))
);
fixed_algorithm!(
    Sbdf3,
    SBDF3,
    "The third-order semi-implicit BDF method.",
    SplitMethod::Sbdf(Sbdf::new(3))
);
fixed_algorithm!(
    Sbdf4,
    SBDF4,
    "The fourth-order semi-implicit BDF method.",
    SplitMethod::Sbdf(Sbdf::new(4))
);
fixed_algorithm!(
    Cnab2,
    CNAB2,
    "The fixed-step second-order Crank--Nicolson--Adams--Bashforth method.",
    SplitMethod::Cnab2
);
fixed_algorithm!(
    Cnlf2,
    CNLF2,
    "The fixed-step second-order Crank--Nicolson--leapfrog method.",
    SplitMethod::Cnlf2
);

impl SplitOdeAlgorithm for Sbdf {
    fn solve<FE, FI, P>(
        &self,
        problem: &SplitOdeProblem<FE, FI, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        FE: Fn(&mut [f64], &[f64], &P, f64),
        FI: Fn(&mut [f64], &[f64], &P, f64),
    {
        integrate_split(problem, options, SplitMethod::Sbdf(*self))
    }
}

#[derive(Clone, Copy)]
enum SplitMethod {
    Sbdf(Sbdf),
    Cnab2,
    Cnlf2,
}

struct Workspace {
    explicit: Vec<f64>,
    implicit: Vec<f64>,
    next_explicit: Vec<f64>,
    next_implicit: Vec<f64>,
    explicit_history: Vec<Vec<f64>>,
    implicit_history: Vec<Vec<f64>>,
    state_history: Vec<Vec<f64>>,
    history_len: usize,
    forcing: Vec<f64>,
    candidate: Vec<f64>,
    stage_state: Vec<f64>,
    matrix: Vec<f64>,
    pivots: Vec<usize>,
    residual: Vec<f64>,
    correction: Vec<f64>,
    perturbed_state: Vec<f64>,
    perturbed_derivative: Vec<f64>,
}

impl Workspace {
    fn new(dimension: usize) -> Self {
        Self {
            explicit: vec![0.0; dimension],
            implicit: vec![0.0; dimension],
            next_explicit: vec![0.0; dimension],
            next_implicit: vec![0.0; dimension],
            explicit_history: (0..3).map(|_| vec![0.0; dimension]).collect(),
            implicit_history: vec![vec![0.0; dimension]],
            state_history: (0..3).map(|_| vec![0.0; dimension]).collect(),
            history_len: 0,
            forcing: vec![0.0; dimension],
            candidate: vec![0.0; dimension],
            stage_state: vec![0.0; dimension],
            matrix: vec![0.0; dimension * dimension],
            pivots: vec![0; dimension],
            residual: vec![0.0; dimension],
            correction: vec![0.0; dimension],
            perturbed_state: vec![0.0; dimension],
            perturbed_derivative: vec![0.0; dimension],
        }
    }

    fn clear_history(&mut self) {
        self.history_len = 0;
    }

    fn accept(&mut self, previous_state: &[f64]) {
        self.state_history.rotate_right(1);
        self.explicit_history.rotate_right(1);
        self.state_history[0].copy_from_slice(previous_state);
        self.explicit_history[0].copy_from_slice(&self.explicit);
        self.implicit_history[0].copy_from_slice(&self.implicit);
        self.history_len = (self.history_len + 1).min(3);
        std::mem::swap(&mut self.explicit, &mut self.next_explicit);
        std::mem::swap(&mut self.implicit, &mut self.next_implicit);
    }
}

fn integrate_split<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    options: &SolveOptions,
    method: SplitMethod,
) -> Result<Solution, SolveError>
where
    FE: Fn(&mut [f64], &[f64], &P, f64),
    FI: Fn(&mut [f64], &[f64], &P, f64),
{
    validate(problem, options)?;
    let SplitMethod::Sbdf(sbdf) = method else {
        return integrate_validated(problem, options, method);
    };
    if !(1..=4).contains(&sbdf.order) {
        return Err(SolveError::InvalidMultistepOrder);
    }
    integrate_validated(problem, options, method)
}

fn integrate_validated<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    options: &SolveOptions,
    method: SplitMethod,
) -> Result<Solution, SolveError>
where
    FE: Fn(&mut [f64], &[f64], &P, f64),
    FI: Fn(&mut [f64], &[f64], &P, f64),
{
    let dimension = problem.dimension();
    let (start, end) = problem.time_span();
    let direction = (end - start).signum();
    let nominal_step = direction
        * options
            .initial_step
            .expect("validated fixed-step solve has an initial step")
            .min(options.max_step)
            .min((end - start).abs());
    let mut step = nominal_step;
    let mut state = problem.initial_state().to_vec();
    let mut workspace = Workspace::new(dimension);
    let mut stats = SolverStats::default();
    evaluate_explicit(problem, &mut workspace.explicit, &state, start, &mut stats)?;
    evaluate_implicit(problem, &mut workspace.implicit, &state, start, &mut stats)?;
    let mut recorder = TrajectoryRecorder::new(&state, start, options);
    let mut start_derivative = vec![0.0; dimension];
    let mut end_derivative = vec![0.0; dimension];
    let mut time = start;
    let mut attempted_steps = 0;

    while direction * (end - time) > 0.0 {
        if attempted_steps == options.max_steps {
            return Err(SolveError::MaxStepsExceeded);
        }
        attempted_steps += 1;
        if direction * (time + step - end) > 0.0 {
            step = end - time;
        }
        if time + step == time {
            return Err(SolveError::StepSizeUnderflow);
        }
        if relative_step_change(step, nominal_step) > 1.0e-12 {
            workspace.clear_history();
        }

        let next_time = time + step;
        prepare_forcing(
            method,
            &state,
            time,
            step,
            problem,
            &mut workspace,
            &mut stats,
        )?;
        let implicit_scale = implicit_scale(method, &workspace);
        workspace.candidate.copy_from_slice(&workspace.forcing);
        for (candidate, derivative) in workspace.candidate.iter_mut().zip(&workspace.implicit) {
            *candidate += implicit_scale * step * derivative;
        }
        solve_implicit(
            problem,
            next_time,
            implicit_scale * step,
            &mut workspace,
            &mut stats,
        )?;

        let previous_state = state;
        let next_state = std::mem::take(&mut workspace.candidate);
        evaluate_explicit(
            problem,
            &mut workspace.next_explicit,
            &next_state,
            next_time,
            &mut stats,
        )?;
        for index in 0..dimension {
            start_derivative[index] = workspace.explicit[index] + workspace.implicit[index];
            end_derivative[index] = workspace.next_explicit[index] + workspace.next_implicit[index];
        }
        let segment = BorrowedHermiteSegment::new(
            time,
            next_time,
            &previous_state,
            &next_state,
            &start_derivative,
            &end_derivative,
        )
        .map_err(|_| SolveError::NonFiniteDerivative)?;
        stats.accepted_steps += 1;
        recorder
            .record_step_dense(
                &previous_state,
                time,
                &next_state,
                next_time,
                next_time == end,
                &segment,
            )
            .map_err(|_| SolveError::NonFiniteDerivative)?;
        if recorder.retains_dense_output() {
            recorder.retain_hermite_segment(
                HermiteSegment::new(
                    time,
                    next_time,
                    previous_state.clone(),
                    next_state.clone(),
                    start_derivative.clone(),
                    end_derivative.clone(),
                )
                .map_err(|_| SolveError::NonFiniteDerivative)?,
            );
        }
        workspace.accept(&previous_state);
        workspace.candidate = previous_state;
        state = next_state;
        time = next_time;
        step = nominal_step;
    }

    Ok(recorder.finish(stats))
}

#[allow(clippy::needless_range_loop)]
fn prepare_forcing<FE, FI, P>(
    method: SplitMethod,
    state: &[f64],
    time: f64,
    step: f64,
    problem: &SplitOdeProblem<FE, FI, P>,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    FE: Fn(&mut [f64], &[f64], &P, f64),
    FI: Fn(&mut [f64], &[f64], &P, f64),
{
    match method {
        SplitMethod::Sbdf(config) => {
            // The pinned mutable-cache path forces the first step to order one,
            // then uses `min(requested_order, integrator.iter + 1)`. This means
            // SBDF3 enters order three on step two and SBDF4 enters order four
            // on step three, with their zero-initialized older cache entries.
            let order = if workspace.history_len == 0 {
                1
            } else {
                config.order.min(workspace.history_len + 2)
            };
            if config.ark {
                for ((stage, value), derivative) in workspace
                    .stage_state
                    .iter_mut()
                    .zip(state)
                    .zip(&workspace.implicit)
                {
                    *stage = value + step * derivative;
                }
                evaluate_explicit(
                    problem,
                    &mut workspace.explicit,
                    &workspace.stage_state,
                    time,
                    stats,
                )?;
            }
            sbdf_forcing(order, state, step, workspace);
        }
        SplitMethod::Cnab2 => {
            if workspace.history_len > 0 {
                let previous_explicit = &workspace.explicit_history[0];
                for index in 0..state.len() {
                    workspace.forcing[index] = state[index]
                        + step
                            * (1.5 * workspace.explicit[index] - 0.5 * previous_explicit[index]
                                + 0.5 * workspace.implicit[index]);
                }
            } else {
                for index in 0..state.len() {
                    workspace.forcing[index] = state[index]
                        + step * (workspace.explicit[index] + 0.5 * workspace.implicit[index]);
                }
            }
        }
        SplitMethod::Cnlf2 => {
            if workspace.history_len > 0 {
                let previous_state = &workspace.state_history[0];
                let previous_implicit = &workspace.implicit_history[0];
                for index in 0..state.len() {
                    workspace.forcing[index] = previous_state[index]
                        + step * (2.0 * workspace.explicit[index] + previous_implicit[index]);
                }
            } else {
                for index in 0..state.len() {
                    workspace.forcing[index] = state[index] + step * workspace.explicit[index];
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::needless_range_loop)]
fn sbdf_forcing(order: usize, state: &[f64], step: f64, workspace: &mut Workspace) {
    match order {
        1 => {
            for index in 0..state.len() {
                workspace.forcing[index] = state[index] + step * workspace.explicit[index];
            }
        }
        2 => {
            let gamma = 2.0 / 3.0;
            for index in 0..state.len() {
                workspace.forcing[index] = gamma
                    * (2.0 * state[index] - 0.5 * workspace.state_history[0][index]
                        + step
                            * (2.0 * workspace.explicit[index]
                                - workspace.explicit_history[0][index]));
            }
        }
        3 => {
            let gamma = 6.0 / 11.0;
            for index in 0..state.len() {
                workspace.forcing[index] = gamma
                    * (3.0 * state[index] - 1.5 * workspace.state_history[0][index]
                        + workspace.state_history[1][index] / 3.0
                        + step
                            * (3.0
                                * (workspace.explicit[index]
                                    - workspace.explicit_history[0][index])
                                + workspace.explicit_history[1][index]));
            }
        }
        4 => {
            let gamma = 12.0 / 25.0;
            for index in 0..state.len() {
                workspace.forcing[index] = gamma
                    * (4.0 * state[index] - 3.0 * workspace.state_history[0][index]
                        + 4.0 / 3.0 * workspace.state_history[1][index]
                        - 0.25 * workspace.state_history[2][index]
                        + step
                            * (4.0 * workspace.explicit[index]
                                - 6.0 * workspace.explicit_history[0][index]
                                + 4.0 * workspace.explicit_history[1][index]
                                - workspace.explicit_history[2][index]));
            }
        }
        _ => unreachable!("SBDF order is validated"),
    }
}

fn implicit_scale(method: SplitMethod, workspace: &Workspace) -> f64 {
    match method {
        SplitMethod::Sbdf(config) => match if workspace.history_len == 0 {
            1
        } else {
            config.order.min(workspace.history_len + 2)
        } {
            1 => 1.0,
            2 => 2.0 / 3.0,
            3 => 6.0 / 11.0,
            _ => 12.0 / 25.0,
        },
        SplitMethod::Cnab2 => 0.5,
        SplitMethod::Cnlf2 => 1.0,
    }
}

fn solve_implicit<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    time: f64,
    scale: f64,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    FE: Fn(&mut [f64], &[f64], &P, f64),
    FI: Fn(&mut [f64], &[f64], &P, f64),
{
    let dimension = workspace.candidate.len();
    let mut factorization_ready = false;
    for _ in 0..MAX_NEWTON_ITERATIONS {
        stats.nonlinear_iterations += 1;
        evaluate_implicit(
            problem,
            &mut workspace.next_implicit,
            &workspace.candidate,
            time,
            stats,
        )?;
        let mut residual_norm: f64 = 0.0;
        for index in 0..dimension {
            workspace.residual[index] = workspace.candidate[index]
                - scale * workspace.next_implicit[index]
                - workspace.forcing[index];
            residual_norm = residual_norm.max(workspace.residual[index].abs());
        }
        if residual_norm <= NEWTON_TOLERANCE * (1.0 + infinity_norm(&workspace.candidate)) {
            return Ok(());
        }
        if !factorization_ready {
            build_factorization(problem, time, scale, workspace, stats)?;
            factorization_ready = true;
        }
        for (correction, residual) in workspace.correction.iter_mut().zip(&workspace.residual) {
            *correction = -*residual;
        }
        solve_factorized(
            &workspace.matrix,
            &workspace.pivots,
            &mut workspace.correction,
            dimension,
        );
        stats.linear_solves += 1;
        for (candidate, correction) in workspace.candidate.iter_mut().zip(&workspace.correction) {
            *candidate += correction;
        }
    }
    Err(SolveError::NonlinearSolveFailed)
}

fn build_factorization<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    time: f64,
    scale: f64,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    FE: Fn(&mut [f64], &[f64], &P, f64),
    FI: Fn(&mut [f64], &[f64], &P, f64),
{
    let dimension = workspace.candidate.len();
    if problem.evaluate_implicit_jacobian(&mut workspace.matrix, &workspace.candidate, time) {
        if !workspace.matrix.iter().all(|value| value.is_finite()) {
            return Err(SolveError::NonFiniteDerivative);
        }
        for row in 0..dimension {
            for column in 0..dimension {
                let index = row * dimension + column;
                workspace.matrix[index] =
                    f64::from(row == column) - scale * workspace.matrix[index];
            }
        }
    } else {
        for column in 0..dimension {
            workspace
                .perturbed_state
                .copy_from_slice(&workspace.candidate);
            let perturbation = f64::EPSILON.sqrt() * workspace.candidate[column].abs().max(1.0);
            workspace.perturbed_state[column] += perturbation;
            evaluate_implicit(
                problem,
                &mut workspace.perturbed_derivative,
                &workspace.perturbed_state,
                time,
                stats,
            )?;
            for row in 0..dimension {
                let derivative = (workspace.perturbed_derivative[row]
                    - workspace.next_implicit[row])
                    / perturbation;
                workspace.matrix[row * dimension + column] =
                    f64::from(row == column) - scale * derivative;
            }
        }
    }
    stats.jacobian_evaluations += 1;
    stats.linear_factorizations += 1;
    factorize(&mut workspace.matrix, &mut workspace.pivots, dimension)
}

fn evaluate_explicit<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    derivative: &mut [f64],
    state: &[f64],
    time: f64,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    FE: Fn(&mut [f64], &[f64], &P, f64),
{
    problem.evaluate_explicit(derivative, state, time);
    stats.rhs_evaluations += 1;
    ensure_finite(derivative)
}

fn evaluate_implicit<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    derivative: &mut [f64],
    state: &[f64],
    time: f64,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    FI: Fn(&mut [f64], &[f64], &P, f64),
{
    problem.evaluate_implicit(derivative, state, time);
    stats.rhs_evaluations += 1;
    ensure_finite(derivative)
}

fn validate<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    options: &SolveOptions,
) -> Result<(), SolveError> {
    if problem.initial_state().is_empty() {
        return Err(SolveError::EmptyState);
    }
    if !problem
        .initial_state()
        .iter()
        .all(|value| value.is_finite())
    {
        return Err(SolveError::NonFiniteInitialState);
    }
    let (start, end) = problem.time_span();
    if !start.is_finite() || !end.is_finite() || start == end {
        return Err(SolveError::InvalidTimeSpan);
    }
    if !options.absolute_tolerance.is_finite()
        || options.absolute_tolerance <= 0.0
        || !options.relative_tolerance.is_finite()
        || options.relative_tolerance <= 0.0
    {
        return Err(SolveError::InvalidTolerance);
    }
    if options.adaptive {
        return Err(SolveError::AdaptiveStepUnsupported);
    }
    if options
        .initial_step
        .is_some_and(|step| !step.is_finite() || step <= 0.0)
    {
        return Err(SolveError::InvalidInitialStep);
    }
    if options.initial_step.is_none() {
        return Err(SolveError::InitialStepRequired);
    }
    if options.max_step.is_nan() || options.max_step <= 0.0 {
        return Err(SolveError::InvalidMaxStep);
    }
    if options.max_steps == 0 {
        return Err(SolveError::InvalidMaxSteps);
    }
    if !options.event_tolerance.is_finite() || options.event_tolerance <= 0.0 {
        return Err(SolveError::InvalidEventTolerance);
    }
    let direction = (end - start).signum();
    if !options.save_at.iter().all(|time| {
        time.is_finite() && direction * (*time - start) >= 0.0 && direction * (end - *time) >= 0.0
    }) || options
        .save_at
        .windows(2)
        .any(|pair| direction * (pair[1] - pair[0]) <= 0.0)
    {
        return Err(SolveError::InvalidSaveAt);
    }
    Ok(())
}

fn ensure_finite(values: &[f64]) -> Result<(), SolveError> {
    values
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteDerivative)
}

fn infinity_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

fn relative_step_change(left: f64, right: f64) -> f64 {
    (left - right).abs() / left.abs().max(right.abs()).max(f64::MIN_POSITIVE)
}
