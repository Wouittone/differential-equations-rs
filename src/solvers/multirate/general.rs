//! Multirate infinitesimal-step and MRI-GARK methods for typed split ODEs.
//!
//! The slow component is the `implicit` half of [`SplitOdeProblem`]; the fast
//! component is its `explicit` half.  This matches `OrdinaryDiffEqMultirate`'s
//! `SplitFunction(fast, slow)` convention.

use super::tableaux::*;
use crate::integrator::{TimeStopSchedule, callback_adjusted_step};
use crate::linear::{factorize, solve_factorized};
use crate::solution::{BorrowedHermiteSegment, DenseSegment, HermiteSegment, TrajectoryRecorder};
use crate::solver::{
    validate_preset_time_sequences, validate_state_time_options, validate_vector_callback_lengths,
};
use crate::solvers::explicit::split_euler::SplitOdeAlgorithm;
use crate::solvers::multistep::tableaux::adams_bashforth;
use crate::tableau::{MisTableau, MriTableau, TableauError, load_tableau};
use crate::{Solution, SolveError, SolveOptions, SolverStats, SplitOdeProblem};

const MAX_NEWTON_ITERATIONS: usize = 12;
const NEWTON_TOLERANCE: f64 = 1.0e-11;

/// Extrapolation sequence used by [`MREEF`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MultirateSequence {
    /// `n_j = j`, matching upstream's `:harmonic` default.
    #[default]
    Harmonic,
    /// `n_j = 2^(j-1)`, matching upstream's `:romberg` option.
    Romberg,
}

/// Multirate explicit Euler extrapolation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mreef {
    m: usize,
    order: usize,
    sequence: MultirateSequence,
}

impl Mreef {
    /// Configures the microstep count, extrapolation order, and sequence.
    pub const fn new(m: usize, order: usize, sequence: MultirateSequence) -> Self {
        Self { m, order, sequence }
    }

    /// Returns the number of fast microsteps per macro step.
    pub const fn microsteps(&self) -> usize {
        self.m
    }

    /// Returns the configured extrapolation order.
    pub const fn order(&self) -> usize {
        self.order
    }
}

impl Default for Mreef {
    fn default() -> Self {
        Self::new(4, 4, MultirateSequence::Harmonic)
    }
}

/// Exact compatibility spelling from OrdinaryDiffEqMultirate.
pub type MREEF = Mreef;

/// Multirate Adams--Bashforth with frozen slow forcing per macro step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mrab {
    order: usize,
    m: usize,
}

impl Mrab {
    /// Configures the Adams--Bashforth order and fast microstep count.
    pub const fn new(order: usize, m: usize) -> Self {
        Self { order, m }
    }

    /// Returns the configured Adams--Bashforth order.
    pub const fn order(&self) -> usize {
        self.order
    }

    /// Returns the nominal-order formula shared with fixed-step Adams solvers.
    /// Startup microsteps load only the lower-order formulas they actually use.
    pub fn tableau(&self) -> Result<&'static crate::tableau::LinearMultistepTableau, SolveError> {
        adams_bashforth(self.order)
    }

    /// Returns the number of fast microsteps per macro step.
    pub const fn microsteps(&self) -> usize {
        self.m
    }
}

impl Default for Mrab {
    fn default() -> Self {
        Self::new(2, 4)
    }
}

/// Exact compatibility spelling from OrdinaryDiffEqMultirate.
pub type MRAB = Mrab;

/// Knoth--Wolke multirate infinitesimal-step method.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mis {
    m: usize,
}

impl Mis {
    /// Configures the number of fast microsteps per macro step.
    pub const fn new(m: usize) -> Self {
        Self { m }
    }

    /// Returns the number of fast microsteps per macro step.
    pub const fn microsteps(&self) -> usize {
        self.m
    }

    /// Returns the lazily parsed Knoth--Wolke coupling tableau.
    pub fn tableau(&self) -> Result<&'static MisTableau, TableauError> {
        load_tableau(&MIS_TABLEAU)
    }
}

impl Default for Mis {
    fn default() -> Self {
        Self::new(4)
    }
}

/// Exact compatibility spelling from OrdinaryDiffEqMultirate.
pub type MIS = Mis;

macro_rules! mri_algorithm {
    ($rust:ident, $exact:ident, $tableau:ident, $docs:literal) => {
        #[doc = $docs]
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub struct $rust {
            m: usize,
        }

        impl $rust {
            /// Configures the number of fast microsteps per macro step.
            pub const fn new(m: usize) -> Self {
                Self { m }
            }

            /// Returns the number of fast microsteps per macro step.
            pub const fn microsteps(&self) -> usize {
                self.m
            }

            /// Returns this method's lazily parsed MRI-GARK coupling tableau.
            pub fn tableau(&self) -> Result<&'static MriTableau, TableauError> {
                load_tableau(&$tableau)
            }
        }

        impl Default for $rust {
            fn default() -> Self {
                Self::new(4)
            }
        }

        #[allow(non_camel_case_types)]
        #[doc = concat!("Exact OrdinaryDiffEq-compatible spelling alias for [`", stringify!($rust), "`].")]
        pub type $exact = $rust;

        impl SplitOdeAlgorithm for $rust {
            fn solve_validated<FE, FI, P>(
                &self,
                problem: &SplitOdeProblem<FE, FI, P>,
                options: &SolveOptions,
            ) -> Result<Solution, SolveError>
            where
                FE: crate::OdeFunction<P>,
                FI: crate::OdeFunction<P>,
            {
                if self.m == 0 {
                    return Err(SolveError::InvalidMultistepOrder);
                }
                integrate_multirate(
                    problem,
                    options,
                    Method::Mri {
                        m: self.m,
                        tableau: self.tableau().map_err(|_| SolveError::InvalidTableau)?,
                    },
                )
            }
        }
    };
}

mri_algorithm!(
    MriGarkErk22a,
    MRIGARKERK22a,
    ERK22A_TABLEAU,
    "Explicit second-order MRI-GARK ERK22a."
);
mri_algorithm!(
    MriGarkErk22b,
    MRIGARKERK22b,
    ERK22B_TABLEAU,
    "Explicit second-order MRI-GARK ERK22b."
);
mri_algorithm!(
    MriGarkErk33a,
    MRIGARKERK33a,
    ERK33A_TABLEAU,
    "Explicit third-order MRI-GARK ERK33a."
);
mri_algorithm!(
    MriGarkErk45a,
    MRIGARKERK45a,
    ERK45A_TABLEAU,
    "Explicit fourth-order MRI-GARK ERK45a."
);
mri_algorithm!(
    MriGarkEsdirk34a,
    MRIGARKESDIRK34a,
    ESDIRK34A_TABLEAU,
    "Third-order implicit-slow MRI-GARK ESDIRK34a."
);
mri_algorithm!(
    MriGarkIrk21a,
    MRIGARKIRK21a,
    IRK21A_TABLEAU,
    "Second-order implicit-slow MRI-GARK IRK21a."
);

impl SplitOdeAlgorithm for Mreef {
    fn solve_validated<FE, FI, P>(
        &self,
        problem: &SplitOdeProblem<FE, FI, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        FE: crate::OdeFunction<P>,
        FI: crate::OdeFunction<P>,
    {
        if self.m == 0 || !(2..=10).contains(&self.order) {
            return Err(SolveError::InvalidMultistepOrder);
        }
        integrate_multirate(
            problem,
            options,
            Method::Mreef {
                m: self.m,
                order: self.order,
                sequence: self.sequence,
            },
        )
    }
}

impl SplitOdeAlgorithm for Mrab {
    fn solve_validated<FE, FI, P>(
        &self,
        problem: &SplitOdeProblem<FE, FI, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        FE: crate::OdeFunction<P>,
        FI: crate::OdeFunction<P>,
    {
        if self.m == 0 || !(1..=5).contains(&self.order) {
            return Err(SolveError::InvalidMultistepOrder);
        }
        if options.adaptive && self.order == 1 {
            return Err(SolveError::AdaptiveStepUnsupported);
        }
        integrate_multirate(
            problem,
            options,
            Method::Mrab {
                m: self.m,
                order: self.order,
            },
        )
    }
}

impl SplitOdeAlgorithm for Mis {
    fn solve_validated<FE, FI, P>(
        &self,
        problem: &SplitOdeProblem<FE, FI, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        FE: crate::OdeFunction<P>,
        FI: crate::OdeFunction<P>,
    {
        if self.m == 0 {
            return Err(SolveError::InvalidMultistepOrder);
        }
        integrate_multirate(
            problem,
            options,
            Method::Mis {
                m: self.m,
                tableau: self.tableau().map_err(|_| SolveError::InvalidTableau)?,
            },
        )
    }
}

#[derive(Clone, Copy)]
enum Method {
    Mreef {
        m: usize,
        order: usize,
        sequence: MultirateSequence,
    },
    Mrab {
        m: usize,
        order: usize,
    },
    Mis {
        m: usize,
        tableau: &'static MisTableau,
    },
    Mri {
        m: usize,
        tableau: &'static MriTableau,
    },
}

impl Method {
    fn controller_order(self) -> usize {
        match self {
            Self::Mreef { order, .. } | Self::Mrab { order, .. } => order,
            Self::Mis { tableau, .. } => tableau.order(),
            Self::Mri { tableau, .. } => tableau.order(),
        }
    }
}

fn integrate_multirate<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    options: &SolveOptions,
    method: Method,
) -> Result<Solution, SolveError>
where
    FE: crate::OdeFunction<P>,
    FI: crate::OdeFunction<P>,
{
    validate(problem, options)?;
    if !options.adaptive && options.initial_step.is_none() {
        return Err(SolveError::InitialStepRequired);
    }
    let (start, end) = problem.time_span();
    let direction = (end - start).signum();
    let maximum_step = options.max_step.min((end - start).abs());
    let dimension = problem.dimension();
    let mut state = problem.initial_state().to_vec();
    let mut candidate = vec![0.0; dimension];
    let mut state_before_effect = vec![0.0; dimension];
    let mut start_derivative = vec![0.0; dimension];
    let mut end_derivative = vec![0.0; dimension];
    let mut dense_endpoint = vec![0.0; dimension];
    let mut stats = SolverStats::default();
    let mut recorder = TrajectoryRecorder::new(&state, start, options);
    let initial = problem.apply_initial_callbacks(&mut state, start)?;
    stats.callback_invocations += initial.invocations;
    stats.rhs_evaluations += initial.rhs_evaluations;
    if initial.state_modified {
        recorder.record_callback(start, problem.initial_state(), &state, initial, true);
    }
    if problem.domain_rejection_factor(&state, start).is_some() {
        return Err(SolveError::InitialStateOutOfDomain);
    }
    if initial.terminate {
        return finish_successful(problem, &mut state, start, recorder, stats);
    }
    evaluate_total(problem, &state, start, &mut start_derivative, &mut stats)?;
    let proposed_step = direction
        * match options.initial_step {
            Some(value) => value.min(maximum_step),
            None => estimate_initial_step(&state, &start_derivative, maximum_step),
        };
    let mut step = callback_adjusted_step(initial, proposed_step, direction, maximum_step);
    let mut time = start;
    let mut attempted = 0usize;
    let mut previous_rejected = false;
    let order = method.controller_order() as f64;
    let mut time_stops = TimeStopSchedule::new(&options.time_stops, start, end);
    while direction * (end - time) > 0.0 {
        if attempted == options.max_steps {
            return Err(SolveError::MaxStepsExceeded);
        }
        attempted += 1;
        let proposed_step = step;
        step = time_stops.clip_step_with(time, step, problem.next_preset_time(time, direction));
        if problem.has_predictive_domain() {
            step = problem.predictive_domain_adjusted_step(
                &state,
                &start_derivative,
                time,
                step,
                options.absolute_tolerance,
                &mut candidate,
            )?;
        }
        if time + step == time {
            return Err(SolveError::StepSizeUnderflow);
        }
        let error_norm = match attempt(
            method,
            problem,
            &state,
            time,
            step,
            &mut candidate,
            options,
            &mut stats,
        ) {
            Ok(error) => error,
            Err(SolveError::NonlinearSolveFailed | SolveError::SingularLinearSystem)
                if options.adaptive =>
            {
                stats.rejected_steps += 1;
                step *= 0.2;
                previous_rejected = true;
                continue;
            }
            Err(error) => return Err(error),
        };
        if !candidate.iter().all(|value| value.is_finite()) {
            return Err(SolveError::NonFiniteDerivative);
        }
        if !options.adaptive || error_norm <= 1.0 {
            let previous_time = time;
            let attempted_time = time + step;
            if let Some(reduction_factor) =
                problem.domain_rejection_factor(&candidate, attempted_time)
            {
                stats.rejected_steps += 1;
                step *= reduction_factor;
                previous_rejected = true;
                continue;
            }
            dense_endpoint.copy_from_slice(&candidate);
            evaluate_total(
                problem,
                &dense_endpoint,
                attempted_time,
                &mut end_derivative,
                &mut stats,
            )?;
            let borrowed = BorrowedHermiteSegment::new(
                previous_time,
                attempted_time,
                &state,
                &dense_endpoint,
                &start_derivative,
                &end_derivative,
            )
            .map_err(|_| SolveError::NonFiniteDerivative)?;
            let mut next_time = attempted_time;
            let mut interpolate = |target: f64, output: &mut [f64]| {
                borrowed
                    .interpolate(target, output)
                    .map_err(|_| SolveError::NonFiniteDerivative)
            };
            let callbacks = problem.apply_step_callbacks(
                &state,
                previous_time,
                &mut candidate,
                &mut next_time,
                &mut state_before_effect,
                options.event_tolerance,
                Some(&mut interpolate),
            )?;
            stats.callback_invocations += callbacks.invocations;
            stats.rhs_evaluations += callbacks.rhs_evaluations;
            stats.accepted_steps += 1;
            let dense_state = if callbacks.invocations == 0 {
                &candidate
            } else {
                &state_before_effect
            };
            recorder
                .record_step_dense(
                    &state,
                    previous_time,
                    dense_state,
                    next_time,
                    next_time == end,
                    &borrowed,
                )
                .map_err(|_| SolveError::NonFiniteDerivative)?;
            if recorder.retains_dense_output() {
                let owned = HermiteSegment::new_bounded(
                    previous_time,
                    attempted_time,
                    next_time,
                    state.clone(),
                    dense_endpoint.clone(),
                    start_derivative.clone(),
                    end_derivative.clone(),
                )
                .map_err(|_| SolveError::NonFiniteDerivative)?;
                recorder.retain_hermite_segment(owned);
            }
            if callbacks.invocations > 0 {
                recorder.record_callback(
                    next_time,
                    &state_before_effect,
                    &candidate,
                    callbacks,
                    next_time == end,
                );
            }
            if callbacks.terminate {
                return finish_successful(problem, &mut candidate, next_time, recorder, stats);
            }
            time = next_time;
            time_stops.accepted(time);
            std::mem::swap(&mut state, &mut candidate);
            if !callbacks.state_modified {
                start_derivative.copy_from_slice(&end_derivative);
            } else {
                evaluate_total(problem, &state, time, &mut start_derivative, &mut stats)?;
            }
            if callbacks.requested_step.is_some() {
                step = callback_adjusted_step(callbacks, step, direction, maximum_step);
            } else if options.adaptive {
                let factor = if error_norm == 0.0 {
                    5.0
                } else if error_norm.is_finite() {
                    (0.9 * error_norm.powf(-1.0 / (order + 1.0))).clamp(0.2, 5.0)
                } else {
                    0.2
                };
                let factor = if previous_rejected {
                    factor.min(1.0)
                } else {
                    factor
                };
                step = callback_adjusted_step(
                    callbacks,
                    direction * step.abs() * factor,
                    direction,
                    maximum_step,
                );
            } else {
                step = callback_adjusted_step(callbacks, proposed_step, direction, maximum_step);
            }
            previous_rejected = false;
        } else {
            stats.rejected_steps += 1;
            let factor = if error_norm.is_finite() {
                (0.9 * error_norm.powf(-1.0 / (order + 1.0))).clamp(0.2, 1.0)
            } else {
                0.2
            };
            step *= factor;
            previous_rejected = true;
        }
    }
    finish_successful(problem, &mut state, time, recorder, stats)
}

fn finish_successful<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    state: &mut [f64],
    time: f64,
    mut recorder: TrajectoryRecorder<'_>,
    stats: SolverStats,
) -> Result<Solution, SolveError>
where
    FE: crate::OdeFunction<P>,
    FI: crate::OdeFunction<P>,
{
    if problem.apply_finalize_callbacks(state, time)? {
        recorder.synchronize_endpoint(time, state);
    }
    Ok(recorder.finish(stats))
}

fn validate<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    options: &SolveOptions,
) -> Result<(), SolveError> {
    validate_state_time_options(problem.initial_state(), problem.time_span(), options)?;
    validate_preset_time_sequences(problem.preset_time_sequences(), problem.time_span())?;
    validate_vector_callback_lengths(problem.vector_callback_lengths())
}

fn estimate_initial_step(state: &[f64], derivative: &[f64], maximum: f64) -> f64 {
    let state_scale = state
        .iter()
        .fold(1.0_f64, |acc, value| acc.max(value.abs()));
    let derivative_scale = derivative
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    if derivative_scale == 0.0 {
        (0.01 * maximum).max(f64::MIN_POSITIVE).min(maximum)
    } else {
        (0.01 * state_scale / derivative_scale)
            .max(f64::MIN_POSITIVE)
            .min(maximum)
    }
}

#[allow(clippy::too_many_arguments)]
fn attempt<FE, FI, P>(
    method: Method,
    problem: &SplitOdeProblem<FE, FI, P>,
    state: &[f64],
    time: f64,
    step: f64,
    candidate: &mut [f64],
    options: &SolveOptions,
    stats: &mut SolverStats,
) -> Result<f64, SolveError>
where
    FE: crate::OdeFunction<P>,
    FI: crate::OdeFunction<P>,
{
    let error = match method {
        Method::Mreef { m, order, sequence } => mreef_step(
            problem, state, time, step, candidate, m, order, sequence, stats,
        )?,
        Method::Mrab { m, order } => {
            mrab_step(problem, state, time, step, candidate, m, order, stats)?
        }
        Method::Mis { m, tableau } => {
            mis_step(problem, state, time, step, candidate, m, tableau, stats)?
        }
        Method::Mri { m, tableau } => {
            mri_step(problem, state, time, step, candidate, m, tableau, stats)?
        }
    };
    Ok(scaled_error(
        &error,
        state,
        candidate,
        options.absolute_tolerance,
        options.relative_tolerance,
    ))
}

fn scaled_error(error: &[f64], old: &[f64], new: &[f64], atol: f64, rtol: f64) -> f64 {
    error
        .iter()
        .zip(old)
        .zip(new)
        .fold(0.0_f64, |norm, ((error, old), new)| {
            norm.max(error.abs() / (atol + rtol * old.abs().max(new.abs())))
        })
}

fn evaluate_fast<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    state: &[f64],
    time: f64,
    output: &mut [f64],
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    FE: crate::OdeFunction<P>,
{
    problem.evaluate_explicit(output, state, time)?;
    stats.rhs_evaluations += 1;
    finite(output)
}

fn evaluate_slow<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    state: &[f64],
    time: f64,
    output: &mut [f64],
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    FI: crate::OdeFunction<P>,
{
    problem.evaluate_implicit(output, state, time)?;
    stats.rhs_evaluations += 1;
    finite(output)
}

fn evaluate_total<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    state: &[f64],
    time: f64,
    output: &mut [f64],
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    FE: crate::OdeFunction<P>,
    FI: crate::OdeFunction<P>,
{
    let mut slow = vec![0.0; output.len()];
    evaluate_fast(problem, state, time, output, stats)?;
    evaluate_slow(problem, state, time, &mut slow, stats)?;
    for (value, slow) in output.iter_mut().zip(slow) {
        *value += slow;
    }
    finite(output)
}

fn finite(values: &[f64]) -> Result<(), SolveError> {
    values
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteDerivative)
}

#[allow(clippy::too_many_arguments)]
fn mreef_step<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    state: &[f64],
    time: f64,
    step: f64,
    candidate: &mut [f64],
    m: usize,
    order: usize,
    sequence: MultirateSequence,
    stats: &mut SolverStats,
) -> Result<Vec<f64>, SolveError>
where
    FE: crate::OdeFunction<P>,
    FI: crate::OdeFunction<P>,
{
    let dimension = state.len();
    let ns: Vec<usize> = (1..=order)
        .map(|index| match sequence {
            MultirateSequence::Harmonic => index,
            MultirateSequence::Romberg => 1usize << (index - 1),
        })
        .collect();
    let mut table = Vec::with_capacity(order);
    for &macro_count in &ns {
        let mut value = state.to_vec();
        let macro_step = step / macro_count as f64;
        let micro_step = macro_step / m as f64;
        let mut slow = vec![0.0; dimension];
        let mut fast = vec![0.0; dimension];
        for macro_index in 0..macro_count {
            let macro_time = time + macro_index as f64 * macro_step;
            evaluate_slow(problem, &value, macro_time, &mut slow, stats)?;
            for micro_index in 0..m {
                let micro_time = macro_time + micro_index as f64 * micro_step;
                evaluate_fast(problem, &value, micro_time, &mut fast, stats)?;
                for ((value, fast), slow) in value.iter_mut().zip(&fast).zip(&slow) {
                    *value += micro_step * (fast + slow);
                }
            }
        }
        table.push(value);
    }
    for k in 1..order {
        for j in (k..order).rev() {
            let denominator = ns[j] as f64 / ns[j - k] as f64 - 1.0;
            let (left, right) = table.split_at_mut(j);
            let previous = &left[j - 1];
            let current = &mut right[0];
            for (current, previous) in current.iter_mut().zip(previous) {
                *current += (*current - previous) / denominator;
            }
        }
    }
    candidate.copy_from_slice(&table[order - 1]);
    Ok(candidate
        .iter()
        .zip(&table[order - 2])
        .map(|(high, low)| high - low)
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn mrab_step<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    state: &[f64],
    time: f64,
    step: f64,
    candidate: &mut [f64],
    m: usize,
    order: usize,
    stats: &mut SolverStats,
) -> Result<Vec<f64>, SolveError>
where
    FE: crate::OdeFunction<P>,
    FI: crate::OdeFunction<P>,
{
    let dimension = state.len();
    let h = step / m as f64;
    let mut slow = vec![0.0; dimension];
    let mut fast = vec![0.0; dimension];
    evaluate_slow(problem, state, time, &mut slow, stats)?;
    let mut value = state.to_vec();
    // Upstream deliberately restarts AB-min(l,k) inside every macro step.
    let mut history: Vec<Vec<f64>> = Vec::with_capacity(order);
    for micro in 0..m {
        evaluate_fast(problem, &value, time + micro as f64 * h, &mut fast, stats)?;
        let combined: Vec<f64> = fast.iter().zip(&slow).map(|(a, b)| a + b).collect();
        history.insert(0, combined);
        history.truncate(order);
        let weights = &adams_bashforth(history.len().min(order))?.beta()[1..];
        for index in 0..dimension {
            value[index] += h * weights
                .iter()
                .zip(history.iter())
                .map(|(weight, derivative)| weight * derivative[index])
                .sum::<f64>();
        }
    }
    candidate.copy_from_slice(&value);
    if order == 1 {
        return Ok(vec![0.0; dimension]);
    }
    let mut error = vec![0.0; dimension];
    if history.len() >= order {
        let high = &adams_bashforth(order)?.beta()[1..];
        let low = &adams_bashforth(order - 1)?.beta()[1..];
        for index in 0..dimension {
            error[index] = h
                * (high
                    .iter()
                    .zip(history.iter())
                    .map(|(weight, derivative)| weight * derivative[index])
                    .sum::<f64>()
                    - low
                        .iter()
                        .zip(history.iter())
                        .map(|(weight, derivative)| weight * derivative[index])
                        .sum::<f64>());
        }
    }
    Ok(error)
}

#[allow(clippy::too_many_arguments)]
fn mis_step<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    state: &[f64],
    time: f64,
    step: f64,
    candidate: &mut [f64],
    m: usize,
    tableau: &'static MisTableau,
    stats: &mut SolverStats,
) -> Result<Vec<f64>, SolveError>
where
    FE: crate::OdeFunction<P>,
    FI: crate::OdeFunction<P>,
{
    let stage_count = tableau.d().len();
    let dimension = state.len();
    let mut stages = vec![vec![0.0; dimension]; stage_count];
    let mut slow = vec![vec![0.0; dimension]; stage_count];
    stages[0].copy_from_slice(state);
    evaluate_slow(problem, state, time, &mut slow[0], stats)?;
    let mut fast = vec![0.0; dimension];
    let mut midpoint_fast = vec![0.0; dimension];
    let mut midpoint = vec![0.0; dimension];
    for stage in 1..stage_count {
        let mut value = state.to_vec();
        for (j, previous_stage) in stages.iter().take(stage).enumerate() {
            for index in 0..dimension {
                value[index] += tableau.alpha()[stage][j] * (previous_stage[index] - state[index]);
            }
        }
        let mut offset = vec![0.0; dimension];
        for (j, (previous_stage, previous_slow)) in stages.iter().zip(&slow).take(stage).enumerate()
        {
            for index in 0..dimension {
                offset[index] += tableau.gamma()[stage][j] / (tableau.d()[stage] * step)
                    * (previous_stage[index] - state[index])
                    + tableau.beta()[stage][j] / tableau.d()[stage] * previous_slow[index];
            }
        }
        let micro_count = ((m as f64 * tableau.d()[stage]).ceil() as usize).max(1);
        let h = tableau.d()[stage] * step / micro_count as f64;
        let slope = (tableau.c()[stage] - tableau.c_tilde()[stage]) / tableau.d()[stage];
        for micro in 0..micro_count {
            let tau = micro as f64 * h;
            evaluate_fast(
                problem,
                &value,
                time + tableau.c_tilde()[stage] * step + slope * tau,
                &mut fast,
                stats,
            )?;
            for index in 0..dimension {
                midpoint[index] = value[index] + 0.5 * h * (offset[index] + fast[index]);
            }
            evaluate_fast(
                problem,
                &midpoint,
                time + tableau.c_tilde()[stage] * step + slope * (tau + 0.5 * h),
                &mut midpoint_fast,
                stats,
            )?;
            for index in 0..dimension {
                value[index] += h * (offset[index] + midpoint_fast[index]);
            }
        }
        stages[stage] = value;
        evaluate_slow(
            problem,
            &stages[stage],
            time + tableau.c()[stage] * step,
            &mut slow[stage],
            stats,
        )?;
    }
    candidate.copy_from_slice(&stages[stage_count - 1]);
    Ok(stages[stage_count - 1]
        .iter()
        .zip(&stages[stage_count - 2])
        .map(|(high, low)| high - low)
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn mri_step<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    state: &[f64],
    time: f64,
    step: f64,
    candidate: &mut [f64],
    m: usize,
    tableau: &MriTableau,
    stats: &mut SolverStats,
) -> Result<Vec<f64>, SolveError>
where
    FE: crate::OdeFunction<P>,
    FI: crate::OdeFunction<P>,
{
    let stages_count = tableau.dc().len();
    let dimension = state.len();
    let mut stages = vec![vec![0.0; dimension]; stages_count + 1];
    let mut slow = vec![vec![0.0; dimension]; stages_count];
    stages[0].copy_from_slice(state);
    let mut c_previous = 0.0;
    for stage in 0..stages_count {
        evaluate_slow(
            problem,
            &stages[stage],
            time + c_previous * step,
            &mut slow[stage],
            stats,
        )?;
        let gamma = tableau.gamma()[stage];
        if gamma == 0.0 {
            stages[stage + 1] = mri_substage(
                problem,
                &stages[stage],
                &slow,
                time,
                step,
                c_previous,
                tableau.dc()[stage],
                &tableau.w0()[stage],
                &tableau.w1()[stage],
                m,
                tableau.inner_order(),
                stats,
            )?;
        } else {
            let mut base = stages[stage].clone();
            for (j, slow_stage) in slow.iter().enumerate().take(stage + 1) {
                let weight = tableau.w0()[stage][j] + 0.5 * tableau.w1()[stage][j];
                for index in 0..dimension {
                    base[index] += step * weight * slow_stage[index];
                }
            }
            let target_time = time + (c_previous + tableau.dc()[stage]) * step;
            stages[stage + 1] = implicit_slow_endpoint(
                problem,
                &base,
                &stages[stage],
                target_time,
                gamma * step,
                stats,
            )?;
        }
        c_previous += tableau.dc()[stage];
    }
    candidate.copy_from_slice(&stages[stages_count]);
    let comparison = if let Some(weights0) = tableau.embedded0() {
        mri_substage(
            problem,
            &stages[stages_count - 1],
            &slow,
            time,
            step,
            c_previous - tableau.dc()[stages_count - 1],
            tableau.dc()[stages_count - 1],
            weights0,
            tableau.embedded1().ok_or(SolveError::InvalidTableau)?,
            m,
            tableau.inner_order(),
            stats,
        )?
    } else {
        stages[stages_count - 1].clone()
    };
    Ok(candidate
        .iter()
        .zip(comparison)
        .map(|(high, low)| high - low)
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn mri_substage<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    start: &[f64],
    slow: &[Vec<f64>],
    macro_time: f64,
    macro_step: f64,
    c_previous: f64,
    dc: f64,
    w0: &[f64],
    w1: &[f64],
    m: usize,
    inner_order: usize,
    stats: &mut SolverStats,
) -> Result<Vec<f64>, SolveError>
where
    FE: crate::OdeFunction<P>,
{
    let dimension = start.len();
    if dc == 0.0 {
        let mut value = start.to_vec();
        for (stage, derivative) in slow.iter().enumerate() {
            let weight =
                w0.get(stage).copied().unwrap_or(0.0) + 0.5 * w1.get(stage).copied().unwrap_or(0.0);
            for index in 0..dimension {
                value[index] += macro_step * weight * derivative[index];
            }
        }
        return Ok(value);
    }
    let h = 1.0 / m as f64;
    let mut value = start.to_vec();
    let mut k1 = vec![0.0; dimension];
    let mut k2 = vec![0.0; dimension];
    let mut k3 = vec![0.0; dimension];
    let mut k4 = vec![0.0; dimension];
    let mut temporary = vec![0.0; dimension];
    for micro in 0..m {
        let tau = micro as f64 * h;
        mri_rate(
            problem, &value, slow, macro_time, macro_step, c_previous, dc, w0, w1, tau, &mut k1,
            stats,
        )?;
        for index in 0..dimension {
            temporary[index] = value[index] + 0.5 * h * k1[index];
        }
        mri_rate(
            problem,
            &temporary,
            slow,
            macro_time,
            macro_step,
            c_previous,
            dc,
            w0,
            w1,
            tau + 0.5 * h,
            &mut k2,
            stats,
        )?;
        match inner_order {
            2 => {
                for index in 0..dimension {
                    value[index] += h * k2[index];
                }
            }
            3 => {
                for index in 0..dimension {
                    temporary[index] = value[index] - h * k1[index] + 2.0 * h * k2[index];
                }
                mri_rate(
                    problem,
                    &temporary,
                    slow,
                    macro_time,
                    macro_step,
                    c_previous,
                    dc,
                    w0,
                    w1,
                    tau + h,
                    &mut k3,
                    stats,
                )?;
                for index in 0..dimension {
                    value[index] += h * (k1[index] + 4.0 * k2[index] + k3[index]) / 6.0;
                }
            }
            _ => {
                for index in 0..dimension {
                    temporary[index] = value[index] + 0.5 * h * k2[index];
                }
                mri_rate(
                    problem,
                    &temporary,
                    slow,
                    macro_time,
                    macro_step,
                    c_previous,
                    dc,
                    w0,
                    w1,
                    tau + 0.5 * h,
                    &mut k3,
                    stats,
                )?;
                for index in 0..dimension {
                    temporary[index] = value[index] + h * k3[index];
                }
                mri_rate(
                    problem,
                    &temporary,
                    slow,
                    macro_time,
                    macro_step,
                    c_previous,
                    dc,
                    w0,
                    w1,
                    tau + h,
                    &mut k4,
                    stats,
                )?;
                for index in 0..dimension {
                    value[index] +=
                        h * (k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index]) / 6.0;
                }
            }
        }
    }
    Ok(value)
}

#[allow(clippy::too_many_arguments)]
fn mri_rate<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    state: &[f64],
    slow: &[Vec<f64>],
    macro_time: f64,
    macro_step: f64,
    c_previous: f64,
    dc: f64,
    w0: &[f64],
    w1: &[f64],
    tau: f64,
    output: &mut [f64],
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    FE: crate::OdeFunction<P>,
{
    evaluate_fast(
        problem,
        state,
        macro_time + (c_previous + tau * dc) * macro_step,
        output,
        stats,
    )?;
    for value in output.iter_mut() {
        *value *= macro_step * dc;
    }
    for (stage, derivative) in slow.iter().enumerate() {
        let weight =
            w0.get(stage).copied().unwrap_or(0.0) + tau * w1.get(stage).copied().unwrap_or(0.0);
        for (value, derivative) in output.iter_mut().zip(derivative) {
            *value += macro_step * weight * derivative;
        }
    }
    finite(output)
}

fn implicit_slow_endpoint<FE, FI, P>(
    problem: &SplitOdeProblem<FE, FI, P>,
    base: &[f64],
    predictor: &[f64],
    time: f64,
    scale: f64,
    stats: &mut SolverStats,
) -> Result<Vec<f64>, SolveError>
where
    FI: crate::OdeFunction<P>,
{
    let dimension = base.len();
    let mut value = predictor.to_vec();
    let mut derivative = vec![0.0; dimension];
    let mut perturbed_derivative = vec![0.0; dimension];
    let mut perturbed = value.clone();
    let mut residual = vec![0.0; dimension];
    let mut correction = vec![0.0; dimension];
    let mut matrix = vec![0.0; dimension * dimension];
    let mut pivots = vec![0usize; dimension];
    for _ in 0..MAX_NEWTON_ITERATIONS {
        evaluate_slow(problem, &value, time, &mut derivative, stats)?;
        let mut norm = 0.0_f64;
        for index in 0..dimension {
            residual[index] = value[index] - base[index] - scale * derivative[index];
            norm = norm.max(residual[index].abs());
        }
        if norm <= NEWTON_TOLERANCE * (1.0 + value.iter().fold(0.0_f64, |n, x| n.max(x.abs()))) {
            return Ok(value);
        }
        if !problem.evaluate_implicit_jacobian(&mut matrix, &value, time) {
            for column in 0..dimension {
                perturbed.copy_from_slice(&value);
                let delta = f64::EPSILON.sqrt() * (1.0 + value[column].abs());
                perturbed[column] += delta;
                evaluate_slow(problem, &perturbed, time, &mut perturbed_derivative, stats)?;
                for row in 0..dimension {
                    matrix[row * dimension + column] =
                        (perturbed_derivative[row] - derivative[row]) / delta;
                }
            }
        }
        stats.jacobian_evaluations += 1;
        for row in 0..dimension {
            for column in 0..dimension {
                matrix[row * dimension + column] *= -scale;
            }
            matrix[row * dimension + row] += 1.0;
            correction[row] = -residual[row];
        }
        factorize(&mut matrix, &mut pivots, dimension)?;
        stats.linear_factorizations += 1;
        solve_factorized(&matrix, &pivots, &mut correction, dimension);
        stats.linear_solves += 1;
        stats.nonlinear_iterations += 1;
        for (value, correction) in value.iter_mut().zip(&correction) {
            *value += correction;
        }
        finite(&value)?;
    }
    Err(SolveError::NonlinearSolveFailed)
}
