//! Explicit symplectic composition methods for partitioned second-order problems.
//!
//! The coefficient vectors are pinned copies of the `SymplecticTableau` data in
//! OrdinaryDiffEqSymplecticRK. A stage is a drift of the position by `bᵢ`
//! followed by a kick of the velocity by `aᵢ`.

use super::function::SecondOrderFunction;
use super::general::{
    SecondOrderOdeProblem, apply_finalize_callbacks, apply_initial_callbacks, apply_step_callbacks,
};
use crate::integrator::{TimeStopSchedule, callback_adjusted_step};
use crate::solver::{
    validate_preset_time_sequences, validate_state_time_options, validate_vector_callback_lengths,
};
use crate::{SolveError, SolveOptions, SolverStats};
use ndarray::IxDyn;
use thiserror::Error;

mod solution;

pub use crate::tableau::SymplecticTableau;
pub use solution::SymplecticSolution;

use crate::tableau::{TableauError, define_symplectic_from_file};
use solution::SymplecticRecorder;

/// A named explicit symplectic composition.
///
/// This trait is a downstream extension point. Implementors return a
/// compile-validated, lazily initialized composition tableau. The method takes
/// `&self` so custom algorithms may retain configuration without requiring
/// `Copy` or global mutable state.
pub trait SymplecticAlgorithm {
    /// Returns the validated, lazily initialized composition tableau.
    fn tableau(&self) -> Result<&'static SymplecticTableau, TableauError>;
}

define_symplectic_from_file!(pub PseudoVerletLeapfrog, "src/tableau/resources/symplectic/pseudoverletleapfrog.json", crate = crate);
define_symplectic_from_file!(pub McAte2, "src/tableau/resources/symplectic/mcate2.json", crate = crate);
define_symplectic_from_file!(pub Ruth3, "src/tableau/resources/symplectic/ruth3.json", crate = crate);
define_symplectic_from_file!(pub McAte3, "src/tableau/resources/symplectic/mcate3.json", crate = crate);
define_symplectic_from_file!(pub CandyRoz4, "src/tableau/resources/symplectic/candyroz4.json", crate = crate);
define_symplectic_from_file!(pub McAte4, "src/tableau/resources/symplectic/mcate4.json", crate = crate);
define_symplectic_from_file!(pub CalvoSanz4, "src/tableau/resources/symplectic/calvosanz4.json", crate = crate);
define_symplectic_from_file!(pub McAte42, "src/tableau/resources/symplectic/mcate42.json", crate = crate);
define_symplectic_from_file!(pub McAte5, "src/tableau/resources/symplectic/mcate5.json", crate = crate);
define_symplectic_from_file!(pub Yoshida6, "src/tableau/resources/symplectic/yoshida6.json", crate = crate);
define_symplectic_from_file!(pub KahanLi6, "src/tableau/resources/symplectic/kahanli6.json", crate = crate);
define_symplectic_from_file!(pub McAte8, "src/tableau/resources/symplectic/mcate8.json", crate = crate);
define_symplectic_from_file!(pub KahanLi8, "src/tableau/resources/symplectic/kahanli8.json", crate = crate);
define_symplectic_from_file!(pub SofSpa10, "src/tableau/resources/symplectic/sofspa10.json", crate = crate);

/// Failure specific to a fixed-step symplectic composition.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
#[non_exhaustive]
pub enum SymplecticSolveError {
    /// Position and velocity partitions differ in size.
    #[error("position and velocity dimensions must match")]
    StateDimensionMismatch,
    /// A common solver validation or execution error.
    #[error("{0}")]
    Solve(
        #[from]
        #[source]
        SolveError,
    ),
}

/// Solves a second-order problem with a pinned alternating drift/kick method.
///
pub fn solve_symplectic<F, P, A>(
    problem: &SecondOrderOdeProblem<F, P>,
    algorithm: A,
    options: &SolveOptions,
) -> Result<SymplecticSolution, SymplecticSolveError>
where
    F: SecondOrderFunction<P>,
    A: SymplecticAlgorithm,
{
    validate(problem, options)?;
    if options.adaptive {
        return Err(SolveError::AdaptiveStepUnsupported.into());
    }
    let fixed_step = options
        .initial_step
        .ok_or(SolveError::InitialStepRequired)?;
    let (start, end) = problem.time_span();
    let tableau = algorithm.tableau().map_err(SolveError::from)?;

    let direction = (end - start).signum();
    let maximum_step = options.max_step.min((end - start).abs());
    let mut step_size = fixed_step.min(maximum_step);
    let dimension = problem.initial_position().len();
    let mut position = problem.initial_position().to_vec();
    let mut velocity = problem.initial_velocity().to_vec();
    let mut stats = SolverStats::default();
    let mut recorder = SymplecticRecorder::new(&position, &velocity, start, options);
    let initial_callbacks = apply_initial_callbacks(problem, &mut velocity, &mut position, start)?;
    stats.callback_invocations += initial_callbacks.invocations;
    stats.rhs_evaluations += initial_callbacks.rhs_evaluations;
    if initial_callbacks.state_modified {
        recorder.record_callback(
            start,
            problem.initial_position(),
            problem.initial_velocity(),
            &position,
            &velocity,
            initial_callbacks,
            true,
        );
    }
    if problem
        .domain_rejection_factor(&velocity, &position, start)
        .is_some()
    {
        return Err(SolveError::InitialStateOutOfDomain.into());
    }
    let mut candidate_position = position.clone();
    let mut candidate_velocity = velocity.clone();
    let mut acceleration = vec![0.0; dimension];
    let mut state_before_position = if problem.has_callbacks() {
        vec![0.0; dimension]
    } else {
        Vec::new()
    };
    let mut state_before_velocity = if problem.has_callbacks() {
        vec![0.0; dimension]
    } else {
        Vec::new()
    };
    if initial_callbacks.terminate {
        return finish_successful(
            problem,
            &mut velocity,
            &mut position,
            start,
            recorder,
            stats,
        );
    }
    step_size = callback_adjusted_step(
        initial_callbacks,
        direction * step_size,
        direction,
        maximum_step,
    )
    .abs();
    let mut time = start;
    let mut steps = 0usize;
    let mut time_stops = TimeStopSchedule::new(&options.time_stops, start, end);

    while direction * (end - time) > 0.0 {
        if steps >= options.max_steps {
            return Err(SolveError::MaxStepsExceeded.into());
        }
        steps += 1;
        let step = time_stops.clip_step_with(
            time,
            direction * step_size,
            problem.next_preset_time(time, direction),
        );
        if time + step == time {
            return Err(SolveError::StepSizeUnderflow.into());
        }
        candidate_position.copy_from_slice(&position);
        candidate_velocity.copy_from_slice(&velocity);
        let previous_time = time;
        stats.rhs_evaluations += perform_step(
            problem,
            tableau,
            &mut candidate_position,
            &mut candidate_velocity,
            &mut acceleration,
            time,
            step,
        )?;
        let mut next_time = time + step;
        if direction * (end - next_time) <= 0.0 {
            next_time = end;
        }
        if let Some(reduction_factor) =
            problem.domain_rejection_factor(&candidate_velocity, &candidate_position, next_time)
        {
            stats.rejected_steps += 1;
            step_size = step.abs() * reduction_factor;
            continue;
        }
        let callback = apply_step_callbacks(
            problem,
            &velocity,
            &position,
            previous_time,
            &mut candidate_velocity,
            &mut candidate_position,
            &mut next_time,
            &mut state_before_velocity,
            &mut state_before_position,
            options.event_tolerance,
            None,
        )?;
        stats.callback_invocations += callback.invocations;
        stats.rhs_evaluations += callback.rhs_evaluations;
        stats.accepted_steps += 1;
        recorder.record_step(
            &position,
            &velocity,
            previous_time,
            if callback.invocations == 0 {
                &candidate_position
            } else {
                &state_before_position
            },
            if callback.invocations == 0 {
                &candidate_velocity
            } else {
                &state_before_velocity
            },
            next_time,
            next_time == end,
        )?;
        if callback.invocations > 0 {
            recorder.record_callback(
                next_time,
                &state_before_position,
                &state_before_velocity,
                &candidate_position,
                &candidate_velocity,
                callback,
                next_time == end,
            );
        }
        time = next_time;
        time_stops.accepted(time);
        std::mem::swap(&mut position, &mut candidate_position);
        std::mem::swap(&mut velocity, &mut candidate_velocity);
        if callback.terminate {
            return finish_successful(problem, &mut velocity, &mut position, time, recorder, stats);
        }
        step_size =
            callback_adjusted_step(callback, direction * step_size, direction, maximum_step).abs();
    }

    finish_successful(problem, &mut velocity, &mut position, time, recorder, stats)
}

fn finish_successful<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    velocity: &mut [f64],
    position: &mut [f64],
    time: f64,
    mut recorder: SymplecticRecorder<'_>,
    stats: SolverStats,
) -> Result<SymplecticSolution, SymplecticSolveError>
where
    F: SecondOrderFunction<P>,
{
    if apply_finalize_callbacks(problem, velocity, position, time)? {
        recorder.synchronize_endpoint(time, position, velocity);
    }
    Ok(recorder.finish(stats, IxDyn(problem.state_shape())))
}

fn validate<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    options: &SolveOptions,
) -> Result<(), SymplecticSolveError> {
    let position = problem.initial_position();
    let velocity = problem.initial_velocity();
    if position.is_empty() {
        return Err(SolveError::EmptyState.into());
    }
    if position.len() != velocity.len() {
        return Err(SymplecticSolveError::StateDimensionMismatch);
    }
    validate_state_time_options(position, problem.time_span(), options)?;
    validate_preset_time_sequences(problem.preset_time_sequences(), problem.time_span())?;
    validate_vector_callback_lengths(problem.vector_callback_lengths())?;
    if !velocity.iter().all(|value| value.is_finite()) {
        return Err(SolveError::NonFiniteInitialState.into());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn perform_step<F, P>(
    problem: &SecondOrderOdeProblem<F, P>,
    tableau: &SymplecticTableau,
    position: &mut [f64],
    velocity: &mut [f64],
    acceleration: &mut [f64],
    time: f64,
    step: f64,
) -> Result<usize, SolveError>
where
    F: SecondOrderFunction<P>,
{
    let mut stage_time = time;
    for (stage, (&kick, &drift)) in tableau.a().iter().zip(tableau.b()).enumerate() {
        for (position, &velocity) in position.iter_mut().zip(&*velocity) {
            *position += drift * step * velocity;
        }
        problem.evaluate_acceleration(acceleration, velocity, position, stage_time)?;
        if !acceleration.iter().all(|value| value.is_finite()) {
            return Err(SolveError::NonFiniteDerivative);
        }
        for (velocity, &acceleration) in velocity.iter_mut().zip(&*acceleration) {
            *velocity += kick * step * acceleration;
        }
        if stage + 1 < tableau.stages() {
            stage_time += kick * step;
        }
    }
    Ok(tableau.stages())
}
