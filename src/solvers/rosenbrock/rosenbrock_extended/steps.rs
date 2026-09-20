use super::workspace::Workspace;
use crate::linear::{factorize, solve_factorized};
use crate::tableau::RosenbrockTableau;
use crate::{OdeProblem, SolveError, SolveOptions, SolverStats};

#[derive(Clone, Copy)]
pub(super) enum AdaptiveErrorEstimator {
    Embedded,
    RichardsonStepDoubling { method_order: i32 },
}

#[allow(clippy::too_many_arguments)]
pub(super) fn perform_rosenbrock32<F, P>(
    problem: &OdeProblem<F, P>,
    candidate: &mut [f64],
    state: &[f64],
    time: f64,
    step: f64,
    options: &SolveOptions,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<f64, SolveError>
where
    F: crate::OdeFunction<P>,
{
    let tableau = workspace.pair_tableau.ok_or(SolveError::InvalidTableau)?;
    let dimension = state.len();
    prepare_factorization(
        problem,
        state,
        time,
        step,
        tableau.gamma(),
        workspace,
        stats,
    )?;
    for index in 0..dimension {
        workspace.right_hand_side[index] = tableau.derivative()[0][0]
            * workspace.current_derivative[index]
            + tableau.time_derivative()[0] * step * workspace.time_derivative[index];
    }
    solve_factorized(
        &workspace.factorization,
        &workspace.pivots,
        &mut workspace.right_hand_side,
        dimension,
    );
    workspace.stages[..dimension].copy_from_slice(&workspace.right_hand_side);
    stats.linear_solves += 1;

    for (index, &value) in state.iter().enumerate() {
        workspace.stage_state[index] =
            value + tableau.state()[1][0] * step * workspace.stages[index];
    }
    evaluate(
        problem,
        &mut workspace.stage_derivative,
        &workspace.stage_state,
        time + tableau.nodes()[1] * step,
        stats,
    )?;
    for index in 0..dimension {
        workspace.right_hand_side[index] = tableau.derivative()[1][1]
            * workspace.stage_derivative[index]
            + tableau.stage()[1][0] * workspace.stages[index];
    }
    solve_factorized(
        &workspace.factorization,
        &workspace.pivots,
        &mut workspace.right_hand_side,
        dimension,
    );
    for (index, &value) in state.iter().enumerate() {
        workspace.stages[dimension + index] =
            workspace.right_hand_side[index] + tableau.post_solve()[1][0] * workspace.stages[index];
        workspace.stage_state[index] =
            value + step * tableau.state()[2][1] * workspace.stages[dimension + index];
    }
    stats.linear_solves += 1;

    evaluate(
        problem,
        &mut workspace.error,
        &workspace.stage_state,
        time + tableau.nodes()[2] * step,
        stats,
    )?;
    for index in 0..dimension {
        workspace.right_hand_side[index] = tableau.derivative()[2][0]
            * workspace.current_derivative[index]
            + tableau.derivative()[2][1] * workspace.stage_derivative[index]
            + tableau.derivative()[2][2] * workspace.error[index]
            + tableau.stage()[2][0] * workspace.stages[index]
            + tableau.stage()[2][1] * workspace.stages[dimension + index]
            + tableau.time_derivative()[2] * step * workspace.time_derivative[index];
    }
    solve_factorized(
        &workspace.factorization,
        &workspace.pivots,
        &mut workspace.right_hand_side,
        dimension,
    );
    workspace.stages[2 * dimension..3 * dimension].copy_from_slice(&workspace.right_hand_side);
    stats.linear_solves += 1;

    for (index, &value) in state.iter().enumerate() {
        candidate[index] = value
            + step
                * (tableau.third_order()[0] * workspace.stages[index]
                    + tableau.third_order()[1] * workspace.stages[dimension + index]
                    + tableau.third_order()[2] * workspace.stages[2 * dimension + index]);
        workspace.error[index] = step
            * (tableau.error()[0] * workspace.stages[index]
                + tableau.error()[1] * workspace.stages[dimension + index]
                + tableau.error()[2] * workspace.stages[2 * dimension + index]);
    }
    Ok(if options.adaptive {
        scaled_error_norm(state, candidate, &workspace.error, options)
    } else {
        0.0
    })
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub(super) fn perform_rodas<F, P>(
    problem: &OdeProblem<F, P>,
    candidate: &mut [f64],
    state: &[f64],
    time: f64,
    step: f64,
    options: &SolveOptions,
    tableau: &'static RosenbrockTableau,
    residual_control: bool,
    adaptive_error_estimator: AdaptiveErrorEstimator,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<f64, SolveError>
where
    F: crate::OdeFunction<P>,
{
    if residual_control && tableau.h().len() != 3 {
        return Err(SolveError::InvalidTableau);
    }
    if options.adaptive && tableau.btilde().is_none() {
        return Err(SolveError::InvalidTableau);
    }
    let dimension = state.len();
    prepare_factorization(
        problem,
        state,
        time,
        step,
        tableau.gamma(),
        workspace,
        stats,
    )?;
    for stage in 0..tableau.stages() {
        workspace.stage_state.copy_from_slice(state);
        for previous in 0..stage {
            let coefficient = tableau.a()[stage][previous];
            if coefficient != 0.0 {
                for component in 0..dimension {
                    workspace.stage_state[component] +=
                        coefficient * workspace.stages[previous * dimension + component];
                }
            }
        }
        if stage == 0 {
            workspace
                .stage_derivative
                .copy_from_slice(&workspace.current_derivative);
        } else {
            evaluate(
                problem,
                &mut workspace.stage_derivative,
                &workspace.stage_state,
                time + tableau.c()[stage] * step,
                stats,
            )?;
        }
        for component in 0..dimension {
            workspace.right_hand_side[component] = workspace.stage_derivative[component]
                + step * tableau.d()[stage] * workspace.time_derivative[component];
        }
        for previous in 0..stage {
            let coefficient = tableau.coupling()[stage][previous] / step;
            if coefficient != 0.0 {
                for component in 0..dimension {
                    workspace.right_hand_side[component] +=
                        coefficient * workspace.stages[previous * dimension + component];
                }
            }
        }
        solve_factorized(
            &workspace.factorization,
            &workspace.pivots,
            &mut workspace.right_hand_side,
            dimension,
        );
        for component in 0..dimension {
            workspace.stages[stage * dimension + component] =
                step * tableau.gamma() * workspace.right_hand_side[component];
        }
        stats.linear_solves += 1;
    }

    candidate.copy_from_slice(state);
    workspace.error.fill(0.0);
    for stage in 0..tableau.stages() {
        for component in 0..dimension {
            let increment = workspace.stages[stage * dimension + component];
            candidate[component] += tableau.b()[stage] * increment;
            if let Some(error) = tableau.btilde() {
                workspace.error[component] += error[stage] * increment;
            }
        }
    }
    let mut error_estimate = if options.adaptive {
        match adaptive_error_estimator {
            AdaptiveErrorEstimator::Embedded => {
                scaled_error_norm(state, candidate, &workspace.error, options)
            }
            AdaptiveErrorEstimator::RichardsonStepDoubling { method_order } => {
                richardson_step_doubling(
                    problem,
                    candidate,
                    state,
                    time,
                    step,
                    options,
                    tableau,
                    method_order,
                    workspace,
                    stats,
                )?
            }
        }
    } else {
        0.0
    };

    // OrdinaryDiffEq's Rodas5Pr performs an additional residual check only
    // when the embedded estimate accepts the step. The three H rows are the
    // pinned Rodas5P dense-output weights; use otherwise-idle stage buffers
    // here so this check remains allocation-free without clobbering the
    // current derivative needed if the integrator rejects and retries.
    if residual_control && options.adaptive && error_estimate < 1.0 {
        let dimension = state.len();
        for component in 0..dimension {
            workspace.error[component] = 0.0;
            workspace.stage_derivative[component] = 0.0;
            workspace.perturbed_derivative[component] = 0.0;
            for stage in 0..tableau.stages() {
                let increment = workspace.stages[stage * dimension + component];
                workspace.error[component] += tableau.h()[0][stage] * increment;
                workspace.stage_derivative[component] += tableau.h()[1][stage] * increment;
                workspace.perturbed_derivative[component] += tableau.h()[2][stage] * increment;
            }
            workspace.perturbed_state[component] = 0.5
                * (state[component]
                    + candidate[component]
                    + 0.5
                        * (workspace.error[component]
                            + 0.5
                                * (workspace.stage_derivative[component]
                                    + 0.5 * workspace.perturbed_derivative[component])));
            workspace.right_hand_side[component] = 0.25
                * (workspace.stage_derivative[component]
                    + workspace.perturbed_derivative[component])
                - state[component]
                + candidate[component];
            workspace.right_hand_side[component] /= step;
        }
        evaluate(
            problem,
            &mut workspace.stage_derivative,
            &workspace.perturbed_state,
            time + 0.5 * step,
            stats,
        )?;
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for component in 0..dimension {
            let residual =
                workspace.right_hand_side[component] - workspace.stage_derivative[component];
            let scale = options.absolute_tolerance
                + options.relative_tolerance * workspace.perturbed_state[component].abs();
            numerator += residual * residual;
            denominator += scale * scale;
        }
        if denominator > 0.0 {
            error_estimate = error_estimate.max((numerator / denominator).sqrt());
        }
    }
    Ok(error_estimate)
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub(super) fn perform_tsit5da<F, P>(
    problem: &OdeProblem<F, P>,
    candidate: &mut [f64],
    state: &[f64],
    time: f64,
    step: f64,
    options: &SolveOptions,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<f64, SolveError>
where
    F: crate::OdeFunction<P>,
{
    let tableau = workspace.tableau.ok_or(SolveError::InvalidTableau)?;
    let error = tableau.btilde().ok_or(SolveError::InvalidTableau)?;
    let dimension = state.len();

    for stage in 0..tableau.stages() {
        workspace.stage_state.copy_from_slice(state);
        for previous in 0..stage {
            let coefficient = tableau.a()[stage][previous];
            if coefficient != 0.0 {
                for component in 0..dimension {
                    workspace.stage_state[component] +=
                        coefficient * workspace.stages[previous * dimension + component];
                }
            }
        }

        if stage == 0 {
            workspace
                .stage_derivative
                .copy_from_slice(&workspace.current_derivative);
        } else {
            evaluate(
                problem,
                &mut workspace.stage_derivative,
                &workspace.stage_state,
                time + tableau.c()[stage] * step,
                stats,
            )?;
        }
        for component in 0..dimension {
            workspace.stages[stage * dimension + component] =
                step * workspace.stage_derivative[component];
        }
    }

    candidate.copy_from_slice(state);
    workspace.error.fill(0.0);
    for stage in 0..tableau.stages() {
        for component in 0..dimension {
            let increment = workspace.stages[stage * dimension + component];
            candidate[component] += tableau.b()[stage] * increment;
            workspace.error[component] += error[stage] * increment;
        }
    }

    Ok(if options.adaptive {
        scaled_error_norm(state, candidate, &workspace.error, options)
    } else {
        0.0
    })
}

/// Estimates local error with two half steps when selected by a method.
///
/// For a method of order `p`, the difference between one full step and two
/// half steps is `(2^p - 1)` times the local error of the refined solution to
/// leading order. The refined solution is retained as the candidate, making
/// this an asymptotically valid fallback rather than a lower-order defect
/// heuristic. This path is intentionally selected per method. `ROS34PW1a`
/// uses it consistently because its published embedded combination has a
/// scalar-linear cancellation blind spot; consistently using Richardson also
/// avoids switching estimators when numerical Jacobians perturb that zero.
#[allow(clippy::too_many_arguments)]
fn richardson_step_doubling<F, P>(
    problem: &OdeProblem<F, P>,
    candidate: &mut [f64],
    state: &[f64],
    time: f64,
    step: f64,
    options: &SolveOptions,
    tableau: &'static RosenbrockTableau,
    method_order: i32,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<f64, SolveError>
where
    F: crate::OdeFunction<P>,
{
    let dimension = state.len();
    let half_step = step / 2.0;
    let mut midpoint = vec![0.0; dimension];
    let mut refined_candidate = vec![0.0; dimension];
    let mut refinement_workspace = Workspace::new(dimension, Some(tableau), None);
    let mut fixed_options = options.clone();
    fixed_options.adaptive = false;

    evaluate(
        problem,
        &mut refinement_workspace.current_derivative,
        state,
        time,
        stats,
    )?;
    perform_rodas(
        problem,
        &mut midpoint,
        state,
        time,
        half_step,
        &fixed_options,
        tableau,
        false,
        AdaptiveErrorEstimator::Embedded,
        &mut refinement_workspace,
        stats,
    )?;

    refinement_workspace.differentiation_valid = false;
    evaluate(
        problem,
        &mut refinement_workspace.current_derivative,
        &midpoint,
        time + half_step,
        stats,
    )?;
    perform_rodas(
        problem,
        &mut refined_candidate,
        &midpoint,
        time + half_step,
        half_step,
        &fixed_options,
        tableau,
        false,
        AdaptiveErrorEstimator::Embedded,
        &mut refinement_workspace,
        stats,
    )?;

    let richardson_denominator = 2.0_f64.powi(method_order) - 1.0;
    for component in 0..dimension {
        workspace.error[component] =
            (refined_candidate[component] - candidate[component]) / richardson_denominator;
    }
    candidate.copy_from_slice(&refined_candidate);
    Ok(scaled_error_norm(
        state,
        candidate,
        &workspace.error,
        options,
    ))
}

#[allow(clippy::too_many_arguments)]
fn prepare_factorization<F, P>(
    problem: &OdeProblem<F, P>,
    state: &[f64],
    time: f64,
    step: f64,
    gamma: f64,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: crate::OdeFunction<P>,
{
    if !workspace.differentiation_valid {
        differentiate(problem, state, time, step, workspace, stats)?;
        workspace.differentiation_valid = true;
    }
    let dimension = state.len();
    for row in 0..dimension {
        for column in 0..dimension {
            workspace.factorization[row * dimension + column] = f64::from(row == column)
                - gamma * step * workspace.jacobian[row * dimension + column];
        }
    }
    factorize(
        &mut workspace.factorization,
        &mut workspace.pivots,
        dimension,
    )?;
    stats.linear_factorizations += 1;
    Ok(())
}

fn differentiate<F, P>(
    problem: &OdeProblem<F, P>,
    state: &[f64],
    time: f64,
    step: f64,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: crate::OdeFunction<P>,
{
    let dimension = state.len();
    if problem.evaluate_jacobian(&mut workspace.jacobian, state, time) {
        ensure_finite(&workspace.jacobian)?;
    } else {
        for column in 0..dimension {
            workspace.perturbed_state.copy_from_slice(state);
            let perturbation = f64::EPSILON.sqrt() * state[column].abs().max(1.0);
            workspace.perturbed_state[column] += perturbation;
            evaluate_unchecked(
                problem,
                &mut workspace.perturbed_derivative,
                &workspace.perturbed_state,
                time,
                stats,
            )?;
            for row in 0..dimension {
                workspace.jacobian[row * dimension + column] =
                    (workspace.perturbed_derivative[row] - workspace.current_derivative[row])
                        / perturbation;
            }
        }
        ensure_finite(&workspace.jacobian)?;
    }
    let (first, second) = crate::stepping::TimeDifferencePolicy::default()
        .probes(time, step)
        .map_err(|_| SolveError::NonFiniteDerivative)?;
    evaluate_unchecked(
        problem,
        &mut workspace.perturbed_derivative,
        state,
        first,
        stats,
    )?;
    workspace
        .time_derivative
        .copy_from_slice(&workspace.perturbed_derivative);
    if let Some(second) = second {
        evaluate_unchecked(
            problem,
            &mut workspace.perturbed_derivative,
            state,
            second,
            stats,
        )?;
    }
    for component in 0..dimension {
        workspace.time_derivative[component] = crate::stepping::time_difference::partial(
            workspace.current_derivative[component],
            workspace.time_derivative[component],
            second.map(|_| workspace.perturbed_derivative[component]),
            first - time,
            second.map(|t| t - time),
        );
    }
    ensure_finite(&workspace.time_derivative)?;
    stats.jacobian_evaluations += 1;
    Ok(())
}

fn scaled_error_norm(
    state: &[f64],
    candidate: &[f64],
    error: &[f64],
    options: &SolveOptions,
) -> f64 {
    let mut squared_norm = 0.0;
    for ((&value, &candidate), &error) in state.iter().zip(candidate).zip(error) {
        let scale = options.absolute_tolerance
            + options.relative_tolerance * value.abs().max(candidate.abs());
        squared_norm += (error / scale).powi(2);
    }
    (squared_norm / state.len() as f64).sqrt()
}

fn evaluate_unchecked<F, P>(
    problem: &OdeProblem<F, P>,
    derivative: &mut [f64],
    state: &[f64],
    time: f64,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: crate::OdeFunction<P>,
{
    problem
        .rhs
        .evaluate(derivative, state, problem.parameters(), time)?;
    stats.rhs_evaluations += 1;
    Ok(())
}

pub(super) fn evaluate<F, P>(
    problem: &OdeProblem<F, P>,
    derivative: &mut [f64],
    state: &[f64],
    time: f64,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: crate::OdeFunction<P>,
{
    evaluate_unchecked(problem, derivative, state, time, stats)?;
    ensure_finite(derivative)
}

fn ensure_finite(values: &[f64]) -> Result<(), SolveError> {
    values
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteDerivative)
}

pub(super) fn estimate_initial_step(
    state: &[f64],
    derivative: &[f64],
    options: &SolveOptions,
    maximum_step: f64,
) -> f64 {
    let mut state_norm = 0.0;
    let mut derivative_norm = 0.0;
    for (value, derivative) in state.iter().zip(derivative) {
        let scale = options.absolute_tolerance + options.relative_tolerance * value.abs();
        state_norm += (value / scale).powi(2);
        derivative_norm += (derivative / scale).powi(2);
    }
    let dimension = state.len() as f64;
    state_norm = (state_norm / dimension).sqrt();
    derivative_norm = (derivative_norm / dimension).sqrt();
    if state_norm < 1.0e-5 || derivative_norm < 1.0e-5 {
        1.0e-6_f64.min(maximum_step)
    } else {
        (0.01 * state_norm / derivative_norm).min(maximum_step)
    }
}
