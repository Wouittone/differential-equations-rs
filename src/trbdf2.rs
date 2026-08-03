use crate::linear::{factorize, solve_factorized};
use crate::solution::TrajectoryRecorder;
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

const SQRT_2: f64 = std::f64::consts::SQRT_2;
const GAMMA: f64 = 2.0 - SQRT_2;
const DIAGONAL: f64 = 1.0 - SQRT_2 / 2.0;
const OMEGA: f64 = SQRT_2 / 4.0;
const ERROR_1: f64 = (1.0 - SQRT_2) / 3.0;
const ERROR_2: f64 = 1.0 / 3.0;
const ERROR_3: f64 = (SQRT_2 - 2.0) / 3.0;
const PREDICT_1: f64 = -SQRT_2 / 2.0;
const PREDICT_2: f64 = 1.0 + SQRT_2 / 2.0;

const MAX_NEWTON_ITERATIONS: usize = 12;
const NEWTON_TOLERANCE: f64 = 1.0e-12;
const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 6.0;

/// The adaptive, second-order TR-BDF2 ESDIRK method for stiff ODEs.
///
/// This is the native ODE form of Julia OrdinaryDiffEq's `TRBDF2`: it uses
/// the Hosea-Shampine three-stage tableau and the method's smoothed embedded
/// error estimate. Both adaptive and fixed-step integration are supported.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Trbdf2;

impl OdeAlgorithm for Trbdf2 {
    fn solve<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        integrate(problem, options)
    }
}

struct Workspace {
    current_derivative: Vec<f64>,
    stage_base: Vec<f64>,
    stage_state: Vec<f64>,
    stage_derivative: Vec<f64>,
    perturbed_state: Vec<f64>,
    perturbed_derivative: Vec<f64>,
    z1: Vec<f64>,
    z2: Vec<f64>,
    z3: Vec<f64>,
    candidate: Vec<f64>,
    residual: Vec<f64>,
    correction: Vec<f64>,
    error: Vec<f64>,
    matrix: Vec<f64>,
    pivots: Vec<usize>,
    factorization_valid: bool,
}

impl Workspace {
    fn new(dimension: usize) -> Self {
        Self {
            current_derivative: vec![0.0; dimension],
            stage_base: vec![0.0; dimension],
            stage_state: vec![0.0; dimension],
            stage_derivative: vec![0.0; dimension],
            perturbed_state: vec![0.0; dimension],
            perturbed_derivative: vec![0.0; dimension],
            z1: vec![0.0; dimension],
            z2: vec![0.0; dimension],
            z3: vec![0.0; dimension],
            candidate: vec![0.0; dimension],
            residual: vec![0.0; dimension],
            correction: vec![0.0; dimension],
            error: vec![0.0; dimension],
            matrix: vec![0.0; dimension * dimension],
            pivots: vec![0; dimension],
            factorization_valid: false,
        }
    }
}

fn integrate<F, P>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
) -> Result<Solution, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let dimension = problem.initial_state().len();
    let (start, end) = problem.time_span();
    let direction = (end - start).signum();
    let interval = (end - start).abs();
    let maximum_step = options.max_step.min(interval);
    let mut state = problem.initial_state().to_vec();
    let mut state_before_effect = if problem.has_callbacks() {
        vec![0.0; dimension]
    } else {
        Vec::new()
    };
    let mut workspace = Workspace::new(dimension);
    let mut stats = SolverStats::default();

    let initial_callbacks = problem.apply_initial_callbacks(&mut state, start)?;
    stats.callback_invocations += initial_callbacks.invocations;
    let mut recorder = TrajectoryRecorder::new(&state, start, options);
    if initial_callbacks.terminate {
        recorder.force_state(start, &state);
        return Ok(recorder.finish(stats));
    }
    evaluate_checked(
        problem,
        &mut workspace.current_derivative,
        &state,
        start,
        &mut stats,
    )?;

    let initial_step = match options.initial_step {
        Some(step) => step.min(maximum_step),
        None if !options.adaptive => return Err(SolveError::InitialStepRequired),
        None => estimate_initial_step(&state, &workspace.current_derivative, options, maximum_step),
    };
    let mut step = direction * initial_step;
    let mut time = start;
    let mut attempts = 0;
    let mut previous_rejected = false;

    while direction * (end - time) > 0.0 {
        if attempts == options.max_steps {
            return Err(SolveError::MaxStepsExceeded);
        }
        attempts += 1;
        if direction * (time + step - end) > 0.0 {
            step = end - time;
        }
        if time + step == time {
            return Err(SolveError::StepSizeUnderflow);
        }

        let error = match perform_step(
            problem,
            &state,
            time,
            step,
            options,
            &mut workspace,
            &mut stats,
        ) {
            Ok(error) => error,
            Err(SolveError::NonlinearSolveFailed | SolveError::SingularLinearSystem)
                if options.adaptive =>
            {
                stats.rejected_steps += 1;
                step *= MIN_FACTOR;
                previous_rejected = true;
                continue;
            }
            Err(error) => return Err(error),
        };

        if error <= 1.0 {
            let previous_time = time;
            let mut next_time = time + step;
            if direction * (end - next_time) <= 0.0 {
                next_time = end;
            }
            let callbacks = problem.apply_step_callbacks(
                &state,
                previous_time,
                &mut workspace.candidate,
                &mut next_time,
                &mut state_before_effect,
            )?;
            stats.callback_invocations += callbacks.invocations;
            time = next_time;
            std::mem::swap(&mut state, &mut workspace.candidate);
            stats.accepted_steps += 1;
            recorder.record_step(
                &workspace.candidate,
                previous_time,
                if callbacks.invocations == 0 {
                    &state
                } else {
                    &state_before_effect
                },
                time,
                time == end,
            );
            if callbacks.invocations > 0 {
                recorder.force_state(time, &state);
            }
            if callbacks.terminate {
                return Ok(recorder.finish(stats));
            }

            if callbacks.invocations == 0 && next_time == previous_time + step {
                for (derivative, &z) in workspace.current_derivative.iter_mut().zip(&workspace.z3) {
                    *derivative = z / step;
                }
            } else {
                evaluate_checked(
                    problem,
                    &mut workspace.current_derivative,
                    &state,
                    time,
                    &mut stats,
                )?;
            }
            if options.adaptive {
                let mut factor = step_factor(error);
                if previous_rejected {
                    factor = factor.min(1.0);
                }
                step = direction * (step.abs() * factor).min(maximum_step);
            }
            previous_rejected = false;
        } else {
            stats.rejected_steps += 1;
            step *= step_factor(error).min(1.0);
            previous_rejected = true;
        }
    }

    Ok(recorder.finish(stats))
}

fn perform_step<F, P>(
    problem: &OdeProblem<F, P>,
    state: &[f64],
    time: f64,
    step: f64,
    options: &SolveOptions,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<f64, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    for (z, &derivative) in workspace.z1.iter_mut().zip(&workspace.current_derivative) {
        *z = step * derivative;
    }

    for (index, &value) in state.iter().enumerate() {
        workspace.stage_base[index] = value + DIAGONAL * workspace.z1[index];
        workspace.z2[index] = workspace.z1[index];
    }
    solve_stage(problem, time + GAMMA * step, step, workspace, 2, stats)?;

    for (index, &value) in state.iter().enumerate() {
        workspace.stage_base[index] =
            value + OMEGA * workspace.z1[index] + OMEGA * workspace.z2[index];
        workspace.z3[index] = PREDICT_1 * workspace.z1[index] + PREDICT_2 * workspace.z2[index];
    }
    solve_stage(problem, time + step, step, workspace, 3, stats)?;

    for index in 0..state.len() {
        workspace.candidate[index] = workspace.stage_base[index] + DIAGONAL * workspace.z3[index];
    }
    if !options.adaptive {
        return Ok(0.0);
    }

    for index in 0..state.len() {
        workspace.error[index] = ERROR_1 * workspace.z1[index]
            + ERROR_2 * workspace.z2[index]
            + ERROR_3 * workspace.z3[index];
    }

    // OrdinaryDiffEq's TRBDF2 defaults `smooth_est=true`: apply the last
    // Newton matrix to the embedded estimate before tolerance scaling.
    if !workspace.factorization_valid {
        workspace.stage_state.copy_from_slice(&workspace.candidate);
        workspace.stage_derivative.copy_from_slice(&workspace.z3);
        for derivative in &mut workspace.stage_derivative {
            *derivative /= step;
        }
        build_factorization(problem, time + step, step, workspace, stats)?;
    }
    solve_factorized(
        &workspace.matrix,
        &workspace.pivots,
        &mut workspace.error,
        state.len(),
    );
    stats.linear_solves += 1;

    let mut squared_norm = 0.0;
    for (index, &value) in state.iter().enumerate() {
        let scale = options.absolute_tolerance
            + options.relative_tolerance * value.abs().max(workspace.candidate[index].abs());
        squared_norm += (workspace.error[index] / scale).powi(2);
    }
    Ok((squared_norm / state.len() as f64).sqrt())
}

fn solve_stage<F, P>(
    problem: &OdeProblem<F, P>,
    stage_time: f64,
    step: f64,
    workspace: &mut Workspace,
    stage: u8,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    workspace.factorization_valid = false;
    let dimension = workspace.z2.len();
    for _ in 0..MAX_NEWTON_ITERATIONS {
        stats.nonlinear_iterations += 1;
        for index in 0..dimension {
            let z = if stage == 2 {
                workspace.z2[index]
            } else {
                workspace.z3[index]
            };
            workspace.stage_state[index] = workspace.stage_base[index] + DIAGONAL * z;
        }
        evaluate_checked(
            problem,
            &mut workspace.stage_derivative,
            &workspace.stage_state,
            stage_time,
            stats,
        )?;
        let mut residual_norm: f64 = 0.0;
        for index in 0..dimension {
            let z = if stage == 2 {
                workspace.z2[index]
            } else {
                workspace.z3[index]
            };
            workspace.residual[index] = z - step * workspace.stage_derivative[index];
            residual_norm = residual_norm.max(workspace.residual[index].abs());
        }
        let state_scale = 1.0 + infinity_norm(&workspace.stage_state);
        if residual_norm <= NEWTON_TOLERANCE * state_scale {
            return Ok(());
        }

        build_factorization(problem, stage_time, step, workspace, stats)?;
        for (correction, &residual) in workspace.correction.iter_mut().zip(&workspace.residual) {
            *correction = -residual;
        }
        solve_factorized(
            &workspace.matrix,
            &workspace.pivots,
            &mut workspace.correction,
            dimension,
        );
        stats.linear_solves += 1;
        let correction = &workspace.correction;
        let z = if stage == 2 {
            &mut workspace.z2
        } else {
            &mut workspace.z3
        };
        for (value, &delta) in z.iter_mut().zip(correction) {
            *value += delta;
        }
    }
    Err(SolveError::NonlinearSolveFailed)
}

fn build_factorization<F, P>(
    problem: &OdeProblem<F, P>,
    evaluation_time: f64,
    step: f64,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let dimension = workspace.stage_state.len();
    workspace.factorization_valid = false;
    if problem.evaluate_jacobian(
        &mut workspace.matrix,
        &workspace.stage_state,
        evaluation_time,
    ) {
        for row in 0..dimension {
            for column in 0..dimension {
                let index = row * dimension + column;
                let derivative = workspace.matrix[index];
                if !derivative.is_finite() {
                    return Err(SolveError::NonFiniteDerivative);
                }
                workspace.matrix[index] = f64::from(row == column) - DIAGONAL * step * derivative;
            }
        }
    } else {
        for column in 0..dimension {
            workspace
                .perturbed_state
                .copy_from_slice(&workspace.stage_state);
            let perturbation = f64::EPSILON.sqrt() * workspace.stage_state[column].abs().max(1.0);
            workspace.perturbed_state[column] += perturbation;
            evaluate_checked(
                problem,
                &mut workspace.perturbed_derivative,
                &workspace.perturbed_state,
                evaluation_time,
                stats,
            )?;
            for row in 0..dimension {
                let derivative = (workspace.perturbed_derivative[row]
                    - workspace.stage_derivative[row])
                    / perturbation;
                if !derivative.is_finite() {
                    return Err(SolveError::NonFiniteDerivative);
                }
                workspace.matrix[row * dimension + column] =
                    f64::from(row == column) - DIAGONAL * step * derivative;
            }
        }
    }
    stats.jacobian_evaluations += 1;
    factorize(&mut workspace.matrix, &mut workspace.pivots, dimension)?;
    workspace.factorization_valid = true;
    Ok(())
}

fn evaluate_checked<F, P>(
    problem: &OdeProblem<F, P>,
    derivative: &mut [f64],
    state: &[f64],
    time: f64,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    (problem.rhs)(derivative, state, problem.parameters(), time);
    stats.rhs_evaluations += 1;
    derivative
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteDerivative)
}

fn estimate_initial_step(
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

fn step_factor(error: f64) -> f64 {
    if error == 0.0 {
        MAX_FACTOR
    } else if error.is_finite() {
        (SAFETY * error.powf(-1.0 / 3.0)).clamp(MIN_FACTOR, MAX_FACTOR)
    } else {
        MIN_FACTOR
    }
}

fn infinity_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::Trbdf2;
    use crate::{CallbackAction, OdeProblem, SaveMode, SolveOptions, solve};

    #[test]
    fn solves_a_stiff_nonautonomous_problem_adaptively() {
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), time: f64| {
                du[0] = -1000.0 * (u[0] - time.cos()) - time.sin();
            },
            vec![1.0],
            (0.0, 1.0),
            (),
        );
        let options = SolveOptions {
            absolute_tolerance: 1.0e-7,
            relative_tolerance: 1.0e-7,
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        let solution = solve(&problem, Trbdf2, &options).unwrap();

        assert!((solution.last_state()[0] - 1.0_f64.cos()).abs() < 3.0e-6);
        assert!(solution.stats().accepted_steps > 0);
        assert!(solution.stats().nonlinear_iterations > 0);
        assert!(solution.stats().linear_solves > 0);
    }

    #[test]
    fn has_second_order_fixed_step_convergence() {
        fn error(step: f64) -> f64 {
            let problem = OdeProblem::new(
                |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
                vec![1.0],
                (0.0, 1.0),
                (),
            );
            let options = SolveOptions {
                adaptive: false,
                initial_step: Some(step),
                save: SaveMode::Endpoints,
                ..SolveOptions::default()
            };
            (solve(&problem, Trbdf2, &options).unwrap().last_state()[0] - std::f64::consts::E).abs()
        }

        let coarse = error(0.1);
        let fine = error(0.05);
        assert!(coarse / fine > 3.7, "observed ratio {}", coarse / fine);
        assert!(coarse / fine < 4.3, "observed ratio {}", coarse / fine);
    }

    #[test]
    fn analytic_jacobian_reduces_rhs_work() {
        fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
            du[0] = -100.0 * u[0];
            du[1] = -2.0 * u[1];
        }
        type TestRhs = fn(&mut [f64], &[f64], &(), f64);
        let numeric = OdeProblem::new(rhs as TestRhs, vec![1.0, 1.0], (0.0, 0.2), ());
        let analytic = OdeProblem::new(rhs as TestRhs, vec![1.0, 1.0], (0.0, 0.2), ())
            .with_jacobian(|jacobian: &mut [f64], _: &[f64], _: &(), _: f64| {
                jacobian.copy_from_slice(&[-100.0, 0.0, 0.0, -2.0]);
            });
        let options = SolveOptions {
            absolute_tolerance: 1.0e-7,
            relative_tolerance: 1.0e-7,
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        let numeric = solve(&numeric, Trbdf2, &options).unwrap();
        let analytic = solve(&analytic, Trbdf2, &options).unwrap();

        assert!((numeric.last_state()[1] - analytic.last_state()[1]).abs() < 1.0e-8);
        assert!(analytic.stats().rhs_evaluations < numeric.stats().rhs_evaluations);
        assert!(analytic.stats().jacobian_evaluations > 0);
    }

    #[test]
    fn supports_backward_integration_callbacks_and_save_at() {
        let problem = OdeProblem::new(
            |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = 1.0,
            vec![1.0],
            (1.0, 0.0),
            (),
        )
        .with_continuous_callback(
            |u: &[f64], _: &(), _: f64| u[0] - 0.5,
            |u: &mut [f64], _: &(), _: f64| {
                u[0] = 2.0;
                CallbackAction::Terminate
            },
        );
        let options = SolveOptions {
            absolute_tolerance: 1.0e-9,
            relative_tolerance: 1.0e-9,
            save_at: vec![0.75, 0.5],
            ..SolveOptions::default()
        };

        let solution = solve(&problem, Trbdf2, &options).unwrap();

        assert_eq!(&solution.times()[..2], &[0.75, 0.5]);
        assert!((solution.times().last().unwrap() - 0.5).abs() < 1.0e-12);
        assert!((solution.state(0).unwrap()[0] - 0.75).abs() < 1.0e-10);
        assert_eq!(solution.last_state(), &[2.0]);
        assert_eq!(solution.stats().callback_invocations, 1);
    }
}
