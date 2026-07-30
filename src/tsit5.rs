use crate::{OdeAlgorithm, OdeProblem, SaveMode, Solution, SolveError, SolveOptions, SolverStats};

const C2: f64 = 0.161;
const C3: f64 = 0.327;
const C4: f64 = 0.9;
const C5: f64 = 0.980_025_540_904_509_7;

const A21: f64 = 0.161;
const A31: f64 = -0.008_480_655_492_356_989;
const A32: f64 = 0.335_480_655_492_357;
const A41: f64 = 2.897_153_057_105_493_5;
const A42: f64 = -6.359_448_489_975_075;
const A43: f64 = 4.362_295_432_869_581_5;
const A51: f64 = 5.325_864_828_439_257;
const A52: f64 = -11.748_883_564_062_828;
const A53: f64 = 7.495_539_342_889_836_5;
const A54: f64 = -0.092_495_066_361_755_25;
const A61: f64 = 5.861_455_442_946_42;
const A62: f64 = -12.920_969_317_847_11;
const A63: f64 = 8.159_367_898_576_159;
const A64: f64 = -0.071_584_973_281_401;
const A65: f64 = -0.028_269_050_394_068_383;
const A71: f64 = 0.096_460_766_818_065_23;
const A72: f64 = 0.01;
const A73: f64 = 0.479_889_650_414_499_6;
const A74: f64 = 1.379_008_574_103_742;
const A75: f64 = -3.290_069_515_436_081;
const A76: f64 = 2.324_710_524_099_774;

const ERROR_WEIGHTS: [f64; 7] = [
    -0.001_780_011_052_225_777,
    -0.000_816_434_459_656_746_9,
    0.007_880_878_010_261_995,
    -0.144_711_007_173_262_9,
    0.582_357_165_452_555_2,
    -0.458_082_105_929_186_97,
    0.015_151_515_151_515_152,
];

const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 10.0;
const ERROR_EXPONENT: f64 = -0.2;

/// The Tsitouras 5/4 explicit Runge-Kutta method.
///
/// `Tsit5` is an adaptive, FSAL (first-same-as-last) method intended for
/// non-stiff ODEs at medium tolerances.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Tsit5;

struct Workspace {
    stages: [Vec<f64>; 7],
    temporary: Vec<f64>,
    candidate: Vec<f64>,
}

impl Workspace {
    fn new(dimension: usize) -> Self {
        Self {
            stages: std::array::from_fn(|_| vec![0.0; dimension]),
            temporary: vec![0.0; dimension],
            candidate: vec![0.0; dimension],
        }
    }
}

impl OdeAlgorithm for Tsit5 {
    fn solve<F, P>(
        &self,
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

        let mut workspace = Workspace::new(dimension);
        let mut state = problem.initial_state().to_vec();
        let mut stats = SolverStats::default();
        evaluate(problem, &mut workspace.stages[0], &state, start, &mut stats)?;

        let step_magnitude = match options.initial_step {
            Some(step) => step.min(maximum_step),
            None if !options.adaptive => return Err(SolveError::InitialStepRequired),
            None => estimate_initial_step(
                problem,
                options,
                &state,
                (start, direction, maximum_step),
                &mut workspace,
                &mut stats,
            )?,
        };
        let mut step = direction * step_magnitude;
        let mut time = start;
        let (mut times, mut values) = initial_output(start, &state);
        let mut attempted_steps = 0;
        let mut previous_step_rejected = false;

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

            perform_step(problem, &state, time, step, &mut workspace, &mut stats)?;
            let error = if options.adaptive {
                error_norm(
                    &workspace.stages,
                    &state,
                    &workspace.candidate,
                    step,
                    options,
                )
            } else {
                0.0
            };

            if error <= 1.0 {
                time += step;
                if direction * (end - time) <= 0.0 {
                    time = end;
                }
                std::mem::swap(&mut state, &mut workspace.candidate);
                workspace.stages.swap(0, 6);
                stats.accepted_steps += 1;

                if options.save == SaveMode::EveryStep || time == end {
                    times.push(time);
                    values.extend_from_slice(&state);
                }

                if options.adaptive {
                    let mut factor = step_factor(error);
                    if previous_step_rejected {
                        factor = factor.min(1.0);
                    }
                    step = direction * (step.abs() * factor).min(maximum_step);
                }
                previous_step_rejected = false;
            } else {
                stats.rejected_steps += 1;
                step *= step_factor(error).min(1.0);
                previous_step_rejected = true;
            }
        }

        Ok(Solution::new(times, values, dimension, stats))
    }
}

fn initial_output(start: f64, state: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut times = Vec::with_capacity(2);
    times.push(start);
    let mut values = Vec::with_capacity(state.len() * 2);
    values.extend_from_slice(state);
    (times, values)
}

fn evaluate<F, P>(
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
    if derivative.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(SolveError::NonFiniteDerivative)
    }
}

fn estimate_initial_step<F, P>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
    state: &[f64],
    integration: (f64, f64, f64),
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<f64, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let (time, direction, maximum_step) = integration;
    let dimension = state.len() as f64;
    let mut state_norm = 0.0;
    let mut derivative_norm = 0.0;

    for (value, derivative) in state.iter().zip(&workspace.stages[0]) {
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

    for ((trial, value), derivative) in workspace
        .temporary
        .iter_mut()
        .zip(state)
        .zip(&workspace.stages[0])
    {
        *trial = value + direction * trial_step * derivative;
    }
    evaluate(
        problem,
        &mut workspace.stages[1],
        &workspace.temporary,
        time + direction * trial_step,
        stats,
    )?;

    let mut second_derivative_norm = 0.0;
    for ((next, initial), value) in workspace.stages[1]
        .iter()
        .zip(&workspace.stages[0])
        .zip(state)
    {
        let scale = options.absolute_tolerance + options.relative_tolerance * value.abs();
        second_derivative_norm += ((next - initial) / scale).powi(2);
    }
    second_derivative_norm = (second_derivative_norm / dimension).sqrt() / trial_step;

    let largest_derivative_norm = derivative_norm.max(second_derivative_norm);
    let accuracy_step = if largest_derivative_norm <= 1.0e-15 {
        (trial_step * 1.0e-3).max(1.0e-6)
    } else {
        (0.01 / largest_derivative_norm).powf(0.2)
    };

    Ok((100.0 * trial_step).min(accuracy_step).min(maximum_step))
}

fn perform_step<F, P>(
    problem: &OdeProblem<F, P>,
    state: &[f64],
    time: f64,
    step: f64,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    combine(
        &mut workspace.temporary,
        state,
        step,
        &[(&workspace.stages[0], A21)],
    );
    evaluate(
        problem,
        &mut workspace.stages[1],
        &workspace.temporary,
        time + C2 * step,
        stats,
    )?;

    combine(
        &mut workspace.temporary,
        state,
        step,
        &[(&workspace.stages[0], A31), (&workspace.stages[1], A32)],
    );
    evaluate(
        problem,
        &mut workspace.stages[2],
        &workspace.temporary,
        time + C3 * step,
        stats,
    )?;

    combine(
        &mut workspace.temporary,
        state,
        step,
        &[
            (&workspace.stages[0], A41),
            (&workspace.stages[1], A42),
            (&workspace.stages[2], A43),
        ],
    );
    evaluate(
        problem,
        &mut workspace.stages[3],
        &workspace.temporary,
        time + C4 * step,
        stats,
    )?;

    combine(
        &mut workspace.temporary,
        state,
        step,
        &[
            (&workspace.stages[0], A51),
            (&workspace.stages[1], A52),
            (&workspace.stages[2], A53),
            (&workspace.stages[3], A54),
        ],
    );
    evaluate(
        problem,
        &mut workspace.stages[4],
        &workspace.temporary,
        time + C5 * step,
        stats,
    )?;

    combine(
        &mut workspace.temporary,
        state,
        step,
        &[
            (&workspace.stages[0], A61),
            (&workspace.stages[1], A62),
            (&workspace.stages[2], A63),
            (&workspace.stages[3], A64),
            (&workspace.stages[4], A65),
        ],
    );
    evaluate(
        problem,
        &mut workspace.stages[5],
        &workspace.temporary,
        time + step,
        stats,
    )?;

    combine(
        &mut workspace.candidate,
        state,
        step,
        &[
            (&workspace.stages[0], A71),
            (&workspace.stages[1], A72),
            (&workspace.stages[2], A73),
            (&workspace.stages[3], A74),
            (&workspace.stages[4], A75),
            (&workspace.stages[5], A76),
        ],
    );
    evaluate(
        problem,
        &mut workspace.stages[6],
        &workspace.candidate,
        time + step,
        stats,
    )
}

fn combine(output: &mut [f64], state: &[f64], step: f64, terms: &[(&[f64], f64)]) {
    for (index, output_value) in output.iter_mut().enumerate() {
        let increment = terms
            .iter()
            .map(|(stage, coefficient)| coefficient * stage[index])
            .sum::<f64>();
        *output_value = state[index] + step * increment;
    }
}

fn error_norm(
    stages: &[Vec<f64>; 7],
    state: &[f64],
    candidate: &[f64],
    step: f64,
    options: &SolveOptions,
) -> f64 {
    let mut squared_norm = 0.0;
    for index in 0..state.len() {
        let error = step
            * stages
                .iter()
                .zip(ERROR_WEIGHTS)
                .map(|(stage, weight)| weight * stage[index])
                .sum::<f64>();
        let scale = options.absolute_tolerance
            + options.relative_tolerance * state[index].abs().max(candidate[index].abs());
        squared_norm += (error / scale).powi(2);
    }
    (squared_norm / state.len() as f64).sqrt()
}

fn step_factor(error: f64) -> f64 {
    if error == 0.0 {
        MAX_FACTOR
    } else if error.is_finite() {
        (SAFETY * error.powf(ERROR_EXPONENT)).clamp(MIN_FACTOR, MAX_FACTOR)
    } else {
        MIN_FACTOR
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{E, TAU};

    use crate::{OdeProblem, SaveMode, SolveError, SolveOptions, Tsit5, solve};

    #[test]
    fn solves_scalar_exponential_growth() {
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
            vec![1.0],
            (0.0, 1.0),
            (),
        );
        let options = SolveOptions {
            absolute_tolerance: 1.0e-11,
            relative_tolerance: 1.0e-11,
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        let solution = solve(&problem, Tsit5, &options).unwrap();

        assert!((solution.last_state()[0] - E).abs() < 2.0e-10);
        assert_eq!(solution.times(), &[0.0, 1.0]);
        assert!(solution.stats().accepted_steps > 0);
    }

    #[test]
    fn solves_a_vector_harmonic_oscillator() {
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| {
                du[0] = u[1];
                du[1] = -u[0];
            },
            vec![1.0, 0.0],
            (0.0, TAU),
            (),
        );
        let options = SolveOptions {
            absolute_tolerance: 1.0e-10,
            relative_tolerance: 1.0e-10,
            ..SolveOptions::default()
        };

        let solution = solve(&problem, Tsit5, &options).unwrap();

        assert!((solution.last_state()[0] - 1.0).abs() < 2.0e-9);
        assert!(solution.last_state()[1].abs() < 2.0e-9);
        assert_eq!(solution.times().len(), solution.stats().accepted_steps + 1);
    }

    #[test]
    fn supports_backward_integration() {
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
            vec![E],
            (1.0, 0.0),
            (),
        );
        let options = SolveOptions {
            absolute_tolerance: 1.0e-10,
            relative_tolerance: 1.0e-10,
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        let solution = solve(&problem, Tsit5, &options).unwrap();

        assert!((solution.last_state()[0] - 1.0).abs() < 2.0e-9);
        assert_eq!(solution.times(), &[1.0, 0.0]);
    }

    #[test]
    fn rejects_an_overly_large_initial_step() {
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
            vec![1.0],
            (0.0, 10.0),
            (),
        );
        let options = SolveOptions {
            absolute_tolerance: 1.0e-12,
            relative_tolerance: 1.0e-12,
            initial_step: Some(10.0),
            ..SolveOptions::default()
        };

        let solution = solve(&problem, Tsit5, &options).unwrap();

        assert!(solution.stats().rejected_steps > 0);
        assert_eq!(
            solution.stats().rhs_evaluations,
            1 + 6 * (solution.stats().accepted_steps + solution.stats().rejected_steps)
        );
    }

    #[test]
    fn reports_non_finite_derivatives() {
        let problem = OdeProblem::new(
            |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = f64::NAN,
            vec![1.0],
            (0.0, 1.0),
            (),
        );

        assert_eq!(
            solve(&problem, Tsit5, &SolveOptions::default()),
            Err(SolveError::NonFiniteDerivative)
        );
    }
}
