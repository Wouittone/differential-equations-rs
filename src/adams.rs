use crate::{OdeAlgorithm, OdeProblem, SaveMode, Solution, SolveError, SolveOptions, SolverStats};

const AB3_WEIGHTS: &[f64] = &[23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0];
const AB4_WEIGHTS: &[f64] = &[55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -9.0 / 24.0];
const AB5_WEIGHTS: &[f64] = &[
    1_901.0 / 720.0,
    -2_774.0 / 720.0,
    2_616.0 / 720.0,
    -1_274.0 / 720.0,
    251.0 / 720.0,
];

struct AdamsBashforth {
    order: usize,
    weights: &'static [f64],
    bootstrap: Bootstrap,
}

#[derive(Clone, Copy)]
enum Bootstrap {
    Ralston,
    Rk4,
}

const AB3_METHOD: AdamsBashforth = AdamsBashforth {
    order: 3,
    weights: AB3_WEIGHTS,
    bootstrap: Bootstrap::Ralston,
};
const AB4_METHOD: AdamsBashforth = AdamsBashforth {
    order: 4,
    weights: AB4_WEIGHTS,
    bootstrap: Bootstrap::Rk4,
};
const AB5_METHOD: AdamsBashforth = AdamsBashforth {
    order: 5,
    weights: AB5_WEIGHTS,
    bootstrap: Bootstrap::Rk4,
};

macro_rules! algorithm {
    ($name:ident, $order:literal, $method:ident) => {
        #[doc = concat!(
                                            "The fixed-step, order-",
                                            stringify!($order),
                                            " Adams–Bashforth method."
                                        )]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        pub struct $name;

        impl OdeAlgorithm for $name {
            fn solve<F, P>(
                &self,
                problem: &OdeProblem<F, P>,
                options: &SolveOptions,
            ) -> Result<Solution, SolveError>
            where
                F: Fn(&mut [f64], &[f64], &P, f64),
            {
                integrate(problem, options, &$method)
            }
        }
    };
}

algorithm!(Ab3, 3, AB3_METHOD);
algorithm!(Ab4, 4, AB4_METHOD);
algorithm!(Ab5, 5, AB5_METHOD);

struct Workspace {
    history: Vec<Vec<f64>>,
    candidate: Vec<f64>,
    temporary: Vec<f64>,
    stage2: Vec<f64>,
    stage3: Vec<f64>,
    stage4: Vec<f64>,
    next_derivative: Vec<f64>,
}

impl Workspace {
    fn new(initial_derivative: Vec<f64>, dimension: usize, order: usize) -> Self {
        let mut history = Vec::with_capacity(order);
        history.push(initial_derivative);
        Self {
            history,
            candidate: vec![0.0; dimension],
            temporary: vec![0.0; dimension],
            stage2: vec![0.0; dimension],
            stage3: vec![0.0; dimension],
            stage4: vec![0.0; dimension],
            next_derivative: vec![0.0; dimension],
        }
    }
}

fn integrate<F, P>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
    method: &AdamsBashforth,
) -> Result<Solution, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    if options.adaptive {
        return Err(SolveError::AdaptiveStepUnsupported);
    }
    let fixed_step = options
        .initial_step
        .ok_or(SolveError::InitialStepRequired)?;
    let dimension = problem.initial_state().len();
    let (start, end) = problem.time_span();
    let direction = (end - start).signum();
    let maximum_step = options.max_step.min(fixed_step);
    let mut state = problem.initial_state().to_vec();
    let mut stats = SolverStats::default();
    let mut initial_derivative = vec![0.0; dimension];
    evaluate(problem, &mut initial_derivative, &state, start, &mut stats)?;
    let mut workspace = Workspace::new(initial_derivative, dimension, method.order);

    let mut time = start;
    let mut times = vec![start];
    let mut values = Vec::with_capacity(2 * dimension);
    values.extend_from_slice(&state);
    let mut steps = 0;

    while direction * (end - time) > 0.0 {
        if steps == options.max_steps {
            return Err(SolveError::MaxStepsExceeded);
        }
        steps += 1;
        let step = direction * maximum_step.min((end - time).abs());
        if time + step == time {
            return Err(SolveError::StepSizeUnderflow);
        }

        if workspace.history.len() < method.order {
            bootstrap_step(
                problem,
                &state,
                time,
                step,
                method.bootstrap,
                &mut workspace,
                &mut stats,
            )?;
        } else {
            for index in 0..dimension {
                let increment = workspace
                    .history
                    .iter()
                    .zip(method.weights)
                    .map(|(derivative, weight)| weight * derivative[index])
                    .sum::<f64>();
                workspace.candidate[index] = state[index] + step * increment;
            }
        }

        time += step;
        if direction * (end - time) <= 0.0 {
            time = end;
        }
        evaluate(
            problem,
            &mut workspace.next_derivative,
            &workspace.candidate,
            time,
            &mut stats,
        )?;
        std::mem::swap(&mut state, &mut workspace.candidate);
        workspace
            .history
            .insert(0, std::mem::take(&mut workspace.next_derivative));
        if workspace.history.len() > method.order {
            workspace.next_derivative = workspace
                .history
                .pop()
                .expect("history exceeds method order");
        } else {
            workspace.next_derivative = vec![0.0; dimension];
        }
        stats.accepted_steps += 1;

        if options.save == SaveMode::EveryStep || time == end {
            times.push(time);
            values.extend_from_slice(&state);
        }
    }

    Ok(Solution::new(times, values, dimension, stats))
}

fn bootstrap_step<F, P>(
    problem: &OdeProblem<F, P>,
    state: &[f64],
    time: f64,
    step: f64,
    method: Bootstrap,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    match method {
        Bootstrap::Ralston => {
            for (index, &value) in state.iter().enumerate() {
                workspace.temporary[index] =
                    value + (2.0 / 3.0) * step * workspace.history[0][index];
            }
            evaluate(
                problem,
                &mut workspace.stage2,
                &workspace.temporary,
                time + (2.0 / 3.0) * step,
                stats,
            )?;
            for (index, &value) in state.iter().enumerate() {
                workspace.candidate[index] = value
                    + (step / 4.0) * (workspace.history[0][index] + 3.0 * workspace.stage2[index]);
            }
        }
        Bootstrap::Rk4 => {
            for (index, &value) in state.iter().enumerate() {
                workspace.temporary[index] = value + 0.5 * step * workspace.history[0][index];
            }
            evaluate(
                problem,
                &mut workspace.stage2,
                &workspace.temporary,
                time + 0.5 * step,
                stats,
            )?;
            for (index, &value) in state.iter().enumerate() {
                workspace.temporary[index] = value + 0.5 * step * workspace.stage2[index];
            }
            evaluate(
                problem,
                &mut workspace.stage3,
                &workspace.temporary,
                time + 0.5 * step,
                stats,
            )?;
            for (index, &value) in state.iter().enumerate() {
                workspace.temporary[index] = value + step * workspace.stage3[index];
            }
            evaluate(
                problem,
                &mut workspace.stage4,
                &workspace.temporary,
                time + step,
                stats,
            )?;
            for (index, &value) in state.iter().enumerate() {
                workspace.candidate[index] = value
                    + (step / 6.0)
                        * (workspace.history[0][index]
                            + 2.0 * workspace.stage2[index]
                            + 2.0 * workspace.stage3[index]
                            + workspace.stage4[index]);
            }
        }
    }
    Ok(())
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
    derivative
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteDerivative)
}

#[cfg(test)]
mod tests {
    use std::f64::consts::E;

    use crate::{Ab3, Ab4, Ab5, OdeProblem, SaveMode, SolveOptions, solve};

    type TestRhs = fn(&mut [f64], &[f64], &(), f64);

    fn exponential() -> OdeProblem<TestRhs, ()> {
        fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
            du[0] = u[0];
        }
        OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
    }

    #[test]
    fn fixed_adams_bashforth_methods_converge() {
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.001),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        let errors = [
            (solve(&exponential(), Ab3, &options).unwrap().last_state()[0] - E).abs(),
            (solve(&exponential(), Ab4, &options).unwrap().last_state()[0] - E).abs(),
            (solve(&exponential(), Ab5, &options).unwrap().last_state()[0] - E).abs(),
        ];

        assert!(errors[0] < 1.0e-8);
        assert!(errors[1] < 1.0e-11);
        assert!(errors[2] < 1.0e-12);
    }
}
