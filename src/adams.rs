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
const AM3_WEIGHTS: &[f64] = &[5.0 / 12.0, 8.0 / 12.0, -1.0 / 12.0];
const AM4_WEIGHTS: &[f64] = &[9.0 / 24.0, 19.0 / 24.0, -5.0 / 24.0, 1.0 / 24.0];
const AM5_WEIGHTS: &[f64] = &[
    251.0 / 720.0,
    646.0 / 720.0,
    -264.0 / 720.0,
    106.0 / 720.0,
    -19.0 / 720.0,
];

#[derive(Clone, Copy)]
struct AdamsMethod {
    order: usize,
    weights: &'static [f64],
    bootstrap: Bootstrap,
    corrector_weights: Option<&'static [f64]>,
    repeating_bootstrap_predictor: bool,
}

#[derive(Clone, Copy)]
enum Bootstrap {
    Ralston,
    Rk4,
}

const AB3_METHOD: AdamsMethod = AdamsMethod {
    order: 3,
    weights: AB3_WEIGHTS,
    bootstrap: Bootstrap::Ralston,
    corrector_weights: None,
    repeating_bootstrap_predictor: false,
};
const AB4_METHOD: AdamsMethod = AdamsMethod {
    order: 4,
    weights: AB4_WEIGHTS,
    bootstrap: Bootstrap::Rk4,
    corrector_weights: None,
    repeating_bootstrap_predictor: false,
};
const AB5_METHOD: AdamsMethod = AdamsMethod {
    order: 5,
    weights: AB5_WEIGHTS,
    bootstrap: Bootstrap::Rk4,
    corrector_weights: None,
    repeating_bootstrap_predictor: false,
};
const ABM32_METHOD: AdamsMethod = AdamsMethod {
    corrector_weights: Some(AM3_WEIGHTS),
    // OrdinaryDiffEq passes step counter 2 into a fresh AB3 predictor cache.
    repeating_bootstrap_predictor: true,
    ..AB3_METHOD
};
const ABM43_METHOD: AdamsMethod = AdamsMethod {
    corrector_weights: Some(AM4_WEIGHTS),
    // OrdinaryDiffEq likewise keeps its fresh AB4 predictor at startup step 3.
    repeating_bootstrap_predictor: true,
    ..AB4_METHOD
};
const ABM54_METHOD: AdamsMethod = AdamsMethod {
    corrector_weights: Some(AM5_WEIGHTS),
    ..AB5_METHOD
};

macro_rules! algorithm {
    ($name:ident, $documentation:literal, $method:ident) => {
        #[doc = $documentation]
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

algorithm!(
    Ab3,
    "The fixed-step, order-3 Adams–Bashforth method.",
    AB3_METHOD
);
algorithm!(
    Ab4,
    "The fixed-step, order-4 Adams–Bashforth method.",
    AB4_METHOD
);
algorithm!(
    Ab5,
    "The fixed-step, order-5 Adams–Bashforth method.",
    AB5_METHOD
);
algorithm!(
    Abm32,
    "The fixed-step, order-3 Adams–Bashforth–Moulton predictor/corrector method.",
    ABM32_METHOD
);
algorithm!(
    Abm43,
    "The fixed-step, order-4 Adams–Bashforth–Moulton predictor/corrector method.",
    ABM43_METHOD
);
algorithm!(
    Abm54,
    "The fixed-step, order-5 Adams–Bashforth–Moulton predictor/corrector method.",
    ABM54_METHOD
);

struct Workspace {
    history: Vec<Vec<f64>>,
    candidate: Vec<f64>,
    temporary: Vec<f64>,
    stage2: Vec<f64>,
    stage3: Vec<f64>,
    stage4: Vec<f64>,
    predicted_derivative: Vec<f64>,
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
            predicted_derivative: vec![0.0; dimension],
            next_derivative: vec![0.0; dimension],
        }
    }
}

fn integrate<F, P>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
    method: &AdamsMethod,
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

        if workspace.history.len() < method.order || method.repeating_bootstrap_predictor {
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

        let mut next_time = time + step;
        if direction * (end - next_time) <= 0.0 {
            next_time = end;
        }
        if let Some(corrector_weights) = method.corrector_weights
            && workspace.history.len() >= method.order - 1
        {
            evaluate(
                problem,
                &mut workspace.predicted_derivative,
                &workspace.candidate,
                next_time,
                &mut stats,
            )?;
            for index in 0..dimension {
                let history_increment = workspace
                    .history
                    .iter()
                    .zip(&corrector_weights[1..])
                    .map(|(derivative, weight)| weight * derivative[index])
                    .sum::<f64>();
                workspace.candidate[index] = state[index]
                    + step
                        * (corrector_weights[0] * workspace.predicted_derivative[index]
                            + history_increment);
            }
        }

        time = next_time;
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

    use crate::{Ab3, Ab4, Ab5, Abm32, Abm43, Abm54, OdeProblem, SaveMode, SolveOptions, solve};

    type TestRhs = fn(&mut [f64], &[f64], &(), f64);

    fn exponential() -> OdeProblem<TestRhs, ()> {
        fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
            du[0] = u[0];
        }
        OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
    }

    #[test]
    fn fixed_adams_methods_converge() {
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
            (solve(&exponential(), Abm32, &options).unwrap().last_state()[0] - E).abs(),
            (solve(&exponential(), Abm43, &options).unwrap().last_state()[0] - E).abs(),
            (solve(&exponential(), Abm54, &options).unwrap().last_state()[0] - E).abs(),
        ];

        assert!(errors[0] < 1.0e-8);
        assert!(errors[1] < 1.0e-11);
        assert!(errors[2] < 1.0e-12);
        assert!(errors[3] < 1.0e-8);
        assert!(errors[4] < 1.0e-11);
        assert!(errors[5] < 1.0e-12);
    }
}
