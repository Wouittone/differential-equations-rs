use crate::integrator::{KernelCapabilities, StepEstimate, StepKernel, integrate};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

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
                integrate(problem, options, AdamsKernel::new(&$method))
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
    history_len: usize,
    temporary: Vec<f64>,
    stage2: Vec<f64>,
    stage3: Vec<f64>,
    stage4: Vec<f64>,
    predicted_derivative: Vec<f64>,
    next_derivative: Vec<f64>,
}

impl Workspace {
    fn new(dimension: usize, order: usize) -> Self {
        Self {
            history: (0..order).map(|_| vec![0.0; dimension]).collect(),
            history_len: 0,
            temporary: vec![0.0; dimension],
            stage2: vec![0.0; dimension],
            stage3: vec![0.0; dimension],
            stage4: vec![0.0; dimension],
            predicted_derivative: vec![0.0; dimension],
            next_derivative: vec![0.0; dimension],
        }
    }
}

struct AdamsKernel {
    method: &'static AdamsMethod,
    workspace: Option<Workspace>,
}

impl AdamsKernel {
    const fn new(method: &'static AdamsMethod) -> Self {
        Self {
            method,
            workspace: None,
        }
    }

    fn workspace(&mut self) -> &mut Workspace {
        self.workspace
            .as_mut()
            .expect("Adams kernel is initialized before stepping")
    }
}

impl<F, P> StepKernel<F, P> for AdamsKernel
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::new(false, self.method.order)
    }

    fn initialize(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        let mut workspace = Workspace::new(state.len(), self.method.order);
        evaluate(problem, &mut workspace.history[0], state, time, stats);
        ensure_finite(&workspace.history[0])?;
        workspace.history_len = 1;
        self.workspace = Some(workspace);
        Ok(())
    }

    fn estimate_initial_step(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: f64,
        _: f64,
        maximum_step: f64,
        _: &mut [f64],
        _: &SolveOptions,
        _: &mut SolverStats,
    ) -> Result<f64, SolveError> {
        Ok(maximum_step)
    }

    fn attempt_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        _: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError> {
        let method = self.method;
        let workspace = self.workspace();
        let used_bootstrap =
            workspace.history_len < method.order || method.repeating_bootstrap_predictor;
        if used_bootstrap {
            bootstrap_step(
                problem,
                state,
                time,
                step,
                method.bootstrap,
                candidate,
                workspace,
                stats,
            );
        } else {
            weighted_update(
                candidate,
                state,
                step,
                None,
                &workspace.history[..workspace.history_len],
                method.weights,
            );
        }

        let used_corrector = match method.corrector_weights {
            Some(corrector_weights) if workspace.history_len >= method.order - 1 => {
                evaluate(
                    problem,
                    &mut workspace.predicted_derivative,
                    candidate,
                    time + step,
                    stats,
                );
                weighted_update(
                    candidate,
                    state,
                    step,
                    Some((&workspace.predicted_derivative, corrector_weights[0])),
                    &workspace.history[..workspace.history_len],
                    &corrector_weights[1..],
                );
                true
            }
            _ => false,
        };
        if used_bootstrap || used_corrector {
            ensure_finite(candidate)?;
        }

        Ok(StepEstimate::new(0.0))
    }

    fn accept_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        _: &[f64],
        state: &[f64],
        time: f64,
        _: f64,
        callback_applied: bool,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        let method = self.method;
        let workspace = self.workspace();
        evaluate(problem, &mut workspace.next_derivative, state, time, stats);
        ensure_finite(&workspace.next_derivative)?;
        if callback_applied {
            workspace.history_len = 1;
        } else {
            workspace.history_len = (workspace.history_len + 1).min(method.order);
            for index in (1..workspace.history_len).rev() {
                workspace.history.swap(index, index - 1);
            }
        }
        std::mem::swap(&mut workspace.history[0], &mut workspace.next_derivative);
        Ok(())
    }

    fn reject_step(&mut self) {}
}

#[allow(clippy::too_many_arguments)]
fn bootstrap_step<F, P>(
    problem: &OdeProblem<F, P>,
    state: &[f64],
    time: f64,
    step: f64,
    method: Bootstrap,
    candidate: &mut [f64],
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) where
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
            );
            for (index, &value) in state.iter().enumerate() {
                candidate[index] = value
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
            );
            for (index, &value) in state.iter().enumerate() {
                workspace.temporary[index] = value + 0.5 * step * workspace.stage2[index];
            }
            evaluate(
                problem,
                &mut workspace.stage3,
                &workspace.temporary,
                time + 0.5 * step,
                stats,
            );
            for (index, &value) in state.iter().enumerate() {
                workspace.temporary[index] = value + step * workspace.stage3[index];
            }
            evaluate(
                problem,
                &mut workspace.stage4,
                &workspace.temporary,
                time + step,
                stats,
            );
            for (index, &value) in state.iter().enumerate() {
                candidate[index] = value
                    + (step / 6.0)
                        * (workspace.history[0][index]
                            + 2.0 * workspace.stage2[index]
                            + 2.0 * workspace.stage3[index]
                            + workspace.stage4[index]);
            }
        }
    }
}

fn weighted_update(
    output: &mut [f64],
    state: &[f64],
    step: f64,
    leading_term: Option<(&[f64], f64)>,
    history: &[Vec<f64>],
    weights: &[f64],
) {
    output.fill(0.0);
    if let Some((derivative, weight)) = leading_term {
        for (increment, derivative) in output.iter_mut().zip(derivative) {
            *increment += weight * derivative;
        }
    }
    for (derivative, weight) in history.iter().zip(weights) {
        for (increment, derivative) in output.iter_mut().zip(derivative) {
            *increment += weight * derivative;
        }
    }
    for (output_value, state_value) in output.iter_mut().zip(state) {
        *output_value = state_value + step * *output_value;
    }
}

fn evaluate<F, P>(
    problem: &OdeProblem<F, P>,
    derivative: &mut [f64],
    state: &[f64],
    time: f64,
    stats: &mut SolverStats,
) where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    (problem.rhs)(derivative, state, problem.parameters(), time);
    stats.rhs_evaluations += 1;
}

fn ensure_finite(values: &[f64]) -> Result<(), SolveError> {
    values
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteDerivative)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::f64::consts::E;

    use crate::{
        Ab3, Ab4, Ab5, Abm32, Abm43, Abm54, CallbackAction, OdeProblem, SaveMode, SolveError,
        SolveOptions, solve,
    };

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

    #[test]
    fn reports_non_finite_bootstrap_derivatives() {
        let calls = Cell::new(0);
        let problem = OdeProblem::new(
            |du: &mut [f64], _: &[f64], _: &(), _: f64| {
                let call = calls.get();
                calls.set(call + 1);
                du[0] = if call == 1 { f64::NAN } else { 1.0 };
            },
            vec![1.0],
            (0.0, 1.0),
            (),
        );
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(1.0),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        assert_eq!(
            solve(&problem, Ab3, &options),
            Err(SolveError::NonFiniteDerivative)
        );
    }

    #[test]
    fn terminating_callback_skips_accepted_history_derivative() {
        let calls = Cell::new(0);
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| {
                calls.set(calls.get() + 1);
                du[0] = u[0];
            },
            vec![1.0],
            (0.0, 1.0),
            (),
        )
        .with_continuous_callback(
            |_: &[f64], _: &(), time| time - 0.5,
            |_: &mut [f64], _: &(), _: f64| CallbackAction::Terminate,
        );
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(1.0),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        let solution = solve(&problem, Ab3, &options).unwrap();

        assert_eq!(solution.stats().accepted_steps, 1);
        assert_eq!(solution.stats().rhs_evaluations, 2);
        assert_eq!(calls.get(), 2);
    }
}
