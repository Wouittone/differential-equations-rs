use crate::{OdeAlgorithm, OdeProblem, SaveMode, Solution, SolveError, SolveOptions, SolverStats};

const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 10.0;

struct Tableau {
    nodes: &'static [f64],
    coefficients: &'static [&'static [f64]],
    weights: &'static [f64],
    error_weights: Option<&'static [f64]>,
    order: usize,
    fsal: bool,
}

const EMPTY: &[f64] = &[];
const EULER_A: &[&[f64]] = &[EMPTY];
const EULER_B: &[f64] = &[1.0];
const EULER_C: &[f64] = &[0.0];

const MIDPOINT_A2: &[f64] = &[0.5];
const MIDPOINT_A: &[&[f64]] = &[EMPTY, MIDPOINT_A2];
const MIDPOINT_B: &[f64] = &[0.0, 1.0];
const MIDPOINT_E: &[f64] = &[-1.0, 1.0];
const MIDPOINT_C: &[f64] = &[0.0, 0.5];

const HEUN_A2: &[f64] = &[1.0];
const HEUN_A: &[&[f64]] = &[EMPTY, HEUN_A2];
const HEUN_B: &[f64] = &[0.5, 0.5];
const HEUN_E: &[f64] = &[-0.5, 0.5];
const HEUN_C: &[f64] = &[0.0, 1.0];

const RALSTON_A2: &[f64] = &[2.0 / 3.0];
const RALSTON_A: &[&[f64]] = &[EMPTY, RALSTON_A2];
const RALSTON_B: &[f64] = &[0.25, 0.75];
const RALSTON_E: &[f64] = &[-0.75, 0.75];
const RALSTON_C: &[f64] = &[0.0, 2.0 / 3.0];

const RK4_A2: &[f64] = &[0.5];
const RK4_A3: &[f64] = &[0.0, 0.5];
const RK4_A4: &[f64] = &[0.0, 0.0, 1.0];
const RK4_A: &[&[f64]] = &[EMPTY, RK4_A2, RK4_A3, RK4_A4];
const RK4_B: &[f64] = &[1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0];
const RK4_C: &[f64] = &[0.0, 0.5, 0.5, 1.0];

const BS3_A2: &[f64] = &[0.5];
const BS3_A3: &[f64] = &[0.0, 0.75];
const BS3_A4: &[f64] = &[2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0];
const BS3_A: &[&[f64]] = &[EMPTY, BS3_A2, BS3_A3, BS3_A4];
const BS3_B: &[f64] = &[2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0];
const BS3_E: &[f64] = &[-5.0 / 72.0, 1.0 / 12.0, 1.0 / 9.0, -1.0 / 8.0];
const BS3_C: &[f64] = &[0.0, 0.5, 0.75, 1.0];

const DP5_A2: &[f64] = &[1.0 / 5.0];
const DP5_A3: &[f64] = &[3.0 / 40.0, 9.0 / 40.0];
const DP5_A4: &[f64] = &[44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0];
const DP5_A5: &[f64] = &[
    19_372.0 / 6_561.0,
    -25_360.0 / 2_187.0,
    64_448.0 / 6_561.0,
    -212.0 / 729.0,
];
const DP5_A6: &[f64] = &[
    9_017.0 / 3_168.0,
    -355.0 / 33.0,
    46_732.0 / 5_247.0,
    49.0 / 176.0,
    -5_103.0 / 18_656.0,
];
const DP5_A7: &[f64] = &[
    35.0 / 384.0,
    0.0,
    500.0 / 1_113.0,
    125.0 / 192.0,
    -2_187.0 / 6_784.0,
    11.0 / 84.0,
];
const DP5_A: &[&[f64]] = &[EMPTY, DP5_A2, DP5_A3, DP5_A4, DP5_A5, DP5_A6, DP5_A7];
const DP5_B: &[f64] = &[
    35.0 / 384.0,
    0.0,
    500.0 / 1_113.0,
    125.0 / 192.0,
    -2_187.0 / 6_784.0,
    11.0 / 84.0,
    0.0,
];
const DP5_E: &[f64] = &[
    35.0 / 384.0 - 5_179.0 / 57_600.0,
    0.0,
    500.0 / 1_113.0 - 7_571.0 / 16_695.0,
    125.0 / 192.0 - 393.0 / 640.0,
    -2_187.0 / 6_784.0 + 92_097.0 / 339_200.0,
    11.0 / 84.0 - 187.0 / 2_100.0,
    -1.0 / 40.0,
];
const DP5_C: &[f64] = &[0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];

const EULER_TABLEAU: Tableau = Tableau {
    nodes: EULER_C,
    coefficients: EULER_A,
    weights: EULER_B,
    error_weights: None,
    order: 1,
    fsal: false,
};
const MIDPOINT_TABLEAU: Tableau = Tableau {
    nodes: MIDPOINT_C,
    coefficients: MIDPOINT_A,
    weights: MIDPOINT_B,
    error_weights: Some(MIDPOINT_E),
    order: 2,
    fsal: false,
};
const HEUN_TABLEAU: Tableau = Tableau {
    nodes: HEUN_C,
    coefficients: HEUN_A,
    weights: HEUN_B,
    error_weights: Some(HEUN_E),
    order: 2,
    fsal: false,
};
const RALSTON_TABLEAU: Tableau = Tableau {
    nodes: RALSTON_C,
    coefficients: RALSTON_A,
    weights: RALSTON_B,
    error_weights: Some(RALSTON_E),
    order: 2,
    fsal: false,
};
const RK4_TABLEAU: Tableau = Tableau {
    nodes: RK4_C,
    coefficients: RK4_A,
    weights: RK4_B,
    error_weights: None,
    order: 4,
    fsal: false,
};
const BS3_TABLEAU: Tableau = Tableau {
    nodes: BS3_C,
    coefficients: BS3_A,
    weights: BS3_B,
    error_weights: Some(BS3_E),
    order: 3,
    fsal: true,
};
const DP5_TABLEAU: Tableau = Tableau {
    nodes: DP5_C,
    coefficients: DP5_A,
    weights: DP5_B,
    error_weights: Some(DP5_E),
    order: 5,
    fsal: true,
};

macro_rules! algorithm {
    ($name:ident, $documentation:literal, $tableau:ident) => {
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
                integrate(problem, options, &$tableau)
            }
        }
    };
}

algorithm!(Euler, "The fixed-step forward Euler method.", EULER_TABLEAU);
algorithm!(
    Midpoint,
    "The adaptive second-order explicit midpoint method with an embedded Euler estimate.",
    MIDPOINT_TABLEAU
);
algorithm!(
    Heun,
    "The adaptive second-order explicit trapezoid (Heun) method.",
    HEUN_TABLEAU
);
algorithm!(
    Ralston,
    "Ralston's adaptive second-order explicit Runge–Kutta method.",
    RALSTON_TABLEAU
);
algorithm!(
    Rk4,
    "The fixed-step classical fourth-order Runge–Kutta method.",
    RK4_TABLEAU
);
algorithm!(
    Bs3,
    "The adaptive Bogacki–Shampine 3/2 method.",
    BS3_TABLEAU
);
algorithm!(Dp5, "The adaptive Dormand–Prince 5/4 method.", DP5_TABLEAU);

struct Workspace {
    stages: Vec<Vec<f64>>,
    temporary: Vec<f64>,
    candidate: Vec<f64>,
}

impl Workspace {
    fn new(stage_count: usize, dimension: usize) -> Self {
        Self {
            stages: (0..stage_count).map(|_| vec![0.0; dimension]).collect(),
            temporary: vec![0.0; dimension],
            candidate: vec![0.0; dimension],
        }
    }
}

fn integrate<F, P>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
    tableau: &Tableau,
) -> Result<Solution, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let adaptive = options.adaptive;
    if adaptive && tableau.error_weights.is_none() {
        return Err(SolveError::AdaptiveStepUnsupported);
    }
    if !adaptive && options.initial_step.is_none() {
        return Err(SolveError::InitialStepRequired);
    }

    let dimension = problem.initial_state().len();
    let (start, end) = problem.time_span();
    let direction = (end - start).signum();
    let interval = (end - start).abs();
    let maximum_step = options.max_step.min(interval);
    let mut state = problem.initial_state().to_vec();
    let mut workspace = Workspace::new(tableau.weights.len(), dimension);
    let mut stats = SolverStats::default();
    evaluate(problem, &mut workspace.stages[0], &state, start, &mut stats)?;

    let step_magnitude = match options.initial_step {
        Some(step) => step.min(maximum_step),
        None => estimate_initial_step(
            problem,
            options,
            &state,
            (start, direction, maximum_step),
            tableau.order,
            &mut workspace,
            &mut stats,
        )?,
    };
    let mut step = direction * step_magnitude;
    let mut time = start;
    let mut times = vec![start];
    let mut values = Vec::with_capacity(2 * dimension);
    values.extend_from_slice(&state);
    let mut attempted_steps = 0;
    let mut previous_step_rejected = false;
    let mut stage_zero_is_current = true;

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
        if !stage_zero_is_current {
            evaluate(problem, &mut workspace.stages[0], &state, time, &mut stats)?;
        }

        perform_step(
            problem,
            &state,
            time,
            step,
            tableau,
            &mut workspace,
            &mut stats,
        )?;
        let error = if adaptive {
            error_norm(
                &workspace.stages,
                &state,
                &workspace.candidate,
                step,
                options,
                tableau.error_weights.expect("checked above"),
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
            if tableau.fsal {
                let last = workspace.stages.len() - 1;
                workspace.stages.swap(0, last);
                stage_zero_is_current = true;
            } else {
                stage_zero_is_current = false;
            }
            stats.accepted_steps += 1;

            if options.save == SaveMode::EveryStep || time == end {
                times.push(time);
                values.extend_from_slice(&state);
            }

            if adaptive {
                let mut factor = step_factor(error, tableau.order);
                if previous_step_rejected {
                    factor = factor.min(1.0);
                }
                step = direction * (step.abs() * factor).min(maximum_step);
            }
            previous_step_rejected = false;
        } else {
            stats.rejected_steps += 1;
            step *= step_factor(error, tableau.order).min(1.0);
            previous_step_rejected = true;
            stage_zero_is_current = true;
        }
    }

    Ok(Solution::new(times, values, dimension, stats))
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

fn estimate_initial_step<F, P>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
    state: &[f64],
    integration: (f64, f64, f64),
    order: usize,
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

    let mut curvature_norm = 0.0;
    for ((next, initial), value) in workspace.stages[1]
        .iter()
        .zip(&workspace.stages[0])
        .zip(state)
    {
        let scale = options.absolute_tolerance + options.relative_tolerance * value.abs();
        curvature_norm += ((next - initial) / scale).powi(2);
    }
    curvature_norm = (curvature_norm / dimension).sqrt() / trial_step;
    let largest = derivative_norm.max(curvature_norm);
    let accuracy_step = if largest <= 1.0e-15 {
        (trial_step * 1.0e-3).max(1.0e-6)
    } else {
        (0.01 / largest).powf(1.0 / order as f64)
    };
    Ok((100.0 * trial_step).min(accuracy_step).min(maximum_step))
}

fn perform_step<F, P>(
    problem: &OdeProblem<F, P>,
    state: &[f64],
    time: f64,
    step: f64,
    tableau: &Tableau,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    for stage_index in 1..workspace.stages.len() {
        combine(
            &mut workspace.temporary,
            state,
            step,
            &workspace.stages[..stage_index],
            tableau.coefficients[stage_index],
        );
        evaluate(
            problem,
            &mut workspace.stages[stage_index],
            &workspace.temporary,
            time + tableau.nodes[stage_index] * step,
            stats,
        )?;
    }
    combine(
        &mut workspace.candidate,
        state,
        step,
        &workspace.stages,
        tableau.weights,
    );
    Ok(())
}

fn combine(output: &mut [f64], state: &[f64], step: f64, stages: &[Vec<f64>], weights: &[f64]) {
    for (index, output_value) in output.iter_mut().enumerate() {
        let increment = stages
            .iter()
            .zip(weights)
            .map(|(stage, weight)| weight * stage[index])
            .sum::<f64>();
        *output_value = state[index] + step * increment;
    }
}

fn error_norm(
    stages: &[Vec<f64>],
    state: &[f64],
    candidate: &[f64],
    step: f64,
    options: &SolveOptions,
    error_weights: &[f64],
) -> f64 {
    let mut squared_norm = 0.0;
    for index in 0..state.len() {
        let error = step
            * stages
                .iter()
                .zip(error_weights)
                .map(|(stage, weight)| weight * stage[index])
                .sum::<f64>();
        let scale = options.absolute_tolerance
            + options.relative_tolerance * state[index].abs().max(candidate[index].abs());
        squared_norm += (error / scale).powi(2);
    }
    (squared_norm / state.len() as f64).sqrt()
}

fn step_factor(error: f64, order: usize) -> f64 {
    if error == 0.0 {
        MAX_FACTOR
    } else if error.is_finite() {
        (SAFETY * error.powf(-1.0 / order as f64)).clamp(MIN_FACTOR, MAX_FACTOR)
    } else {
        MIN_FACTOR
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::E;

    use crate::{
        Bs3, Dp5, Euler, Heun, Midpoint, OdeProblem, Ralston, Rk4, SaveMode, SolveError,
        SolveOptions, solve,
    };

    type TestRhs = fn(&mut [f64], &[f64], &(), f64);

    fn exponential() -> OdeProblem<TestRhs, ()> {
        fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
            du[0] = u[0];
        }

        OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
    }

    fn adaptive_options() -> SolveOptions {
        SolveOptions {
            absolute_tolerance: 1.0e-9,
            relative_tolerance: 1.0e-9,
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        }
    }

    #[test]
    fn adaptive_embedded_methods_solve_exponential_growth() {
        for endpoint in [
            solve(&exponential(), Midpoint, &adaptive_options())
                .unwrap()
                .last_state()[0],
            solve(&exponential(), Heun, &adaptive_options())
                .unwrap()
                .last_state()[0],
            solve(&exponential(), Ralston, &adaptive_options())
                .unwrap()
                .last_state()[0],
            solve(&exponential(), Bs3, &adaptive_options())
                .unwrap()
                .last_state()[0],
            solve(&exponential(), Dp5, &adaptive_options())
                .unwrap()
                .last_state()[0],
        ] {
            assert!((endpoint - E).abs() < 2.0e-7);
        }
    }

    #[test]
    fn fixed_methods_have_expected_convergence() {
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.001),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };
        let euler_error =
            (solve(&exponential(), Euler, &options).unwrap().last_state()[0] - E).abs();
        let rk4_error = (solve(&exponential(), Rk4, &options).unwrap().last_state()[0] - E).abs();

        assert!(euler_error < 0.002);
        assert!(rk4_error < 1.0e-12);
    }

    #[test]
    fn fixed_only_methods_reject_adaptive_configuration() {
        assert_eq!(
            solve(&exponential(), Euler, &SolveOptions::default()),
            Err(SolveError::AdaptiveStepUnsupported)
        );
        assert_eq!(
            solve(&exponential(), Rk4, &SolveOptions::default()),
            Err(SolveError::AdaptiveStepUnsupported)
        );
    }
}
