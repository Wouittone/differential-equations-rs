use crate::{OdeAlgorithm, OdeProblem, SaveMode, Solution, SolveError, SolveOptions, SolverStats};
use std::marker::PhantomData;

const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 10.0;

/// Coefficients and method properties for an explicit Runge–Kutta method.
///
/// `COEFFICIENTS[i]` is the strictly lower-triangular row for stage `i`, so it
/// must contain exactly `i` entries. All other coefficient arrays must contain
/// one entry per stage. [`ExplicitRungeKutta`] validates these invariants before
/// solving.
pub trait ButcherTableau {
    const NODES: &'static [f64];
    const COEFFICIENTS: &'static [&'static [f64]];
    const WEIGHTS: &'static [f64];
    const ERROR_WEIGHTS: Option<&'static [f64]>;
    const ORDER: usize;
    const FSAL: bool;
}

/// The centralized explicit Runge–Kutta solver for a [`ButcherTableau`].
///
/// Named algorithms such as [`Rk4`](crate::Rk4) are lightweight facades over
/// this type. It can also be instantiated with a user-defined tableau marker.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExplicitRungeKutta<T> {
    marker: PhantomData<fn() -> T>,
}

impl<T> ExplicitRungeKutta<T> {
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

impl<T> Default for ExplicitRungeKutta<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> OdeAlgorithm for ExplicitRungeKutta<T>
where
    T: ButcherTableau,
{
    fn solve<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        integrate::<F, P, T>(problem, options)
    }
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

const RKM_A2: &[f64] = &[0.167_266_187_050_662];
const RKM_A3: &[f64] = &[0.0, 0.484_574_582_244_783];
const RKM_A4: &[f64] = &[0.0, 0.0, 0.536_909_403_373_491];
const RKM_A5: &[f64] = &[0.0, 0.0, 0.0, 0.082_069_535_961_948];
const RKM_A6: &[f64] = &[0.0, 0.0, 0.0, 0.0, 0.853_923_000_035_347];
const RKM_A: &[&[f64]] = &[EMPTY, RKM_A2, RKM_A3, RKM_A4, RKM_A5, RKM_A6];
const RKM_B: &[f64] = &[
    -0.028_289_441_132_839,
    0.463_968_918_564_71,
    -0.434_414_348_751_899,
    0.693_796_229_087_598,
    0.0,
    0.304_938_642_232_43,
];
const RKM_C: &[f64] = &[
    0.0,
    0.167_266_187_050_662,
    0.484_574_582_244_783,
    0.536_909_403_373_491,
    0.082_069_535_961_948,
    0.853_923_000_035_347,
];

const RALSTON4_A2: &[f64] = &[0.4];
const RALSTON4_A3: &[f64] = &[0.296_977_609_247_753_57, 0.158_759_644_971_035_84];
const RALSTON4_A4: &[f64] = &[
    0.218_100_388_225_920_04,
    -3.050_965_148_692_930_6,
    3.832_864_760_467_010_5,
];
const RALSTON4_A: &[&[f64]] = &[EMPTY, RALSTON4_A2, RALSTON4_A3, RALSTON4_A4];
const RALSTON4_B: &[f64] = &[
    0.174_760_282_262_690_4,
    -0.551_480_662_878_733,
    1.205_535_599_396_523_5,
    0.171_184_781_219_519_02,
];
const RALSTON4_C: &[f64] = &[0.0, 0.4, 0.455_737_254_218_789_4, 1.0];

const ALSHINA2_E: &[f64] = &[1.0, 0.0];
const ALSHINA3_A3: &[f64] = &[0.0, 0.75];
const ALSHINA3_A: &[&[f64]] = &[EMPTY, MIDPOINT_A2, ALSHINA3_A3];
const ALSHINA3_B: &[f64] = &[2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0];
const ALSHINA3_E: &[f64] = &[0.0, 4.0 / 9.0, 0.0];
const ALSHINA3_C: &[f64] = &[0.0, 0.5, 0.75];

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

const SSPRK22_A: &[&[f64]] = &[EMPTY, HEUN_A2];
const SSPRK22_B: &[f64] = HEUN_B;
const SSPRK22_C: &[f64] = HEUN_C;

const SSPRK33_A2: &[f64] = &[1.0];
const SSPRK33_A3: &[f64] = &[0.25, 0.25];
const SSPRK33_A: &[&[f64]] = &[EMPTY, SSPRK33_A2, SSPRK33_A3];
const SSPRK33_B: &[f64] = &[1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0];
const SSPRK33_C: &[f64] = &[0.0, 1.0, 0.5];

const SSPRK43_A2: &[f64] = &[0.5];
const SSPRK43_A3: &[f64] = &[0.5, 0.5];
const SSPRK43_A4: &[f64] = &[1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0];
const SSPRK43_A: &[&[f64]] = &[EMPTY, SSPRK43_A2, SSPRK43_A3, SSPRK43_A4];
const SSPRK43_B: &[f64] = &[1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 0.5];
const SSPRK43_E: &[f64] = &[-1.0 / 12.0, -1.0 / 12.0, -1.0 / 12.0, 0.25];
const SSPRK43_C: &[f64] = &[0.0, 0.5, 1.0, 0.5];

macro_rules! algorithm {
    (
        $name:ident,
        $documentation:literal,
        nodes = $nodes:ident,
        coefficients = $coefficients:ident,
        weights = $weights:ident,
        error_weights = $error_weights:expr,
        order = $order:literal,
        fsal = $fsal:literal
    ) => {
        #[doc = $documentation]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        pub struct $name;

        impl ButcherTableau for $name {
            const NODES: &'static [f64] = $nodes;
            const COEFFICIENTS: &'static [&'static [f64]] = $coefficients;
            const WEIGHTS: &'static [f64] = $weights;
            const ERROR_WEIGHTS: Option<&'static [f64]> = $error_weights;
            const ORDER: usize = $order;
            const FSAL: bool = $fsal;
        }

        impl OdeAlgorithm for $name {
            fn solve<F, P>(
                &self,
                problem: &OdeProblem<F, P>,
                options: &SolveOptions,
            ) -> Result<Solution, SolveError>
            where
                F: Fn(&mut [f64], &[f64], &P, f64),
            {
                ExplicitRungeKutta::<Self>::new().solve(problem, options)
            }
        }
    };
}

algorithm!(
    Euler,
    "The fixed-step forward Euler method.",
    nodes = EULER_C,
    coefficients = EULER_A,
    weights = EULER_B,
    error_weights = None,
    order = 1,
    fsal = false
);
algorithm!(
    Midpoint,
    "The adaptive second-order explicit midpoint method with an embedded Euler estimate.",
    nodes = MIDPOINT_C,
    coefficients = MIDPOINT_A,
    weights = MIDPOINT_B,
    error_weights = Some(MIDPOINT_E),
    order = 2,
    fsal = false
);
algorithm!(
    Heun,
    "The adaptive second-order explicit trapezoid (Heun) method.",
    nodes = HEUN_C,
    coefficients = HEUN_A,
    weights = HEUN_B,
    error_weights = Some(HEUN_E),
    order = 2,
    fsal = false
);
algorithm!(
    Ralston,
    "Ralston's adaptive second-order explicit Runge–Kutta method.",
    nodes = RALSTON_C,
    coefficients = RALSTON_A,
    weights = RALSTON_B,
    error_weights = Some(RALSTON_E),
    order = 2,
    fsal = false
);
algorithm!(
    Rk4,
    "The fixed-step classical fourth-order Runge–Kutta method.",
    nodes = RK4_C,
    coefficients = RK4_A,
    weights = RK4_B,
    error_weights = None,
    order = 4,
    fsal = false
);
algorithm!(
    Rkm,
    "The fixed-step six-stage, fourth-order Mead–Renaut Runge–Kutta method.",
    nodes = RKM_C,
    coefficients = RKM_A,
    weights = RKM_B,
    error_weights = None,
    order = 4,
    fsal = false
);
algorithm!(
    Ralston4,
    "Ralston's fixed-step four-stage, fourth-order Runge–Kutta method.",
    nodes = RALSTON4_C,
    coefficients = RALSTON4_A,
    weights = RALSTON4_B,
    error_weights = None,
    order = 4,
    fsal = false
);
algorithm!(
    Alshina2,
    "The adaptive optimal two-stage, second-order Alshina method.",
    nodes = RALSTON_C,
    coefficients = RALSTON_A,
    weights = RALSTON_B,
    error_weights = Some(ALSHINA2_E),
    order = 2,
    fsal = false
);
algorithm!(
    Alshina3,
    "The adaptive optimal three-stage, third-order Alshina method.",
    nodes = ALSHINA3_C,
    coefficients = ALSHINA3_A,
    weights = ALSHINA3_B,
    error_weights = Some(ALSHINA3_E),
    order = 3,
    fsal = false
);
algorithm!(
    Bs3,
    "The adaptive Bogacki–Shampine 3/2 method.",
    nodes = BS3_C,
    coefficients = BS3_A,
    weights = BS3_B,
    error_weights = Some(BS3_E),
    order = 3,
    fsal = true
);
algorithm!(
    Dp5,
    "The adaptive Dormand–Prince 5/4 method.",
    nodes = DP5_C,
    coefficients = DP5_A,
    weights = DP5_B,
    error_weights = Some(DP5_E),
    order = 5,
    fsal = true
);
algorithm!(
    SspRk22,
    "The fixed-step two-stage, second-order SSP Runge–Kutta method.",
    nodes = SSPRK22_C,
    coefficients = SSPRK22_A,
    weights = SSPRK22_B,
    error_weights = None,
    order = 2,
    fsal = false
);
algorithm!(
    SspRk33,
    "The fixed-step three-stage, third-order SSP Runge–Kutta method.",
    nodes = SSPRK33_C,
    coefficients = SSPRK33_A,
    weights = SSPRK33_B,
    error_weights = None,
    order = 3,
    fsal = false
);
algorithm!(
    SspRk43,
    "The adaptive four-stage, third-order SSP Runge–Kutta method.",
    nodes = SSPRK43_C,
    coefficients = SSPRK43_A,
    weights = SSPRK43_B,
    error_weights = Some(SSPRK43_E),
    order = 3,
    fsal = false
);

struct Workspace {
    // Flat stage-major storage: every stage is one contiguous component array.
    // The other work vectors remain separate arrays rather than per-component
    // structs, keeping the hot saxpy-style loops friendly to SIMD.
    stages: Vec<f64>,
    stage_count: usize,
    dimension: usize,
    temporary: Vec<f64>,
    candidate: Vec<f64>,
}

impl Workspace {
    fn new(stage_count: usize, dimension: usize) -> Self {
        Self {
            stages: vec![0.0; stage_count * dimension],
            stage_count,
            dimension,
            temporary: vec![0.0; dimension],
            candidate: vec![0.0; dimension],
        }
    }

    fn stage(&self, index: usize) -> &[f64] {
        let start = index * self.dimension;
        &self.stages[start..start + self.dimension]
    }

    fn swap_stages(&mut self, left: usize, right: usize) {
        let left_start = left * self.dimension;
        let right_start = right * self.dimension;
        for offset in 0..self.dimension {
            self.stages.swap(left_start + offset, right_start + offset);
        }
    }
}

fn validate_tableau<T: ButcherTableau>() -> Result<(), SolveError> {
    let stage_count = T::WEIGHTS.len();
    let structurally_valid = stage_count > 0
        && T::ORDER > 0
        && T::NODES.first() == Some(&0.0)
        && T::NODES.len() == stage_count
        && T::COEFFICIENTS.len() == stage_count
        && T::COEFFICIENTS
            .iter()
            .enumerate()
            .all(|(stage, row)| row.len() == stage)
        && T::ERROR_WEIGHTS.is_none_or(|weights| weights.len() == stage_count);
    let coefficients_finite = T::NODES.iter().all(|value| value.is_finite())
        && T::WEIGHTS.iter().all(|value| value.is_finite())
        && T::COEFFICIENTS
            .iter()
            .flat_map(|row| row.iter())
            .all(|value| value.is_finite())
        && T::ERROR_WEIGHTS.is_none_or(|weights| weights.iter().all(|value| value.is_finite()));
    let fsal_valid = !T::FSAL
        || (T::NODES.last() == Some(&1.0)
            && T::WEIGHTS.last() == Some(&0.0)
            && T::COEFFICIENTS
                .last()
                .is_some_and(|last_row| *last_row == &T::WEIGHTS[..stage_count - 1]));

    (structurally_valid && coefficients_finite && fsal_valid)
        .then_some(())
        .ok_or(SolveError::InvalidTableau)
}

fn integrate<F, P, T>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
) -> Result<Solution, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
    T: ButcherTableau,
{
    validate_tableau::<T>()?;
    let adaptive = options.adaptive;
    if adaptive && T::ERROR_WEIGHTS.is_none() {
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
    let mut workspace = Workspace::new(T::WEIGHTS.len(), dimension);
    let mut stats = SolverStats::default();
    evaluate(
        problem,
        &mut workspace.stages[..dimension],
        &state,
        start,
        &mut stats,
    );
    ensure_finite(&workspace.stages[..dimension])?;

    let step_magnitude = match options.initial_step {
        Some(step) => step.min(maximum_step),
        None => estimate_initial_step(
            problem,
            options,
            &state,
            (start, direction, maximum_step),
            T::ORDER,
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
            evaluate(
                problem,
                &mut workspace.stages[..dimension],
                &state,
                time,
                &mut stats,
            );
        }

        perform_step::<F, P, T>(problem, &state, time, step, &mut workspace, &mut stats);
        ensure_finite(&workspace.candidate)?;
        let error = if adaptive {
            error_norm(
                &workspace.stages,
                dimension,
                (&state, &workspace.candidate),
                step,
                options,
                T::ERROR_WEIGHTS.expect("checked above"),
                &mut workspace.temporary,
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
            if T::FSAL {
                workspace.swap_stages(0, workspace.stage_count - 1);
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
                let mut factor = step_factor(error, T::ORDER);
                if previous_step_rejected {
                    factor = factor.min(1.0);
                }
                step = direction * (step.abs() * factor).min(maximum_step);
            }
            previous_step_rejected = false;
        } else {
            stats.rejected_steps += 1;
            step *= step_factor(error, T::ORDER).min(1.0);
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
    for (value, derivative) in state.iter().zip(workspace.stage(0)) {
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
        .zip(&workspace.stages[..workspace.dimension])
    {
        *trial = value + direction * trial_step * derivative;
    }
    evaluate(
        problem,
        &mut workspace.stages[workspace.dimension..2 * workspace.dimension],
        &workspace.temporary,
        time + direction * trial_step,
        stats,
    );
    ensure_finite(&workspace.stages[workspace.dimension..2 * workspace.dimension])?;

    let mut curvature_norm = 0.0;
    for ((next, initial), value) in workspace.stages[workspace.dimension..2 * workspace.dimension]
        .iter()
        .zip(&workspace.stages[..workspace.dimension])
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

fn perform_step<F, P, T>(
    problem: &OdeProblem<F, P>,
    state: &[f64],
    time: f64,
    step: f64,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) where
    F: Fn(&mut [f64], &[f64], &P, f64),
    T: ButcherTableau,
{
    for stage_index in 1..workspace.stage_count {
        combine(
            &mut workspace.temporary,
            state,
            step,
            &workspace.stages,
            workspace.dimension,
            stage_index,
            T::COEFFICIENTS[stage_index],
        );
        let start = stage_index * workspace.dimension;
        evaluate(
            problem,
            &mut workspace.stages[start..start + workspace.dimension],
            &workspace.temporary,
            time + T::NODES[stage_index] * step,
            stats,
        );
    }
    combine(
        &mut workspace.candidate,
        state,
        step,
        &workspace.stages,
        workspace.dimension,
        workspace.stage_count,
        T::WEIGHTS,
    );
}

fn combine(
    output: &mut [f64],
    state: &[f64],
    step: f64,
    stages: &[f64],
    dimension: usize,
    stage_count: usize,
    weights: &[f64],
) {
    output.fill(0.0);
    for (stage_index, weight) in weights.iter().take(stage_count).enumerate() {
        let start = stage_index * dimension;
        let stage = &stages[start..start + dimension];
        for (increment, stage_value) in output.iter_mut().zip(stage) {
            *increment += weight * stage_value;
        }
    }
    for (output_value, state_value) in output.iter_mut().zip(state) {
        *output_value = state_value + step * *output_value;
    }
}

fn error_norm(
    stages: &[f64],
    dimension: usize,
    states: (&[f64], &[f64]),
    step: f64,
    options: &SolveOptions,
    error_weights: &[f64],
    error_buffer: &mut [f64],
) -> f64 {
    let (state, candidate) = states;
    error_buffer.fill(0.0);
    for (stage_index, weight) in error_weights.iter().enumerate() {
        let start = stage_index * dimension;
        let stage = &stages[start..start + dimension];
        for (error, stage_value) in error_buffer.iter_mut().zip(stage) {
            *error += weight * stage_value;
        }
    }
    let mut squared_norm = 0.0;
    for ((error, state), candidate) in error_buffer.iter().zip(state).zip(candidate) {
        let error = step * error;
        let scale = options.absolute_tolerance
            + options.relative_tolerance * state.abs().max(candidate.abs());
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

    use super::{ButcherTableau, ExplicitRungeKutta};
    use crate::{
        Alshina2, Alshina3, Bs3, Dp5, Euler, Heun, Midpoint, OdeProblem, Ralston, Ralston4, Rk4,
        Rkm, SaveMode, SolveError, SolveOptions, SspRk22, SspRk33, SspRk43, solve,
    };

    type TestRhs = fn(&mut [f64], &[f64], &(), f64);

    struct CustomEuler;

    impl ButcherTableau for CustomEuler {
        const NODES: &'static [f64] = &[0.0];
        const COEFFICIENTS: &'static [&'static [f64]] = &[&[]];
        const WEIGHTS: &'static [f64] = &[1.0];
        const ERROR_WEIGHTS: Option<&'static [f64]> = None;
        const ORDER: usize = 1;
        const FSAL: bool = false;
    }

    struct MalformedTableau;

    impl ButcherTableau for MalformedTableau {
        const NODES: &'static [f64] = &[0.0, 1.0];
        const COEFFICIENTS: &'static [&'static [f64]] = &[&[]];
        const WEIGHTS: &'static [f64] = &[1.0];
        const ERROR_WEIGHTS: Option<&'static [f64]> = None;
        const ORDER: usize = 1;
        const FSAL: bool = false;
    }

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
        let rkm_error = (solve(&exponential(), Rkm, &options).unwrap().last_state()[0] - E).abs();
        let ralston4_error = (solve(&exponential(), Ralston4, &options)
            .unwrap()
            .last_state()[0]
            - E)
            .abs();
        let alshina2_error = (solve(&exponential(), Alshina2, &options)
            .unwrap()
            .last_state()[0]
            - E)
            .abs();
        let alshina3_error = (solve(&exponential(), Alshina3, &options)
            .unwrap()
            .last_state()[0]
            - E)
            .abs();

        assert!(euler_error < 0.002);
        assert!(rk4_error < 1.0e-12);
        assert!(rkm_error < 1.0e-12);
        assert!(ralston4_error < 1.0e-12);
        assert!(alshina2_error < 1.0e-6);
        assert!(alshina3_error < 1.0e-9);
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

    #[test]
    fn named_solver_is_a_facade_over_the_generic_kernel() {
        let problem = exponential();
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.01),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        let named = solve(&problem, Rk4, &options).unwrap();
        let generic = solve(&problem, ExplicitRungeKutta::<Rk4>::new(), &options).unwrap();

        assert_eq!(named, generic);
    }

    #[test]
    fn supports_custom_tableaus_and_rejects_malformed_ones() {
        let problem = exponential();
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.01),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        let custom = solve(&problem, ExplicitRungeKutta::<CustomEuler>::new(), &options).unwrap();
        let named = solve(&problem, Euler, &options).unwrap();

        assert_eq!(custom, named);
        assert_eq!(
            solve(
                &problem,
                ExplicitRungeKutta::<MalformedTableau>::new(),
                &options,
            ),
            Err(SolveError::InvalidTableau)
        );
    }

    #[test]
    fn reports_non_finite_stage_derivatives() {
        let problem = OdeProblem::new(
            |du: &mut [f64], _: &[f64], _: &(), time: f64| {
                du[0] = if time == 0.0 { 1.0 } else { f64::NAN };
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
            solve(&problem, Rk4, &options),
            Err(SolveError::NonFiniteDerivative)
        );
    }

    #[test]
    fn ssp_methods_solve_exponential_growth() {
        let fixed = SolveOptions {
            adaptive: false,
            initial_step: Some(0.001),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        let endpoints = [
            solve(&exponential(), SspRk22, &fixed).unwrap().last_state()[0],
            solve(&exponential(), SspRk33, &fixed).unwrap().last_state()[0],
            solve(&exponential(), SspRk43, &adaptive_options())
                .unwrap()
                .last_state()[0],
        ];

        assert!((endpoints[0] - E).abs() < 1.0e-6);
        assert!((endpoints[1] - E).abs() < 1.0e-9);
        assert!((endpoints[2] - E).abs() < 2.0e-7);
    }
}
