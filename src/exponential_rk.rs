use crate::integrator::{
    ControllerConfig, KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 6.0;

/// Common metadata for exponential Runge--Kutta algorithms.
pub trait ExponentialAlgorithm: OdeAlgorithm {
    /// Classical convergence order of the method.
    fn order(&self) -> usize;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Scheme {
    LawsonEuler,
    NorsettEuler,
    Etdrk2,
    Etdrk3,
    Etdrk4,
    HochOst4,
    Exp4,
    Epirk4s3A,
    Epirk4s3B,
    Epirk5s3,
    Exprb53s3,
    Epirk5P1,
    Epirk5P2,
    Etd2,
    Exprb32,
    Exprb43,
}

impl Scheme {
    const fn order(self) -> usize {
        match self {
            Self::LawsonEuler | Self::NorsettEuler => 1,
            Self::Etdrk2 | Self::Etd2 => 2,
            Self::Etdrk3 | Self::Exprb32 => 3,
            Self::Etdrk4
            | Self::HochOst4
            | Self::Exp4
            | Self::Epirk4s3A
            | Self::Epirk4s3B
            | Self::Exprb43 => 4,
            Self::Epirk5s3 | Self::Exprb53s3 | Self::Epirk5P1 | Self::Epirk5P2 => 5,
        }
    }

    const fn adaptive(self) -> bool {
        matches!(self, Self::Exprb32 | Self::Exprb43)
    }
}

macro_rules! exponential_algorithm {
    ($name:ident, $scheme:ident, $order:expr, $documentation:literal) => {
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
                solve_scheme(problem, options, Scheme::$scheme)
            }
        }

        impl ExponentialAlgorithm for $name {
            fn order(&self) -> usize {
                $order
            }
        }
    };
}

exponential_algorithm!(
    LawsonEuler,
    LawsonEuler,
    1,
    "First-order Lawson exponential Euler method."
);
exponential_algorithm!(
    NorsettEuler,
    NorsettEuler,
    1,
    "First-order Nørsett exponential Euler method."
);
/// Exact spelling alias for [`NorsettEuler`].
pub type ETD1 = NorsettEuler;
/// Value constructor for the exact [`NorsettEuler`] spelling alias.
#[allow(non_upper_case_globals)]
pub const ETD1: NorsettEuler = NorsettEuler;
exponential_algorithm!(
    ETDRK2,
    Etdrk2,
    2,
    "Second-order exponential Runge--Kutta method."
);
exponential_algorithm!(
    ETDRK3,
    Etdrk3,
    3,
    "Third-order exponential Runge--Kutta method."
);
exponential_algorithm!(
    ETDRK4,
    Etdrk4,
    4,
    "Fourth-order exponential Runge--Kutta method."
);
exponential_algorithm!(
    HochOst4,
    HochOst4,
    4,
    "Hochbruck--Ostermann stiff-order-four method."
);
exponential_algorithm!(Exp4, Exp4, 4, "Hochbruck--Lubich--Selhofer Exp4 method.");
exponential_algorithm!(
    EPIRK4s3A,
    Epirk4s3A,
    4,
    "Three-stage stiff-order-four EPIRK method A."
);
exponential_algorithm!(
    EPIRK4s3B,
    Epirk4s3B,
    4,
    "Three-stage stiff-order-four EPIRK method B."
);
exponential_algorithm!(
    EPIRK5s3,
    Epirk5s3,
    5,
    "Three-stage fifth-order horizontal EPIRK method."
);
exponential_algorithm!(
    EXPRB53s3,
    Exprb53s3,
    5,
    "Three-stage fifth-order exponential Rosenbrock method."
);
exponential_algorithm!(EPIRK5P1, Epirk5P1, 5, "Fifth-order EPIRK method P1.");
exponential_algorithm!(EPIRK5P2, Epirk5P2, 5, "Fifth-order EPIRK method P2.");
exponential_algorithm!(
    ETD2,
    Etd2,
    2,
    "Second-order multistep exponential time-differencing method."
);
exponential_algorithm!(
    Exprb32,
    Exprb32,
    3,
    "Adaptive embedded exponential Rosenbrock 3(2) method."
);
exponential_algorithm!(
    Exprb43,
    Exprb43,
    4,
    "Adaptive embedded exponential Rosenbrock 4(3) method."
);

fn solve_scheme<F, P>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
    scheme: Scheme,
) -> Result<Solution, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    drive_integration(
        problem,
        options,
        ExponentialKernel::new(problem.initial_state().len(), scheme),
    )
}

struct ExponentialKernel {
    scheme: Scheme,
    derivative: Vec<f64>,
    jacobian: Vec<f64>,
    previous_nonlinear: Option<Vec<f64>>,
    pending_nonlinear: Vec<f64>,
}

impl ExponentialKernel {
    fn new(dimension: usize, scheme: Scheme) -> Self {
        Self {
            scheme,
            derivative: vec![0.0; dimension],
            jacobian: vec![0.0; dimension * dimension],
            previous_nonlinear: None,
            pending_nonlinear: vec![0.0; dimension],
        }
    }
}

impl<F, P> StepKernel<F, P> for ExponentialKernel
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::with_controller(
            self.scheme.adaptive(),
            ControllerConfig::proportional(
                self.scheme.order(),
                SAFETY,
                MIN_FACTOR,
                MAX_FACTOR,
                MIN_FACTOR,
            ),
        )
    }

    fn initialize(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        evaluate(problem, &mut self.derivative, state, time, stats)
    }

    fn estimate_initial_step(
        &mut self,
        _: &OdeProblem<F, P>,
        state: &[f64],
        _: f64,
        _: f64,
        maximum_step: f64,
        _: &mut [f64],
        options: &SolveOptions,
        _: &mut SolverStats,
    ) -> Result<f64, SolveError> {
        let mut scale = 0.0_f64;
        for (state, derivative) in state.iter().zip(&self.derivative) {
            let tolerance = options.absolute_tolerance + options.relative_tolerance * state.abs();
            scale = scale.max((derivative / tolerance).abs());
        }
        let estimate = if scale == 0.0 { 1.0e-3 } else { 0.01 / scale };
        Ok(estimate.clamp(f64::EPSILON, maximum_step))
    }

    fn attempt_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        options: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError> {
        compute_jacobian(
            problem,
            state,
            time,
            &self.derivative,
            &mut self.jacobian,
            stats,
        )?;
        let result = perform_step(
            self.scheme,
            problem,
            state,
            time,
            step,
            &self.derivative,
            &self.jacobian,
            self.previous_nonlinear.as_deref(),
            stats,
        )?;
        candidate.copy_from_slice(&result.state);
        self.pending_nonlinear = result.current_nonlinear;
        let error_norm = if options.adaptive {
            scaled_error_norm(&result.error, state, candidate, options)
        } else {
            0.0
        };
        Ok(StepEstimate::new(error_norm))
    }

    fn accept_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        _: &[f64],
        state: &[f64],
        time: f64,
        _: f64,
        _: bool,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        if self.scheme == Scheme::Etd2 {
            self.previous_nonlinear = Some(self.pending_nonlinear.clone());
        }
        evaluate(problem, &mut self.derivative, state, time, stats)
    }

    fn reject_step(&mut self) {}
}

struct StepResult {
    state: Vec<f64>,
    error: Vec<f64>,
    current_nonlinear: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
fn perform_step<F, P>(
    scheme: Scheme,
    problem: &OdeProblem<F, P>,
    state: &[f64],
    time: f64,
    step: f64,
    f0: &[f64],
    jacobian: &[f64],
    previous_nonlinear: Option<&[f64]>,
    stats: &mut SolverStats,
) -> Result<StepResult, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let au = mat_vec(jacobian, state);
    let nonlinear0 = subtract(f0, &au);
    let (state, error) = match scheme {
        Scheme::LawsonEuler => (lawson_euler(jacobian, state, &nonlinear0, step), vec![]),
        Scheme::NorsettEuler => (norsett_euler(jacobian, state, f0, step), vec![]),
        Scheme::Etdrk2 => (
            etdrk2(problem, jacobian, state, time, step, f0, &nonlinear0, stats)?,
            vec![],
        ),
        Scheme::Etdrk3 => (
            etdrk3(problem, jacobian, state, time, step, f0, &au, stats)?,
            vec![],
        ),
        Scheme::Etdrk4 => (
            etdrk4(problem, jacobian, state, time, step, f0, &au, stats)?,
            vec![],
        ),
        Scheme::HochOst4 => (
            hoch_ost4(problem, jacobian, state, time, step, f0, &au, stats)?,
            vec![],
        ),
        Scheme::Exp4 => (
            exp4(problem, jacobian, state, time, step, f0, stats)?,
            vec![],
        ),
        Scheme::Epirk4s3A => (
            epirk4s3(problem, jacobian, state, time, step, f0, false, stats)?,
            vec![],
        ),
        Scheme::Epirk4s3B => (
            epirk4s3(problem, jacobian, state, time, step, f0, true, stats)?,
            vec![],
        ),
        Scheme::Epirk5s3 => (
            epirk5s3(problem, jacobian, state, time, step, f0, stats)?,
            vec![],
        ),
        Scheme::Exprb53s3 => (
            exprb53s3(problem, jacobian, state, time, step, f0, stats)?,
            vec![],
        ),
        Scheme::Epirk5P1 => (
            epirk5p(problem, jacobian, state, time, step, f0, false, stats)?,
            vec![],
        ),
        Scheme::Epirk5P2 => (
            epirk5p(problem, jacobian, state, time, step, f0, true, stats)?,
            vec![],
        ),
        Scheme::Etd2 => (
            etd2(jacobian, state, step, &nonlinear0, previous_nonlinear),
            vec![],
        ),
        Scheme::Exprb32 => exprb32(problem, jacobian, state, time, step, f0, stats)?,
        Scheme::Exprb43 => exprb43(problem, jacobian, state, time, step, f0, &au, stats)?,
    };
    if !state.iter().all(|value| value.is_finite()) {
        return Err(SolveError::NonFiniteDerivative);
    }
    Ok(StepResult {
        state,
        error,
        current_nonlinear: nonlinear0,
    })
}

fn lawson_euler(a: &[f64], u: &[f64], g: &[f64], h: f64) -> Vec<f64> {
    let mut shifted = u.to_vec();
    axpy(&mut shifted, h, g);
    exp_action(a, h, &shifted)
}

fn norsett_euler(a: &[f64], u: &[f64], f0: &[f64], h: f64) -> Vec<f64> {
    add(u, &phi_term(a, h, 1, h, f0))
}

fn etd2(a: &[f64], u: &[f64], h: f64, g: &[f64], previous: Option<&[f64]>) -> Vec<f64> {
    let mut output = exp_action(a, h, u);
    if let Some(previous) = previous {
        axpy(&mut output, 1.0, &phi_term(a, h, 1, h, g));
        axpy(&mut output, 1.0, &phi_term(a, h, 2, h, g));
        axpy(&mut output, -1.0, &phi_term(a, h, 2, h, previous));
    } else {
        axpy(&mut output, 1.0, &phi_term(a, h, 1, h, g));
    }
    output
}

#[allow(clippy::too_many_arguments)]
fn etdrk2<F, P>(
    problem: &OdeProblem<F, P>,
    a: &[f64],
    u: &[f64],
    t: f64,
    h: f64,
    f0: &[f64],
    g0: &[f64],
    stats: &mut SolverStats,
) -> Result<Vec<f64>, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let u2 = norsett_euler(a, u, f0, h);
    let f2 = evaluate_new(problem, &u2, t + h, stats)?;
    let g2 = subtract(&f2, &mat_vec(a, &u2));
    let difference = subtract(&g2, g0);
    Ok(add(&u2, &phi_term(a, h, 2, h, &difference)))
}

#[allow(clippy::too_many_arguments)]
fn etdrk3<F, P>(
    problem: &OdeProblem<F, P>,
    a: &[f64],
    u: &[f64],
    t: f64,
    h: f64,
    f1: &[f64],
    au: &[f64],
    stats: &mut SolverStats,
) -> Result<Vec<f64>, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let u2 = add(u, &phi_term(a, h / 2.0, 1, h / 2.0, f1));
    let f2 = shifted_stage(problem, a, &u2, u, au, t + h / 2.0, stats)?;
    let mut stage = scale(&f2, 2.0);
    axpy(&mut stage, -1.0, f1);
    let u3 = add(u, &phi_term(a, h, 1, h, &stage));
    let f3 = shifted_stage(problem, a, &u3, u, au, t + h, stats)?;
    let mut increment = phi_term(a, h, 1, h, f1);
    axpy(&mut increment, -3.0, &phi_term(a, h, 2, h, f1));
    axpy(&mut increment, 4.0, &phi_term(a, h, 3, h, f1));
    axpy(&mut increment, 4.0, &phi_term(a, h, 2, h, &f2));
    axpy(&mut increment, -8.0, &phi_term(a, h, 3, h, &f2));
    axpy(&mut increment, -1.0, &phi_term(a, h, 2, h, &f3));
    axpy(&mut increment, 4.0, &phi_term(a, h, 3, h, &f3));
    Ok(add(u, &increment))
}

#[allow(clippy::too_many_arguments)]
fn etdrk4<F, P>(
    problem: &OdeProblem<F, P>,
    a: &[f64],
    u: &[f64],
    t: f64,
    h: f64,
    f1: &[f64],
    au: &[f64],
    stats: &mut SolverStats,
) -> Result<Vec<f64>, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let half = h / 2.0;
    let u2 = add(u, &phi_term(a, half, 1, half, f1));
    let f2 = shifted_stage(problem, a, &u2, u, au, t + half, stats)?;
    let u3 = add(u, &phi_term(a, half, 1, half, &f2));
    let f3 = shifted_stage(problem, a, &u3, u, au, t + half, stats)?;
    let mut u4 = u2.clone();
    let mut combination = scale(&f3, 2.0);
    axpy(&mut combination, -1.0, f1);
    axpy(&mut combination, -1.0, au);
    axpy(&mut combination, 1.0, &mat_vec(a, &u2));
    axpy(&mut u4, 1.0, &phi_term(a, half, 1, half, &combination));
    let f4 = shifted_stage(problem, a, &u4, u, au, t + h, stats)?;
    let mut increment = phi_term(a, h, 1, h, f1);
    axpy(&mut increment, -3.0, &phi_term(a, h, 2, h, f1));
    axpy(&mut increment, 4.0, &phi_term(a, h, 3, h, f1));
    for stage in [&f2, &f3] {
        axpy(&mut increment, 2.0, &phi_term(a, h, 2, h, stage));
        axpy(&mut increment, -4.0, &phi_term(a, h, 3, h, stage));
    }
    axpy(&mut increment, -1.0, &phi_term(a, h, 2, h, &f4));
    axpy(&mut increment, 4.0, &phi_term(a, h, 3, h, &f4));
    Ok(add(u, &increment))
}

#[allow(clippy::too_many_arguments)]
fn hoch_ost4<F, P>(
    problem: &OdeProblem<F, P>,
    a: &[f64],
    u: &[f64],
    t: f64,
    h: f64,
    f1: &[f64],
    au: &[f64],
    stats: &mut SolverStats,
) -> Result<Vec<f64>, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let half = h / 2.0;
    let p1h = |v: &[f64]| phi_term(a, half, 1, 1.0, v);
    let p2h = |v: &[f64]| phi_term(a, half, 2, 1.0, v);
    let p3h = |v: &[f64]| phi_term(a, half, 3, 1.0, v);
    let p1 = |v: &[f64]| phi_term(a, h, 1, 1.0, v);
    let p2 = |v: &[f64]| phi_term(a, h, 2, 1.0, v);
    let p3 = |v: &[f64]| phi_term(a, h, 3, 1.0, v);
    let u2 = add(u, &scale(&p1h(f1), half));
    let f2 = shifted_stage(problem, a, &u2, u, au, t + half, stats)?;
    let mut i3 = scale(&p1h(f1), 0.5);
    axpy(&mut i3, -1.0, &p2h(f1));
    axpy(&mut i3, 1.0, &p2h(&f2));
    let u3 = add(u, &scale(&i3, h));
    let f3 = shifted_stage(problem, a, &u3, u, au, t + half, stats)?;
    let mut i4 = p1(f1);
    axpy(&mut i4, -2.0, &p2(f1));
    axpy(&mut i4, 1.0, &p2(&f2));
    axpy(&mut i4, 1.0, &p2(&f3));
    let u4 = add(u, &scale(&i4, h));
    let f4 = shifted_stage(problem, a, &u4, u, au, t + h, stats)?;
    let a52 = |v: &[f64]| combine(&[(0.5, p2h(v)), (-1.0, p3(v)), (0.25, p2(v)), (-0.5, p3h(v))]);
    let a54 = |v: &[f64]| combine(&[(0.25, p2h(v)), (-1.0, a52(v))]);
    let mut i5 = scale(&p1h(f1), 0.5);
    axpy(&mut i5, -2.0, &a52(f1));
    axpy(&mut i5, -1.0, &a54(f1));
    let sum23 = add(&f2, &f3);
    axpy(&mut i5, 1.0, &a52(&sum23));
    axpy(&mut i5, 1.0, &a54(&f4));
    let u5 = add(u, &scale(&i5, h));
    let f5 = shifted_stage(problem, a, &u5, u, au, t + half, stats)?;
    let mut increment = p1(f1);
    axpy(&mut increment, -3.0, &p2(f1));
    axpy(&mut increment, 4.0, &p3(f1));
    axpy(&mut increment, -1.0, &p2(&f4));
    axpy(&mut increment, 4.0, &p3(&f4));
    axpy(&mut increment, 4.0, &p2(&f5));
    axpy(&mut increment, -8.0, &p3(&f5));
    Ok(add(u, &scale(&increment, h)))
}

#[allow(clippy::too_many_arguments)]
fn exp4<F, P>(
    problem: &OdeProblem<F, P>,
    a: &[f64],
    u: &[f64],
    t: f64,
    h: f64,
    f0: &[f64],
    stats: &mut SolverStats,
) -> Result<Vec<f64>, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let k1: Vec<_> = [h / 3.0, 2.0 * h / 3.0, h]
        .into_iter()
        .map(|tau| phi_term(a, tau, 1, 1.0, f0))
        .collect();
    let w4 = combine(&[
        (-7.0 / 300.0, k1[0].clone()),
        (97.0 / 150.0, k1[1].clone()),
        (-37.0 / 300.0, k1[2].clone()),
    ]);
    let u4 = add(u, &scale(&w4, h));
    let d4 = remainder(problem, a, &u4, u, f0, t + h, stats)?;
    let k2: Vec<_> = [h / 3.0, 2.0 * h / 3.0, h]
        .into_iter()
        .map(|tau| phi_term(a, tau, 1, 1.0, &d4))
        .collect();
    let mut w7 = combine(&[
        (59.0 / 300.0, k1[0].clone()),
        (-7.0 / 75.0, k1[1].clone()),
        (269.0 / 300.0, k1[2].clone()),
    ]);
    for value in &k2 {
        axpy(&mut w7, 2.0 / 3.0, value);
    }
    let u7 = add(u, &scale(&w7, h));
    let d7 = remainder(problem, a, &u7, u, f0, t + h, stats)?;
    let k7 = phi_term(a, h / 3.0, 1, 1.0, &d7);
    let mut increment = k1[2].clone();
    axpy(&mut increment, 1.0, &k2[0]);
    axpy(&mut increment, -4.0 / 3.0, &k2[1]);
    axpy(&mut increment, 1.0, &k2[2]);
    axpy(&mut increment, 1.0 / 6.0, &k7);
    Ok(add(u, &scale(&increment, h)))
}

#[allow(clippy::too_many_arguments)]
fn epirk4s3<F, P>(
    problem: &OdeProblem<F, P>,
    a: &[f64],
    u: &[f64],
    t: f64,
    h: f64,
    f0: &[f64],
    variant_b: bool,
    stats: &mut SolverStats,
) -> Result<Vec<f64>, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let (c2, c3, phi_order, r2c, r3c, q2c, q3c) = if variant_b {
        (0.5, 0.75, 2, 54.0, -16.0, -324.0, 144.0)
    } else {
        (0.5, 2.0 / 3.0, 1, 32.0, -13.5, -144.0, 81.0)
    };
    let (k2, k3) = if variant_b {
        (
            phi_term(a, c2 * h, phi_order, 8.0 * (c2 * h).powi(2) / (3.0 * h), f0),
            phi_term(
                a,
                c3 * h,
                phi_order,
                16.0 * (c3 * h).powi(2) / (9.0 * h),
                f0,
            ),
        )
    } else {
        (
            phi_term(a, c2 * h, phi_order, c2 * h, f0),
            phi_term(a, c3 * h, phi_order, c3 * h, f0),
        )
    };
    let u2 = add(u, &k2);
    let u3 = add(u, &k3);
    let r2 = remainder(problem, a, &u2, u, f0, t + c2 * h, stats)?;
    let r3 = remainder(problem, a, &u3, u, f0, t + c3 * h, stats)?;
    let b4 = combine(&[(r2c / h.powi(2), r2.clone()), (r3c / h.powi(2), r3.clone())]);
    let b5 = combine(&[(q2c / h.powi(3), r2), (q3c / h.powi(3), r3)]);
    let mut increment = phi_term(a, h, 1, h, f0);
    axpy(&mut increment, 1.0, &phi_term(a, h, 3, h.powi(3), &b4));
    axpy(&mut increment, 1.0, &phi_term(a, h, 4, h.powi(4), &b5));
    Ok(add(u, &increment))
}

#[allow(clippy::too_many_arguments)]
fn epirk5s3<F, P>(
    problem: &OdeProblem<F, P>,
    a: &[f64],
    u: &[f64],
    t: f64,
    h: f64,
    f0: &[f64],
    stats: &mut SolverStats,
) -> Result<Vec<f64>, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let c2 = 48.0 / 55.0;
    let mut k2 = phi_term(a, c2 * h, 2, (55.0 / (8.0 * h)) * (c2 * h).powi(2), f0);
    axpy(
        &mut k2,
        1.0,
        &phi_term(
            a,
            c2 * h,
            3,
            (-3025.0 / (192.0 * h * h)) * (c2 * h).powi(3),
            f0,
        ),
    );
    let u2 = add(u, &k2);
    let r2 = remainder(problem, a, &u2, u, f0, t + c2 * h, stats)?;
    let c3 = 4.0 / 9.0;
    let mut k3 = phi_term(a, c3 * h, 1, (53.0 / 5.0) * (c3 * h), f0);
    axpy(
        &mut k3,
        1.0,
        &phi_term(a, c3 * h, 2, (-648.0 / (5.0 * h)) * (c3 * h).powi(2), f0),
    );
    let b3 = combine(&[
        (2916.0 / (5.0 * h * h), f0.to_vec()),
        (32065.0 / (1152.0 * h * h), r2.clone()),
    ]);
    axpy(&mut k3, 1.0, &phi_term(a, c3 * h, 3, (c3 * h).powi(3), &b3));
    let u3 = add(u, &k3);
    let r3 = remainder(problem, a, &u3, u, f0, t + c3 * h, stats)?;
    let b4 = combine(&[
        (-166375.0 / (61056.0 * h * h), r2.clone()),
        (2187.0 / (106.0 * h * h), r3.clone()),
    ]);
    let b5 = combine(&[
        (499125.0 / (27136.0 * h.powi(3)), r2),
        (-2187.0 / (106.0 * h.powi(3)), r3),
    ]);
    let mut increment = phi_term(a, h, 1, h, f0);
    axpy(&mut increment, 1.0, &phi_term(a, h, 3, h.powi(3), &b4));
    axpy(&mut increment, 1.0, &phi_term(a, h, 4, h.powi(4), &b5));
    Ok(add(u, &increment))
}

#[allow(clippy::too_many_arguments)]
fn exprb53s3<F, P>(
    problem: &OdeProblem<F, P>,
    a: &[f64],
    u: &[f64],
    t: f64,
    h: f64,
    f0: &[f64],
    stats: &mut SolverStats,
) -> Result<Vec<f64>, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let k2 = phi_term(a, h / 2.0, 1, h / 2.0, f0);
    let k3base = phi_term(a, 0.9 * h, 1, 0.9 * h, f0);
    let u2 = add(u, &k2);
    let r2 = remainder(problem, a, &u2, u, f0, t + h / 2.0, stats)?;
    let mut u3 = add(u, &k3base);
    axpy(
        &mut u3,
        216.0 / (25.0 * h * h),
        &phi_term(a, h / 2.0, 3, (h / 2.0).powi(3), &r2),
    );
    axpy(
        &mut u3,
        8.0 / (h * h),
        &phi_term(a, 0.9 * h, 3, (0.9 * h).powi(3), &r2),
    );
    let r3 = remainder(problem, a, &u3, u, f0, t + 0.9 * h, stats)?;
    let b4 = combine(&[
        (18.0 / (h * h), r2.clone()),
        (-250.0 / (81.0 * h * h), r3.clone()),
    ]);
    let b5 = combine(&[(-60.0 / h.powi(3), r2), (500.0 / (27.0 * h.powi(3)), r3)]);
    let mut increment = phi_term(a, h, 1, h, f0);
    axpy(&mut increment, 1.0, &phi_term(a, h, 3, h.powi(3), &b4));
    axpy(&mut increment, 1.0, &phi_term(a, h, 4, h.powi(4), &b5));
    Ok(add(u, &increment))
}

#[allow(clippy::too_many_arguments)]
fn epirk5p<F, P>(
    problem: &OdeProblem<F, P>,
    a: &[f64],
    u: &[f64],
    t: f64,
    h: f64,
    f0: &[f64],
    variant_2: bool,
    stats: &mut SolverStats,
) -> Result<Vec<f64>, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let (g11, g21, g32, g33, a22, b2, b31, b32, b33, second_order) = if variant_2 {
        (
            0.46629408528088196,
            0.8821791265336387,
            0.9207491648814003,
            0.7979156183266452,
            2.8062037328933126,
            2.528063102562463,
            -0.12848678265700557,
            -0.16102803317280918,
            5.267263316169096,
            2,
        )
    } else {
        (
            0.351_295_926_950_581_9,
            0.8440547201165713,
            0.7111109536436687,
            0.6237811195337149,
            2.377391534041867,
            1.7897526753236234,
            0.0,
            0.0,
            9.358546505792617,
            1,
        )
    };
    let k11 = phi_term(a, g11 * h, 1, g11 * h, f0);
    let k21 = phi_term(a, g21 * h, 1, g21 * h, f0);
    let k31 = phi_term(a, h, 1, h, f0);
    let u1 = add(u, &k11);
    let r1 = remainder(problem, a, &u1, u, f0, t + g11 * h, stats)?;
    let k2 = phi_term(
        a,
        g32 * h,
        second_order,
        (g32 * h).powi(second_order as i32),
        &r1,
    );
    let mut u2 = add(u, &k21);
    axpy(&mut u2, a22 / if variant_2 { h } else { 1.0 }, &k2);
    let r2 = remainder(problem, a, &u2, u, f0, t + g21 * h, stats)?;
    let mut dr = r2;
    axpy(&mut dr, -2.0, &r1);
    let mut output = add(u, &k31);
    axpy(&mut output, b2 / if variant_2 { h } else { 1.0 }, &k2);
    if variant_2 {
        axpy(
            &mut output,
            b31 * (g33 * h),
            &phi_action(a, g33 * h, 1, &dr),
        );
        axpy(
            &mut output,
            b32 / h * (g33 * h).powi(2),
            &phi_action(a, g33 * h, 2, &dr),
        );
        axpy(
            &mut output,
            b33 / (h * h) * (g33 * h).powi(3),
            &phi_action(a, g33 * h, 3, &dr),
        );
    } else {
        axpy(
            &mut output,
            b33 / (h * h),
            &phi_term(a, g33 * h, 3, (g33 * h).powi(3), &dr),
        );
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn exprb32<F, P>(
    problem: &OdeProblem<F, P>,
    a: &[f64],
    u: &[f64],
    t: f64,
    h: f64,
    f1: &[f64],
    stats: &mut SolverStats,
) -> Result<(Vec<f64>, Vec<f64>), SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let w1 = phi_action(a, h, 1, f1);
    let u2 = add(u, &scale(&w1, h));
    let au = mat_vec(a, u);
    let f2 = shifted_stage(problem, a, &u2, u, &au, t + h, stats)?;
    let w13 = phi_action(a, h, 3, f1);
    let w23 = phi_action(a, h, 3, &f2);
    let mut increment = scale(&w1, h);
    axpy(&mut increment, -2.0 * h, &w13);
    axpy(&mut increment, 2.0 * h, &w23);
    let candidate = add(u, &increment);
    let mut error = scale(&w23, 2.0 * h);
    axpy(&mut error, -2.0 * h, &w13);
    Ok((candidate, error))
}

#[allow(clippy::too_many_arguments)]
fn exprb43<F, P>(
    problem: &OdeProblem<F, P>,
    a: &[f64],
    u: &[f64],
    t: f64,
    h: f64,
    f1: &[f64],
    au: &[f64],
    stats: &mut SolverStats,
) -> Result<(Vec<f64>, Vec<f64>), SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let w1h = phi_action(a, h / 2.0, 1, f1);
    let w11 = phi_action(a, h, 1, f1);
    let w13 = phi_action(a, h, 3, f1);
    let w14 = phi_action(a, h, 4, f1);
    let u2 = add(u, &scale(&w1h, h / 2.0));
    let f2 = shifted_stage(problem, a, &u2, u, au, t + h / 2.0, stats)?;
    let w21 = phi_action(a, h, 1, &f2);
    let w23 = phi_action(a, h, 3, &f2);
    let w24 = phi_action(a, h, 4, &f2);
    let u3 = add(u, &scale(&w21, h));
    let f3 = shifted_stage(problem, a, &u3, u, au, t + h, stats)?;
    let w33 = phi_action(a, h, 3, &f3);
    let w34 = phi_action(a, h, 4, &f3);
    let mut increment = scale(&w11, h);
    for (coefficient, value) in [
        (-14.0, &w13),
        (36.0, &w14),
        (16.0, &w23),
        (-48.0, &w24),
        (-2.0, &w33),
        (12.0, &w34),
    ] {
        axpy(&mut increment, coefficient * h, value);
    }
    let candidate = add(u, &increment);
    let mut error = scale(&w14, 36.0 * h);
    axpy(&mut error, -48.0 * h, &w24);
    axpy(&mut error, 12.0 * h, &w34);
    Ok((candidate, error))
}

fn shifted_stage<F, P>(
    problem: &OdeProblem<F, P>,
    a: &[f64],
    stage: &[f64],
    base: &[f64],
    au: &[f64],
    time: f64,
    stats: &mut SolverStats,
) -> Result<Vec<f64>, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let mut value = evaluate_new(problem, stage, time, stats)?;
    axpy(&mut value, -1.0, &mat_vec(a, stage));
    axpy(&mut value, 1.0, au);
    debug_assert_eq!(base.len(), value.len());
    Ok(value)
}

fn remainder<F, P>(
    problem: &OdeProblem<F, P>,
    a: &[f64],
    stage: &[f64],
    base: &[f64],
    f0: &[f64],
    time: f64,
    stats: &mut SolverStats,
) -> Result<Vec<f64>, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let mut value = evaluate_new(problem, stage, time, stats)?;
    axpy(&mut value, -1.0, f0);
    axpy(&mut value, -1.0, &mat_vec(a, &subtract(stage, base)));
    Ok(value)
}

fn evaluate_new<F, P>(
    problem: &OdeProblem<F, P>,
    state: &[f64],
    time: f64,
    stats: &mut SolverStats,
) -> Result<Vec<f64>, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let mut output = vec![0.0; state.len()];
    evaluate(problem, &mut output, state, time, stats)?;
    Ok(output)
}

fn evaluate<F, P>(
    problem: &OdeProblem<F, P>,
    output: &mut [f64],
    state: &[f64],
    time: f64,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    (problem.rhs)(output, state, problem.parameters(), time);
    stats.rhs_evaluations += 1;
    if output.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(SolveError::NonFiniteDerivative)
    }
}

fn compute_jacobian<F, P>(
    problem: &OdeProblem<F, P>,
    state: &[f64],
    time: f64,
    f0: &[f64],
    jacobian: &mut [f64],
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    stats.jacobian_evaluations += 1;
    if problem.evaluate_jacobian(jacobian, state, time) {
        return jacobian
            .iter()
            .all(|v| v.is_finite())
            .then_some(())
            .ok_or(SolveError::NonFiniteDerivative);
    }
    let n = state.len();
    let mut perturbed = state.to_vec();
    let mut derivative = vec![0.0; n];
    for column in 0..n {
        let delta = f64::EPSILON.sqrt() * (1.0 + state[column].abs());
        perturbed[column] += delta;
        evaluate(problem, &mut derivative, &perturbed, time, stats)?;
        for row in 0..n {
            jacobian[row * n + column] = (derivative[row] - f0[row]) / delta;
        }
        perturbed[column] = state[column];
    }
    Ok(())
}

fn scaled_error_norm(
    error: &[f64],
    state: &[f64],
    candidate: &[f64],
    options: &SolveOptions,
) -> f64 {
    if error.is_empty() {
        return 0.0;
    }
    let sum = error
        .iter()
        .zip(state)
        .zip(candidate)
        .map(|((e, u), v)| {
            let scale =
                options.absolute_tolerance + options.relative_tolerance * u.abs().max(v.abs());
            (e / scale).powi(2)
        })
        .sum::<f64>();
    (sum / error.len() as f64).sqrt()
}

fn phi_term(a: &[f64], scale_a: f64, order: usize, coefficient: f64, input: &[f64]) -> Vec<f64> {
    scale(&phi_action(a, scale_a, order, input), coefficient)
}

fn phi_action(a: &[f64], scale_a: f64, order: usize, input: &[f64]) -> Vec<f64> {
    assert!(order > 0);
    let n = input.len();
    let size = n + order;
    let mut augmented = vec![0.0; size * size];
    for row in 0..n {
        for column in 0..n {
            augmented[row * size + column] = scale_a * a[row * n + column];
        }
    }
    for row in 0..n {
        augmented[row * size + n] = input[row];
    }
    for index in 0..order.saturating_sub(1) {
        augmented[(n + index) * size + n + index + 1] = 1.0;
    }
    let exponential = matrix_exp(&augmented, size);
    (0..n)
        .map(|row| exponential[row * size + size - 1])
        .collect()
}

fn exp_action(a: &[f64], scale_a: f64, input: &[f64]) -> Vec<f64> {
    let n = input.len();
    let scaled: Vec<_> = a.iter().map(|value| scale_a * value).collect();
    mat_vec(&matrix_exp(&scaled, n), input)
}

pub(crate) fn matrix_exp(matrix: &[f64], n: usize) -> Vec<f64> {
    let norm = (0..n)
        .map(|row| {
            (0..n)
                .map(|column| matrix[row * n + column].abs())
                .sum::<f64>()
        })
        .fold(0.0, f64::max);
    let squarings = if norm <= 0.5 {
        0
    } else {
        (norm / 0.5).log2().ceil() as u32
    };
    let divisor = 2.0_f64.powi(squarings as i32);
    let scaled: Vec<_> = matrix.iter().map(|value| value / divisor).collect();
    let mut result = identity(n);
    let mut term = identity(n);
    for k in 1..=128 {
        term = mat_mul(&term, &scaled, n);
        for value in &mut term {
            *value /= k as f64;
        }
        let term_norm = term
            .iter()
            .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
        for (result, value) in result.iter_mut().zip(&term) {
            *result += value;
        }
        if term_norm
            <= f64::EPSILON
                * result
                    .iter()
                    .fold(1.0_f64, |maximum, value| maximum.max(value.abs()))
        {
            break;
        }
    }
    for _ in 0..squarings {
        result = mat_mul(&result, &result, n);
    }
    result
}

pub(crate) fn identity(n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        out[i * n + i] = 1.0;
    }
    out
}
pub(crate) fn mat_mul(left: &[f64], right: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..n {
            let value = left[i * n + k];
            for j in 0..n {
                out[i * n + j] += value * right[k * n + j];
            }
        }
    }
    out
}
pub(crate) fn mat_vec(matrix: &[f64], vector: &[f64]) -> Vec<f64> {
    let n = vector.len();
    (0..n)
        .map(|row| {
            (0..n)
                .map(|column| matrix[row * n + column] * vector[column])
                .sum()
        })
        .collect()
}
fn add(left: &[f64], right: &[f64]) -> Vec<f64> {
    left.iter().zip(right).map(|(l, r)| l + r).collect()
}
fn subtract(left: &[f64], right: &[f64]) -> Vec<f64> {
    left.iter().zip(right).map(|(l, r)| l - r).collect()
}
fn scale(vector: &[f64], factor: f64) -> Vec<f64> {
    vector.iter().map(|value| factor * value).collect()
}
fn axpy(output: &mut [f64], factor: f64, input: &[f64]) {
    for (value, input) in output.iter_mut().zip(input) {
        *value += factor * input;
    }
}
fn combine(terms: &[(f64, Vec<f64>)]) -> Vec<f64> {
    let mut output = vec![0.0; terms.first().map_or(0, |(_, v)| v.len())];
    for (coefficient, value) in terms {
        axpy(&mut output, *coefficient, value);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::{ETD1, NorsettEuler, matrix_exp, phi_action};
    use std::any::TypeId;

    #[test]
    fn matrix_functions_match_scalar_definitions() {
        let exponential = matrix_exp(&[1.0], 1)[0];
        let phi1 = phi_action(&[1.0], 1.0, 1, &[1.0])[0];
        let phi2 = phi_action(&[1.0], 1.0, 2, &[1.0])[0];
        assert!((exponential - std::f64::consts::E).abs() < 1.0e-14);
        assert!((phi1 - (std::f64::consts::E - 1.0)).abs() < 1.0e-14);
        assert!((phi2 - (std::f64::consts::E - 2.0)).abs() < 1.0e-14);
    }

    #[test]
    fn etd1_is_an_exact_type_alias() {
        assert_eq!(TypeId::of::<ETD1>(), TypeId::of::<NorsettEuler>());
    }
}
