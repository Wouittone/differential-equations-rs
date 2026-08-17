//! Explicit stabilized Runge--Kutta methods for regular ODE problems.
//!
//! The compact polynomial recurrences in this module are recovered from
//! OrdinaryDiffEqStabilizedRK at commit
//! `34a1983869d1235e8fb5680aafc47cd41da428b3`.  Each implemented method
//! estimates the Jacobian spectral radius by a matrix-free power iteration,
//! chooses a stage count from its own stability bound, and advances with its
//! method-specific Chebyshev, Legendre, or Gegenbauer recurrence.
//!
//! Algorithms whose genuine implementation still needs large coefficient
//! tables, multistep history, or a split implicit problem are intentionally not
//! exported: silently substituting an unrelated method would give those names
//! dishonest numerical semantics.

use crate::integrator::{
    KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

const SPECTRAL_ITERATIONS: usize = 12;
const SPECTRAL_SAFETY: f64 = 1.2;
const MAX_POLYNOMIAL_STAGES: usize = 200;
const MAX_RKMC2_STAGES: usize = 1_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StabilizedFamily {
    Rkc,
    Rkl1,
    Rkl2,
    Rkg1,
    Rkg2,
    Rkmc2,
}

impl StabilizedFamily {
    const fn order(self) -> usize {
        match self {
            Self::Rkl1 | Self::Rkg1 => 1,
            Self::Rkc | Self::Rkl2 | Self::Rkg2 | Self::Rkmc2 => 2,
        }
    }

    fn stages(self, scaled_radius: f64) -> usize {
        let scaled_radius = scaled_radius.max(0.0);
        match self {
            Self::Rkc => ((1.54 * scaled_radius + 1.0).sqrt().floor() as usize + 1)
                .clamp(2, MAX_POLYNOMIAL_STAGES),
            Self::Rkl1 => {
                odd_stage_count((((1.0 + 4.0 * scaled_radius).sqrt() - 1.0) / 2.0).ceil() as usize)
            }
            Self::Rkl2 => {
                odd_stage_count((((9.0 + 8.0 * scaled_radius).sqrt() - 1.0) / 2.0).ceil() as usize)
            }
            Self::Rkg1 => (((9.0 + 16.0 * scaled_radius).sqrt() - 3.0) / 2.0)
                .ceil()
                .max(2.0) as usize,
            Self::Rkg2 => (((25.0 + 24.0 * scaled_radius).sqrt() - 3.0) / 2.0)
                .ceil()
                .max(3.0) as usize,
            Self::Rkmc2 => (-0.830_678_217_871_279_5
                + 1.854_788_782_583_655_3 * scaled_radius.powf(0.533_871_357_807_877))
            .ceil()
            .max(3.0) as usize,
        }
        .min(match self {
            Self::Rkmc2 => MAX_RKMC2_STAGES,
            _ => MAX_POLYNOMIAL_STAGES,
        })
    }
}

fn odd_stage_count(stages: usize) -> usize {
    let stages = stages.max(3);
    if stages % 2 == 0 { stages + 1 } else { stages }.min(MAX_POLYNOMIAL_STAGES - 1)
}

struct StabilizedKernel {
    family: StabilizedFamily,
    first_derivative: Vec<f64>,
    last_derivative: Vec<f64>,
    derivative: Vec<f64>,
    previous_two: Vec<f64>,
    previous_one: Vec<f64>,
    next_stage: Vec<f64>,
    eigenvector: Vec<f64>,
    perturbed_state: Vec<f64>,
    perturbed_derivative: Vec<f64>,
}

impl StabilizedKernel {
    fn new(family: StabilizedFamily, dimension: usize) -> Self {
        Self {
            family,
            first_derivative: vec![0.0; dimension],
            last_derivative: vec![0.0; dimension],
            derivative: vec![0.0; dimension],
            previous_two: vec![0.0; dimension],
            previous_one: vec![0.0; dimension],
            next_stage: vec![0.0; dimension],
            eigenvector: vec![0.0; dimension],
            perturbed_state: vec![0.0; dimension],
            perturbed_derivative: vec![0.0; dimension],
        }
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
        finite(derivative)
    }

    fn spectral_radius<F, P>(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<f64, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        if vector_norm(&self.eigenvector) <= f64::MIN_POSITIVE {
            self.eigenvector.copy_from_slice(&self.first_derivative);
            if vector_norm(&self.eigenvector) <= f64::MIN_POSITIVE {
                self.eigenvector.fill(1.0);
            }
        }

        let perturbation = f64::EPSILON.sqrt() * vector_norm(state).max(1.0);
        let mut previous_radius = 0.0;
        let mut radius = 0.0;
        for iteration in 0..SPECTRAL_ITERATIONS {
            let direction_norm = vector_norm(&self.eigenvector);
            let scale = perturbation / direction_norm.max(f64::MIN_POSITIVE);
            for ((perturbed, value), direction) in self
                .perturbed_state
                .iter_mut()
                .zip(state)
                .zip(&self.eigenvector)
            {
                *perturbed = value + scale * direction;
            }
            Self::evaluate(
                problem,
                &mut self.perturbed_derivative,
                &self.perturbed_state,
                time,
                stats,
            )?;
            for ((direction, perturbed), base) in self
                .eigenvector
                .iter_mut()
                .zip(&self.perturbed_derivative)
                .zip(&self.first_derivative)
            {
                *direction = perturbed - base;
            }
            radius = vector_norm(&self.eigenvector) / perturbation;
            if radius == 0.0 {
                break;
            }
            if iteration > 0 && (radius - previous_radius).abs() <= 0.01 * radius.max(1.0) {
                break;
            }
            previous_radius = radius;
        }
        Ok(SPECTRAL_SAFETY * radius)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_recurrence<F, P>(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        stages: usize,
        candidate: &mut [f64],
        stats: &mut SolverStats,
    ) -> Result<(), SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        match self.family {
            StabilizedFamily::Rkc => {
                self.run_rkc(problem, state, time, step, stages, candidate, stats)
            }
            StabilizedFamily::Rkl1 => self.run_orthogonal_polynomial(
                problem,
                state,
                time,
                step,
                stages,
                candidate,
                stats,
                OrthogonalPolynomial::Legendre1,
            ),
            StabilizedFamily::Rkl2 => self.run_orthogonal_polynomial(
                problem,
                state,
                time,
                step,
                stages,
                candidate,
                stats,
                OrthogonalPolynomial::Legendre2,
            ),
            StabilizedFamily::Rkg1 => self.run_orthogonal_polynomial(
                problem,
                state,
                time,
                step,
                stages,
                candidate,
                stats,
                OrthogonalPolynomial::Gegenbauer1,
            ),
            StabilizedFamily::Rkg2 => self.run_orthogonal_polynomial(
                problem,
                state,
                time,
                step,
                stages,
                candidate,
                stats,
                OrthogonalPolynomial::Gegenbauer2,
            ),
            StabilizedFamily::Rkmc2 => {
                self.run_rkmc2(problem, state, time, step, stages, candidate, stats)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_rkc<F, P>(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        stages: usize,
        candidate: &mut [f64],
        stats: &mut SolverStats,
    ) -> Result<(), SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        let degree = stages as f64;
        let omega_zero_minus_one = (2.0 / 13.0) / degree.powi(2);
        let omega_zero = 1.0 + omega_zero_minus_one;
        let omega_zero_squared_minus_one = omega_zero_minus_one * (omega_zero_minus_one + 2.0);
        let root = omega_zero_squared_minus_one.sqrt();
        let argument = degree * (omega_zero + root).ln();
        let omega_one = argument.sinh() * omega_zero_squared_minus_one
            / (argument.cosh() * degree * root - omega_zero * argument.sinh());
        let mut b_previous = 1.0 / (2.0 * omega_zero).powi(2);
        let mut b_previous_two = b_previous;

        self.previous_two.copy_from_slice(state);
        let mut mu_tilde = omega_one * b_previous;
        for ((output, value), derivative) in self
            .previous_one
            .iter_mut()
            .zip(state)
            .zip(&self.first_derivative)
        {
            *output = value + step * mu_tilde * derivative;
        }
        let mut theta_previous_two = 0.0;
        let mut theta_previous = mu_tilde;
        let mut value_previous = omega_zero;
        let mut value_previous_two = 1.0;
        let mut first_previous = 1.0;
        let mut first_previous_two = 0.0;
        let mut second_previous = 0.0;
        let mut second_previous_two = 0.0;

        for stage in 2..=stages {
            let value = 2.0 * omega_zero * value_previous - value_previous_two;
            let first =
                2.0 * omega_zero * first_previous - first_previous_two + 2.0 * value_previous;
            let second =
                2.0 * omega_zero * second_previous - second_previous_two + 4.0 * first_previous;
            let b = second / first.powi(2);
            let nu_tilde = 1.0 - value_previous * b_previous;
            let mu = 2.0 * omega_zero * b / b_previous;
            let nu = -b / b_previous_two;
            mu_tilde = mu * omega_one / omega_zero;
            Self::evaluate(
                problem,
                &mut self.derivative,
                &self.previous_one,
                time + step * theta_previous,
                stats,
            )?;
            for ((((output, previous), previous_two), initial), (derivative, first_derivative)) in
                self.next_stage
                    .iter_mut()
                    .zip(&self.previous_one)
                    .zip(&self.previous_two)
                    .zip(state)
                    .zip(self.derivative.iter().zip(&self.first_derivative))
            {
                *output = mu * previous
                    + nu * previous_two
                    + (1.0 - mu - nu) * initial
                    + step * mu_tilde * (derivative - nu_tilde * first_derivative);
            }
            let theta = mu * theta_previous + nu * theta_previous_two + mu_tilde * (1.0 - nu_tilde);
            if stage < stages {
                std::mem::swap(&mut self.previous_two, &mut self.previous_one);
                std::mem::swap(&mut self.previous_one, &mut self.next_stage);
                theta_previous_two = theta_previous;
                theta_previous = theta;
                b_previous_two = b_previous;
                b_previous = b;
                value_previous_two = value_previous;
                value_previous = value;
                first_previous_two = first_previous;
                first_previous = first;
                second_previous_two = second_previous;
                second_previous = second;
            }
        }
        candidate.copy_from_slice(&self.next_stage);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_orthogonal_polynomial<F, P>(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        stages: usize,
        candidate: &mut [f64],
        stats: &mut SolverStats,
        polynomial: OrthogonalPolynomial,
    ) -> Result<(), SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        let degree = stages as f64;
        let omega_one = match polynomial {
            OrthogonalPolynomial::Legendre1 => 2.0 / (degree.powi(2) + degree),
            OrthogonalPolynomial::Legendre2 => 4.0 / (degree.powi(2) + degree - 2.0),
            OrthogonalPolynomial::Gegenbauer1 => 4.0 / (degree * (degree + 3.0)),
            OrthogonalPolynomial::Gegenbauer2 => 6.0 / ((degree + 4.0) * (degree - 1.0)),
        };
        let first_mu_tilde = match polynomial {
            OrthogonalPolynomial::Legendre1 => omega_one,
            OrthogonalPolynomial::Legendre2 => omega_one / 3.0,
            OrthogonalPolynomial::Gegenbauer1 => omega_one,
            OrthogonalPolynomial::Gegenbauer2 => omega_one,
        };

        self.previous_two.copy_from_slice(state);
        for ((output, value), derivative) in self
            .previous_one
            .iter_mut()
            .zip(state)
            .zip(&self.first_derivative)
        {
            *output = value + step * first_mu_tilde * derivative;
        }
        let mut theta_previous_two = 0.0;
        let mut theta_previous = first_mu_tilde;

        for stage in 2..=stages {
            let stage_f64 = stage as f64;
            let (mu, nu, mu_tilde, gamma_tilde) = match polynomial {
                OrthogonalPolynomial::Legendre1 => {
                    let mu = (2.0 * stage_f64 - 1.0) / stage_f64;
                    let nu = -(stage_f64 - 1.0) / stage_f64;
                    (mu, nu, mu * omega_one, 0.0)
                }
                OrthogonalPolynomial::Legendre2 => {
                    let b = legendre_two_b(stage);
                    let b_previous = legendre_two_b(stage - 1);
                    let b_previous_two = legendre_two_b(stage - 2);
                    let mu = (2.0 * stage_f64 - 1.0) / stage_f64 * b / b_previous;
                    let nu = -(stage_f64 - 1.0) / stage_f64 * b / b_previous_two;
                    let mu_tilde = mu * omega_one;
                    (mu, nu, mu_tilde, -(1.0 - b_previous) * mu_tilde)
                }
                OrthogonalPolynomial::Gegenbauer1 => {
                    let b = gegenbauer_one_b(stage);
                    let b_previous = gegenbauer_one_b(stage - 1);
                    let b_previous_two = gegenbauer_one_b(stage - 2);
                    let mu = (2.0 * stage_f64 + 1.0) / stage_f64 * b / b_previous;
                    let nu = -(stage_f64 + 1.0) / stage_f64 * b / b_previous_two;
                    (mu, nu, mu * omega_one, 0.0)
                }
                OrthogonalPolynomial::Gegenbauer2 => {
                    let b = gegenbauer_two_b(stage);
                    let b_previous = gegenbauer_two_b(stage - 1);
                    let b_previous_two = gegenbauer_two_b(stage - 2);
                    let mu = (2.0 * stage_f64 + 1.0) / stage_f64 * b / b_previous;
                    let nu = -(stage_f64 + 1.0) / stage_f64 * b / b_previous_two;
                    let mu_tilde = mu * omega_one;
                    let a_previous = 1.0 - (stage * (stage + 1) / 2) as f64 * b_previous;
                    (mu, nu, mu_tilde, -mu_tilde * a_previous)
                }
            };
            Self::evaluate(
                problem,
                &mut self.derivative,
                &self.previous_one,
                time + step * theta_previous,
                stats,
            )?;
            for ((((output, previous), previous_two), initial), (derivative, first_derivative)) in
                self.next_stage
                    .iter_mut()
                    .zip(&self.previous_one)
                    .zip(&self.previous_two)
                    .zip(state)
                    .zip(self.derivative.iter().zip(&self.first_derivative))
            {
                *output = mu * previous
                    + nu * previous_two
                    + (1.0 - mu - nu) * initial
                    + step * mu_tilde * derivative
                    + step * gamma_tilde * first_derivative;
            }
            let theta = mu * theta_previous + nu * theta_previous_two + mu_tilde + gamma_tilde;
            if stage < stages {
                std::mem::swap(&mut self.previous_two, &mut self.previous_one);
                std::mem::swap(&mut self.previous_one, &mut self.next_stage);
                theta_previous_two = theta_previous;
                theta_previous = theta;
            }
        }
        candidate.copy_from_slice(&self.next_stage);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_rkmc2<F, P>(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        stages: usize,
        candidate: &mut [f64],
        stats: &mut SolverStats,
    ) -> Result<(), SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        let degree = stages as f64;
        let sign = if stages % 2 == 0 { 1.0 } else { -1.0 };
        let mut alpha_low = 1.0e-10;
        let mut alpha_high = 2.0;
        for _ in 0..50 {
            let alpha: f64 = 0.5 * (alpha_low + alpha_high);
            let value = 1.0
                + sign / (degree * (degree - 2.0))
                + alpha.cosh()
                + (degree * alpha).cosh() / (2.0 * degree)
                - ((degree - 2.0) * alpha).cosh() / (2.0 * (degree - 2.0))
                - (1.0 + ((degree - 1.0) * alpha).cosh()).powi(2)
                    / ((degree - 1.0) * ((degree - 1.0) * alpha).sinh() / alpha.sinh());
            if value > 0.0 {
                alpha_low = alpha;
            } else {
                alpha_high = alpha;
            }
        }
        let alpha = 0.5 * (alpha_low + alpha_high);
        let omega_zero = alpha.cosh();
        let omega_one = (1.0 + ((degree - 1.0) * alpha).cosh())
            / ((degree - 1.0) * ((degree - 1.0) * alpha).sinh() / alpha.sinh());

        let mut chebyshev_previous_two = 1.0;
        let mut chebyshev_previous = omega_zero;
        let mut b_previous_two = 1.0 / (1.0 + chebyshev_previous_two);
        let mut b_previous = 1.0 / (1.0 + chebyshev_previous);
        self.previous_two.copy_from_slice(state);
        let mu_tilde_first = b_previous * omega_one;
        for ((output, value), derivative) in self
            .previous_one
            .iter_mut()
            .zip(state)
            .zip(&self.first_derivative)
        {
            *output = value + step * mu_tilde_first * derivative;
        }
        let mut theta_previous_two = 0.0;
        let mut theta_previous = mu_tilde_first;
        let mut b_final = b_previous;

        for stage in 2..=stages {
            let chebyshev = 2.0 * omega_zero * chebyshev_previous - chebyshev_previous_two;
            let b = 1.0 / (1.0 + chebyshev);
            b_final = b;
            let mu = 2.0 * omega_zero * b / b_previous;
            let nu = -b / b_previous_two;
            let mu_tilde = 2.0 * omega_one * b / b_previous;
            Self::evaluate(
                problem,
                &mut self.derivative,
                &self.previous_one,
                time + step * theta_previous,
                stats,
            )?;
            for ((((output, previous), previous_two), initial), (derivative, first_derivative)) in
                self.next_stage
                    .iter_mut()
                    .zip(&self.previous_one)
                    .zip(&self.previous_two)
                    .zip(state)
                    .zip(self.derivative.iter().zip(&self.first_derivative))
            {
                *output = (1.0 - mu - nu) * initial
                    + mu * previous
                    + nu * previous_two
                    + step * mu_tilde * (derivative - b_previous * first_derivative);
            }
            let theta =
                mu * theta_previous + nu * theta_previous_two + mu_tilde * (1.0 - b_previous);
            if stage < stages {
                std::mem::swap(&mut self.previous_two, &mut self.previous_one);
                std::mem::swap(&mut self.previous_one, &mut self.next_stage);
                theta_previous_two = theta_previous;
                theta_previous = theta;
                chebyshev_previous_two = chebyshev_previous;
                chebyshev_previous = chebyshev;
                b_previous_two = b_previous;
                b_previous = b;
            }
        }

        let gamma = b_previous / (2.0 * degree * omega_one);
        let delta = -b_previous / (2.0 * (degree - 2.0) * omega_one);
        for ((((output, initial), final_stage), older_stage), first_derivative) in candidate
            .iter_mut()
            .zip(state)
            .zip(&self.next_stage)
            .zip(&self.previous_two)
            .zip(&self.first_derivative)
        {
            *output = (1.0 - gamma / b_final - delta / b_previous_two) * initial
                + (gamma / b_final) * final_stage
                + (delta / b_previous_two) * older_stage
                + step * b_previous * first_derivative;
        }
        Ok(())
    }
}

impl<F, P> StepKernel<F, P> for StabilizedKernel
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::new(true, self.family.order())
    }

    fn initialize(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        Self::evaluate(problem, &mut self.first_derivative, state, time, stats)
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
        options: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError> {
        let radius = self.spectral_radius(problem, state, time, stats)?;
        let stages = self.family.stages(step.abs() * radius);
        self.run_recurrence(problem, state, time, step, stages, candidate, stats)?;
        finite(candidate)?;
        Self::evaluate(
            problem,
            &mut self.last_derivative,
            candidate,
            time + step,
            stats,
        )?;

        if !options.adaptive {
            return Ok(StepEstimate::new(0.0));
        }
        for ((((error, initial), final_state), first), last) in self
            .derivative
            .iter_mut()
            .zip(state)
            .zip(candidate.iter())
            .zip(&self.first_derivative)
            .zip(&self.last_derivative)
        {
            *error = match self.family {
                StabilizedFamily::Rkc => {
                    (4.0 * (initial - final_state) + 2.0 * step * (first + last)) / 5.0
                }
                StabilizedFamily::Rkmc2 => (initial - final_state + step * last) / 10.0,
                StabilizedFamily::Rkl1
                | StabilizedFamily::Rkl2
                | StabilizedFamily::Rkg1
                | StabilizedFamily::Rkg2 => final_state - (initial + step * first),
            };
        }
        let divisor = match self.family {
            StabilizedFamily::Rkl1
            | StabilizedFamily::Rkl2
            | StabilizedFamily::Rkg1
            | StabilizedFamily::Rkg2 => stages as f64,
            StabilizedFamily::Rkc | StabilizedFamily::Rkmc2 => 1.0,
        };
        Ok(StepEstimate::new(
            scaled_error_norm(&self.derivative, state, candidate, options) / divisor,
        ))
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
        if callback_applied {
            Self::evaluate(problem, &mut self.first_derivative, state, time, stats)?;
            self.eigenvector.fill(0.0);
        } else {
            self.first_derivative.copy_from_slice(&self.last_derivative);
        }
        Ok(())
    }

    fn reject_step(&mut self) {}
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OrthogonalPolynomial {
    Legendre1,
    Legendre2,
    Gegenbauer1,
    Gegenbauer2,
}

fn legendre_two_b(stage: usize) -> f64 {
    if stage <= 2 {
        1.0 / 3.0
    } else {
        let stage = stage as f64;
        (stage * stage + stage - 2.0) / (2.0 * stage * (stage + 1.0))
    }
}

fn gegenbauer_one_b(stage: usize) -> f64 {
    2.0 / ((stage + 1) * (stage + 2)) as f64
}

fn gegenbauer_two_b(stage: usize) -> f64 {
    match stage {
        0 => 1.0,
        1 => 1.0 / 3.0,
        _ => {
            let stage = stage as f64;
            4.0 * (stage - 1.0) * (stage + 4.0)
                / (3.0 * stage * (stage + 1.0) * (stage + 2.0) * (stage + 3.0))
        }
    }
}

fn vector_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn scaled_error_norm(
    error: &[f64],
    initial: &[f64],
    candidate: &[f64],
    options: &SolveOptions,
) -> f64 {
    let sum = error
        .iter()
        .zip(initial)
        .zip(candidate)
        .map(|((error, initial), candidate)| {
            let scale = options.absolute_tolerance
                + options.relative_tolerance * initial.abs().max(candidate.abs());
            (error / scale).powi(2)
        })
        .sum::<f64>();
    (sum / error.len() as f64).sqrt()
}

fn finite(values: &[f64]) -> Result<(), SolveError> {
    values
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteDerivative)
}

macro_rules! implemented_method {
    ($name:ident, $family:expr, $documentation:literal) => {
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
                drive_integration(
                    problem,
                    options,
                    StabilizedKernel::new($family, problem.initial_state().len()),
                )
            }
        }
    };
}

implemented_method!(
    RKC,
    StabilizedFamily::Rkc,
    "Second-order Runge--Kutta--Chebyshev method with variable stage count."
);
implemented_method!(
    RKL1,
    StabilizedFamily::Rkl1,
    "First-order Runge--Kutta--Legendre super-time-stepping method."
);
implemented_method!(
    RKL2,
    StabilizedFamily::Rkl2,
    "Second-order Runge--Kutta--Legendre super-time-stepping method."
);
implemented_method!(
    RKG1,
    StabilizedFamily::Rkg1,
    "First-order Runge--Kutta--Gegenbauer super-time-stepping method."
);
implemented_method!(
    RKG2,
    StabilizedFamily::Rkg2,
    "Second-order Runge--Kutta--Gegenbauer super-time-stepping method."
);
implemented_method!(
    RKMC2,
    StabilizedFamily::Rkmc2,
    "Second-order monotone Runge--Kutta--Chebyshev method."
);
