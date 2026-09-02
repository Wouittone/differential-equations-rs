//! Explicit stabilized Runge--Kutta methods for regular ODE problems.
//!
//! The compact polynomial recurrences in this module are recovered from
//! OrdinaryDiffEqStabilizedRK at commit
//! `34a1983869d1235e8fb5680aafc47cd41da428b3`.  Each implemented method
//! estimates the Jacobian spectral radius by a matrix-free power iteration,
//! chooses a stage count from its own stability bound, and advances with its
//! method-specific Chebyshev, Legendre, or Gegenbauer recurrence.
//!
//! The two-step TSRKC recurrences keep their accepted-step history inside the
//! kernel. ROCK2, ROCK4, SERK2, ESERK4, and ESERK5 select their full
//! degree-indexed coefficient banks from the compile-time resources in
//! `src/tableau/resources/methods/stabilized`. No degree subset or substitute
//! recurrence is used.

use super::coefficient_data::{
    ESERK4_DEGREES, ESERK4_ERROR_COMBINATION, ESERK4_SOLUTION_COMBINATION, ESERK4_WEIGHTS,
    ESERK5_DEGREES, ESERK5_ERROR_COMBINATION, ESERK5_SOLUTION_COMBINATION, ESERK5_WEIGHTS,
    ROCK2_DEGREES, ROCK2_FINISH_FIRST, ROCK2_FINISH_SECOND, ROCK2_RECURRENCE, ROCK4_DEGREES,
    ROCK4_FINISH_A, ROCK4_FINISH_B, ROCK4_FINISH_ERROR, ROCK4_RECURRENCE, SERK2_DEGREES,
    SERK2_WEIGHTS,
};
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
    Rock2,
    Rock4,
    Serk2,
    Eserk4,
    Eserk5,
    Tsrkc2,
    Tsrkc3,
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
            Self::Rkc
            | Self::Rock2
            | Self::Serk2
            | Self::Tsrkc2
            | Self::Rkl2
            | Self::Rkg2
            | Self::Rkmc2 => 2,
            Self::Rock4 | Self::Eserk4 => 4,
            Self::Eserk5 => 5,
            Self::Tsrkc3 => 3,
        }
    }

    fn stages(self, scaled_radius: f64) -> Option<usize> {
        let scaled_radius = scaled_radius.max(0.0);
        let stages = match self {
            Self::Rkc => ((1.54 * scaled_radius + 1.0).sqrt().floor() as usize + 1)
                .clamp(2, MAX_POLYNOMIAL_STAGES),
            Self::Rock2
            | Self::Rock4
            | Self::Serk2
            | Self::Eserk4
            | Self::Eserk5
            | Self::Tsrkc2
            | Self::Tsrkc3 => return None,
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
        };
        Some(stages.min(if self == Self::Rkmc2 {
            MAX_RKMC2_STAGES
        } else {
            MAX_POLYNOMIAL_STAGES
        }))
    }
}

fn odd_stage_count(stages: usize) -> usize {
    let stages = stages.max(3);
    if stages % 2 == 0 { stages + 1 } else { stages }.min(MAX_POLYNOMIAL_STAGES - 1)
}

fn select_rock_degree(degrees: &[usize], requested: usize) -> (usize, usize, usize) {
    let mut start = 0;
    for (index, &degree) in degrees.iter().enumerate() {
        if degree >= requested {
            return (index, degree, start);
        }
        start += 2 * degree - 1;
    }
    let index = degrees.len() - 1;
    (index, degrees[index], start - (2 * degrees[index] - 1))
}

fn select_serk_degree(degrees: &[usize], requested: usize) -> (usize, usize) {
    let mut start = 0;
    for &degree in degrees {
        if degree >= requested {
            return (degree, start);
        }
        start += degree + 1;
    }
    let degree = degrees[degrees.len() - 1];
    (degree, start - degree - 1)
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
    previous_accepted_state: Option<Vec<f64>>,
    previous_accepted_step: Option<f64>,
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
            previous_accepted_state: None,
            previous_accepted_step: None,
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
        F: crate::OdeFunction<P>,
    {
        problem
            .rhs
            .evaluate(derivative, state, problem.parameters(), time)?;
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
        F: crate::OdeFunction<P>,
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
        F: crate::OdeFunction<P>,
    {
        match self.family {
            StabilizedFamily::Rkc => {
                self.run_rkc(problem, state, time, step, stages, candidate, stats)
            }
            StabilizedFamily::Tsrkc2
            | StabilizedFamily::Tsrkc3
            | StabilizedFamily::Rock2
            | StabilizedFamily::Rock4
            | StabilizedFamily::Serk2
            | StabilizedFamily::Eserk4
            | StabilizedFamily::Eserk5 => Err(SolveError::InvalidTableau),
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
        F: crate::OdeFunction<P>,
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
    fn run_rock2<F, P>(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        scaled_radius: f64,
        candidate: &mut [f64],
        stats: &mut SolverStats,
    ) -> Result<(), SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        let requested_total = (((1.5 + scaled_radius) / 0.811).sqrt().floor() as usize + 1)
            .clamp(1, MAX_POLYNOMIAL_STAGES);
        let requested_degree = requested_total.max(3) - 2;
        let (degree_index, degree, start) = select_rock_degree(ROCK2_DEGREES, requested_degree);

        let first_coefficient = ROCK2_RECURRENCE[start];
        let mut time_previous = time + step * first_coefficient;
        let mut time_previous_two = time_previous;
        let mut time_previous_three = time;
        self.previous_two.copy_from_slice(state);
        for ((output, value), derivative) in self
            .previous_one
            .iter_mut()
            .zip(state)
            .zip(&self.first_derivative)
        {
            *output = value + step * first_coefficient * derivative;
        }

        if degree == 1 {
            self.next_stage.copy_from_slice(&self.previous_one);
        } else {
            for stage in 2..=degree {
                let coefficient = start + (stage - 2) * 2 + 1;
                let mu = ROCK2_RECURRENCE[coefficient];
                let kappa = ROCK2_RECURRENCE[coefficient + 1];
                let nu = -1.0 - kappa;
                Self::evaluate(
                    problem,
                    &mut self.derivative,
                    &self.previous_one,
                    time_previous,
                    stats,
                )?;
                for (((output, previous), previous_two), derivative) in self
                    .next_stage
                    .iter_mut()
                    .zip(&self.previous_one)
                    .zip(&self.previous_two)
                    .zip(&self.derivative)
                {
                    *output = step * mu * derivative - nu * previous - kappa * previous_two;
                }
                time_previous = step * mu - nu * time_previous_two - kappa * time_previous_three;
                time_previous_three = time_previous_two;
                time_previous_two = time_previous;
                if stage < degree {
                    std::mem::swap(&mut self.previous_two, &mut self.previous_one);
                    std::mem::swap(&mut self.previous_one, &mut self.next_stage);
                }
            }
        }

        let finish_first = step * ROCK2_FINISH_FIRST[degree_index];
        let finish_second = step * ROCK2_FINISH_SECOND[degree_index];
        Self::evaluate(
            problem,
            &mut self.derivative,
            &self.next_stage,
            time_previous,
            stats,
        )?;
        for ((output, value), derivative) in self
            .previous_one
            .iter_mut()
            .zip(&self.next_stage)
            .zip(&self.derivative)
        {
            *output = value + finish_first * derivative;
        }
        Self::evaluate(
            problem,
            &mut self.perturbed_derivative,
            &self.previous_one,
            time_previous + finish_first,
            stats,
        )?;
        for ((((output, intermediate), first), second), error) in candidate
            .iter_mut()
            .zip(&self.previous_one)
            .zip(&self.derivative)
            .zip(&self.perturbed_derivative)
            .zip(self.eigenvector.iter_mut())
        {
            *error = finish_second * (second - first);
            *output = intermediate + finish_first * second + *error;
        }
        self.derivative.copy_from_slice(&self.eigenvector);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_rock4<F, P>(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        scaled_radius: f64,
        candidate: &mut [f64],
        stats: &mut SolverStats,
    ) -> Result<(), SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        let requested_total =
            (((3.0 + scaled_radius) / 0.353).sqrt().floor() as usize + 1).clamp(1, 152);
        let requested_degree = requested_total.max(5) - 4;
        let (degree_index, degree, start) = select_rock_degree(ROCK4_DEGREES, requested_degree);
        let first_coefficient = ROCK4_RECURRENCE[start];
        let mut recurrence_time = time + step * first_coefficient;
        let mut time_previous_two = recurrence_time;
        let mut time_previous_three = time;
        self.previous_two.copy_from_slice(state);
        for ((output, value), derivative) in self
            .previous_one
            .iter_mut()
            .zip(state)
            .zip(&self.first_derivative)
        {
            *output = value + step * first_coefficient * derivative;
        }
        if degree == 1 {
            self.next_stage.copy_from_slice(&self.previous_one);
        } else {
            for stage in 2..=degree {
                let coefficient = start + (stage - 2) * 2 + 1;
                let mu = ROCK4_RECURRENCE[coefficient];
                let kappa = ROCK4_RECURRENCE[coefficient + 1];
                let nu = -1.0 - kappa;
                Self::evaluate(
                    problem,
                    &mut self.derivative,
                    &self.previous_one,
                    recurrence_time,
                    stats,
                )?;
                for (((output, previous), previous_two), derivative) in self
                    .next_stage
                    .iter_mut()
                    .zip(&self.previous_one)
                    .zip(&self.previous_two)
                    .zip(&self.derivative)
                {
                    *output = step * mu * derivative - nu * previous - kappa * previous_two;
                }
                recurrence_time = step * mu - nu * time_previous_two - kappa * time_previous_three;
                time_previous_three = time_previous_two;
                time_previous_two = recurrence_time;
                if stage < degree {
                    std::mem::swap(&mut self.previous_two, &mut self.previous_one);
                    std::mem::swap(&mut self.previous_one, &mut self.next_stage);
                }
            }
        }

        let a_offset = degree_index * 6;
        let b_offset = degree_index * 4;
        let error_offset = degree_index * 5;
        let a21 = step * ROCK4_FINISH_A[a_offset];
        let a31 = step * ROCK4_FINISH_A[a_offset + 1];
        let a32 = step * ROCK4_FINISH_A[a_offset + 2];
        let a41 = step * ROCK4_FINISH_A[a_offset + 3];
        let a42 = step * ROCK4_FINISH_A[a_offset + 4];
        let a43 = step * ROCK4_FINISH_A[a_offset + 5];
        let b1 = step * ROCK4_FINISH_B[b_offset];
        let b2 = step * ROCK4_FINISH_B[b_offset + 1];
        let b3 = step * ROCK4_FINISH_B[b_offset + 2];
        let b4 = step * ROCK4_FINISH_B[b_offset + 3];
        let error1 = step * (ROCK4_FINISH_ERROR[error_offset] - ROCK4_FINISH_B[b_offset]);
        let error2 = step * (ROCK4_FINISH_ERROR[error_offset + 1] - ROCK4_FINISH_B[b_offset + 1]);
        let error3 = step * (ROCK4_FINISH_ERROR[error_offset + 2] - ROCK4_FINISH_B[b_offset + 2]);
        let error4 = step * (ROCK4_FINISH_ERROR[error_offset + 3] - ROCK4_FINISH_B[b_offset + 3]);
        let error5 = step * ROCK4_FINISH_ERROR[error_offset + 4];

        Self::evaluate(
            problem,
            &mut self.derivative,
            &self.next_stage,
            recurrence_time,
            stats,
        )?;
        for ((((((accumulator, stage_two), stage_three), stage_four), error), base), derivative) in
            candidate
                .iter_mut()
                .zip(&mut self.previous_one)
                .zip(&mut self.previous_two)
                .zip(&mut self.perturbed_state)
                .zip(&mut self.eigenvector)
                .zip(&self.next_stage)
                .zip(&self.derivative)
        {
            *stage_two = base + a21 * derivative;
            *stage_three = base + a31 * derivative;
            *stage_four = base + a41 * derivative;
            *accumulator = base + b1 * derivative;
            *error = error1 * derivative;
        }

        Self::evaluate(
            problem,
            &mut self.derivative,
            &self.previous_one,
            recurrence_time + a21,
            stats,
        )?;
        for (((accumulator, stage_three), stage_four), (error, derivative)) in candidate
            .iter_mut()
            .zip(&mut self.previous_two)
            .zip(&mut self.perturbed_state)
            .zip(self.eigenvector.iter_mut().zip(&self.derivative))
        {
            *stage_three += a32 * derivative;
            *stage_four += a42 * derivative;
            *accumulator += b2 * derivative;
            *error += error2 * derivative;
        }
        Self::evaluate(
            problem,
            &mut self.derivative,
            &self.previous_two,
            recurrence_time + a31 + a32,
            stats,
        )?;
        for ((accumulator, stage_four), (error, derivative)) in candidate
            .iter_mut()
            .zip(&mut self.perturbed_state)
            .zip(self.eigenvector.iter_mut().zip(&self.derivative))
        {
            *stage_four += a43 * derivative;
            *accumulator += b3 * derivative;
            *error += error3 * derivative;
        }
        Self::evaluate(
            problem,
            &mut self.derivative,
            &self.perturbed_state,
            recurrence_time + a41 + a42 + a43,
            stats,
        )?;
        for ((accumulator, error), derivative) in candidate
            .iter_mut()
            .zip(self.eigenvector.iter_mut())
            .zip(&self.derivative)
        {
            *accumulator += b4 * derivative;
            *error += error4 * derivative;
        }
        Self::evaluate(
            problem,
            &mut self.perturbed_derivative,
            candidate,
            time + step,
            stats,
        )?;
        for (error, derivative) in self.eigenvector.iter_mut().zip(&self.perturbed_derivative) {
            *error += error5 * derivative;
        }
        self.derivative.copy_from_slice(&self.eigenvector);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_serk2<F, P>(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        scaled_radius: f64,
        candidate: &mut [f64],
        stats: &mut SolverStats,
    ) -> Result<(), SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        let requested = ((scaled_radius / 0.8).sqrt().floor() as usize + 1).min(250);
        let (degree, start) = select_serk_degree(SERK2_DEGREES, requested);
        let internal_degree = degree / 10;
        let alpha = 2.5 / (degree * degree) as f64;
        self.previous_one.copy_from_slice(state);
        self.previous_two.copy_from_slice(state);
        for (sum, value) in self.perturbed_state.iter_mut().zip(state) {
            *sum = SERK2_WEIGHTS[start] * value;
        }

        for block in 0..10 {
            let first_time =
                time + (1 + block * internal_degree * internal_degree) as f64 * alpha * step;
            Self::evaluate(
                problem,
                &mut self.derivative,
                &self.previous_one,
                first_time,
                stats,
            )?;
            for ((output, previous), derivative) in self
                .next_stage
                .iter_mut()
                .zip(&self.previous_one)
                .zip(&self.derivative)
            {
                *output = previous + alpha * step * derivative;
            }
            let first_weight = start + block * internal_degree + 1;
            for (sum, value) in self.perturbed_state.iter_mut().zip(&self.next_stage) {
                *sum += SERK2_WEIGHTS[first_weight] * value;
            }
            std::mem::swap(&mut self.previous_two, &mut self.previous_one);
            std::mem::swap(&mut self.previous_one, &mut self.next_stage);

            for stage in 2..=internal_degree {
                let stage_time = time
                    + (stage * stage + block * internal_degree * internal_degree) as f64
                        * alpha
                        * step;
                Self::evaluate(
                    problem,
                    &mut self.derivative,
                    &self.previous_one,
                    stage_time,
                    stats,
                )?;
                for (((output, previous), previous_two), derivative) in self
                    .next_stage
                    .iter_mut()
                    .zip(&self.previous_one)
                    .zip(&self.previous_two)
                    .zip(&self.derivative)
                {
                    *output = 2.0 * previous - previous_two + 2.0 * alpha * step * derivative;
                }
                let weight = start + stage + block * internal_degree;
                for (sum, value) in self.perturbed_state.iter_mut().zip(&self.next_stage) {
                    *sum += SERK2_WEIGHTS[weight] * value;
                }
                if stage < internal_degree || block < 9 {
                    std::mem::swap(&mut self.previous_two, &mut self.previous_one);
                    std::mem::swap(&mut self.previous_one, &mut self.next_stage);
                }
            }
        }
        candidate.copy_from_slice(&self.perturbed_state);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_eserk<F, P>(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        scaled_radius: f64,
        candidate: &mut [f64],
        stats: &mut SolverStats,
        fifth_order: bool,
    ) -> Result<(), SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        let (degrees, solution_combination, error_combination, weights, requested, subdivisions) =
            if fifth_order {
                (
                    ESERK5_DEGREES,
                    ESERK5_SOLUTION_COMBINATION,
                    ESERK5_ERROR_COMBINATION,
                    ESERK5_WEIGHTS,
                    ((scaled_radius / 0.98).sqrt().floor() as usize + 1).min(2_000),
                    5,
                )
            } else {
                (
                    ESERK4_DEGREES,
                    ESERK4_SOLUTION_COMBINATION,
                    ESERK4_ERROR_COMBINATION,
                    ESERK4_WEIGHTS,
                    (scaled_radius.sqrt().floor() as usize + 1).min(4_000),
                    4,
                )
            };
        let (degree, start) = select_serk_degree(degrees, requested);
        let internal_degree = if fifth_order {
            match degree {
                0..=20 => 2,
                21..=50 => 5,
                51..=100 => 10,
                101..=500 => 50,
                501..=1_000 => 100,
                _ => 200,
            }
        } else {
            match degree {
                0..=20 => 2,
                21..=100 => 10,
                101..=500 => 25,
                501..=1_000 => 100,
                _ => 200,
            }
        };
        let alpha = if fifth_order {
            100.0 / (49 * degree * degree) as f64
        } else {
            2.0 / (degree * degree) as f64
        };
        candidate.fill(0.0);
        self.eigenvector.fill(0.0);

        for subdivision in 1..=subdivisions {
            let substep = step / subdivision as f64;
            let mut substep_time = time;
            for repetition in 1..=subdivision {
                self.previous_one.copy_from_slice(if repetition == 1 {
                    state
                } else {
                    &self.perturbed_state
                });
                self.previous_two.fill(0.0);
                for (sum, value) in self.perturbed_state.iter_mut().zip(&self.previous_one) {
                    *sum = weights[start] * value;
                }
                let mut stage_time = substep_time;
                for stage in 1..=degree {
                    Self::evaluate(
                        problem,
                        &mut self.derivative,
                        &self.previous_one,
                        stage_time,
                        stats,
                    )?;
                    if stage % internal_degree == 1 {
                        for ((output, previous), derivative) in self
                            .next_stage
                            .iter_mut()
                            .zip(&self.previous_one)
                            .zip(&self.derivative)
                        {
                            *output = previous + alpha * substep * derivative;
                        }
                    } else {
                        for (((output, previous), previous_two), derivative) in self
                            .next_stage
                            .iter_mut()
                            .zip(&self.previous_one)
                            .zip(&self.previous_two)
                            .zip(&self.derivative)
                        {
                            *output =
                                2.0 * previous - previous_two + 2.0 * alpha * substep * derivative;
                        }
                    }
                    let block = stage / internal_degree;
                    stage_time = substep_time
                        + alpha
                            * (stage * stage + block * internal_degree * internal_degree) as f64
                            * substep;
                    for (sum, value) in self.perturbed_state.iter_mut().zip(&self.next_stage) {
                        *sum += weights[start + stage] * value;
                    }
                    if stage < degree {
                        std::mem::swap(&mut self.previous_two, &mut self.previous_one);
                        std::mem::swap(&mut self.previous_one, &mut self.next_stage);
                    }
                }
                if repetition < subdivision {
                    substep_time += substep;
                }
            }
            let solution_factor = solution_combination[subdivision - 1] as f64;
            let error_factor = error_combination[subdivision - 1] as f64;
            for ((output, error), sum) in candidate
                .iter_mut()
                .zip(self.eigenvector.iter_mut())
                .zip(&self.perturbed_state)
            {
                *output += solution_factor * sum;
                *error += error_factor * sum;
            }
        }
        let denominator = if fifth_order { 24.0 } else { 6.0 };
        for (output, error) in candidate.iter_mut().zip(self.eigenvector.iter_mut()) {
            *output /= denominator;
            *error /= denominator;
        }
        self.derivative.copy_from_slice(&self.eigenvector);
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
        F: crate::OdeFunction<P>,
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
    fn run_tsrkc2<F, P>(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        scaled_radius: f64,
        candidate: &mut [f64],
        stats: &mut SolverStats,
    ) -> Result<(), SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        let q = self
            .previous_accepted_step
            .map_or(0.0, |previous| previous / step);
        let one_minus_q = 1.0 - q;
        let stages = ((1.0
            + 0.759_782_816_506_459
                * scaled_radius
                * (one_minus_q + (1.0 + q * (q - 0.598_626_091_572_911)).sqrt()))
        .sqrt()
        .floor() as usize)
            + 1;
        let stages = stages.max(2);
        let degree = stages as f64;
        let t_star = 1.1_f64;
        let acosh_t_star = t_star.acosh();
        let sinh_acosh_t_star = acosh_t_star.sinh();
        let omega_zero_minus_one = 2.0 * (acosh_t_star / (2.0 * degree)).sinh().powi(2);
        let omega_zero = 1.0 + omega_zero_minus_one;
        let omega_zero_squared_minus_one = omega_zero_minus_one * (omega_zero_minus_one + 2.0);
        let derivative_t_star = degree * sinh_acosh_t_star / omega_zero_squared_minus_one.sqrt();
        let second_derivative_t_star = (degree.powi(2) * t_star - omega_zero * derivative_t_star)
            / omega_zero_squared_minus_one;
        let omega_one = (one_minus_q * derivative_t_star
            + ((one_minus_q * derivative_t_star).powi(2)
                + 4.0 * q * t_star * second_derivative_t_star)
                .sqrt())
            / (2.0 * second_derivative_t_star);

        self.previous_two.copy_from_slice(state);
        let mut mu_tilde = omega_one / omega_zero;
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
        let mut chebyshev_previous_two = 1.0;
        let mut chebyshev_previous = omega_zero;
        let mut chebyshev = 2.0 * omega_zero * chebyshev_previous - chebyshev_previous_two;

        for stage in 2..=stages {
            let mu = 2.0 * omega_zero * chebyshev_previous / chebyshev;
            let nu = -chebyshev_previous_two / chebyshev;
            mu_tilde = mu * omega_one / omega_zero;
            Self::evaluate(
                problem,
                &mut self.derivative,
                &self.previous_one,
                time + step * theta_previous,
                stats,
            )?;
            for (((output, previous), previous_two), derivative) in self
                .next_stage
                .iter_mut()
                .zip(&self.previous_one)
                .zip(&self.previous_two)
                .zip(&self.derivative)
            {
                *output = mu * previous + nu * previous_two + step * mu_tilde * derivative;
            }
            if stage < stages {
                std::mem::swap(&mut self.previous_two, &mut self.previous_one);
                std::mem::swap(&mut self.previous_one, &mut self.next_stage);
                let theta = mu * theta_previous + nu * theta_previous_two + mu_tilde;
                theta_previous_two = theta_previous;
                theta_previous = theta;
                chebyshev_previous_two = chebyshev_previous;
                chebyshev_previous = chebyshev;
                chebyshev = 2.0 * omega_zero * chebyshev_previous - chebyshev_previous_two;
            }
        }

        let gain = (1.0 + q) * t_star / (q * t_star + omega_one * derivative_t_star);
        let older = self.previous_accepted_state.as_deref().unwrap_or(state);
        for ((output, older), recurrence) in candidate.iter_mut().zip(older).zip(&self.next_stage) {
            *output = (1.0 - gain) * older + gain * recurrence;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_tsrkc3<F, P>(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        scaled_radius: f64,
        candidate: &mut [f64],
        stats: &mut SolverStats,
    ) -> Result<(), SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        let q = self
            .previous_accepted_step
            .map_or(0.0, |previous| previous / step);
        if q < 0.49 {
            let stages = ((1.54 * scaled_radius + 1.0).sqrt().floor() as usize + 1).max(2);
            return self.run_rkc(problem, state, time, step, stages, candidate, stats);
        }

        let one_minus_q = 1.0 - q;
        let one_plus_q = 1.0 + q;
        let stages = ((4.0
            + 1.267_029_788_142_009
                * scaled_radius
                * (one_minus_q + (1.0 + q * (0.442_562_207_455_629_63 + q)).sqrt()))
        .sqrt()
        .floor() as usize)
            + 1;
        let stages = stages.max(3);
        let degree = stages as f64;
        let degree_squared = degree * degree;
        let t_star = 1.25_f64;
        let acosh_t_star = t_star.acosh();
        let sinh_acosh_t_star = acosh_t_star.sinh();
        let scaled_acosh = acosh_t_star / degree;
        let omega_zero_minus_one = 2.0 * (scaled_acosh / 2.0).sinh().powi(2);
        let omega_zero = 1.0 + omega_zero_minus_one;
        let omega_zero_squared = omega_zero * omega_zero;
        let omega_zero_squared_minus_one = omega_zero_minus_one * (omega_zero_minus_one + 2.0);
        let derivative_t_star = degree * sinh_acosh_t_star / omega_zero_squared_minus_one.sqrt();
        let second_derivative_t_star = (degree_squared * t_star - omega_zero * derivative_t_star)
            / omega_zero_squared_minus_one;
        let third_derivative_t_star =
            ((1.0 + 2.0 * omega_zero_squared + degree_squared * omega_zero_squared_minus_one)
                * derivative_t_star
                - 3.0 * degree_squared * omega_zero * t_star)
                / omega_zero_squared_minus_one.powi(2);
        let omega_one = (one_minus_q * second_derivative_t_star
            + ((one_minus_q * second_derivative_t_star).powi(2)
                + 4.0 * q * derivative_t_star * third_derivative_t_star)
                .sqrt())
            / (2.0 * third_derivative_t_star);

        let mut b_previous =
            ((degree - 2.0) * scaled_acosh).sinh() / (4.0 * ((degree - 1.0) * scaled_acosh).sinh());
        let mut b = 15.0 / (8.0 * omega_zero).powi(2);
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
        let mut value = 2.0 * omega_zero * value_previous - value_previous_two;
        let mut first =
            2.0 * omega_zero * first_previous - first_previous_two + 2.0 * value_previous;
        let mut second =
            2.0 * omega_zero * second_previous - second_previous_two + 4.0 * first_previous;

        for stage in 2..=stages {
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
            if stage < stages {
                std::mem::swap(&mut self.previous_two, &mut self.previous_one);
                std::mem::swap(&mut self.previous_one, &mut self.next_stage);
                let theta =
                    mu * theta_previous + nu * theta_previous_two + mu_tilde * (1.0 - nu_tilde);
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
                value = 2.0 * omega_zero * value_previous - value_previous_two;
                first =
                    2.0 * omega_zero * first_previous - first_previous_two + 2.0 * value_previous;
                second =
                    2.0 * omega_zero * second_previous - second_previous_two + 4.0 * first_previous;
                b = second / first.powi(2);
            }
        }

        let a = one_plus_q / (q * derivative_t_star + omega_one * second_derivative_t_star);
        let history_gain = (a * derivative_t_star - 1.0) / q;
        let recurrence_gain = a / (b * omega_one);
        let current_gain = 1.0 - history_gain - recurrence_gain;
        let older = self.previous_accepted_state.as_deref().unwrap_or(state);
        for (((output, current), older), recurrence) in candidate
            .iter_mut()
            .zip(state)
            .zip(older)
            .zip(&self.next_stage)
        {
            *output = current_gain * current + history_gain * older + recurrence_gain * recurrence;
        }
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
        F: crate::OdeFunction<P>,
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
    F: crate::OdeFunction<P>,
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
        let scaled_radius = step.abs() * radius;
        let stages = self.family.stages(scaled_radius);
        match self.family {
            StabilizedFamily::Rock2 => {
                self.run_rock2(problem, state, time, step, scaled_radius, candidate, stats)?
            }
            StabilizedFamily::Rock4 => {
                self.run_rock4(problem, state, time, step, scaled_radius, candidate, stats)?
            }
            StabilizedFamily::Serk2 => {
                self.run_serk2(problem, state, time, step, scaled_radius, candidate, stats)?
            }
            StabilizedFamily::Eserk4 => self.run_eserk(
                problem,
                state,
                time,
                step,
                scaled_radius,
                candidate,
                stats,
                false,
            )?,
            StabilizedFamily::Eserk5 => self.run_eserk(
                problem,
                state,
                time,
                step,
                scaled_radius,
                candidate,
                stats,
                true,
            )?,
            StabilizedFamily::Tsrkc2 => {
                self.run_tsrkc2(problem, state, time, step, scaled_radius, candidate, stats)?
            }
            StabilizedFamily::Tsrkc3 => {
                self.run_tsrkc3(problem, state, time, step, scaled_radius, candidate, stats)?
            }
            _ => {
                self.run_recurrence(
                    problem,
                    state,
                    time,
                    step,
                    stages.ok_or(SolveError::InvalidTableau)?,
                    candidate,
                    stats,
                )?;
            }
        }
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
        if matches!(
            self.family,
            StabilizedFamily::Rock2
                | StabilizedFamily::Rock4
                | StabilizedFamily::Eserk4
                | StabilizedFamily::Eserk5
        ) {
            return Ok(StepEstimate::new(scaled_error_norm(
                &self.derivative,
                state,
                candidate,
                options,
            )));
        }
        for (index, ((((error, initial), final_state), first), last)) in self
            .derivative
            .iter_mut()
            .zip(state)
            .zip(candidate.iter())
            .zip(&self.first_derivative)
            .zip(&self.last_derivative)
            .enumerate()
        {
            *error = match self.family {
                StabilizedFamily::Rock2
                | StabilizedFamily::Rock4
                | StabilizedFamily::Eserk4
                | StabilizedFamily::Eserk5 => return Err(SolveError::InvalidTableau),
                StabilizedFamily::Serk2 => final_state - initial - step * last,
                StabilizedFamily::Rkc => {
                    (4.0 * (initial - final_state) + 2.0 * step * (first + last)) / 5.0
                }
                StabilizedFamily::Tsrkc2 => (initial - final_state + step * last) / 3.0,
                StabilizedFamily::Tsrkc3 => {
                    let q = self
                        .previous_accepted_step
                        .map_or(0.0, |previous| previous / step);
                    if q < 0.49 {
                        3.0 / 5.0 * (2.0 * (initial - final_state) + step * (first + last))
                    } else {
                        let older = self
                            .previous_accepted_state
                            .as_ref()
                            .ok_or(SolveError::InvalidMultistepHistory)?;
                        let one_plus_q = 1.0 + q;
                        3.0 / 5.0
                            * (initial / q
                                - older[index] / (q * one_plus_q.powi(2))
                                - final_state * (2.0 + q) / one_plus_q.powi(2)
                                + step * last / one_plus_q)
                    }
                }
                StabilizedFamily::Rkmc2 => (initial - final_state + step * last) / 10.0,
                StabilizedFamily::Rkl1
                | StabilizedFamily::Rkl2
                | StabilizedFamily::Rkg1
                | StabilizedFamily::Rkg2 => final_state - (initial + step * first),
            };
        }
        let divisor = match self.family {
            StabilizedFamily::Rock2
            | StabilizedFamily::Rock4
            | StabilizedFamily::Eserk4
            | StabilizedFamily::Eserk5 => return Err(SolveError::InvalidTableau),
            StabilizedFamily::Serk2 => 1.0,
            StabilizedFamily::Rkl1
            | StabilizedFamily::Rkl2
            | StabilizedFamily::Rkg1
            | StabilizedFamily::Rkg2 => stages.ok_or(SolveError::InvalidTableau)? as f64,
            StabilizedFamily::Rkc
            | StabilizedFamily::Tsrkc2
            | StabilizedFamily::Tsrkc3
            | StabilizedFamily::Rkmc2 => 1.0,
        };
        Ok(StepEstimate::new(
            scaled_error_norm(&self.derivative, state, candidate, options) / divisor,
        ))
    }

    fn accept_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        previous_state: &[f64],
        state: &[f64],
        time: f64,
        accepted_step: f64,
        callback_applied: bool,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        if callback_applied {
            Self::evaluate(problem, &mut self.first_derivative, state, time, stats)?;
            self.eigenvector.fill(0.0);
            self.previous_accepted_state = None;
            self.previous_accepted_step = None;
        } else {
            self.first_derivative.copy_from_slice(&self.last_derivative);
            if matches!(
                self.family,
                StabilizedFamily::Tsrkc2 | StabilizedFamily::Tsrkc3
            ) {
                self.previous_accepted_state = Some(previous_state.to_vec());
                self.previous_accepted_step = Some(accepted_step);
            }
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
            fn solve_validated<F, P>(
                &self,
                problem: &OdeProblem<F, P>,
                options: &SolveOptions,
            ) -> Result<Solution, SolveError>
            where
                F: crate::OdeFunction<P>,
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
    ROCK2,
    StabilizedFamily::Rock2,
    "Second-order orthogonal-polynomial ROCK method with a tabulated finishing procedure."
);
implemented_method!(
    ROCK4,
    StabilizedFamily::Rock4,
    "Fourth-order orthogonal-polynomial ROCK method with a tabulated finishing procedure."
);
implemented_method!(
    SERK2,
    StabilizedFamily::Serk2,
    "Second-order stabilized explicit Runge--Kutta method with tabulated finishing weights."
);
implemented_method!(
    ESERK4,
    StabilizedFamily::Eserk4,
    "Fourth-order extrapolated stabilized explicit Runge--Kutta method."
);
implemented_method!(
    ESERK5,
    StabilizedFamily::Eserk5,
    "Fifth-order extrapolated stabilized explicit Runge--Kutta method."
);
implemented_method!(
    TSRKC2,
    StabilizedFamily::Tsrkc2,
    "Two-step second-order Runge--Kutta--Chebyshev method."
);
implemented_method!(
    TSRKC3,
    StabilizedFamily::Tsrkc3,
    "Two-step third-order Runge--Kutta--Chebyshev method."
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

#[cfg(test)]
mod tests {
    use super::{ESERK4, ESERK5, ROCK2, ROCK4, SERK2, TSRKC2, TSRKC3};
    use crate::{OdeAlgorithm, OdeProblem, SaveMode, SolveOptions, solve};

    type ScalarRhs = fn(&mut [f64], &[f64], &(), f64);

    fn fixed_options(step: f64) -> SolveOptions {
        SolveOptions {
            adaptive: false,
            initial_step: Some(step),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        }
    }

    fn exponential() -> OdeProblem<ScalarRhs, ()> {
        fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
            du[0] = u[0];
        }
        OdeProblem::new(rhs as ScalarRhs, vec![1.0], (0.0, 1.0), ())
    }

    fn convergence_ratio<A: OdeAlgorithm + Copy>(algorithm: A) -> f64 {
        let exact = std::f64::consts::E;
        let endpoint = |step| {
            solve(&exponential(), algorithm, &fixed_options(step))
                .expect("two-step convergence solve failed")
                .last_state()[0]
        };
        (endpoint(0.1) - exact).abs() / (endpoint(0.05) - exact).abs()
    }

    #[test]
    fn two_step_methods_recover_their_formal_orders() {
        assert!(convergence_ratio(TSRKC2).log2() > 1.5);
        assert!(convergence_ratio(TSRKC3).log2() > 2.5);
    }

    #[test]
    fn tabulated_methods_recover_their_formal_orders() {
        assert!(convergence_ratio(ROCK2).log2() > 1.5);
        assert!(convergence_ratio(SERK2).log2() > 1.5);
        assert!(convergence_ratio(ROCK4).log2() > 3.5);
        assert!(convergence_ratio(ESERK4).log2() > 3.5);
        assert!(convergence_ratio(ESERK5).log2() > 4.5);
    }

    #[test]
    fn two_step_methods_stabilize_a_real_negative_mode() {
        fn rhs(du: &mut [f64], u: &[f64], _: &(), time: f64) {
            du[0] = -40.0 * (u[0] - time.cos()) - time.sin();
        }
        let problem = OdeProblem::new(rhs as ScalarRhs, vec![1.0], (0.0, 1.0), ());
        for endpoint in [
            solve(&problem, TSRKC2, &fixed_options(0.05))
                .expect("TSRKC2 stiff solve failed")
                .last_state()[0],
            solve(&problem, TSRKC3, &fixed_options(0.05))
                .expect("TSRKC3 stiff solve failed")
                .last_state()[0],
            solve(&problem, ROCK2, &fixed_options(0.05))
                .expect("ROCK2 stiff solve failed")
                .last_state()[0],
            solve(&problem, ROCK4, &fixed_options(0.05))
                .expect("ROCK4 stiff solve failed")
                .last_state()[0],
            solve(&problem, SERK2, &fixed_options(0.05))
                .expect("SERK2 stiff solve failed")
                .last_state()[0],
            solve(&problem, ESERK4, &fixed_options(0.05))
                .expect("ESERK4 stiff solve failed")
                .last_state()[0],
            solve(&problem, ESERK5, &fixed_options(0.05))
                .expect("ESERK5 stiff solve failed")
                .last_state()[0],
        ] {
            assert!((endpoint - 1.0_f64.cos()).abs() < 1.5e-3);
        }
    }
}
