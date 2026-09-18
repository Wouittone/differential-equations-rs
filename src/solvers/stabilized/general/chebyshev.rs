use super::family::StabilizedFamily;
use super::orthogonal::OrthogonalPolynomial;
use super::workspace::StabilizedKernel;
use crate::{OdeProblem, SolveError, SolverStats};

impl StabilizedKernel {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_recurrence<F, P>(
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
    pub(super) fn run_tsrkc2<F, P>(
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
    pub(super) fn run_tsrkc3<F, P>(
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
