use super::workspace::StabilizedKernel;
use crate::{OdeProblem, SolveError, SolverStats};

pub(super) enum OrthogonalPolynomial {
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

impl StabilizedKernel {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_orthogonal_polynomial<F, P>(
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
}
