use super::family::{MAX_POLYNOMIAL_STAGES, StabilizedFamily};
use super::workspace::StabilizedKernel;
use crate::solvers::stabilized::resources::{
    eserk4_tableau_for_degree, eserk5_tableau_for_degree, rock2_tableau_for_degree,
    rock4_tableau_for_degree, serk2_tableau_for_degree,
};
use crate::{OdeProblem, SolveError, SolverStats};

impl StabilizedKernel {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_rock2<F, P>(
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
        let tableau =
            rock2_tableau_for_degree(requested_degree).map_err(|_| SolveError::InvalidTableau)?;
        let degree = tableau.degree();
        let recurrence = tableau.recurrence();
        let first_coefficient = recurrence.first_stage();
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
            for (stage_index, coefficients) in recurrence.stages().iter().enumerate() {
                let stage = stage_index + 2;
                let mu = coefficients.mu();
                let kappa = coefficients.kappa();
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

        let finish_first = step * tableau.finish_first();
        let finish_second = step * tableau.finish_second();
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
    pub(super) fn run_rock4<F, P>(
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
        let tableau =
            rock4_tableau_for_degree(requested_degree).map_err(|_| SolveError::InvalidTableau)?;
        let degree = tableau.degree();
        let recurrence = tableau.recurrence();
        let first_coefficient = recurrence.first_stage();
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
            for (stage_offset, recurrence_stage) in recurrence.stages().iter().enumerate() {
                let stage = stage_offset + 2;
                let mu = recurrence_stage.mu();
                let kappa = recurrence_stage.kappa();
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

        let a = tableau.finishing_a();
        let b = tableau.b();
        let b_hat = tableau.b_hat();
        let a21 = step * a[1][0];
        let a31 = step * a[2][0];
        let a32 = step * a[2][1];
        let a41 = step * a[3][0];
        let a42 = step * a[3][1];
        let a43 = step * a[3][2];
        let b1 = step * b[0];
        let b2 = step * b[1];
        let b3 = step * b[2];
        let b4 = step * b[3];
        let error1 = step * (b_hat[0] - b[0]);
        let error2 = step * (b_hat[1] - b[1]);
        let error3 = step * (b_hat[2] - b[2]);
        let error4 = step * (b_hat[3] - b[3]);
        let error5 = step * b_hat[4];

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
    pub(super) fn run_serk2<F, P>(
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
        let tableau =
            serk2_tableau_for_degree(requested).map_err(|_| SolveError::InvalidTableau)?;
        let subdivisions = tableau.subdivisions();
        let internal_degree = tableau.internal_degree();
        let weights = tableau.weights();
        let alpha = tableau.alpha();
        self.previous_one.copy_from_slice(state);
        self.previous_two.copy_from_slice(state);
        for (sum, value) in self.perturbed_state.iter_mut().zip(state) {
            *sum = weights[0] * value;
        }

        for block in 0..subdivisions {
            let block_node = block * internal_degree * internal_degree;
            // Each right-hand-side evaluation belongs to the state supplied as
            // its input. Because the SERK recurrence state at stage `j - 1`
            // has abscissa `(j - 1)^2`, using the output state's `j^2` node
            // would reduce nonautonomous problems to first order. This is an
            // intentional correction to the pinned SciML implementation; the
            // forced multi-stage regression below locks in the order condition.
            let first_time = time + block_node as f64 * alpha * step;
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
            let first_weight = block * internal_degree + 1;
            for (sum, value) in self.perturbed_state.iter_mut().zip(&self.next_stage) {
                *sum += weights[first_weight] * value;
            }
            std::mem::swap(&mut self.previous_two, &mut self.previous_one);
            std::mem::swap(&mut self.previous_one, &mut self.next_stage);

            for stage in 2..=internal_degree {
                let input_stage = stage - 1;
                let stage_time =
                    time + (input_stage * input_stage + block_node) as f64 * alpha * step;
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
                let weight = stage + block * internal_degree;
                for (sum, value) in self.perturbed_state.iter_mut().zip(&self.next_stage) {
                    *sum += weights[weight] * value;
                }
                if stage < internal_degree || block + 1 < subdivisions {
                    std::mem::swap(&mut self.previous_two, &mut self.previous_one);
                    std::mem::swap(&mut self.previous_one, &mut self.next_stage);
                }
            }
        }
        candidate.copy_from_slice(&self.perturbed_state);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_eserk<F, P>(
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
        let tableau = match self.family {
            StabilizedFamily::Eserk4 => {
                let requested = (scaled_radius.sqrt().floor() as usize + 1).min(4_000);
                eserk4_tableau_for_degree(requested)
            }
            StabilizedFamily::Eserk5 => {
                let requested = ((scaled_radius / 0.98).sqrt().floor() as usize + 1).min(2_000);
                eserk5_tableau_for_degree(requested)
            }
            _ => return Err(SolveError::InvalidTableau),
        }
        .map_err(|_| SolveError::InvalidTableau)?;
        let degree = tableau.degree();
        let internal_degree = tableau.internal_degree();
        let alpha = tableau.alpha();
        let subdivisions = tableau.subdivisions();
        let solution_combination = tableau.solution_combination();
        let error_combination = tableau.error_combination();
        let combination_denominator = tableau.combination_denominator();
        let weights = tableau.weights();
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
                    *sum = weights[0] * value;
                }
                for (stage, &weight) in weights.iter().enumerate().take(degree + 1).skip(1) {
                    // The supplied recurrence state is at the input node of
                    // this stage. The pinned SciML implementation advances
                    // this clock to the output node instead, which loses the
                    // advertised order on nonautonomous equations.
                    let block = (stage - 1) / internal_degree;
                    let local_stage = (stage - 1) % internal_degree;
                    let stage_time = substep_time
                        + alpha
                            * (block * internal_degree * internal_degree
                                + local_stage * local_stage) as f64
                            * substep;
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
                    for (sum, value) in self.perturbed_state.iter_mut().zip(&self.next_stage) {
                        *sum += weight * value;
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
        for (output, error) in candidate.iter_mut().zip(self.eigenvector.iter_mut()) {
            *output /= combination_denominator;
            *error /= combination_denominator;
        }
        self.derivative.copy_from_slice(&self.eigenvector);
        Ok(())
    }
}
