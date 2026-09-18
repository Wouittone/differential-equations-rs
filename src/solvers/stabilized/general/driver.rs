use super::family::StabilizedFamily;
use super::workspace::{StabilizedKernel, finite};
use crate::integrator::{KernelCapabilities, StepEstimate, StepKernel};
use crate::{OdeProblem, SolveError, SolveOptions, SolverStats};

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
            StabilizedFamily::Eserk4 | StabilizedFamily::Eserk5 => {
                self.run_eserk(problem, state, time, step, scaled_radius, candidate, stats)?
            }
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
