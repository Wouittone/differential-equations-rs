use super::{StepError, StepFailure, StepStatistics, checked_time, endpoint_step, finite};
use crate::tableau::RungeKuttaNystromTableau;
use super::state_buffer::StateBuffer;

/// Explicit acceleration dependence contract for RKN formulas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccelerationPolicy {
    /// Acceleration is independent of velocity; stage velocity is unspecified.
    VelocityIndependent,
    /// Advance stage velocities using the tableau's velocity-stage matrix.
    VelocityDependent,
    /// Deliberately reproduce historical host formulas that freeze stage velocity.
    LegacyFrozenVelocity,
}
/// Borrowed RKN candidate and unscaled acceleration stages.
#[derive(Debug)]
pub struct RknStepView<'a> {
    /// Accepted start time.
    pub start_time: f64,
    /// Candidate time.
    pub end_time: f64,
    /// Candidate position.
    pub position: &'a [f64],
    /// Candidate velocity.
    pub velocity: &'a [f64],
    /// Signed position embedded error (includes step squared).
    pub position_error: Option<&'a [f64]>,
    /// Signed velocity embedded error (includes step).
    pub velocity_error: Option<&'a [f64]>,
    /// Stage-major acceleration evaluations, empty for a zero step.
    pub accelerations: &'a [f64],
    /// Cumulative work statistics.
    pub statistics: StepStatistics,
    dimension: usize,
}
impl RknStepView<'_> {
    /// Acceleration at a stage, or `None` for an invalid index.
    pub fn stage(&self, index: usize) -> Option<&[f64]> {
        let start = index.checked_mul(self.dimension)?;
        self.accelerations
            .get(start..start.checked_add(self.dimension)?)
    }
}
/// Reusable RKN state/workspace with separate position and velocity partitions.
#[derive(Debug)]
pub struct RknStepper<'a> {
    tableau: &'a RungeKuttaNystromTableau,
    policy: AccelerationPolicy,
    time: f64,
    position: StateBuffer<'a>,
    velocity: StateBuffer<'a>,
    next_position: Vec<f64>,
    next_velocity: Vec<f64>,
    position_error: Vec<f64>,
    velocity_error: Vec<f64>,
    stage_position: Vec<f64>,
    stage_velocity: Vec<f64>,
    stages: Vec<f64>,
    initial_acceleration: Vec<f64>,
    initial_valid: bool,
    pending: Option<f64>,
    stats: StepStatistics,
}
impl<'a> RknStepper<'a> {
    /// Allocate a fixed-size workspace. Velocity-dependent formulas require A_velocity.
    pub fn new(
        tableau: &'a RungeKuttaNystromTableau,
        policy: AccelerationPolicy,
        time: f64,
        position: &[f64],
        velocity: &[f64],
    ) -> Result<Self, StepFailure> {
        Self::with_storage(tableau, policy, time,
            StateBuffer::Owned(position.to_vec()), StateBuffer::Owned(velocity.to_vec()))
    }
    /// Borrow position and velocity buffers without copying or allocating them.
    ///
    /// Scratch storage is allocated once. Accept copies the candidate into
    /// these buffers; rejection and failed attempts leave them unchanged.
    /// Arrays and dynamic slices use the same path. Both exclusive borrows
    /// remain active until the stepper is dropped; dimensions cannot resize.
    pub fn from_buffers(
        tableau: &'a RungeKuttaNystromTableau,
        policy: AccelerationPolicy,
        time: f64,
        position: &'a mut [f64],
        velocity: &'a mut [f64],
    ) -> Result<Self, StepFailure> {
        Self::with_storage(tableau, policy, time,
            StateBuffer::Borrowed(position), StateBuffer::Borrowed(velocity))
    }
    fn with_storage(
        tableau: &'a RungeKuttaNystromTableau,
        policy: AccelerationPolicy,
        time: f64,
        position: StateBuffer<'a>,
        velocity: StateBuffer<'a>,
    ) -> Result<Self, StepFailure> {
        let n = position.len();
        if n == 0 || n != velocity.len() {
            return Err(StepFailure::Dimension);
        }
        finite(&position)?;
        finite(&velocity)?;
        finite(&[time])?;
        if policy == AccelerationPolicy::VelocityDependent && tableau.a_velocity().is_none() {
            return Err(StepFailure::UnsupportedTableau);
        }
        let count = n
            .checked_mul(tableau.stages())
            .ok_or(StepFailure::Dimension)?;
        Ok(Self {
            tableau,
            policy,
            time,
            position,
            velocity,
            next_position: vec![0.; n],
            next_velocity: vec![0.; n],
            position_error: vec![0.; n],
            velocity_error: vec![0.; n],
            stage_position: vec![0.; n],
            stage_velocity: vec![0.; n],
            stages: vec![0.; count],
            initial_acceleration: vec![0.; n],
            initial_valid: false,
            pending: None,
            stats: StepStatistics::default(),
        })
    }
    /// Accepted position.
    pub fn position(&self) -> &[f64] {
        &self.position
    }
    /// Accepted velocity.
    pub fn velocity(&self) -> &[f64] {
        &self.velocity
    }
    /// Accepted time.
    pub fn time(&self) -> f64 {
        self.time
    }
    /// Method coefficients.
    pub fn tableau(&self) -> &'a RungeKuttaNystromTableau {
        self.tableau
    }
    /// Cumulative work counts.
    pub fn statistics(&self) -> StepStatistics {
        self.stats
    }
    /// Forget cached initial acceleration after RHS/control/parameter changes.
    pub fn invalidate_derivative(&mut self) {
        self.initial_valid = false;
    }
    /// Cached acceleration at the accepted state, if available.
    pub fn current_derivative(&self) -> Option<&[f64]> {
        self.initial_valid.then_some(&self.initial_acceleration)
    }
    /// Inject the complete acceleration at the accepted state.
    pub fn inject_derivative(&mut self, acceleration: &[f64]) -> Result<(), StepFailure> {
        if self.pending.is_some() {
            return Err(StepFailure::PendingCandidate);
        }
        if acceleration.len() != self.position.len() {
            return Err(StepFailure::Dimension);
        }
        finite(acceleration)?;
        self.initial_acceleration.copy_from_slice(acceleration);
        self.initial_valid = true;
        Ok(())
    }
    /// Same-shape restart; no allocations and no retained trajectory.
    pub fn reset(
        &mut self,
        time: f64,
        position: &[f64],
        velocity: &[f64],
    ) -> Result<(), StepFailure> {
        if position.len() != self.position.len() || velocity.len() != self.velocity.len() {
            return Err(StepFailure::Dimension);
        }
        finite(&[time])?;
        finite(position)?;
        finite(velocity)?;
        self.time = time;
        self.position.copy_from_slice(position);
        self.velocity.copy_from_slice(velocity);
        self.pending = None;
        self.initial_valid = false;
        Ok(())
    }
    /// Clamp a signed proposal to an endpoint.
    pub fn attempt_to<F, E>(
        &mut self,
        endpoint: f64,
        proposal: f64,
        rhs: &mut F,
    ) -> Result<RknStepView<'_>, StepError<E>>
    where
        F: FnMut(f64, &[f64], &[f64], &mut [f64]) -> Result<(), E>,
    {
        let step = endpoint_step(self.time, endpoint, proposal)?;
        self.attempt(step, rhs)
    }
    /// Evaluate acceleration stages once, exposing them to external controllers.
    pub fn attempt<F, E>(&mut self, step: f64, rhs: &mut F) -> Result<RknStepView<'_>, StepError<E>>
    where
        F: FnMut(f64, &[f64], &[f64], &mut [f64]) -> Result<(), E>,
    {
        if self.pending.is_some() {
            return Err(StepFailure::PendingCandidate.into());
        }
        let end = checked_time(self.time, step)?;
        self.stats.attempts += 1;
        let n = self.position.len();
        let h2 = step * step;
        self.position_error.fill(0.);
        self.velocity_error.fill(0.);
        for k in 0..n {
            self.next_position[k] = self.position[k] + step * self.velocity[k];
            self.next_velocity[k] = self.velocity[k];
        }
        if step != 0. {
            for i in 0..self.tableau.stages() {
                let start = i * n;
                if i == 0 && self.tableau.c()[0] == 0. && self.initial_valid {
                    self.stages[..n].copy_from_slice(&self.initial_acceleration);
                } else {
                    for k in 0..n {
                        self.stage_position[k] =
                            self.position[k] + step * self.tableau.c()[i] * self.velocity[k];
                        self.stage_velocity[k] = self.velocity[k];
                    }
                    for j in 0..i {
                        let a = self.tableau.a()[i][j];
                        let av = if self.policy == AccelerationPolicy::VelocityDependent {
                            self.tableau.a_velocity().expect("validated policy")[i][j]
                        } else {
                            0.
                        };
                        for k in 0..n {
                            self.stage_position[k] += h2 * a * self.stages[j * n + k];
                            self.stage_velocity[k] += step * av * self.stages[j * n + k];
                        }
                    }
                    finite(&self.stage_position)?;
                    finite(&self.stage_velocity)?;
                    let stage_time = self.time + step * self.tableau.c()[i];
                    finite(&[stage_time])?;
                    self.stats.rhs_evaluations += 1;
                    rhs(
                        stage_time,
                        &self.stage_position,
                        &self.stage_velocity,
                        &mut self.stages[start..start + n],
                    )
                    .map_err(StepError::User)?;
                    finite(&self.stages[start..start + n])?;
                    if i == 0 && self.tableau.c()[0] == 0. {
                        self.initial_acceleration.copy_from_slice(&self.stages[..n]);
                        self.initial_valid = true;
                    }
                }
                for k in 0..n {
                    let acc = self.stages[start + k];
                    self.next_position[k] += h2 * self.tableau.b()[i] * acc;
                    self.next_velocity[k] += step * self.tableau.b_velocity()[i] * acc;
                    self.position_error[k] += h2 * self.tableau.error().map_or(0., |w| w[i]) * acc;
                    self.velocity_error[k] +=
                        step * self.tableau.velocity_error().map_or(0., |w| w[i]) * acc;
                }
            }
        }
        finite(&self.next_position)?;
        finite(&self.next_velocity)?;
        finite(&self.position_error)?;
        finite(&self.velocity_error)?;
        self.pending = Some(step);
        Ok(RknStepView {
            start_time: self.time,
            end_time: end,
            position: &self.next_position,
            velocity: &self.next_velocity,
            position_error: self.tableau.error().map(|_| self.position_error.as_slice()),
            velocity_error: self
                .tableau
                .velocity_error()
                .map(|_| self.velocity_error.as_slice()),
            accelerations: if step == 0. { &[] } else { &self.stages },
            statistics: self.stats,
            dimension: n,
        })
    }
    /// Commit a candidate; RKN acceleration is not presumed FSAL.
    pub fn accept(&mut self) -> Result<(), StepFailure> {
        let step = self.pending.take().ok_or(StepFailure::NoCandidate)?;
        self.position.copy_from_slice(&self.next_position);
        self.velocity.copy_from_slice(&self.next_velocity);
        self.time += step;
        if step != 0. {
            self.initial_valid = false;
        }
        self.stats.accepted_steps += 1;
        Ok(())
    }
    /// Discard a candidate, retaining initial acceleration for another attempt.
    pub fn reject(&mut self) -> Result<(), StepFailure> {
        self.pending.take().ok_or(StepFailure::NoCandidate)?;
        self.stats.rejected_steps += 1;
        Ok(())
    }
}
