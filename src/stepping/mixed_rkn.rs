use super::{
    AccelerationPolicy, RknStepView, RknStepper, StepError, StepFailure, StepStatistics,
    checked_time, finite,
};
use crate::tableau::RungeKuttaNystromTableau;

/// Borrowed physical and auxiliary results of a mixed RKN attempt.
#[derive(Debug)]
pub struct MixedRknStepView<'a> {
    /// Physical position/velocity candidate and acceleration stages.
    pub physical: RknStepView<'a>,
    /// First-order auxiliary candidate (STM, sensitivities or other variables).
    pub auxiliary: &'a [f64],
    /// Auxiliary embedded error using the velocity companion weights.
    pub auxiliary_error: Option<&'a [f64]>,
    /// Stage-major first-order auxiliary derivatives.
    pub auxiliary_derivatives: &'a [f64],
}
/// Partitioned second-order physical and first-order augmented integration.
///
/// The physical RKN position stages use `A`, velocity stages use `A_velocity`;
/// first-order auxiliary stages also use `A_velocity`, and update/error weights
/// are `b_velocity`/`velocity_error`. This requires a velocity-dependent tableau.
/// The combined callback receives each physical/auxiliary stage and computes
/// acceleration plus auxiliary derivatives once, so STM propagation need not
/// recompute the primary force. Coupling in either direction is permitted.
/// Formula order for a chosen coupled system remains a numerical property of
/// its partitioned coefficients; no missing velocity-stage matrix is invented.
///
/// Auxiliary errors are exposed separately: hosts choose joint, block, masked,
/// maximum or RMS control explicitly. Reset/rejection/acceptance are atomic
/// across both partitions; storage is allocated only on construction.
#[derive(Debug)]
pub struct MixedRknStepper<'a> {
    physical: RknStepper<'a>,
    auxiliary: Vec<f64>,
    candidate: Vec<f64>,
    error: Vec<f64>,
    stage_state: Vec<f64>,
    derivatives: Vec<f64>,
    pending: bool,
}
impl<'a> MixedRknStepper<'a> {
    /// Construct a mixed system using the RKN velocity companion for auxiliary ODEs.
    pub fn new(
        tableau: &'a RungeKuttaNystromTableau,
        time: f64,
        position: &[f64],
        velocity: &[f64],
        auxiliary: &[f64],
    ) -> Result<Self, StepFailure> {
        if auxiliary.is_empty() {
            return Err(StepFailure::Dimension);
        }
        finite(auxiliary)?;
        let m = auxiliary.len();
        let count = m
            .checked_mul(tableau.stages())
            .ok_or(StepFailure::Dimension)?;
        Ok(Self {
            physical: RknStepper::new(
                tableau,
                AccelerationPolicy::VelocityDependent,
                time,
                position,
                velocity,
            )?,
            auxiliary: auxiliary.to_vec(),
            candidate: vec![0.; m],
            error: vec![0.; m],
            stage_state: vec![0.; m],
            derivatives: vec![0.; count],
            pending: false,
        })
    }
    /// Accepted physical position.
    pub fn position(&self) -> &[f64] {
        self.physical.position()
    }
    /// Accepted physical velocity.
    pub fn velocity(&self) -> &[f64] {
        self.physical.velocity()
    }
    /// Accepted first-order auxiliary state.
    pub fn auxiliary(&self) -> &[f64] {
        &self.auxiliary
    }
    /// Accepted time shared by both partitions.
    pub fn time(&self) -> f64 {
        self.physical.time()
    }
    /// Counts of combined force/auxiliary calls, not per partition.
    pub fn statistics(&self) -> StepStatistics {
        self.physical.statistics()
    }
    /// Invalidate joint stage-zero cache after any RHS or parameter change.
    pub fn invalidate_derivative(&mut self) {
        self.physical.invalidate_derivative();
    }
    /// Restart all partitions without reallocating; invalid input changes nothing.
    pub fn reset(
        &mut self,
        time: f64,
        position: &[f64],
        velocity: &[f64],
        auxiliary: &[f64],
    ) -> Result<(), StepFailure> {
        if auxiliary.len() != self.auxiliary.len() {
            return Err(StepFailure::Dimension);
        }
        finite(auxiliary)?;
        self.physical.reset(time, position, velocity)?;
        self.auxiliary.copy_from_slice(auxiliary);
        self.pending = false;
        Ok(())
    }
    /// Attempt one coupled step with one combined callback per physical stage.
    pub fn attempt<F, E>(
        &mut self,
        step: f64,
        rhs: &mut F,
    ) -> Result<MixedRknStepView<'_>, StepError<E>>
    where
        F: FnMut(f64, &[f64], &[f64], &[f64], &mut [f64], &mut [f64]) -> Result<(), E>,
    {
        if self.pending {
            return Err(StepFailure::PendingCandidate.into());
        }
        let end = checked_time(self.time(), step)?;
        let step = end - self.time();
        let tableau = self.physical.tableau();
        let m = self.auxiliary.len();
        let mut stage =
            usize::from(self.physical.current_derivative().is_some() && tableau.c()[0] == 0.);
        let auxiliary = &self.auxiliary;
        let derivatives = &mut self.derivatives;
        let stage_state = &mut self.stage_state;
        self.physical
            .attempt(step, &mut |time, q, v, a| {
                let start = stage * m;
                stage_state.copy_from_slice(auxiliary);
                for j in 0..stage {
                    let weight = tableau.a_velocity().expect("validated companion")[stage][j];
                    if weight != 0. {
                        for k in 0..m {
                            stage_state[k] += step * weight * derivatives[j * m + k];
                        }
                    }
                }
                finite(stage_state).map_err(StepError::Solver)?;
                rhs(
                    time,
                    q,
                    v,
                    stage_state,
                    a,
                    &mut derivatives[start..start + m],
                )
                .map_err(StepError::User)?;
                finite(&derivatives[start..start + m]).map_err(StepError::Solver)?;
                stage += 1;
                Ok::<(), StepError<E>>(())
            })
            .map_err(|e| match e {
                StepError::Solver(e) => StepError::Solver(e),
                StepError::User(e) => e,
            })?;
        self.candidate.copy_from_slice(&self.auxiliary);
        self.error.fill(0.);
        if step != 0. {
            for i in 0..tableau.stages() {
                for k in 0..m {
                    self.candidate[k] +=
                        step * tableau.b_velocity()[i] * self.derivatives[i * m + k];
                    self.error[k] += step
                        * tableau.velocity_error().map_or(0., |w| w[i])
                        * self.derivatives[i * m + k];
                }
            }
        }
        if let Err(e) = finite(&self.candidate).and_then(|_| finite(&self.error)) {
            self.physical.reject()?;
            return Err(e.into());
        }
        self.pending = true;
        Ok(MixedRknStepView {
            physical: self
                .physical
                .pending_view()
                .expect("successful physical attempt"),
            auxiliary: &self.candidate,
            auxiliary_error: tableau.velocity_error().map(|_| self.error.as_slice()),
            auxiliary_derivatives: if step == 0. { &[] } else { &self.derivatives },
        })
    }
    /// Commit both partitions together.
    pub fn accept(&mut self) -> Result<(), StepFailure> {
        if !self.pending {
            return Err(StepFailure::NoCandidate);
        }
        self.physical.accept()?;
        self.auxiliary.copy_from_slice(&self.candidate);
        self.pending = false;
        Ok(())
    }
    /// Discard both candidates while preserving accepted state and stage-zero data.
    pub fn reject(&mut self) -> Result<(), StepFailure> {
        if !self.pending {
            return Err(StepFailure::NoCandidate);
        }
        self.physical.reject()?;
        self.pending = false;
        Ok(())
    }
}

impl MixedRknStepper<'_> {
    /// Endpoint-clamped attempt preserving signed propagation direction.
    pub fn attempt_to<F, E>(
        &mut self,
        endpoint: f64,
        proposal: f64,
        rhs: &mut F,
    ) -> Result<MixedRknStepView<'_>, StepError<E>>
    where
        F: FnMut(f64, &[f64], &[f64], &[f64], &mut [f64], &mut [f64]) -> Result<(), E>,
    {
        let step = super::endpoint_step(self.time(), endpoint, proposal)?;
        self.attempt(step, rhs)
    }
    /// Copy every accepted partition to correctly sized caller buffers atomically.
    pub fn copy_state_into(
        &self,
        position: &mut [f64],
        velocity: &mut [f64],
        auxiliary: &mut [f64],
    ) -> Result<(), StepFailure> {
        if auxiliary.len() != self.auxiliary.len() {
            return Err(StepFailure::Dimension);
        }
        self.physical.copy_state_into(position, velocity)?;
        auxiliary.copy_from_slice(&self.auxiliary);
        Ok(())
    }
    /// Clear joint evaluation counters without discarding caches.
    pub fn clear_statistics(&mut self) {
        self.physical.clear_statistics();
    }
}
