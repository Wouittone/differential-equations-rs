use super::state_buffer::StateBuffer;
use super::{StepError, StepFailure, StepStatistics, checked_time, endpoint_step, finite};
use crate::tableau::{RungeKuttaKind, RungeKuttaTableau};

/// Borrowed results of one explicit Runge--Kutta attempt.
#[derive(Debug)]
pub struct StepView<'a> {
    /// Accepted state before this attempt.
    pub previous_state: &'a [f64],
    /// Time at the accepted state.
    pub start_time: f64,
    /// Time at the candidate state.
    pub end_time: f64,
    /// Candidate state, not yet committed.
    pub candidate: &'a [f64],
    /// Signed component local-error estimate, if the method supplies one.
    /// The tableau's error_estimator_kind distinguishes embedded differences
    /// from direct residual formulas.
    pub component_error: Option<&'a [f64]>,
    /// Optional secondary estimator (for methods with two error formulas).
    pub second_error: Option<&'a [f64]>,
    /// Stage-major unscaled RHS evaluations; empty for a zero step.
    pub stage_derivatives: &'a [f64],
    /// Cumulative statistics including this attempt.
    pub statistics: StepStatistics,
    dimension: usize,
}
impl StepView<'_> {
    /// Return one RHS stage, or `None` for an invalid index.
    pub fn stage(&self, index: usize) -> Option<&[f64]> {
        let start = index.checked_mul(self.dimension)?;
        self.stage_derivatives
            .get(start..start.checked_add(self.dimension)?)
    }
}

/// Persistent explicit RK workspace borrowing a validated per-instance tableau.
///
/// Construction allocates a compact scratch buffer that packs the candidate,
/// temporary stage data, embedded errors, and cached derivative into a single
/// contiguous allocation. Attempt, accept, reject, reset and derivative
/// injection never allocate. Specialized fitted formulas are rejected; their
/// frequency-dependent weights require a dedicated kernel. Both embedded error
/// vectors remain exposed so hosts can implement their method-specific norm
/// formulas exactly.
#[derive(Debug)]
pub struct ExplicitRungeKuttaStepper<'a> {
    tableau: &'a RungeKuttaTableau,
    time: f64,
    state: StateBuffer<'a>,
    workspace: Vec<f64>,
    candidate_offset: usize,
    error_offset: usize,
    second_error_offset: Option<usize>,
    temporary_offset: usize,
    stages_offset: usize,
    derivative_offset: usize,
    derivative_valid: bool,
    pending: Option<f64>,
    stats: StepStatistics,
}
impl<'a> ExplicitRungeKuttaStepper<'a> {
    fn workspace_slice(&self, offset: usize, len: usize) -> &[f64] {
        &self.workspace[offset..offset + len]
    }
    fn workspace_slice_mut(&mut self, offset: usize, len: usize) -> &mut [f64] {
        &mut self.workspace[offset..offset + len]
    }
    fn candidate_slice(&self) -> &[f64] {
        self.workspace_slice(self.candidate_offset, self.state.len())
    }
    fn error_slice(&self) -> &[f64] {
        self.workspace_slice(self.error_offset, self.state.len())
    }
    fn second_error_slice(&self) -> Option<&[f64]> {
        let len = self.state.len();
        self.second_error_offset
            .map(|offset| self.workspace_slice(offset, len))
    }
    fn stages_slice(&self) -> &[f64] {
        let n = self.state.len();
        let stage_len = n * self.tableau.stages();
        self.workspace_slice(self.stages_offset, stage_len)
    }
    fn derivative_slice(&self) -> &[f64] {
        self.workspace_slice(self.derivative_offset, self.state.len())
    }
    fn derivative_slice_mut(&mut self) -> &mut [f64] {
        self.workspace_slice_mut(self.derivative_offset, self.state.len())
    }
    /// Construct once; all initial state components and time must be finite.
    pub fn new(
        tableau: &'a RungeKuttaTableau,
        time: f64,
        state: &[f64],
    ) -> Result<Self, StepFailure> {
        Self::with_storage(tableau, time, StateBuffer::Owned(state.to_vec()))
    }
    /// Borrow caller-owned accepted state, including a const-sized array.
    ///
    /// The state is neither copied nor allocated at construction. Scratch
    /// storage is allocated once. Successful acceptance copies the candidate
    /// into this buffer; rejection and failed attempts leave it unchanged.
    /// The exclusive borrow lasts until the stepper is dropped. No state
    /// resizing is allowed; use `reset` for another state of the same length.
    pub fn from_buffer(
        tableau: &'a RungeKuttaTableau,
        time: f64,
        state: &'a mut [f64],
    ) -> Result<Self, StepFailure> {
        Self::with_storage(tableau, time, StateBuffer::Borrowed(state))
    }
    fn with_storage(
        tableau: &'a RungeKuttaTableau,
        time: f64,
        state: StateBuffer<'a>,
    ) -> Result<Self, StepFailure> {
        if state.is_empty() {
            return Err(StepFailure::Dimension);
        }
        finite(&state)?;
        finite(&[time])?;
        if tableau.kind() != RungeKuttaKind::Explicit || !tableau.fitted_weights().is_empty() {
            return Err(StepFailure::UnsupportedTableau);
        }
        let n = state.len();
        let stage_len = n
            .checked_mul(tableau.stages())
            .ok_or(StepFailure::Dimension)?;
        let second_error_len = usize::from(tableau.second_error().is_some()) * n;
        let candidate_len = n;
        let error_len = n;
        let temporary_len = n;
        let derivative_len = n;
        let workspace_len = candidate_len
            + error_len
            + second_error_len
            + temporary_len
            + stage_len
            + derivative_len;
        let mut workspace = vec![0.; workspace_len];
        let candidate_offset = 0;
        let error_offset = candidate_len;
        let second_error_offset = if tableau.second_error().is_some() {
            Some(error_offset + error_len)
        } else {
            None
        };
        let temporary_offset = second_error_offset
            .map_or(error_offset + error_len, |offset| offset + second_error_len);
        let stages_offset = temporary_offset + temporary_len;
        let derivative_offset = stages_offset + stage_len;
        debug_assert_eq!(derivative_offset + derivative_len, workspace_len);
        workspace.fill(0.);
        Ok(Self {
            tableau,
            time,
            state,
            workspace,
            candidate_offset,
            error_offset,
            second_error_offset,
            temporary_offset,
            stages_offset,
            derivative_offset,
            derivative_valid: false,
            pending: None,
            stats: StepStatistics::default(),
        })
    }
    /// Validated method used by this workspace.
    pub fn tableau(&self) -> &'a RungeKuttaTableau {
        self.tableau
    }
    /// Last accepted state; pending attempts do not change it.
    pub fn state(&self) -> &[f64] {
        &self.state
    }
    /// Last accepted time.
    pub fn time(&self) -> f64 {
        self.time
    }
    /// Cumulative statistics; reset preserves these counters.
    pub fn statistics(&self) -> StepStatistics {
        self.stats
    }
    /// Clear work counters without changing state or caches.
    pub fn clear_statistics(&mut self) {
        self.stats = StepStatistics::default();
    }
    /// Forget a cached derivative after parameter, control or RHS changes.
    pub fn invalidate_derivative(&mut self) {
        self.derivative_valid = false;
    }
    /// Derivative at the accepted state, only when a valid cache exists.
    pub fn current_derivative(&self) -> Option<&[f64]> {
        self.derivative_valid.then_some(self.derivative_slice())
    }
    /// Inject the complete RHS at the accepted state (including control terms).
    pub fn inject_derivative(&mut self, derivative: &[f64]) -> Result<(), StepFailure> {
        if self.pending.is_some() {
            return Err(StepFailure::PendingCandidate);
        }
        if derivative.len() != self.state.len() {
            return Err(StepFailure::Dimension);
        }
        finite(derivative)?;
        self.derivative_slice_mut().copy_from_slice(derivative);
        self.derivative_valid = true;
        Ok(())
    }
    /// Restart at a same-sized state, discarding pending output and FSAL cache.
    pub fn reset(&mut self, time: f64, state: &[f64]) -> Result<(), StepFailure> {
        if state.len() != self.state.len() {
            return Err(StepFailure::Dimension);
        }
        finite(&[time])?;
        finite(state)?;
        self.time = time;
        self.state.copy_from_slice(state);
        self.pending = None;
        self.derivative_valid = false;
        Ok(())
    }
    /// Make an endpoint-clamped attempt using a signed next-step proposal.
    pub fn attempt_to<F, E>(
        &mut self,
        endpoint: f64,
        proposal: f64,
        rhs: &mut F,
    ) -> Result<StepView<'_>, StepError<E>>
    where
        F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), E>,
    {
        let step = endpoint_step(self.time, endpoint, proposal)?;
        self.attempt(step, rhs)
    }
    /// Evaluate a candidate. Failure leaves accepted state and time unchanged.
    pub fn attempt<F, E>(&mut self, step: f64, rhs: &mut F) -> Result<StepView<'_>, StepError<E>>
    where
        F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), E>,
    {
        if self.pending.is_some() {
            return Err(StepFailure::PendingCandidate.into());
        }
        let end = checked_time(self.time, step)?;
        // Advance state over the same representable interval as the public clock.
        let step = end - self.time;
        let n = self.state.len();
        self.stats.attempts += 1;
        for k in 0..n {
            self.workspace[self.candidate_offset + k] = self.state[k];
        }
        for k in 0..n {
            self.workspace[self.error_offset + k] = 0.0;
        }
        if let Some(offset) = self.second_error_offset {
            for k in 0..n {
                self.workspace[offset + k] = 0.0;
            }
        }
        for k in 0..n {
            self.workspace[self.temporary_offset + k] = 0.0;
        }
        for k in 0..(n * self.tableau.stages()) {
            self.workspace[self.stages_offset + k] = 0.0;
        }
        if step != 0.0 {
            for i in 0..self.tableau.stages() {
                let start = i * n;
                let stage_start = self.stages_offset + start;
                if i == 0 && self.tableau.c()[0] == 0.0 && self.derivative_valid {
                    for k in 0..n {
                        self.workspace[stage_start + k] = self.workspace[self.derivative_offset + k];
                    }
                } else {
                    for k in 0..n {
                        self.workspace[self.temporary_offset + k] = self.state[k];
                    }
                    for (j, &a) in self
                        .tableau
                        .stage_row(i)
                        .expect("validated explicit tableau")
                        .iter()
                        .enumerate()
                    {
                        if a != 0.0 {
                            for k in 0..n {
                                self.workspace[self.temporary_offset + k] +=
                                    step * a * self.workspace[self.stages_offset + j * n + k];
                            }
                        }
                    }
                    let temporary = &self.workspace[self.temporary_offset..self.temporary_offset + n];
                    finite(temporary)?;
                    let stage_time = self.time + self.tableau.c()[i] * step;
                    finite(&[stage_time])?;
                    self.stats.rhs_evaluations += 1;
                    let (prefix, stage_tail) = self.workspace.split_at_mut(stage_start);
                    let temporary2 = &prefix[self.temporary_offset..self.temporary_offset + n];
                    let stage = &mut stage_tail[..n];
                    rhs(stage_time, temporary2, stage).map_err(StepError::User)?;
                    finite(stage)?;
                    if i == 0 && self.tableau.c()[0] == 0.0 {
                        for k in 0..n {
                            self.workspace[self.derivative_offset + k] =
                                self.workspace[self.stages_offset + k];
                        }
                        self.derivative_valid = true;
                    }
                }
                let b = self.tableau.b()[i];
                let e = self.tableau.error().map_or(0., |w| w[i]);
                let e2 = self.tableau.second_error().map_or(0., |w| w[i]);
                for k in 0..n {
                    self.workspace[self.candidate_offset + k] +=
                        step * b * self.workspace[stage_start + k];
                    self.workspace[self.error_offset + k] +=
                        step * e * self.workspace[stage_start + k];
                }
                if let Some(offset) = self.second_error_offset {
                    for k in 0..n {
                        self.workspace[offset + k] += step * e2 * self.workspace[stage_start + k];
                    }
                }
            }
            finite(self.candidate_slice())?;
            finite(self.error_slice())?;
            if let Some(offset) = self.second_error_offset {
                finite(&self.workspace[offset..offset + n])?;
            }
        }
        self.pending = Some(step);
        Ok(StepView {
            previous_state: &self.state,
            start_time: self.time,
            end_time: end,
            candidate: self.candidate_slice(),
            component_error: self.tableau.error().map(|_| self.error_slice()),
            second_error: self
                .tableau
                .second_error()
                .map(|_| self.second_error_slice().expect("allocated when present")),
            stage_derivatives: if step == 0.0 { &[] } else { self.stages_slice() },
            statistics: self.stats,
            dimension: n,
        })
    }
    /// Commit the pending candidate and advance the accepted time.
    pub fn accept(&mut self) -> Result<(), StepFailure> {
        let step = self.pending.take().ok_or(StepFailure::NoCandidate)?;
        let n = self.state.len();
        for k in 0..n {
            self.state[k] = self.workspace[self.candidate_offset + k];
        }
        self.time += step;
        self.stats.accepted_steps += 1;
        if step != 0.0 {
            self.derivative_valid = self.tableau.fsal();
            if self.derivative_valid {
                let start = (self.tableau.stages() - 1) * n;
                for k in 0..n {
                    self.workspace[self.derivative_offset + k] =
                        self.workspace[self.stages_offset + start + k];
                }
            }
        }
        Ok(())
    }
    /// Discard a candidate; keep accepted state and its derivative cache.
    pub fn reject(&mut self) -> Result<(), StepFailure> {
        self.pending.take().ok_or(StepFailure::NoCandidate)?;
        self.stats.rejected_steps += 1;
        Ok(())
    }
}

impl ExplicitRungeKuttaStepper<'_> {
    pub(super) fn pending_step(&self) -> Option<f64> {
        self.pending
    }
    pub(super) fn attempt_with_norm<F, N, E>(
        &mut self,
        endpoint: f64,
        proposal: f64,
        rhs: &mut F,
        norm: &mut N,
    ) -> Result<f64, super::IntegrationError<E>>
    where
        F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), E>,
        N: FnMut(&[f64], &[f64], &[f64]) -> Result<f64, E>,
    {
        if self.tableau.error().is_none() {
            return Err(super::IntegrationError::MissingErrorEstimate);
        }
        self.attempt_to(endpoint, proposal, rhs)?;
        // Match the existing resource RK kernel's maximum-of-norms policy.
        // Hosts needing a compound estimator drive the raw StepView directly.
        let estimate = (|| {
            let primary = norm(&self.state, self.candidate_slice(), self.error_slice())?;
            if primary.is_nan() || primary < 0.0 {
                return Ok(primary);
            }
            if self.tableau.second_error().is_some() {
                let secondary = norm(
                    &self.state,
                    self.candidate_slice(),
                    self.second_error_slice().expect("allocated when present"),
                )?;
                if secondary.is_nan() || secondary < 0.0 {
                    return Ok(secondary);
                }
                Ok(primary.max(secondary))
            } else {
                Ok(primary)
            }
        })();
        match estimate {
            Ok(e) => Ok(e),
            Err(e) => {
                self.pending = None;
                Err(StepError::User(e).into())
            }
        }
    }
    /// Copy the final accepted state into a same-sized caller buffer, without resizing.
    pub fn copy_state_into(&self, output: &mut [f64]) -> Result<(), StepFailure> {
        if output.len() != self.state.len() {
            return Err(StepFailure::Dimension);
        }
        output.copy_from_slice(&self.state);
        Ok(())
    }
}
