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
    /// Signed component embedded errors, if the method supplies them.
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
/// Construction allocates six state vectors and one stage-major vector.
/// Attempt, accept, reject, reset and derivative injection never allocate.
/// Specialized fitted formulas are rejected; their frequency-dependent weights
/// require a dedicated kernel. Both embedded error vectors remain exposed so
/// hosts can implement their method-specific norm formulas exactly.
#[derive(Debug)]
pub struct ExplicitRungeKuttaStepper<'a> {
    tableau: &'a RungeKuttaTableau,
    time: f64,
    state: StateBuffer<'a>,
    candidate: Vec<f64>,
    error: Vec<f64>,
    second_error: Vec<f64>,
    temporary: Vec<f64>,
    stages: Vec<f64>,
    derivative: Vec<f64>,
    derivative_valid: bool,
    pending: Option<f64>,
    stats: StepStatistics,
}
impl<'a> ExplicitRungeKuttaStepper<'a> {
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
        Ok(Self {
            tableau,
            time,
            state,
            candidate: vec![0.; n],
            error: vec![0.; n],
            second_error: vec![0.; n],
            temporary: vec![0.; n],
            stages: vec![0.; stage_len],
            derivative: vec![0.; n],
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
        self.derivative_valid.then_some(&self.derivative)
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
        self.derivative.copy_from_slice(derivative);
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
        self.stats.attempts += 1;
        self.candidate.copy_from_slice(&self.state);
        self.error.fill(0.);
        self.second_error.fill(0.);
        let n = self.state.len();
        if step != 0.0 {
            for i in 0..self.tableau.stages() {
                let start = i * n;
                if i == 0 && self.tableau.c()[0] == 0.0 && self.derivative_valid {
                    self.stages[..n].copy_from_slice(&self.derivative);
                } else {
                    self.temporary.copy_from_slice(&self.state);
                    for (j, &a) in self
                        .tableau
                        .stage_row(i)
                        .expect("validated explicit tableau")
                        .iter()
                        .enumerate()
                    {
                        if a != 0.0 {
                            for k in 0..n {
                                self.temporary[k] += step * a * self.stages[j * n + k];
                            }
                        }
                    }
                    finite(&self.temporary)?;
                    let stage_time = self.time + self.tableau.c()[i] * step;
                    finite(&[stage_time])?;
                    self.stats.rhs_evaluations += 1;
                    rhs(
                        stage_time,
                        &self.temporary,
                        &mut self.stages[start..start + n],
                    )
                    .map_err(StepError::User)?;
                    finite(&self.stages[start..start + n])?;
                    if i == 0 && self.tableau.c()[0] == 0.0 {
                        self.derivative.copy_from_slice(&self.stages[..n]);
                        self.derivative_valid = true;
                    }
                }
                let b = self.tableau.b()[i];
                let e = self.tableau.error().map_or(0., |w| w[i]);
                let e2 = self.tableau.second_error().map_or(0., |w| w[i]);
                for k in 0..n {
                    self.candidate[k] += step * b * self.stages[start + k];
                    self.error[k] += step * e * self.stages[start + k];
                    self.second_error[k] += step * e2 * self.stages[start + k];
                }
            }
            finite(&self.candidate)?;
            finite(&self.error)?;
            finite(&self.second_error)?;
        }
        self.pending = Some(step);
        Ok(StepView {
            previous_state: &self.state,
            start_time: self.time,
            end_time: end,
            candidate: &self.candidate,
            component_error: self.tableau.error().map(|_| self.error.as_slice()),
            second_error: self
                .tableau
                .second_error()
                .map(|_| self.second_error.as_slice()),
            stage_derivatives: if step == 0.0 { &[] } else { &self.stages },
            statistics: self.stats,
            dimension: n,
        })
    }
    /// Commit the pending candidate and advance the accepted time.
    pub fn accept(&mut self) -> Result<(), StepFailure> {
        let step = self.pending.take().ok_or(StepFailure::NoCandidate)?;
        self.state.copy_from_slice(&self.candidate);
        self.time += step;
        self.stats.accepted_steps += 1;
        if step != 0.0 {
            self.derivative_valid = self.tableau.fsal();
            if self.derivative_valid {
                let n = self.state.len();
                let start = (self.tableau.stages() - 1) * n;
                self.derivative
                    .copy_from_slice(&self.stages[start..start + n]);
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
        match norm(&self.state, &self.candidate, &self.error) {
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
