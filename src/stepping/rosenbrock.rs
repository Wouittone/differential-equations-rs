use super::state_buffer::StateBuffer;
use super::{StepError, StepFailure, StepStatistics, TimeDifferencePolicy, checked_time, finite};
use crate::linear::{factorize, solve_factorized};
use crate::tableau::{RosenbrockKind, RosenbrockTableau};

/// Borrowed fallible Jacobian or explicit-time-partial hook.
/// Jacobians are row-major `out[row*n+column] = df_row/dy_column`.
/// Time partials hold the supplied state fixed; they are not `df/dt + J*f`.
pub type DerivativeHook<'a, E> = dyn FnMut(f64, &[f64], &mut [f64]) -> Result<(), E> + 'a;
/// Attempt results distinguishing solved increments from RHS evaluations.
#[derive(Debug)]
pub struct RosenbrockStepView<'a> {
    /// Accepted state before this attempt.
    pub previous_state: &'a [f64],
    /// Accepted start time.
    pub start_time: f64,
    /// Candidate time.
    pub end_time: f64,
    /// Uncommitted candidate state.
    pub candidate: &'a [f64],
    /// Signed embedded component error, if supplied by the tableau.
    pub component_error: Option<&'a [f64]>,
    /// Solved stage increments k, including the `h*gamma` scale.
    pub solved_stages: &'a [f64],
    /// Actual unscaled RHS values at stage states, not solved increments.
    pub stage_derivatives: &'a [f64],
    /// Row-major state Jacobian at the accepted state.
    pub jacobian: &'a [f64],
    /// Explicit time partial at fixed state.
    pub time_partial: &'a [f64],
    /// Cumulative factorization/solve/evaluation statistics.
    pub statistics: StepStatistics,
    dimension: usize,
}
impl RosenbrockStepView<'_> {
    /// Solved stage increment, not an RHS evaluation.
    pub fn stage(&self, index: usize) -> Option<&[f64]> {
        let start = index.checked_mul(self.dimension)?;
        self.solved_stages
            .get(start..start.checked_add(self.dimension)?)
    }
    /// Unscaled right-hand side evaluated at a stage.
    pub fn rhs_stage(&self, index: usize) -> Option<&[f64]> {
        let start = index.checked_mul(self.dimension)?;
        self.stage_derivatives
            .get(start..start.checked_add(self.dimension)?)
    }
}
/// Allocation-free repeated attempts for transformed Rosenbrock/Rodas tableaus.
///
/// Uses `(I-h*gamma*J) k_i = h*gamma*(f_i+h*d_i*f_t+C_i*k/h)`.
/// Hybrid implicit/explicit tableaus and specialized low-storage pairs require
/// their existing solver kernels and are rejected by this type. Each attempt
/// factorizes once and solves once per stage. Accepted state, derivatives and
/// matrices are reusable; accepted state can be owned or borrowed. Hooks are
/// borrowed only for the duration of an attempt.
/// Tableaus requiring Richardson estimation remain usable for fixed steps, but
/// expose no component error. Algorithm wrappers such as Rodas5Pr's residual
/// control cannot be inferred from its shared Rodas5P tableau and are not applied.
#[derive(Debug)]
pub struct RosenbrockStepper<'a> {
    tableau: &'a RosenbrockTableau,
    time: f64,
    state: StateBuffer<'a>,
    candidate: Vec<f64>,
    error: Vec<f64>,
    stages: Vec<f64>,
    rhs_stages: Vec<f64>,
    stage_state: Vec<f64>,
    current: Vec<f64>,
    jacobian: Vec<f64>,
    factors: Vec<f64>,
    pivots: Vec<usize>,
    time_partial: Vec<f64>,
    probe_state: Vec<f64>,
    probe_rhs: Vec<f64>,
    linear_rhs: Vec<f64>,
    derivative_valid: bool,
    differentiation_valid: bool,
    pending: Option<f64>,
    stats: StepStatistics,
    time_policy: TimeDifferencePolicy,
}
impl<'a> RosenbrockStepper<'a> {
    /// Construct fixed-size state and matrix workspace from a borrowed tableau.
    pub fn new(
        tableau: &'a RosenbrockTableau,
        time: f64,
        state: &[f64],
    ) -> Result<Self, StepFailure> {
        Self::with_storage(tableau, time, StateBuffer::Owned(state.to_vec()))
    }
    /// Borrow caller-owned accepted state; only scratch/matrix storage is allocated.
    /// Acceptance updates this buffer, while failed/rejected attempts leave it unchanged.
    pub fn from_buffer(
        tableau: &'a RosenbrockTableau,
        time: f64,
        state: &'a mut [f64],
    ) -> Result<Self, StepFailure> {
        Self::with_storage(tableau, time, StateBuffer::Borrowed(state))
    }
    fn with_storage(
        tableau: &'a RosenbrockTableau,
        time: f64,
        state: StateBuffer<'a>,
    ) -> Result<Self, StepFailure> {
        if state.is_empty() {
            return Err(StepFailure::Dimension);
        }
        finite(&state)?;
        finite(&[time])?;
        if tableau.kind() != RosenbrockKind::Rosenbrock {
            return Err(StepFailure::UnsupportedTableau);
        }
        let n = state.len();
        let nn = n.checked_mul(n).ok_or(StepFailure::Dimension)?;
        let sn = n
            .checked_mul(tableau.stages())
            .ok_or(StepFailure::Dimension)?;
        Ok(Self {
            tableau,
            time,
            state,
            candidate: vec![0.; n],
            error: vec![0.; n],
            stages: vec![0.; sn],
            rhs_stages: vec![0.; sn],
            stage_state: vec![0.; n],
            current: vec![0.; n],
            jacobian: vec![0.; nn],
            factors: vec![0.; nn],
            pivots: vec![0; n],
            time_partial: vec![0.; n],
            probe_state: vec![0.; n],
            probe_rhs: vec![0.; n],
            linear_rhs: vec![0.; n],
            derivative_valid: false,
            differentiation_valid: false,
            pending: None,
            stats: StepStatistics::default(),
            time_policy: TimeDifferencePolicy::default(),
        })
    }
    /// Last accepted time.
    pub fn time(&self) -> f64 {
        self.time
    }
    /// Last accepted state.
    pub fn state(&self) -> &[f64] {
        &self.state
    }
    /// Cumulative work.
    pub fn statistics(&self) -> StepStatistics {
        self.stats
    }
    /// Method coefficients.
    pub fn tableau(&self) -> &'a RosenbrockTableau {
        self.tableau
    }
    /// Set physical scale/domain policy and invalidate differentiation caches.
    pub fn set_time_difference_policy(&mut self, policy: TimeDifferencePolicy) {
        self.time_policy = policy;
        self.differentiation_valid = false;
    }
    /// Invalidate all evaluation caches after control, parameter or hook changes.
    pub fn invalidate_derivative(&mut self) {
        self.derivative_valid = false;
        self.differentiation_valid = false;
    }
    /// Accepted-state derivative cache when available (Rodas is not assumed FSAL).
    pub fn current_derivative(&self) -> Option<&[f64]> {
        self.derivative_valid.then_some(&self.current)
    }
    /// Inject the full accepted-state RHS; Jacobian and time partial are invalidated.
    pub fn inject_derivative(&mut self, derivative: &[f64]) -> Result<(), StepFailure> {
        if self.pending.is_some() {
            return Err(StepFailure::PendingCandidate);
        }
        if derivative.len() != self.state.len() {
            return Err(StepFailure::Dimension);
        }
        finite(derivative)?;
        self.current.copy_from_slice(derivative);
        self.derivative_valid = true;
        self.differentiation_valid = false;
        Ok(())
    }
    /// Same-sized restart discards all caches and pending output, reusing storage.
    pub fn reset(&mut self, time: f64, state: &[f64]) -> Result<(), StepFailure> {
        if state.len() != self.state.len() {
            return Err(StepFailure::Dimension);
        }
        finite(state)?;
        finite(&[time])?;
        self.state.copy_from_slice(state);
        self.time = time;
        self.pending = None;
        self.invalidate_derivative();
        Ok(())
    }
    /// Attempt with optional analytic Jacobian and explicit time partial.
    ///
    /// Missing hooks use finite differences. Both hooks can borrow mutable caller
    /// data and propagate the same original application error type as the RHS.
    /// Differentiation is reused after rejection; changing hooks requires explicit
    /// invalidation. A failed hook leaves accepted state unchanged.
    pub fn attempt<F, E>(
        &mut self,
        step: f64,
        rhs: &mut F,
        mut jacobian: Option<&mut DerivativeHook<'_, E>>,
        mut time_partial: Option<&mut DerivativeHook<'_, E>>,
    ) -> Result<RosenbrockStepView<'_>, StepError<E>>
    where
        F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), E>,
    {
        if self.pending.is_some() {
            return Err(StepFailure::PendingCandidate.into());
        }
        let end = checked_time(self.time, step)?;
        // Advance state over the same representable interval as the public clock.
        let step = end - self.time;
        self.stats.attempts += 1;
        let n = self.state.len();
        self.candidate.copy_from_slice(&self.state);
        self.error.fill(0.);
        if step != 0. {
            if !self.derivative_valid {
                self.stats.rhs_evaluations += 1;
                rhs(self.time, &self.state, &mut self.current).map_err(StepError::User)?;
                finite(&self.current)?;
                self.derivative_valid = true;
            }
            if !self.differentiation_valid {
                self.stats.jacobian_evaluations += 1;
                if let Some(hook) = jacobian.as_mut() {
                    hook(self.time, &self.state, &mut self.jacobian).map_err(StepError::User)?;
                } else {
                    for column in 0..n {
                        self.probe_state.copy_from_slice(&self.state);
                        self.probe_state[column] +=
                            f64::EPSILON.sqrt() * self.state[column].abs().max(1.);
                        finite(&self.probe_state)?;
                        let delta = self.probe_state[column] - self.state[column];
                        if delta == 0. {
                            return Err(StepFailure::TimeResolution.into());
                        }
                        self.stats.rhs_evaluations += 1;
                        rhs(self.time, &self.probe_state, &mut self.probe_rhs)
                            .map_err(StepError::User)?;
                        finite(&self.probe_rhs)?;
                        for row in 0..n {
                            self.jacobian[row * n + column] =
                                (self.probe_rhs[row] - self.current[row]) / delta;
                        }
                    }
                }
                finite(&self.jacobian)?;
                if let Some(hook) = time_partial.as_mut() {
                    hook(self.time, &self.state, &mut self.time_partial)
                        .map_err(StepError::User)?;
                } else {
                    let (first, second) = self.time_policy.probes(self.time, step)?;
                    self.stats.rhs_evaluations += 1;
                    rhs(first, &self.state, &mut self.time_partial).map_err(StepError::User)?;
                    finite(&self.time_partial)?;
                    if let Some(second) = second {
                        self.stats.rhs_evaluations += 1;
                        rhs(second, &self.state, &mut self.probe_rhs).map_err(StepError::User)?;
                        finite(&self.probe_rhs)?;
                    }
                    for k in 0..n {
                        self.time_partial[k] = super::time_difference::partial(
                            self.current[k],
                            self.time_partial[k],
                            second.map(|_| self.probe_rhs[k]),
                            first - self.time,
                            second.map(|t| t - self.time),
                        );
                    }
                }
                finite(&self.time_partial)?;
                self.differentiation_valid = true;
            }
            for row in 0..n {
                for col in 0..n {
                    self.factors[row * n + col] = f64::from(row == col)
                        - step * self.tableau.gamma() * self.jacobian[row * n + col];
                }
            }
            finite(&self.factors)?;
            self.stats.factorizations += 1;
            factorize(&mut self.factors, &mut self.pivots, n)
                .map_err(|_| StepFailure::SingularSystem)?;
            for i in 0..self.tableau.stages() {
                let start = i * n;
                self.stage_state.copy_from_slice(&self.state);
                for j in 0..i {
                    let a = self.tableau.a()[i][j];
                    if a != 0. {
                        for k in 0..n {
                            self.stage_state[k] += a * self.stages[j * n + k];
                        }
                    }
                }
                finite(&self.stage_state)?;
                if i == 0 && self.tableau.c()[i] == 0. {
                    self.rhs_stages[..n].copy_from_slice(&self.current);
                } else {
                    let time = self.time + self.tableau.c()[i] * step;
                    finite(&[time])?;
                    self.stats.rhs_evaluations += 1;
                    rhs(
                        time,
                        &self.stage_state,
                        &mut self.rhs_stages[start..start + n],
                    )
                    .map_err(StepError::User)?;
                    finite(&self.rhs_stages[start..start + n])?;
                }
                for k in 0..n {
                    self.linear_rhs[k] = self.rhs_stages[start + k]
                        + step * self.tableau.d()[i] * self.time_partial[k];
                }
                for j in 0..i {
                    let c = self.tableau.coupling()[i][j] / step;
                    if c != 0. {
                        for k in 0..n {
                            self.linear_rhs[k] += c * self.stages[j * n + k];
                        }
                    }
                }
                solve_factorized(&self.factors, &self.pivots, &mut self.linear_rhs, n);
                self.stats.linear_solves += 1;
                for k in 0..n {
                    self.stages[start + k] = step * self.tableau.gamma() * self.linear_rhs[k];
                }
                finite(&self.stages[start..start + n])?;
                for k in 0..n {
                    self.candidate[k] += self.tableau.b()[i] * self.stages[start + k];
                    self.error[k] +=
                        self.tableau.btilde().map_or(0., |w| w[i]) * self.stages[start + k];
                }
            }
            finite(&self.candidate)?;
            finite(&self.error)?;
        }
        self.pending = Some(step);
        Ok(RosenbrockStepView {
            previous_state: &self.state,
            start_time: self.time,
            end_time: end,
            candidate: &self.candidate,
            component_error: match self.tableau.error_estimator() {
                differential_equations_tableau_core::RosenbrockErrorEstimator::Embedded => {
                    self.tableau.btilde().map(|_| self.error.as_slice())
                }
                _ => None,
            },
            solved_stages: if step == 0. { &[] } else { &self.stages },
            stage_derivatives: if step == 0. { &[] } else { &self.rhs_stages },
            jacobian: if step == 0. { &[] } else { &self.jacobian },
            time_partial: if step == 0. { &[] } else { &self.time_partial },
            statistics: self.stats,
            dimension: n,
        })
    }
    /// Commit state and discard accepted-state differentiation caches.
    pub fn accept(&mut self) -> Result<(), StepFailure> {
        let step = self.pending.take().ok_or(StepFailure::NoCandidate)?;
        self.state.copy_from_slice(&self.candidate);
        self.time += step;
        self.stats.accepted_steps += 1;
        if step != 0. {
            self.invalidate_derivative();
        }
        Ok(())
    }
    /// Reject state; retain accepted-state derivative, Jacobian and time partial.
    pub fn reject(&mut self) -> Result<(), StepFailure> {
        self.pending.take().ok_or(StepFailure::NoCandidate)?;
        self.stats.rejected_steps += 1;
        Ok(())
    }
}

impl RosenbrockStepper<'_> {
    /// Endpoint-clamped attempt preserving signed propagation direction.
    pub fn attempt_to<F, E>(
        &mut self,
        endpoint: f64,
        proposal: f64,
        rhs: &mut F,
        jacobian: Option<&mut DerivativeHook<'_, E>>,
        time_partial: Option<&mut DerivativeHook<'_, E>>,
    ) -> Result<RosenbrockStepView<'_>, StepError<E>>
    where
        F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), E>,
    {
        let step = super::endpoint_step(self.time, endpoint, proposal)?;
        self.attempt(step, rhs, jacobian, time_partial)
    }
    /// Copy accepted state to an existing correctly sized output buffer.
    pub fn copy_state_into(&self, output: &mut [f64]) -> Result<(), StepFailure> {
        if output.len() != self.state.len() {
            return Err(StepFailure::Dimension);
        }
        output.copy_from_slice(&self.state);
        Ok(())
    }
    /// Clear work counters, retaining accepted state and valid caches.
    pub fn clear_statistics(&mut self) {
        self.stats = StepStatistics::default();
    }
}
