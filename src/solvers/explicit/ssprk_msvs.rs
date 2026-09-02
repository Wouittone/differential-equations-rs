//! Shu--Osher multistep SSP methods from OrdinaryDiffEqSSPRK.
//!
//! `SSPRKMSVS32` and `SSPRKMSVS43` are respectively three-step, second-order
//! and four-step, third-order methods. Although the pinned Julia types carry
//! the adaptive-algorithm marker, their documentation and implementations
//! provide no error estimator and require a fixed time step. This port
//! therefore deliberately exposes fixed-step integration only.

use crate::callback::CallbackOutcome;
use crate::integrator::{KernelCapabilities, StepEstimate, StepKernel, integrate};
use crate::solution::{BorrowedHermiteSegment, DenseSegment, HermiteSegment, TrajectoryRecorder};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

/// The second-order, three-step strong-stability-preserving linear multistep
/// method `SSPRKMSVS32`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SspRkMsvs32;

/// Upper-case spelling retained for callers matching the upstream algorithm
/// name.  The canonical Rust type follows the crate's existing `SspRk*`
/// naming convention.
#[allow(non_camel_case_types)]
pub type SSPRKMSVS32 = SspRkMsvs32;

/// The third-order, four-step strong-stability-preserving linear multistep
/// method `SSPRKMSVS43`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SspRkMsvs43;

/// Upper-case spelling retained for callers matching the upstream algorithm
/// name. The canonical Rust type follows the crate's existing `SspRk*`
/// naming convention.
#[allow(non_camel_case_types)]
pub type SSPRKMSVS43 = SspRkMsvs43;

struct Msvs32Kernel {
    /// Derivative at the accepted state at the start of the trial step.
    derivative: Vec<f64>,
    /// Derivative at the candidate endpoint, retained for the next step.
    next_derivative: Vec<f64>,
    dense_endpoint_state: Vec<f64>,
    dense_endpoint_prepared: bool,
    /// Temporary Euler state used by the two-stage startup procedure.
    euler_state: Vec<f64>,
    /// State two accepted steps before the current state (`u_2` upstream).
    u_2: Vec<f64>,
    /// State one accepted step before the current state (`u_1` upstream).
    u_1: Vec<f64>,
    /// Number of accepted kernel steps, including startup steps.
    step: usize,
    /// Magnitude of the previous accepted step, for a clipped final step.
    last_step: f64,
}

struct Msvs43Kernel {
    /// Derivative at the accepted state at the start of the trial step.
    derivative: Vec<f64>,
    /// Derivative at the candidate endpoint, retained for the next step.
    next_derivative: Vec<f64>,
    dense_endpoint_state: Vec<f64>,
    dense_endpoint_prepared: bool,
    /// Temporary Euler state used by the two-stage startup procedure.
    euler_state: Vec<f64>,
    /// Accepted states one, two, and three steps behind the current state.
    u_1: Vec<f64>,
    u_2: Vec<f64>,
    u_3: Vec<f64>,
    /// Derivatives paired with the three stored history states.
    k1: Vec<f64>,
    k2: Vec<f64>,
    k3: Vec<f64>,
    /// Number of accepted kernel steps, including startup steps.
    step: usize,
    /// Magnitude of the previous accepted step, for a clipped final step.
    last_step: f64,
}

#[allow(clippy::too_many_arguments)]
fn apply_msvs_callbacks<F, P>(
    problem: &OdeProblem<F, P>,
    previous_state: &[f64],
    previous_time: f64,
    state: &mut [f64],
    time: &mut f64,
    state_before_effect: &mut [f64],
    event_tolerance: f64,
    start_derivative: &[f64],
    endpoint_derivative: &[f64],
    endpoint_state: &mut [f64],
    endpoint_prepared: &mut bool,
) -> Result<CallbackOutcome, SolveError>
where
    F: crate::OdeFunction<P>,
{
    if !problem.has_callbacks() {
        *endpoint_prepared = false;
        return problem.apply_step_callbacks(
            previous_state,
            previous_time,
            state,
            time,
            state_before_effect,
            event_tolerance,
            None,
        );
    }
    endpoint_state.copy_from_slice(state);
    *endpoint_prepared = true;
    if !problem.has_continuous_callbacks() {
        return problem.apply_step_callbacks(
            previous_state,
            previous_time,
            state,
            time,
            state_before_effect,
            event_tolerance,
            None,
        );
    }
    let segment = BorrowedHermiteSegment::new(
        previous_time,
        *time,
        previous_state,
        endpoint_state,
        start_derivative,
        endpoint_derivative,
    )
    .map_err(|_| SolveError::NonFiniteDerivative)?;
    let mut interpolate = |sample_time: f64, output: &mut [f64]| {
        segment
            .interpolate(sample_time, output)
            .map_err(|_| SolveError::NonFiniteDerivative)
    };
    problem.apply_step_callbacks(
        previous_state,
        previous_time,
        state,
        time,
        state_before_effect,
        event_tolerance,
        Some(&mut interpolate),
    )
}

#[allow(clippy::too_many_arguments)]
fn record_msvs_step(
    previous_state: &[f64],
    state: &[f64],
    start_derivative: &[f64],
    endpoint_derivative: &[f64],
    endpoint_state: &mut [f64],
    endpoint_prepared: &mut bool,
    previous_time: f64,
    attempted_time: f64,
    time: f64,
    final_time: bool,
    recorder: &mut TrajectoryRecorder<'_>,
) -> Result<bool, SolveError> {
    if !recorder.needs_dense_sampling() && !recorder.retains_dense_output() {
        *endpoint_prepared = false;
        return Ok(false);
    }
    if !*endpoint_prepared {
        endpoint_state.copy_from_slice(state);
    }
    let segment = BorrowedHermiteSegment::new(
        previous_time,
        attempted_time,
        previous_state,
        endpoint_state,
        start_derivative,
        endpoint_derivative,
    )
    .map_err(|_| SolveError::NonFiniteDerivative)?;
    recorder
        .record_step_dense(
            previous_state,
            previous_time,
            state,
            time,
            final_time,
            &segment,
        )
        .map_err(|_| SolveError::NonFiniteDerivative)?;
    if recorder.retains_dense_output() {
        let segment = HermiteSegment::new_bounded(
            previous_time,
            attempted_time,
            time,
            previous_state.to_vec(),
            endpoint_state.to_vec(),
            start_derivative.to_vec(),
            endpoint_derivative.to_vec(),
        )
        .map_err(|_| SolveError::NonFiniteDerivative)?;
        recorder.retain_hermite_segment(segment);
    }
    *endpoint_prepared = false;
    Ok(true)
}

impl Msvs32Kernel {
    fn new(dimension: usize) -> Self {
        Self {
            derivative: vec![0.0; dimension],
            next_derivative: vec![0.0; dimension],
            dense_endpoint_state: vec![0.0; dimension],
            dense_endpoint_prepared: false,
            euler_state: vec![0.0; dimension],
            u_2: vec![0.0; dimension],
            u_1: vec![0.0; dimension],
            step: 1,
            last_step: 0.0,
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
        Ok(())
    }

    fn ensure_finite(values: &[f64]) -> Result<(), SolveError> {
        values
            .iter()
            .all(|value| value.is_finite())
            .then_some(())
            .ok_or(SolveError::NonFiniteDerivative)
    }
}

impl Msvs43Kernel {
    fn new(dimension: usize) -> Self {
        Self {
            derivative: vec![0.0; dimension],
            next_derivative: vec![0.0; dimension],
            dense_endpoint_state: vec![0.0; dimension],
            dense_endpoint_prepared: false,
            euler_state: vec![0.0; dimension],
            u_1: vec![0.0; dimension],
            u_2: vec![0.0; dimension],
            u_3: vec![0.0; dimension],
            k1: vec![0.0; dimension],
            k2: vec![0.0; dimension],
            k3: vec![0.0; dimension],
            step: 1,
            last_step: 0.0,
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
        Ok(())
    }

    fn ensure_finite(values: &[f64]) -> Result<(), SolveError> {
        values
            .iter()
            .all(|value| value.is_finite())
            .then_some(())
            .ok_or(SolveError::NonFiniteDerivative)
    }
}

impl<F, P> StepKernel<F, P> for Msvs32Kernel
where
    F: crate::OdeFunction<P>,
{
    fn has_custom_dense_output(&self) -> bool {
        true
    }

    fn capabilities(&self) -> KernelCapabilities {
        // The pinned constructor has no error estimator and explicitly
        // requires fixed timestep operation.
        KernelCapabilities::new(false, 2)
    }

    fn initialize(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        Self::evaluate(problem, &mut self.derivative, state, time, stats)?;
        Self::ensure_finite(&self.derivative)?;
        self.u_2.copy_from_slice(state);
        self.u_1.copy_from_slice(state);
        self.step = 1;
        self.last_step = 0.0;
        Ok(())
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
        _: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError> {
        // A fixed-step solve may clip its final step by a tiny roundoff
        // remainder.  The multistep history term is not valid for that
        // partial interval, so use the pinned two-stage SSP startup there.
        let startup = self.step < 3 || (self.last_step > 0.0 && step.abs() < 0.5 * self.last_step);
        if startup {
            // Upstream startup: forward Euler followed by Heun using both
            // endpoint evaluations at t + dt.
            for ((output, value), derivative) in
                self.euler_state.iter_mut().zip(state).zip(&self.derivative)
            {
                *output = *value + step * derivative;
            }
            Self::ensure_finite(&self.euler_state)?;
            Self::evaluate(
                problem,
                &mut self.next_derivative,
                &self.euler_state,
                time + step,
                stats,
            )?;
            Self::ensure_finite(&self.next_derivative)?;
            for (((output, value), euler), derivative) in candidate
                .iter_mut()
                .zip(state)
                .zip(&self.euler_state)
                .zip(&self.next_derivative)
            {
                *output = (*value + *euler + step * derivative) * 0.5;
            }

            // The first startup step stores the initial state as u_2.  The
            // second startup step needs no additional history in this compact
            // cache; after it is accepted, u_2 is shifted to state one.
        } else {
            // Constant-step MSVS32 uses Ω = 2, i.e. the pinned expression
            // ((Ω² - 1)/Ω²) * (u_n + Ω/(Ω - 1) dt f_n) + u_{n-2}/Ω².
            for (((output, value), derivative), old) in candidate
                .iter_mut()
                .zip(state)
                .zip(&self.derivative)
                .zip(&self.u_2)
            {
                *output = 0.75 * (*value + 2.0 * step * derivative) + 0.25 * old;
            }
            Self::ensure_finite(candidate)?;
        }

        // The upstream perform-step evaluates the endpoint derivative for
        // FSAL use by the next accepted step.
        Self::evaluate(
            problem,
            &mut self.next_derivative,
            candidate,
            time + step,
            stats,
        )?;
        Self::ensure_finite(&self.next_derivative)?;
        Ok(StepEstimate::new(0.0))
    }

    fn apply_step_callbacks(
        &mut self,
        problem: &OdeProblem<F, P>,
        previous_state: &[f64],
        previous_time: f64,
        state: &mut [f64],
        time: &mut f64,
        state_before_effect: &mut [f64],
        event_tolerance: f64,
        _: &mut SolverStats,
    ) -> Result<CallbackOutcome, SolveError> {
        apply_msvs_callbacks(
            problem,
            previous_state,
            previous_time,
            state,
            time,
            state_before_effect,
            event_tolerance,
            &self.derivative,
            &self.next_derivative,
            &mut self.dense_endpoint_state,
            &mut self.dense_endpoint_prepared,
        )
    }

    fn record_dense_step(
        &mut self,
        _: &OdeProblem<F, P>,
        previous_state: &[f64],
        state: &[f64],
        previous_time: f64,
        attempted_time: f64,
        time: f64,
        final_time: bool,
        recorder: &mut TrajectoryRecorder<'_>,
        _: &mut SolverStats,
    ) -> Result<bool, SolveError> {
        record_msvs_step(
            previous_state,
            state,
            &self.derivative,
            &self.next_derivative,
            &mut self.dense_endpoint_state,
            &mut self.dense_endpoint_prepared,
            previous_time,
            attempted_time,
            time,
            final_time,
            recorder,
        )
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
            // The Julia cache does not itself expose callback history reset,
            // but retaining pre-event states would make the next MSVS step
            // depend on a discontinuous trajectory.  Restarting the pinned
            // startup sequence is the sound regular-ODE behavior.
            self.u_2.copy_from_slice(state);
            self.u_1.copy_from_slice(state);
            self.step = 1;
            self.last_step = 0.0;
            Self::evaluate(problem, &mut self.derivative, state, time, stats)?;
            Self::ensure_finite(&self.derivative)?;
            return Ok(());
        }

        if self.step == 1 {
            self.u_2.copy_from_slice(previous_state);
        } else if self.step == 2 {
            self.u_1.copy_from_slice(previous_state);
        } else {
            self.u_2.copy_from_slice(&self.u_1);
            self.u_1.copy_from_slice(previous_state);
        }
        self.derivative.copy_from_slice(&self.next_derivative);
        self.last_step = accepted_step.abs();
        self.step += 1;
        Ok(())
    }

    fn reject_step(&mut self) {}
}

impl<F, P> StepKernel<F, P> for Msvs43Kernel
where
    F: crate::OdeFunction<P>,
{
    fn has_custom_dense_output(&self) -> bool {
        true
    }

    fn capabilities(&self) -> KernelCapabilities {
        // The pinned constructor has no error estimator and explicitly
        // requires fixed timestep operation.
        KernelCapabilities::new(false, 3)
    }

    fn initialize(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        Self::evaluate(problem, &mut self.derivative, state, time, stats)?;
        Self::ensure_finite(&self.derivative)?;
        self.u_1.copy_from_slice(state);
        self.u_2.copy_from_slice(state);
        self.u_3.copy_from_slice(state);
        self.k1.fill(0.0);
        self.k2.fill(0.0);
        self.k3.fill(0.0);
        self.step = 1;
        self.last_step = 0.0;
        Ok(())
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
        _: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError> {
        // The pinned algorithm starts with three SSPRK22 steps. A tiny clipped
        // final interval cannot use constant-step multistep history either.
        let startup = self.step < 4 || (self.last_step > 0.0 && step.abs() < 0.5 * self.last_step);
        if startup {
            for ((output, value), derivative) in
                self.euler_state.iter_mut().zip(state).zip(&self.derivative)
            {
                *output = *value + step * derivative;
            }
            Self::ensure_finite(&self.euler_state)?;
            Self::evaluate(
                problem,
                &mut self.next_derivative,
                &self.euler_state,
                time + step,
                stats,
            )?;
            Self::ensure_finite(&self.next_derivative)?;
            for (((output, value), euler), derivative) in candidate
                .iter_mut()
                .zip(state)
                .zip(&self.euler_state)
                .zip(&self.next_derivative)
            {
                *output = (*value + *euler + step * derivative) * 0.5;
            }
        } else {
            // Pinned constant-step recurrence:
            // 16/27 * (u_n + 3 h f_n) + 11/27 * (u_{n-3} + 12/11 h f_{n-3}).
            for ((((output, value), derivative), old), old_derivative) in candidate
                .iter_mut()
                .zip(state)
                .zip(&self.derivative)
                .zip(&self.u_3)
                .zip(&self.k3)
            {
                *output = (16.0 / 27.0) * (*value + 3.0 * step * derivative)
                    + (11.0 / 27.0) * (*old + (12.0 / 11.0) * step * old_derivative);
            }
        }
        Self::ensure_finite(candidate)?;

        Self::evaluate(
            problem,
            &mut self.next_derivative,
            candidate,
            time + step,
            stats,
        )?;
        Self::ensure_finite(&self.next_derivative)?;
        Ok(StepEstimate::new(0.0))
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
            self.u_1.copy_from_slice(state);
            self.u_2.copy_from_slice(state);
            self.u_3.copy_from_slice(state);
            self.k1.fill(0.0);
            self.k2.fill(0.0);
            self.k3.fill(0.0);
            self.step = 1;
            self.last_step = 0.0;
            Self::evaluate(problem, &mut self.derivative, state, time, stats)?;
            Self::ensure_finite(&self.derivative)?;
            return Ok(());
        }

        match self.step {
            1 => {
                self.u_3.copy_from_slice(previous_state);
                Self::evaluate(problem, &mut self.k3, previous_state, time, stats)?;
                Self::ensure_finite(&self.k3)?;
            }
            2 => {
                self.u_2.copy_from_slice(previous_state);
                Self::evaluate(problem, &mut self.k2, previous_state, time, stats)?;
                Self::ensure_finite(&self.k2)?;
            }
            3 => {
                self.u_1.copy_from_slice(previous_state);
                Self::evaluate(problem, &mut self.k1, previous_state, time, stats)?;
                Self::ensure_finite(&self.k1)?;
            }
            _ => {
                self.u_3.copy_from_slice(&self.u_2);
                self.u_2.copy_from_slice(&self.u_1);
                self.u_1.copy_from_slice(previous_state);
                self.k3.copy_from_slice(&self.k2);
                self.k2.copy_from_slice(&self.k1);
                self.k1.copy_from_slice(&self.derivative);
            }
        }
        self.derivative.copy_from_slice(&self.next_derivative);
        self.last_step = accepted_step.abs();
        self.step += 1;
        Ok(())
    }

    fn reject_step(&mut self) {}

    fn apply_step_callbacks(
        &mut self,
        problem: &OdeProblem<F, P>,
        previous_state: &[f64],
        previous_time: f64,
        state: &mut [f64],
        time: &mut f64,
        state_before_effect: &mut [f64],
        event_tolerance: f64,
        _: &mut SolverStats,
    ) -> Result<CallbackOutcome, SolveError> {
        apply_msvs_callbacks(
            problem,
            previous_state,
            previous_time,
            state,
            time,
            state_before_effect,
            event_tolerance,
            &self.derivative,
            &self.next_derivative,
            &mut self.dense_endpoint_state,
            &mut self.dense_endpoint_prepared,
        )
    }

    fn record_dense_step(
        &mut self,
        _: &OdeProblem<F, P>,
        previous_state: &[f64],
        state: &[f64],
        previous_time: f64,
        attempted_time: f64,
        time: f64,
        final_time: bool,
        recorder: &mut TrajectoryRecorder<'_>,
        _: &mut SolverStats,
    ) -> Result<bool, SolveError> {
        record_msvs_step(
            previous_state,
            state,
            &self.derivative,
            &self.next_derivative,
            &mut self.dense_endpoint_state,
            &mut self.dense_endpoint_prepared,
            previous_time,
            attempted_time,
            time,
            final_time,
            recorder,
        )
    }
}

impl OdeAlgorithm for SspRkMsvs32 {
    fn solve_validated<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        integrate(
            problem,
            options,
            Msvs32Kernel::new(problem.initial_state().len()),
        )
    }
}

impl OdeAlgorithm for SspRkMsvs43 {
    fn solve_validated<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        integrate(
            problem,
            options,
            Msvs43Kernel::new(problem.initial_state().len()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{SspRkMsvs32, SspRkMsvs43};
    use crate::{CallbackAction, OdeProblem, SaveMode, SolveError, SolveOptions, solve};

    type TestRhs = fn(&mut [f64], &[f64], &(), f64);

    fn exponential_rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
        du[0] = u[0];
    }

    fn exponential() -> OdeProblem<TestRhs, ()> {
        OdeProblem::new(exponential_rhs, vec![1.0], (0.0, 1.0), ())
    }

    fn fixed(step: f64) -> SolveOptions {
        SolveOptions {
            adaptive: false,
            initial_step: Some(step),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        }
    }

    #[test]
    fn msvs32_is_second_order_after_startup() {
        let coarse = solve(&exponential(), SspRkMsvs32, &fixed(0.1)).unwrap();
        let fine = solve(&exponential(), SspRkMsvs32, &fixed(0.05)).unwrap();
        let coarse_error = (coarse.last_state()[0] - 1.0_f64.exp()).abs();
        let fine_error = (fine.last_state()[0] - 1.0_f64.exp()).abs();
        assert!(
            fine_error < coarse_error / 3.0,
            "{coarse_error} -> {fine_error}"
        );
    }

    #[test]
    fn msvs43_is_third_order_after_startup() {
        let coarse = solve(&exponential(), SspRkMsvs43, &fixed(0.1)).unwrap();
        let fine = solve(&exponential(), SspRkMsvs43, &fixed(0.05)).unwrap();
        let coarse_error = (coarse.last_state()[0] - 1.0_f64.exp()).abs();
        let fine_error = (fine.last_state()[0] - 1.0_f64.exp()).abs();
        assert!(
            fine_error < coarse_error / 7.0,
            "{coarse_error} -> {fine_error}"
        );
    }

    #[test]
    fn msvs32_supports_backward_and_rejects_adaptive_mode() {
        let problem = OdeProblem::new(exponential_rhs, vec![1.0_f64.exp()], (1.0, 0.0), ());
        let mut options = fixed(0.05);
        options.save = crate::SaveMode::EveryStep;
        let solution = solve(&problem, SspRkMsvs32, &options).unwrap();
        assert!(
            (solution.last_state()[0] - 1.0).abs() < 2.0e-3,
            "backward endpoint = {}",
            solution.last_state()[0]
        );
        assert_eq!(
            solve(&exponential(), SspRkMsvs32, &SolveOptions::default()),
            Err(SolveError::AdaptiveStepUnsupported)
        );
    }

    #[test]
    fn msvs43_supports_backward_and_rejects_adaptive_mode() {
        let problem = OdeProblem::new(exponential_rhs, vec![1.0_f64.exp()], (1.0, 0.0), ());
        let solution = solve(&problem, SspRkMsvs43, &fixed(0.05)).unwrap();
        assert!(
            (solution.last_state()[0] - 1.0).abs() < 5.0e-4,
            "backward endpoint = {}",
            solution.last_state()[0]
        );
        assert_eq!(
            solve(&exponential(), SspRkMsvs43, &SolveOptions::default()),
            Err(SolveError::AdaptiveStepUnsupported)
        );
    }

    #[test]
    fn msvs32_resets_history_after_callbacks_and_honors_save_at() {
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -u[0],
            vec![1.0],
            (0.0, 1.0),
            (),
        )
        .with_discrete_callback(
            |state, _: &(), time| time >= 0.5 && state[0] < 5.0,
            |state, _: &(), _| {
                state[0] += 10.0;
                CallbackAction::Continue
            },
        );
        let mut options = fixed(0.1);
        options.save = SaveMode::Endpoints;
        options.save_at = vec![0.25, 0.5, 0.75];
        let solution = solve(&problem, SspRkMsvs32, &options).unwrap();
        assert_eq!(solution.stats().callback_invocations, 1);
        assert_eq!(solution.times(), &[0.25, 0.5, 0.5, 0.75]);
        assert!(solution.state(2).unwrap()[0] - solution.state(1).unwrap()[0] > 9.0);
        assert!(solution.last_state()[0] > 3.0);
    }
}
