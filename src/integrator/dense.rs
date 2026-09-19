use super::kernel::StepKernel;
use crate::callback::CallbackOutcome;
use crate::solution::{BorrowedHermiteSegment, HermiteSegment, TrajectoryRecorder};
use crate::{OdeProblem, SolveError, SolverStats};

pub(super) struct DefaultDenseState {
    start_derivative: Vec<f64>,
    endpoint_state: Vec<f64>,
    endpoint_derivative: Vec<f64>,
    start_derivative_valid: bool,
    pub(super) prepared: bool,
}

impl DefaultDenseState {
    pub(super) fn new(dimension: usize, enabled: bool) -> Self {
        let size = if enabled { dimension } else { 0 };
        Self {
            start_derivative: vec![0.0; size],
            endpoint_state: vec![0.0; size],
            endpoint_derivative: vec![0.0; size],
            start_derivative_valid: false,
            prepared: false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare<F, P, K>(
        &mut self,
        kernel: &mut K,
        problem: &OdeProblem<F, P>,
        previous_state: &[f64],
        previous_time: f64,
        endpoint_state: &[f64],
        endpoint_time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError>
    where
        F: crate::OdeFunction<P>,
        K: StepKernel<F, P>,
    {
        self.endpoint_state.copy_from_slice(endpoint_state);
        if !self.start_derivative_valid {
            kernel.evaluate_dense_derivative(
                problem,
                &mut self.start_derivative,
                previous_state,
                previous_time,
                stats,
            )?;
        }
        kernel.evaluate_dense_derivative(
            problem,
            &mut self.endpoint_derivative,
            &self.endpoint_state,
            endpoint_time,
            stats,
        )?;
        self.prepared = true;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn apply_callbacks<F, P, K>(
        &mut self,
        kernel: &mut K,
        problem: &OdeProblem<F, P>,
        previous_state: &[f64],
        previous_time: f64,
        state: &mut [f64],
        time: &mut f64,
        state_before_effect: &mut [f64],
        event_tolerance: f64,
        enabled: bool,
        stats: &mut SolverStats,
    ) -> Result<CallbackOutcome, SolveError>
    where
        F: crate::OdeFunction<P>,
        K: StepKernel<F, P>,
    {
        if !enabled {
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
        let endpoint_time = *time;
        self.prepare(
            kernel,
            problem,
            previous_state,
            previous_time,
            state,
            endpoint_time,
            stats,
        )?;
        let segment = BorrowedHermiteSegment::new(
            previous_time,
            endpoint_time,
            previous_state,
            &self.endpoint_state,
            &self.start_derivative,
            &self.endpoint_derivative,
        )
        .map_err(|_| SolveError::NonFiniteDerivative)?;
        let mut interpolate = |sample_time: f64, output: &mut [f64]| {
            crate::solution::DenseSegment::interpolate(&segment, sample_time, output)
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
    pub(super) fn record(
        &self,
        previous_state: &[f64],
        previous_time: f64,
        attempted_time: f64,
        state: &[f64],
        time: f64,
        final_time: bool,
        recorder: &mut TrajectoryRecorder<'_>,
    ) -> Result<(), SolveError> {
        debug_assert!(self.prepared);
        let segment = BorrowedHermiteSegment::new(
            previous_time,
            attempted_time,
            previous_state,
            &self.endpoint_state,
            &self.start_derivative,
            &self.endpoint_derivative,
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
            recorder.retain_hermite_segment(
                HermiteSegment::new_bounded(
                    previous_time,
                    attempted_time,
                    time,
                    previous_state.to_vec(),
                    self.endpoint_state.clone(),
                    self.start_derivative.clone(),
                    self.endpoint_derivative.clone(),
                )
                .map_err(|_| SolveError::NonFiniteDerivative)?,
            );
        }
        Ok(())
    }

    pub(super) fn accepted(&mut self, callback_applied: bool) {
        if callback_applied || !self.prepared {
            self.start_derivative_valid = false;
        } else {
            std::mem::swap(&mut self.start_derivative, &mut self.endpoint_derivative);
            self.start_derivative_valid = true;
        }
        self.prepared = false;
    }
}
