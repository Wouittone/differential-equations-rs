use std::marker::PhantomData;

use super::super::tableaux::ROSENBROCK23_32_TABLEAU;
use super::methods::ExtendedRosenbrockMethod;
use super::steps::{estimate_initial_step, evaluate};
use super::workspace::Workspace;
use crate::callback::CallbackOutcome;
use crate::integrator::{ControllerConfig, KernelCapabilities, StepEstimate, StepKernel};
use crate::solution::{
    BorrowedHermiteSegment, BorrowedRungeKuttaSegment, BorrowedStiffSegment, HermiteSegment,
    RungeKuttaSegment, StiffSegment, TrajectoryRecorder,
};
use crate::tableau::load_tableau;
use crate::{OdeProblem, SolveError, SolveOptions, SolverStats};

const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 6.0;

pub(crate) struct ExtendedRosenbrockKernel<M> {
    workspace: Workspace,
    dense_endpoint_prepared: bool,
    method: PhantomData<M>,
}

impl<M: ExtendedRosenbrockMethod> ExtendedRosenbrockKernel<M> {
    pub(crate) fn new(dimension: usize) -> Result<Self, SolveError> {
        let tableau = M::load_resource()?;
        let pair_tableau = M::SPECIAL_DENSE
            .then(|| load_tableau(&ROSENBROCK23_32_TABLEAU))
            .transpose()
            .map_err(|_| SolveError::InvalidTableau)?;
        if let Some(tableau) = tableau {
            if tableau.kind() != M::RESOURCE_KIND || (M::ADAPTIVE && tableau.btilde().is_none()) {
                return Err(SolveError::InvalidTableau);
            }
        } else if !M::SPECIAL_DENSE {
            return Err(SolveError::InvalidTableau);
        }
        Ok(Self {
            workspace: Workspace::new(dimension, tableau, pair_tableau),
            dense_endpoint_prepared: false,
            method: PhantomData,
        })
    }

    fn dense_order(&self) -> usize {
        self.workspace
            .tableau
            .map_or(0, |tableau| tableau.h().len())
    }

    /// Returns a conservative infinity-norm estimate of the current Jacobian.
    ///
    /// Rosenbrock attempts already form this matrix for their linear solves,
    /// so automatic composites can reuse it without another right-hand-side
    /// evaluation or Jacobian allocation.
    pub(crate) fn stiffness_estimate(&self) -> Option<f64> {
        let dimension = self.workspace.current_derivative.len();
        if dimension == 0 || self.workspace.jacobian.len() != dimension * dimension {
            return None;
        }

        let mut estimate = 0.0_f64;
        for row in self.workspace.jacobian.chunks_exact(dimension) {
            let row_sum = row.iter().map(|value| value.abs()).sum::<f64>();
            if !row_sum.is_finite() {
                return Some(row_sum);
            }
            estimate = estimate.max(row_sum);
        }
        Some(estimate)
    }

    fn prepare_stiff_corrections(&mut self)
    where
        M: ExtendedRosenbrockMethod,
    {
        let Some(tableau) = self.workspace.tableau else {
            return;
        };
        let dimension = self.workspace.current_derivative.len();
        for (row, coefficients) in tableau.h().iter().enumerate() {
            for component in 0..dimension {
                let mut correction = 0.0;
                for (stage, coefficient) in coefficients.iter().enumerate() {
                    correction +=
                        coefficient * self.workspace.stages[stage * dimension + component];
                }
                self.workspace.dense_corrections[row * dimension + component] = correction;
            }
        }
    }
}

impl<F, P, M> StepKernel<F, P> for ExtendedRosenbrockKernel<M>
where
    F: crate::OdeFunction<P>,
    M: ExtendedRosenbrockMethod,
{
    fn has_custom_dense_output(&self) -> bool {
        true
    }

    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::with_controller(
            M::ADAPTIVE,
            ControllerConfig::proportional(
                M::ERROR_ORDER,
                SAFETY,
                MIN_FACTOR,
                MAX_FACTOR,
                MIN_FACTOR,
            ),
        )
    }

    fn initialize(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        evaluate(
            problem,
            &mut self.workspace.current_derivative,
            state,
            time,
            stats,
        )
    }

    fn estimate_initial_step(
        &mut self,
        _: &OdeProblem<F, P>,
        state: &[f64],
        _: f64,
        _: f64,
        maximum_step: f64,
        _: &mut [f64],
        options: &SolveOptions,
        _: &mut SolverStats,
    ) -> Result<f64, SolveError> {
        Ok(estimate_initial_step(
            state,
            &self.workspace.current_derivative,
            options,
            maximum_step,
        ))
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
        self.dense_endpoint_prepared = false;
        Ok(StepEstimate::new(M::perform_step(
            problem,
            state,
            time,
            step,
            candidate,
            options,
            &mut self.workspace,
            stats,
        )?))
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
        stats: &mut SolverStats,
    ) -> Result<CallbackOutcome, SolveError> {
        let attempted_time = *time;
        self.workspace.dense_endpoint_state.copy_from_slice(state);
        if M::SPECIAL_DENSE {
            let dimension = previous_state.len();
            let segment = BorrowedRungeKuttaSegment::new(
                previous_time,
                attempted_time,
                previous_state,
                &self.workspace.dense_endpoint_state,
                &self.workspace.stages[..2 * dimension],
                self.workspace
                    .pair_tableau
                    .ok_or(SolveError::InvalidTableau)?
                    .dense(),
            )
            .map_err(|_| SolveError::NonFiniteDerivative)?;
            let mut interpolate = |sample_time: f64, output: &mut [f64]| {
                crate::solution::DenseSegment::interpolate(&segment, sample_time, output)
                    .map_err(|_| SolveError::NonFiniteDerivative)
            };
            return problem.apply_step_callbacks(
                previous_state,
                previous_time,
                state,
                time,
                state_before_effect,
                event_tolerance,
                Some(&mut interpolate),
            );
        }
        if self.dense_order() > 0 {
            self.prepare_stiff_corrections();
            let dimension = previous_state.len();
            let segment = BorrowedStiffSegment::new(
                previous_time,
                attempted_time,
                previous_state,
                &self.workspace.dense_endpoint_state,
                &self.workspace.dense_corrections[..self.dense_order() * dimension],
                self.dense_order(),
            )
            .map_err(|_| SolveError::NonFiniteDerivative)?;
            let mut interpolate = |sample_time: f64, output: &mut [f64]| {
                crate::solution::DenseSegment::interpolate(&segment, sample_time, output)
                    .map_err(|_| SolveError::NonFiniteDerivative)
            };
            return problem.apply_step_callbacks(
                previous_state,
                previous_time,
                state,
                time,
                state_before_effect,
                event_tolerance,
                Some(&mut interpolate),
            );
        }
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
        evaluate(
            problem,
            &mut self.workspace.dense_endpoint_derivative,
            &self.workspace.dense_endpoint_state,
            attempted_time,
            stats,
        )?;
        self.dense_endpoint_prepared = true;
        let segment = BorrowedHermiteSegment::new(
            previous_time,
            attempted_time,
            previous_state,
            &self.workspace.dense_endpoint_state,
            &self.workspace.current_derivative,
            &self.workspace.dense_endpoint_derivative,
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

    fn record_dense_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        previous_state: &[f64],
        state: &[f64],
        previous_time: f64,
        attempted_time: f64,
        time: f64,
        final_time: bool,
        recorder: &mut TrajectoryRecorder<'_>,
        stats: &mut SolverStats,
    ) -> Result<bool, SolveError> {
        let dimension = previous_state.len();
        if M::SPECIAL_DENSE {
            let segment = BorrowedRungeKuttaSegment::new(
                previous_time,
                attempted_time,
                previous_state,
                &self.workspace.dense_endpoint_state,
                &self.workspace.stages[..2 * dimension],
                self.workspace
                    .pair_tableau
                    .ok_or(SolveError::InvalidTableau)?
                    .dense(),
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
                recorder.retain_runge_kutta_segment(
                    RungeKuttaSegment::new(
                        previous_time,
                        attempted_time,
                        time,
                        previous_state,
                        &self.workspace.dense_endpoint_state,
                        &self.workspace.stages[..2 * dimension],
                        self.workspace
                            .pair_tableau
                            .ok_or(SolveError::InvalidTableau)?
                            .dense(),
                    )
                    .map_err(|_| SolveError::NonFiniteDerivative)?,
                );
            }
            return Ok(true);
        }
        if self.dense_order() > 0 {
            self.prepare_stiff_corrections();
            let corrections = &self.workspace.dense_corrections[..self.dense_order() * dimension];
            let segment = BorrowedStiffSegment::new(
                previous_time,
                attempted_time,
                previous_state,
                &self.workspace.dense_endpoint_state,
                corrections,
                self.dense_order(),
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
                recorder.retain_stiff_segment(
                    StiffSegment::new(
                        previous_time,
                        attempted_time,
                        time,
                        previous_state,
                        &self.workspace.dense_endpoint_state,
                        corrections,
                        self.dense_order(),
                    )
                    .map_err(|_| SolveError::NonFiniteDerivative)?,
                );
            }
            return Ok(true);
        }
        if !self.dense_endpoint_prepared {
            evaluate(
                problem,
                &mut self.workspace.dense_endpoint_derivative,
                &self.workspace.dense_endpoint_state,
                attempted_time,
                stats,
            )?;
            self.dense_endpoint_prepared = true;
        }
        let segment = BorrowedHermiteSegment::new(
            previous_time,
            attempted_time,
            previous_state,
            &self.workspace.dense_endpoint_state,
            &self.workspace.current_derivative,
            &self.workspace.dense_endpoint_derivative,
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
                    self.workspace.dense_endpoint_state.clone(),
                    self.workspace.current_derivative.clone(),
                    self.workspace.dense_endpoint_derivative.clone(),
                )
                .map_err(|_| SolveError::NonFiniteDerivative)?,
            );
        }
        Ok(true)
    }

    fn accept_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        _: &[f64],
        state: &[f64],
        time: f64,
        _: f64,
        callback_applied: bool,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        self.workspace.differentiation_valid = false;
        if self.dense_endpoint_prepared && !callback_applied {
            std::mem::swap(
                &mut self.workspace.current_derivative,
                &mut self.workspace.dense_endpoint_derivative,
            );
            self.dense_endpoint_prepared = false;
            return Ok(());
        }
        self.dense_endpoint_prepared = false;
        evaluate(
            problem,
            &mut self.workspace.current_derivative,
            state,
            time,
            stats,
        )
    }

    fn reject_step(&mut self) {}
}
