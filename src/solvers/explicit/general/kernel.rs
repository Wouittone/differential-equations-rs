use crate::callback::CallbackOutcome;
use crate::integrator::{
    KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::solution::{
    BorrowedHermiteSegment, BorrowedRungeKuttaSegment, HermiteSegment, RungeKuttaSegment,
    TrajectoryRecorder, interpolate_runge_kutta,
};
use crate::tableau::{RungeKuttaKind, RungeKuttaTableau};
use crate::{OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

use super::helpers::{
    endpoint_stage_stiffness_estimate, ensure_finite, error_norm, estimate_initial_step, evaluate,
    perform_lazy_dense_stages, perform_step,
};
use super::tableau_access::{ResourceTableau, TableauAccess};
use super::workspace::Workspace;

pub(super) fn integrate_resource<F, P>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
    tableau: &'static RungeKuttaTableau,
) -> Result<Solution, SolveError>
where
    F: crate::OdeFunction<P>,
{
    if tableau.kind() != RungeKuttaKind::Explicit {
        return Err(SolveError::InvalidTableau);
    }
    drive_integration(
        problem,
        options,
        ExplicitKernel::new(ResourceTableau(tableau), problem.initial_state().len()),
    )
}

pub(crate) struct ExplicitKernel<T> {
    tableau: T,
    workspace: Workspace,
    stage_zero_is_current: bool,
    dense_endpoint_state: Vec<f64>,
    dense_endpoint_prepared: bool,
    dense_stages_prepared: bool,
    stiffness_estimate: Option<f64>,
}

impl<T: TableauAccess> ExplicitKernel<T> {
    pub(crate) fn new(tableau: T, dimension: usize) -> Self {
        Self::with_stiffness_detection(tableau, dimension, false)
    }

    /// Builds the explicit branch of an automatic composite.
    ///
    /// Only automatic composites retain the penultimate endpoint-stage state;
    /// ordinary explicit solves therefore pay no storage or diagnostic cost.
    pub(crate) fn new_for_automatic(tableau: T, dimension: usize) -> Self {
        Self::with_stiffness_detection(tableau, dimension, true)
    }

    fn with_stiffness_detection(tableau: T, dimension: usize, enabled: bool) -> Self {
        Self {
            tableau,
            workspace: Workspace::new(
                tableau.weights().len() + tableau.lazy_stage_count(),
                dimension,
                enabled,
            ),
            stage_zero_is_current: false,
            dense_endpoint_state: vec![0.0; dimension],
            dense_endpoint_prepared: false,
            dense_stages_prepared: false,
            stiffness_estimate: None,
        }
    }

    /// Returns the endpoint-stage secant estimate from the last attempted
    /// step, when this kernel was created for an automatic composite.
    pub(crate) fn stiffness_estimate(&self) -> Option<f64> {
        self.stiffness_estimate
    }
}

impl<F, P, T> StepKernel<F, P> for ExplicitKernel<T>
where
    F: crate::OdeFunction<P>,
    T: TableauAccess,
{
    fn has_custom_dense_output(&self) -> bool {
        true
    }

    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::new(self.tableau.error_weights().is_some(), self.tableau.order())
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
            &mut self.workspace.stages[..self.workspace.dimension],
            state,
            time,
            stats,
        )?;
        ensure_finite(&self.workspace.stages[..self.workspace.dimension])?;
        self.stage_zero_is_current = true;
        Ok(())
    }

    fn estimate_initial_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        direction: f64,
        maximum_step: f64,
        candidate: &mut [f64],
        options: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<f64, SolveError> {
        estimate_initial_step(
            problem,
            options,
            (state, candidate),
            (time, direction, maximum_step),
            self.tableau.order(),
            &mut self.workspace,
            stats,
        )
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
        if !self.stage_zero_is_current {
            evaluate(
                problem,
                &mut self.workspace.stages[..self.workspace.dimension],
                state,
                time,
                stats,
            )?;
            ensure_finite(&self.workspace.stages[..self.workspace.dimension])?;
        }
        perform_step(
            problem,
            state,
            time,
            step,
            candidate,
            &mut self.workspace,
            stats,
            self.tableau,
        )?;
        self.stiffness_estimate =
            endpoint_stage_stiffness_estimate(&self.workspace, self.tableau.weights().len());
        self.dense_stages_prepared = false;
        ensure_finite(candidate)?;
        let error = if options.adaptive {
            let error_weights = self
                .tableau
                .error_weights()
                .ok_or(SolveError::AdaptiveStepUnsupported)?;
            let primary_error = error_norm(
                &self.workspace.stages,
                self.workspace.dimension,
                (state, candidate),
                step,
                options,
                error_weights,
                &mut self.workspace.temporary,
            );
            self.tableau
                .second_error_weights()
                .map_or(primary_error, |weights| {
                    primary_error.max(error_norm(
                        &self.workspace.stages,
                        self.workspace.dimension,
                        (state, candidate),
                        step,
                        options,
                        weights,
                        &mut self.workspace.temporary,
                    ))
                })
        } else {
            0.0
        };
        Ok(StepEstimate::new(error))
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
        let Some(coefficients) = self.tableau.dense_coefficients() else {
            if !problem.has_continuous_callbacks() {
                self.dense_endpoint_prepared = false;
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
            self.dense_endpoint_state.copy_from_slice(state);
            evaluate(problem, &mut self.workspace.temporary, state, *time, stats)?;
            ensure_finite(&self.workspace.temporary)?;
            self.dense_endpoint_prepared = true;
            let attempted_time = *time;
            let segment = BorrowedHermiteSegment::new(
                previous_time,
                attempted_time,
                previous_state,
                &self.dense_endpoint_state,
                self.workspace.stage(0),
                &self.workspace.temporary,
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
        };
        self.dense_endpoint_prepared = false;
        let attempted_time = *time;
        if self.tableau.lazy_stage_count() != 0 && problem.has_continuous_callbacks() {
            perform_lazy_dense_stages(
                problem,
                previous_state,
                previous_time,
                attempted_time - previous_time,
                &mut self.workspace,
                stats,
                self.tableau,
            )?;
            self.dense_stages_prepared = true;
        }
        let stages = &self.workspace.stages;
        let mut interpolate = |sample_time: f64, output: &mut [f64]| {
            interpolate_runge_kutta(
                previous_time,
                attempted_time,
                previous_state,
                stages,
                coefficients,
                sample_time,
                output,
            )
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
        if let Some(coefficients) = self.tableau.dense_coefficients() {
            stats.dense_output_evaluations += 1;
            if !self.dense_stages_prepared && self.tableau.lazy_stage_count() != 0 {
                perform_lazy_dense_stages(
                    problem,
                    previous_state,
                    previous_time,
                    attempted_time - previous_time,
                    &mut self.workspace,
                    stats,
                    self.tableau,
                )?;
            }
            let segment = BorrowedRungeKuttaSegment::new(
                previous_time,
                attempted_time,
                previous_state,
                state,
                &self.workspace.stages,
                coefficients,
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
                let segment = RungeKuttaSegment::new(
                    previous_time,
                    attempted_time,
                    time,
                    previous_state,
                    state,
                    &self.workspace.stages,
                    coefficients,
                )
                .map_err(|_| SolveError::NonFiniteDerivative)?;
                recorder.retain_runge_kutta_segment(segment);
            }
            self.dense_stages_prepared = false;
        } else {
            if !recorder.needs_dense_sampling() && !recorder.retains_dense_output() {
                self.dense_endpoint_prepared = false;
                return Ok(false);
            }
            stats.dense_output_evaluations += 1;
            if !self.dense_endpoint_prepared {
                self.dense_endpoint_state.copy_from_slice(state);
                evaluate(
                    problem,
                    &mut self.workspace.temporary,
                    state,
                    attempted_time,
                    stats,
                )?;
                ensure_finite(&self.workspace.temporary)?;
            }
            let segment = BorrowedHermiteSegment::new(
                previous_time,
                attempted_time,
                previous_state,
                &self.dense_endpoint_state,
                self.workspace.stage(0),
                &self.workspace.temporary,
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
                    self.dense_endpoint_state.clone(),
                    self.workspace.stage(0).to_vec(),
                    self.workspace.temporary.clone(),
                )
                .map_err(|_| SolveError::NonFiniteDerivative)?;
                recorder.retain_hermite_segment(segment);
            }
            self.dense_endpoint_prepared = false;
        }
        Ok(true)
    }

    fn accept_step(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: &[f64],
        _: f64,
        _: f64,
        callback_applied: bool,
        _: &mut SolverStats,
    ) -> Result<(), SolveError> {
        if self.tableau.fsal() && !callback_applied {
            self.workspace
                .swap_stages(0, self.tableau.weights().len() - 1);
            self.stage_zero_is_current = true;
        } else {
            self.stage_zero_is_current = false;
        }
        Ok(())
    }

    fn reject_step(&mut self) {
        self.stage_zero_is_current = true;
        self.dense_stages_prepared = false;
    }
}
