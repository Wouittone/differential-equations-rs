use crate::callback::CallbackOutcome;
use crate::integrator::{
    KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::solution::{
    BorrowedHermiteSegment, BorrowedRungeKuttaSegment, HermiteSegment, RungeKuttaCoefficients,
    RungeKuttaSegment, TrajectoryRecorder, interpolate_runge_kutta,
};
use crate::tableau::{LazyTableau, RungeKuttaKind, RungeKuttaTableau, load_tableau};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};
use std::marker::PhantomData;

// Compatibility reexports for the historical `explicit::general` façade.
/// Coefficients and method properties for an explicit Runge–Kutta method.
///
/// `COEFFICIENTS[i]` is the strictly lower-triangular row for stage `i`, so it must contain exactly `i` entries.
/// All other coefficient arrays must contain one entry per stage.
/// [`ExplicitRungeKutta`] validates these invariants before solving.
pub trait ButcherTableau {
    /// Stage nodes `c` in evaluation order.
    const NODES: &'static [f64];
    /// Strictly lower-triangular stage rows `A`.
    const COEFFICIENTS: &'static [&'static [f64]];
    /// Primary solution weights `b`.
    const WEIGHTS: &'static [f64];
    /// Optional embedded or direct stage-combination error weights.
    const ERROR_WEIGHTS: Option<&'static [f64]>;
    /// A second embedded error estimator, combined with the first by taking
    /// the larger scaled norm. Most methods use only [`Self::ERROR_WEIGHTS`].
    const SECOND_ERROR_WEIGHTS: Option<&'static [f64]> = None;
    /// Classical convergence order of the primary method.
    const ORDER: usize;
    /// Whether the method has the first-same-as-last stage property.
    const FSAL: bool;
    /// Optional method-specific continuous-extension coefficients.
    ///
    /// Each row corresponds to one RK stage and stores `r0, r1, ...` for the
    /// stage weight `theta * (r0 + r1*theta + ...)`.
    const DENSE_COEFFICIENTS: Option<&'static [&'static [f64]]> = None;
    /// Optional stages evaluated lazily only when the continuous extension is
    /// requested by saving, root localization, or retained dense output.
    const LAZY_DENSE_STAGES: &'static [LazyDenseStage] = &[];
}

/// One sparse explicit stage used only by a method-specific continuous
/// extension.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LazyDenseStage {
    node: f64,
    coefficients: &'static [(usize, f64)],
}

impl LazyDenseStage {
    /// Creates a lazy dense stage at `node` from zero-based prior-stage
    /// coefficient pairs.
    pub const fn new(node: f64, coefficients: &'static [(usize, f64)]) -> Self {
        Self { node, coefficients }
    }

    /// Returns the stage node within the step.
    pub const fn node(self) -> f64 {
        self.node
    }

    /// Returns `(prior_stage_index, weight)` pairs for this sparse stage.
    pub const fn coefficients(self) -> &'static [(usize, f64)] {
        self.coefficients
    }
}

/// The centralized explicit Runge–Kutta solver for a [`ButcherTableau`].
///
/// Named algorithms such as [`Rk4`] are lightweight facades over
/// this type. It can also be instantiated with a user-defined tableau marker.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExplicitRungeKutta<T> {
    marker: PhantomData<fn() -> T>,
}

/// Upstream-compatible name for a user-supplied explicit Runge--Kutta
/// tableau marker.
pub type ExplicitRK<T> = ExplicitRungeKutta<T>;

/// A generic explicit Runge--Kutta solver backed by a lazy text resource.
///
/// Use [`LazyTableau`] with `include_str!` to define downstream methods
/// without a procedural macro or source-level coefficient constants.
#[derive(Clone, Copy)]
pub struct ResourceExplicitRungeKutta {
    resource: &'static LazyTableau,
}

impl ResourceExplicitRungeKutta {
    /// Creates a solver referring to a lazily parsed tableau resource.
    pub const fn new(resource: &'static LazyTableau) -> Self {
        Self { resource }
    }

    /// Loads and returns the method tableau.
    pub fn tableau(self) -> Result<&'static RungeKuttaTableau, crate::tableau::TableauError> {
        load_tableau(self.resource)
    }
}

impl std::fmt::Debug for ResourceExplicitRungeKutta {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("ResourceExplicitRungeKutta { .. }")
    }
}

impl OdeAlgorithm for ResourceExplicitRungeKutta {
    fn solve_validated<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        let tableau = load_tableau(self.resource).map_err(|_| SolveError::InvalidTableau)?;
        integrate_resource(problem, options, tableau)
    }
}

impl<T> ExplicitRungeKutta<T> {
    /// Creates the zero-sized generic solver for tableau marker `T`.
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

impl<T> Default for ExplicitRungeKutta<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> OdeAlgorithm for ExplicitRungeKutta<T>
where
    T: ButcherTableau,
{
    fn solve_validated<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        integrate::<F, P, T>(problem, options)
    }
}

crate::define_explicit_rk_from_file!(pub Rkm, "tableaux/explicit/rkm.json", crate = crate);
crate::define_explicit_rk_from_file!(pub Rko65, "tableaux/explicit/rko65.json", crate = crate);
crate::define_explicit_rk_from_file!(pub Msrk5, "tableaux/explicit/msrk5.json", crate = crate);
crate::define_explicit_rk_from_file!(pub Msrk6, "tableaux/explicit/msrk6.json", crate = crate);
crate::define_explicit_rk_from_file!(pub Stepanov5, "tableaux/explicit/stepanov5.json", crate = crate);
crate::define_explicit_rk_from_file!(pub Sir54, "tableaux/explicit/sir54.json", crate = crate);
crate::define_explicit_rk_from_file!(pub Ralston4, "tableaux/explicit/ralston4.json", crate = crate);
crate::define_explicit_rk_from_file!(pub Alshina3, "tableaux/explicit/alshina3.json", crate = crate);
crate::define_explicit_rk_from_file!(pub Alshina6, "tableaux/explicit/alshina6.json", crate = crate);
crate::define_explicit_rk_from_file!(pub Bs3, "tableaux/explicit/bs3.json", crate = crate);
crate::define_explicit_rk_from_file!(pub Dp5, "tableaux/explicit/dp5.json", crate = crate);
crate::define_explicit_rk_from_file!(pub OwrenZen3, "tableaux/explicit/owren_zen3.json", crate = crate);
crate::define_explicit_rk_from_file!(pub OwrenZen4, "tableaux/explicit/owren_zen4.json", crate = crate);
crate::define_explicit_rk_from_file!(pub OwrenZen5, "tableaux/explicit/owren_zen5.json", crate = crate);
crate::define_explicit_rk_from_file!(pub Bs5, "tableaux/explicit/bs5.json", crate = crate);
crate::define_explicit_rk_from_file!(pub SspRk22, "tableaux/explicit/ssp_rk22.json", crate = crate);
crate::define_explicit_rk_from_file!(pub SspRk33, "tableaux/explicit/ssp_rk33.json", crate = crate);
crate::define_explicit_rk_from_file!(pub SspRk43, "tableaux/explicit/ssp_rk43.json", crate = crate);
crate::define_explicit_rk_from_file!(pub Psrk3p5q4, "tableaux/explicit/psrk3p5q4.json", crate = crate);
crate::define_explicit_rk_from_file!(pub Psrk3p6q5, "tableaux/explicit/psrk3p6q5.json", crate = crate);
crate::define_explicit_rk_from_file!(pub Psrk4p7q6, "tableaux/explicit/psrk4p7q6.json", crate = crate);
crate::define_explicit_rk_from_file!(pub Euler, "tableaux/explicit/euler.json", crate = crate);
crate::define_explicit_rk_from_file!(pub Midpoint, "tableaux/explicit/midpoint.json", crate = crate);
crate::define_explicit_rk_from_file!(pub Heun, "tableaux/explicit/heun.json", crate = crate);
crate::define_explicit_rk_from_file!(pub Ralston, "tableaux/explicit/ralston.json", crate = crate);
crate::define_explicit_rk_from_file!(pub Rk4, "tableaux/explicit/rk4.json", crate = crate);
crate::define_explicit_rk_from_file!(pub Alshina2, "tableaux/explicit/alshina2.json", crate = crate);

struct Workspace {
    // Flat stage-major storage: every stage is one contiguous component array.
    // The other work vectors remain separate arrays rather than per-component
    // structs, keeping the hot saxpy-style loops friendly to SIMD.
    stages: Vec<f64>,
    dimension: usize,
    temporary: Vec<f64>,
}

impl Workspace {
    fn new(stage_count: usize, dimension: usize) -> Self {
        Self {
            stages: vec![0.0; stage_count * dimension],
            dimension,
            temporary: vec![0.0; dimension],
        }
    }

    fn stage(&self, index: usize) -> &[f64] {
        let start = index * self.dimension;
        &self.stages[start..start + self.dimension]
    }

    fn swap_stages(&mut self, left: usize, right: usize) {
        let left_start = left * self.dimension;
        let right_start = right * self.dimension;
        for offset in 0..self.dimension {
            self.stages.swap(left_start + offset, right_start + offset);
        }
    }
}

fn validate_tableau<T: ButcherTableau>() -> Result<(), SolveError> {
    let stage_count = T::WEIGHTS.len();
    let dense_stage_count = stage_count + T::LAZY_DENSE_STAGES.len();
    let structurally_valid = stage_count > 0
        && T::ORDER > 0
        && T::NODES.first() == Some(&0.0)
        && T::NODES.len() == stage_count
        && T::COEFFICIENTS.len() == stage_count
        && T::COEFFICIENTS
            .iter()
            .enumerate()
            .all(|(stage, row)| row.len() == stage)
        && T::ERROR_WEIGHTS.is_none_or(|weights| weights.len() == stage_count);
    let error_estimators_valid = T::SECOND_ERROR_WEIGHTS
        .is_none_or(|weights| T::ERROR_WEIGHTS.is_some() && weights.len() == stage_count);
    let lazy_dense_stages_valid = (T::LAZY_DENSE_STAGES.is_empty()
        || T::DENSE_COEFFICIENTS.is_some())
        && T::LAZY_DENSE_STAGES
            .iter()
            .enumerate()
            .all(|(offset, stage)| {
                stage.node.is_finite()
                    && !stage.coefficients.is_empty()
                    && stage.coefficients.iter().all(|&(index, coefficient)| {
                        index < stage_count + offset && coefficient.is_finite()
                    })
            });
    let coefficients_finite = T::NODES.iter().all(|value| value.is_finite())
        && T::WEIGHTS.iter().all(|value| value.is_finite())
        && T::COEFFICIENTS
            .iter()
            .flat_map(|row| row.iter())
            .all(|value| value.is_finite())
        && T::ERROR_WEIGHTS.is_none_or(|weights| weights.iter().all(|value| value.is_finite()));
    let second_error_estimator_finite =
        T::SECOND_ERROR_WEIGHTS.is_none_or(|weights| weights.iter().all(|value| value.is_finite()));
    let dense_coefficients_valid = T::DENSE_COEFFICIENTS.is_none_or(|rows| {
        rows.len() == dense_stage_count
            && rows.iter().enumerate().all(|(stage, row)| {
                let endpoint_weight = T::WEIGHTS.get(stage).copied().unwrap_or(0.0);
                let coefficient_scale = row.iter().map(|value| value.abs()).sum::<f64>();
                !row.is_empty()
                    && row.iter().all(|coefficient| coefficient.is_finite())
                    && (row.iter().sum::<f64>() - endpoint_weight).abs()
                        <= 1.0e-12 * (1.0 + endpoint_weight.abs())
                            + 64.0 * f64::EPSILON * coefficient_scale
            })
    });
    let fsal_valid = !T::FSAL
        || (stage_count > 0
            && T::NODES.last() == Some(&1.0)
            && T::WEIGHTS.last() == Some(&0.0)
            && T::COEFFICIENTS
                .last()
                .is_some_and(|last_row| *last_row == &T::WEIGHTS[..stage_count - 1]));

    (structurally_valid
        && error_estimators_valid
        && lazy_dense_stages_valid
        && coefficients_finite
        && second_error_estimator_finite
        && dense_coefficients_valid
        && fsal_valid)
        .then_some(())
        .ok_or(SolveError::InvalidTableau)
}

fn integrate<F, P, T>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
) -> Result<Solution, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
    T: ButcherTableau,
{
    validate_tableau::<T>()?;
    drive_integration(
        problem,
        options,
        ExplicitKernel::new(
            StaticTableau::<T>(PhantomData),
            problem.initial_state().len(),
        ),
    )
}

fn integrate_resource<F, P>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
    tableau: &'static RungeKuttaTableau,
) -> Result<Solution, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
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

trait TableauAccess: Copy {
    fn order(self) -> usize;
    fn fsal(self) -> bool;
    fn nodes(self) -> &'static [f64];
    fn weights(self) -> &'static [f64];
    fn stage_row(self, stage: usize) -> &'static [f64];
    fn error_weights(self) -> Option<&'static [f64]>;
    fn second_error_weights(self) -> Option<&'static [f64]>;
    fn dense_coefficients(self) -> Option<RungeKuttaCoefficients>;
    fn lazy_stage_count(self) -> usize;
    fn lazy_stage(self, stage: usize) -> (f64, &'static [(usize, f64)]);
}

struct StaticTableau<T>(PhantomData<fn() -> T>);

impl<T> Clone for StaticTableau<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for StaticTableau<T> {}

impl<T: ButcherTableau> TableauAccess for StaticTableau<T> {
    fn order(self) -> usize {
        T::ORDER
    }
    fn fsal(self) -> bool {
        T::FSAL
    }
    fn nodes(self) -> &'static [f64] {
        T::NODES
    }
    fn weights(self) -> &'static [f64] {
        T::WEIGHTS
    }
    fn stage_row(self, stage: usize) -> &'static [f64] {
        T::COEFFICIENTS[stage]
    }
    fn error_weights(self) -> Option<&'static [f64]> {
        T::ERROR_WEIGHTS
    }
    fn second_error_weights(self) -> Option<&'static [f64]> {
        T::SECOND_ERROR_WEIGHTS
    }
    fn dense_coefficients(self) -> Option<RungeKuttaCoefficients> {
        T::DENSE_COEFFICIENTS.map(RungeKuttaCoefficients::Static)
    }
    fn lazy_stage_count(self) -> usize {
        T::LAZY_DENSE_STAGES.len()
    }
    fn lazy_stage(self, stage: usize) -> (f64, &'static [(usize, f64)]) {
        let stage = T::LAZY_DENSE_STAGES[stage];
        (stage.node(), stage.coefficients())
    }
}

#[derive(Clone, Copy)]
struct ResourceTableau(&'static RungeKuttaTableau);

impl TableauAccess for ResourceTableau {
    fn order(self) -> usize {
        self.0.order()
    }
    fn fsal(self) -> bool {
        self.0.fsal()
    }
    fn nodes(self) -> &'static [f64] {
        self.0.c()
    }
    fn weights(self) -> &'static [f64] {
        self.0.b()
    }
    fn stage_row(self, stage: usize) -> &'static [f64] {
        self.0.stage_row(stage)
    }
    fn error_weights(self) -> Option<&'static [f64]> {
        self.0.error()
    }
    fn second_error_weights(self) -> Option<&'static [f64]> {
        self.0.second_error()
    }
    fn dense_coefficients(self) -> Option<RungeKuttaCoefficients> {
        self.0.dense().map(RungeKuttaCoefficients::Resource)
    }
    fn lazy_stage_count(self) -> usize {
        self.0.lazy_dense_stages().len()
    }
    fn lazy_stage(self, stage: usize) -> (f64, &'static [(usize, f64)]) {
        let stage = &self.0.lazy_dense_stages()[stage];
        (stage.node(), stage.coefficients())
    }
}

struct ExplicitKernel<T> {
    tableau: T,
    workspace: Workspace,
    stage_zero_is_current: bool,
    dense_endpoint_state: Vec<f64>,
    dense_endpoint_prepared: bool,
    dense_stages_prepared: bool,
}

impl<T: TableauAccess> ExplicitKernel<T> {
    fn new(tableau: T, dimension: usize) -> Self {
        Self {
            tableau,
            workspace: Workspace::new(
                tableau.weights().len() + tableau.lazy_stage_count(),
                dimension,
            ),
            stage_zero_is_current: false,
            dense_endpoint_state: vec![0.0; dimension],
            dense_endpoint_prepared: false,
            dense_stages_prepared: false,
        }
    }
}

impl<F, P, T> StepKernel<F, P> for ExplicitKernel<T>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
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
        );
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
            );
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
        );
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
            evaluate(problem, &mut self.workspace.temporary, state, *time, stats);
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
            if !self.dense_endpoint_prepared {
                self.dense_endpoint_state.copy_from_slice(state);
                evaluate(
                    problem,
                    &mut self.workspace.temporary,
                    state,
                    attempted_time,
                    stats,
                );
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

fn evaluate<F, P>(
    problem: &OdeProblem<F, P>,
    derivative: &mut [f64],
    state: &[f64],
    time: f64,
    stats: &mut SolverStats,
) where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    (problem.rhs)(derivative, state, problem.parameters(), time);
    stats.rhs_evaluations += 1;
}

fn ensure_finite(values: &[f64]) -> Result<(), SolveError> {
    values
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteDerivative)
}

fn estimate_initial_step<F, P>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
    states: (&[f64], &mut [f64]),
    integration: (f64, f64, f64),
    order: usize,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
) -> Result<f64, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    let (state, scratch) = states;
    let (time, direction, maximum_step) = integration;
    let dimension = state.len() as f64;
    let mut state_norm = 0.0;
    let mut derivative_norm = 0.0;
    for (value, derivative) in state.iter().zip(workspace.stage(0)) {
        let scale = options.absolute_tolerance + options.relative_tolerance * value.abs();
        state_norm += (value / scale).powi(2);
        derivative_norm += (derivative / scale).powi(2);
    }
    state_norm = (state_norm / dimension).sqrt();
    derivative_norm = (derivative_norm / dimension).sqrt();
    let trial_step = if state_norm < 1.0e-5 || derivative_norm < 1.0e-5 {
        1.0e-6
    } else {
        0.01 * state_norm / derivative_norm
    }
    .min(maximum_step);

    for ((trial, value), derivative) in workspace
        .temporary
        .iter_mut()
        .zip(state)
        .zip(&workspace.stages[..workspace.dimension])
    {
        *trial = value + direction * trial_step * derivative;
    }
    evaluate(
        problem,
        scratch,
        &workspace.temporary,
        time + direction * trial_step,
        stats,
    );
    ensure_finite(scratch)?;

    let mut curvature_norm = 0.0;
    for ((next, initial), value) in scratch
        .iter()
        .zip(&workspace.stages[..workspace.dimension])
        .zip(state)
    {
        let scale = options.absolute_tolerance + options.relative_tolerance * value.abs();
        curvature_norm += ((next - initial) / scale).powi(2);
    }
    curvature_norm = (curvature_norm / dimension).sqrt() / trial_step;
    let largest = derivative_norm.max(curvature_norm);
    let accuracy_step = if largest <= 1.0e-15 {
        (trial_step * 1.0e-3).max(1.0e-6)
    } else {
        (0.01 / largest).powf(1.0 / order as f64)
    };
    Ok((100.0 * trial_step).min(accuracy_step).min(maximum_step))
}

#[allow(clippy::too_many_arguments)]
fn perform_step<F, P, T>(
    problem: &OdeProblem<F, P>,
    state: &[f64],
    time: f64,
    step: f64,
    candidate: &mut [f64],
    workspace: &mut Workspace,
    stats: &mut SolverStats,
    tableau: T,
) where
    F: Fn(&mut [f64], &[f64], &P, f64),
    T: TableauAccess,
{
    let stage_count = tableau.weights().len();
    for stage_index in 1..stage_count {
        combine(
            &mut workspace.temporary,
            state,
            step,
            &workspace.stages,
            workspace.dimension,
            stage_index,
            tableau.stage_row(stage_index),
        );
        let start = stage_index * workspace.dimension;
        evaluate(
            problem,
            &mut workspace.stages[start..start + workspace.dimension],
            &workspace.temporary,
            time + tableau.nodes()[stage_index] * step,
            stats,
        );
    }
    combine(
        candidate,
        state,
        step,
        &workspace.stages,
        workspace.dimension,
        stage_count,
        tableau.weights(),
    );
}

fn perform_lazy_dense_stages<F, P, T>(
    problem: &OdeProblem<F, P>,
    state: &[f64],
    time: f64,
    step: f64,
    workspace: &mut Workspace,
    stats: &mut SolverStats,
    tableau: T,
) -> Result<(), SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
    T: TableauAccess,
{
    let base_stage_count = tableau.weights().len();
    for offset in 0..tableau.lazy_stage_count() {
        let (node, coefficients) = tableau.lazy_stage(offset);
        workspace.temporary.copy_from_slice(state);
        for &(source, coefficient) in coefficients {
            let source_start = source * workspace.dimension;
            for (value, derivative) in workspace
                .temporary
                .iter_mut()
                .zip(&workspace.stages[source_start..source_start + workspace.dimension])
            {
                *value += step * coefficient * derivative;
            }
        }
        let target = base_stage_count + offset;
        let target_start = target * workspace.dimension;
        evaluate(
            problem,
            &mut workspace.stages[target_start..target_start + workspace.dimension],
            &workspace.temporary,
            time + node * step,
            stats,
        );
        ensure_finite(&workspace.stages[target_start..target_start + workspace.dimension])?;
    }
    Ok(())
}

fn combine(
    output: &mut [f64],
    state: &[f64],
    step: f64,
    stages: &[f64],
    dimension: usize,
    stage_count: usize,
    weights: &[f64],
) {
    output.fill(0.0);
    for (stage_index, weight) in weights.iter().take(stage_count).enumerate() {
        let start = stage_index * dimension;
        let stage = &stages[start..start + dimension];
        for (increment, stage_value) in output.iter_mut().zip(stage) {
            *increment += weight * stage_value;
        }
    }
    for (output_value, state_value) in output.iter_mut().zip(state) {
        *output_value = state_value + step * *output_value;
    }
}

fn error_norm(
    stages: &[f64],
    dimension: usize,
    states: (&[f64], &[f64]),
    step: f64,
    options: &SolveOptions,
    error_weights: &[f64],
    error_buffer: &mut [f64],
) -> f64 {
    let (state, candidate) = states;
    error_buffer.fill(0.0);
    for (stage_index, weight) in error_weights.iter().enumerate() {
        let start = stage_index * dimension;
        let stage = &stages[start..start + dimension];
        for (error, stage_value) in error_buffer.iter_mut().zip(stage) {
            *error += weight * stage_value;
        }
    }
    let mut squared_norm = 0.0;
    for ((error, state), candidate) in error_buffer.iter().zip(state).zip(candidate) {
        let error = step * error;
        let scale = options.absolute_tolerance
            + options.relative_tolerance * state.abs().max(candidate.abs());
        squared_norm += (error / scale).powi(2);
    }
    (squared_norm / state.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use std::f64::consts::E;

    use super::{
        Alshina2, Alshina3, Alshina6, Bs3, Dp5, Euler, Heun, Midpoint, Ralston, Ralston4, Rk4, Rkm,
        SspRk22, SspRk33, SspRk43,
    };
    use super::{Bs5, ButcherTableau, ExplicitRungeKutta, OwrenZen3, OwrenZen4, OwrenZen5};
    use crate::{OdeProblem, SaveMode, SolveError, SolveOptions, solve};

    type TestRhs = fn(&mut [f64], &[f64], &(), f64);

    struct CustomEuler;

    impl ButcherTableau for CustomEuler {
        const NODES: &'static [f64] = &[0.0];
        const COEFFICIENTS: &'static [&'static [f64]] = &[&[]];
        const WEIGHTS: &'static [f64] = &[1.0];
        const ERROR_WEIGHTS: Option<&'static [f64]> = None;
        const ORDER: usize = 1;
        const FSAL: bool = false;
    }

    struct MalformedTableau;

    impl ButcherTableau for MalformedTableau {
        const NODES: &'static [f64] = &[0.0, 1.0];
        const COEFFICIENTS: &'static [&'static [f64]] = &[&[]];
        const WEIGHTS: &'static [f64] = &[1.0];
        const ERROR_WEIGHTS: Option<&'static [f64]> = None;
        const ORDER: usize = 1;
        const FSAL: bool = false;
    }

    struct EmptyFsalTableau;

    impl ButcherTableau for EmptyFsalTableau {
        const NODES: &'static [f64] = &[];
        const COEFFICIENTS: &'static [&'static [f64]] = &[];
        const WEIGHTS: &'static [f64] = &[];
        const ERROR_WEIGHTS: Option<&'static [f64]> = None;
        const ORDER: usize = 1;
        const FSAL: bool = true;
    }

    struct SingleStageAdaptiveTableau;

    impl ButcherTableau for SingleStageAdaptiveTableau {
        const NODES: &'static [f64] = &[0.0];
        const COEFFICIENTS: &'static [&'static [f64]] = &[&[]];
        const WEIGHTS: &'static [f64] = &[1.0];
        const ERROR_WEIGHTS: Option<&'static [f64]> = Some(&[0.0]);
        const ORDER: usize = 1;
        const FSAL: bool = false;
    }

    struct DualEstimatorHeun;

    impl ButcherTableau for DualEstimatorHeun {
        const NODES: &'static [f64] = &[0.0, 1.0];
        const COEFFICIENTS: &'static [&'static [f64]] = &[&[], &[1.0]];
        const WEIGHTS: &'static [f64] = &[0.5, 0.5];
        const ERROR_WEIGHTS: Option<&'static [f64]> = Some(&[0.0, 0.0]);
        const SECOND_ERROR_WEIGHTS: Option<&'static [f64]> = Some(&[-0.5, 0.5]);
        const ORDER: usize = 2;
        const FSAL: bool = false;
    }

    struct MalformedSecondEstimator;

    impl ButcherTableau for MalformedSecondEstimator {
        const NODES: &'static [f64] = &[0.0];
        const COEFFICIENTS: &'static [&'static [f64]] = &[&[]];
        const WEIGHTS: &'static [f64] = &[1.0];
        const ERROR_WEIGHTS: Option<&'static [f64]> = Some(&[0.0]);
        const SECOND_ERROR_WEIGHTS: Option<&'static [f64]> = Some(&[]);
        const ORDER: usize = 1;
        const FSAL: bool = false;
    }

    struct MalformedDenseTableau;

    impl ButcherTableau for MalformedDenseTableau {
        const NODES: &'static [f64] = &[0.0];
        const COEFFICIENTS: &'static [&'static [f64]] = &[&[]];
        const WEIGHTS: &'static [f64] = &[1.0];
        const ERROR_WEIGHTS: Option<&'static [f64]> = None;
        const ORDER: usize = 1;
        const FSAL: bool = false;
        const DENSE_COEFFICIENTS: Option<&'static [&'static [f64]]> = Some(&[&[]]);
    }

    fn exponential() -> OdeProblem<TestRhs, ()> {
        fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
            du[0] = u[0];
        }

        OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
    }

    fn adaptive_options() -> SolveOptions {
        SolveOptions {
            absolute_tolerance: 1.0e-9,
            relative_tolerance: 1.0e-9,
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        }
    }

    #[test]
    fn adaptive_embedded_methods_solve_exponential_growth() {
        for endpoint in [
            solve(&exponential(), Midpoint, &adaptive_options())
                .unwrap()
                .last_state()[0],
            solve(&exponential(), Heun, &adaptive_options())
                .unwrap()
                .last_state()[0],
            solve(&exponential(), Ralston, &adaptive_options())
                .unwrap()
                .last_state()[0],
            solve(&exponential(), Bs3, &adaptive_options())
                .unwrap()
                .last_state()[0],
            solve(&exponential(), Dp5, &adaptive_options())
                .unwrap()
                .last_state()[0],
        ] {
            assert!((endpoint - E).abs() < 2.0e-7);
        }
    }

    fn fixed_endpoint<T: crate::OdeAlgorithm>(algorithm: T, step: f64) -> f64 {
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(step),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };
        solve(&exponential(), algorithm, &options)
            .unwrap()
            .last_state()[0]
    }

    fn convergence_ratio<T: crate::OdeAlgorithm + Copy>(algorithm: T, step: f64) -> f64 {
        let coarse = (fixed_endpoint(algorithm, step) - E).abs();
        let fine = (fixed_endpoint(algorithm, step / 2.0) - E).abs();
        coarse / fine
    }

    #[test]
    fn owren_zen_and_bs5_have_their_expected_orders() {
        let ratios = [
            convergence_ratio(OwrenZen3, 0.1),
            convergence_ratio(OwrenZen4, 0.1),
            convergence_ratio(OwrenZen5, 0.1),
            convergence_ratio(Bs5, 0.1),
        ];
        assert!(ratios[0] > 7.0);
        assert!(ratios[1] > 14.0);
        assert!(ratios[2] > 25.0);
        assert!(ratios[3] > 25.0);
    }

    #[test]
    fn owren_zen_and_bs5_adaptive_solvers_reach_tight_tolerance() {
        for endpoint in [
            solve(&exponential(), OwrenZen3, &adaptive_options())
                .unwrap()
                .last_state()[0],
            solve(&exponential(), OwrenZen4, &adaptive_options())
                .unwrap()
                .last_state()[0],
            solve(&exponential(), OwrenZen5, &adaptive_options())
                .unwrap()
                .last_state()[0],
            solve(&exponential(), Bs5, &adaptive_options())
                .unwrap()
                .last_state()[0],
        ] {
            assert!((endpoint - E).abs() < 2.0e-7);
        }
    }

    #[test]
    fn bs5_retains_both_upstream_error_estimators() {
        let tableau = Bs5.tableau().unwrap();
        assert!(tableau.error().is_some());
        assert!(tableau.second_error().is_some());
        assert_ne!(tableau.error(), tableau.second_error());
    }

    #[test]
    fn fixed_methods_have_expected_convergence() {
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.001),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };
        let euler_error =
            (solve(&exponential(), Euler, &options).unwrap().last_state()[0] - E).abs();
        let rk4_error = (solve(&exponential(), Rk4, &options).unwrap().last_state()[0] - E).abs();
        let rkm_error = (solve(&exponential(), Rkm, &options).unwrap().last_state()[0] - E).abs();
        let ralston4_error = (solve(&exponential(), Ralston4, &options)
            .unwrap()
            .last_state()[0]
            - E)
            .abs();
        let alshina2_error = (solve(&exponential(), Alshina2, &options)
            .unwrap()
            .last_state()[0]
            - E)
            .abs();
        let alshina3_error = (solve(&exponential(), Alshina3, &options)
            .unwrap()
            .last_state()[0]
            - E)
            .abs();
        let alshina6_error = (solve(&exponential(), Alshina6, &options)
            .unwrap()
            .last_state()[0]
            - E)
            .abs();

        assert!(euler_error < 0.002);
        assert!(rk4_error < 1.0e-12);
        assert!(rkm_error < 1.0e-12);
        assert!(ralston4_error < 1.0e-12);
        assert!(alshina2_error < 1.0e-6);
        assert!(alshina3_error < 1.0e-9);
        assert!(alshina6_error < 1.0e-12);
        assert!(convergence_ratio(Alshina6, 0.1) > 40.0);
    }

    #[test]
    fn fixed_only_methods_reject_adaptive_configuration() {
        assert_eq!(
            solve(&exponential(), Euler, &SolveOptions::default()),
            Err(SolveError::AdaptiveStepUnsupported)
        );
        assert_eq!(
            solve(&exponential(), Rk4, &SolveOptions::default()),
            Err(SolveError::AdaptiveStepUnsupported)
        );
        assert_eq!(
            solve(&exponential(), Alshina6, &SolveOptions::default()),
            Err(SolveError::AdaptiveStepUnsupported)
        );
    }

    #[test]
    fn named_solver_uses_its_validated_resource_tableau() {
        let problem = exponential();
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.01),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        let named = solve(&problem, Rk4, &options).unwrap();
        assert_eq!(Rk4.tableau().unwrap().name(), "Rk4");
        assert!((named.last_state()[0] - E).abs() < 1.0e-8);
    }

    #[test]
    fn supports_custom_tableaus_and_rejects_malformed_ones() {
        let problem = exponential();
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.01),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        let custom = solve(&problem, ExplicitRungeKutta::<CustomEuler>::new(), &options).unwrap();
        let named = solve(&problem, Euler, &options).unwrap();

        assert_eq!(custom, named);
        assert_eq!(
            solve(
                &problem,
                ExplicitRungeKutta::<MalformedTableau>::new(),
                &options,
            ),
            Err(SolveError::InvalidTableau)
        );
        assert_eq!(
            solve(
                &problem,
                ExplicitRungeKutta::<EmptyFsalTableau>::new(),
                &options,
            ),
            Err(SolveError::InvalidTableau)
        );
        assert_eq!(
            solve(
                &problem,
                ExplicitRungeKutta::<MalformedSecondEstimator>::new(),
                &adaptive_options(),
            ),
            Err(SolveError::InvalidTableau)
        );
        assert_eq!(
            solve(
                &problem,
                ExplicitRungeKutta::<MalformedDenseTableau>::new(),
                &options,
            ),
            Err(SolveError::InvalidTableau)
        );
        assert!(
            solve(
                &problem,
                ExplicitRungeKutta::<SingleStageAdaptiveTableau>::new(),
                &adaptive_options(),
            )
            .is_ok()
        );
    }

    #[test]
    fn combines_two_error_estimators_by_their_maximum_norm() {
        let options = SolveOptions {
            absolute_tolerance: 1.0e-9,
            relative_tolerance: 1.0e-9,
            initial_step: Some(1.0),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        let solution = solve(
            &exponential(),
            ExplicitRungeKutta::<DualEstimatorHeun>::new(),
            &options,
        )
        .unwrap();

        assert!(solution.stats().rejected_steps > 0);
    }

    #[test]
    fn reports_non_finite_stage_derivatives() {
        let problem = OdeProblem::new(
            |du: &mut [f64], _: &[f64], _: &(), time: f64| {
                du[0] = if time == 0.0 { 1.0 } else { f64::NAN };
            },
            vec![1.0],
            (0.0, 1.0),
            (),
        );
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(1.0),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        assert_eq!(
            solve(&problem, Rk4, &options),
            Err(SolveError::NonFiniteDerivative)
        );
    }

    #[test]
    fn ssp_methods_solve_exponential_growth() {
        let fixed = SolveOptions {
            adaptive: false,
            initial_step: Some(0.001),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };

        let endpoints = [
            solve(&exponential(), SspRk22, &fixed).unwrap().last_state()[0],
            solve(&exponential(), SspRk33, &fixed).unwrap().last_state()[0],
            solve(&exponential(), SspRk43, &adaptive_options())
                .unwrap()
                .last_state()[0],
        ];

        assert!((endpoints[0] - E).abs() < 1.0e-6);
        assert!((endpoints[1] - E).abs() < 1.0e-9);
        assert!((endpoints[2] - E).abs() < 2.0e-7);
    }
}
