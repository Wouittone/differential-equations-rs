//! Shared in-flight kernel switching for automatic solver facades.

use super::policy::AutoSwitchDetector;
use super::{AutoSwitchConfig, AutomaticBranch};
use crate::callback::CallbackOutcome;
use crate::integrator::{
    KernelCapabilities, KernelTransition, RejectionReason, StepEstimate, StepKernel,
    integrate as drive_integration,
};
use crate::solution::TrajectoryRecorder;
use crate::solvers::explicit::general::{ExplicitKernel, ResourceTableau};
use crate::solvers::rosenbrock::rosenbrock_extended::{
    ExtendedRosenbrockKernel, ExtendedRosenbrockMethod,
};
use crate::tableau::{RungeKuttaKind, RungeKuttaTableau};
use crate::{
    AutomaticPairIncompatibility, OdeProblem, Solution, SolveError, SolveOptions, SolverStats,
};

mod private {
    use super::ExtendedRosenbrockMethod;

    #[allow(private_bounds)] // Public only inside this private sealing module.
    pub trait Sealed: ExtendedRosenbrockMethod {}

    impl<T> Sealed for T where T: ExtendedRosenbrockMethod {}
}

/// A built-in stiff algorithm that can participate in an automatic pair.
///
/// This sealed marker is implemented only by stateless, `Copy` built-in
/// Rosenbrock method values. The value passed to an automatic facade selects
/// the method type; automatic pairs do not currently accept per-instance stiff
/// solver configuration. Solving returns a typed error before evaluating the
/// problem when the selected method lacks an adaptive estimator or a required
/// switch-back diagnostic.
///
/// A future public kernel extension surface can add downstream-defined
/// branches without exposing the internal integration driver as a 1.0
/// compatibility commitment.
pub trait AutomaticStiffAlgorithm: private::Sealed + Copy + 'static {}

impl<T> AutomaticStiffAlgorithm for T where T: private::Sealed + Copy + 'static {}

pub(crate) fn solve_automatic<F, P, A>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
    explicit_tableau: &'static RungeKuttaTableau,
    _stiff_algorithm: A,
    config: AutoSwitchConfig,
) -> Result<Solution, SolveError>
where
    F: crate::OdeFunction<P>,
    A: AutomaticStiffAlgorithm,
{
    debug_assert!(config.validate().is_ok());
    let kernel =
        SwitchingKernel::<A>::new(explicit_tableau, config, problem.initial_state().len())?;
    drive_integration(problem, options, kernel)
}

struct SwitchingKernel<A>
where
    A: AutomaticStiffAlgorithm,
{
    explicit_tableau: &'static RungeKuttaTableau,
    explicit: Option<ExplicitKernel<ResourceTableau>>,
    stiff: Option<ExtendedRosenbrockKernel<A>>,
    explicit_initialized: bool,
    stiff_initialized: bool,
    detector: AutoSwitchDetector,
    attempt_branch: AutomaticBranch,
    transition: KernelTransition,
    switches: usize,
    dimension: usize,
}

impl<A> SwitchingKernel<A>
where
    A: AutomaticStiffAlgorithm,
{
    fn new(
        explicit_tableau: &'static RungeKuttaTableau,
        config: AutoSwitchConfig,
        dimension: usize,
    ) -> Result<Self, SolveError> {
        if explicit_tableau.kind() != RungeKuttaKind::Explicit {
            return Err(SolveError::InvalidTableau);
        }
        if explicit_tableau.error().is_none() || !A::ADAPTIVE {
            return Err(incompatible(
                AutomaticPairIncompatibility::NonAdaptiveComponent,
            ));
        }
        if config.allow_switch_back() && !A::HAS_STIFFNESS_ESTIMATE {
            return Err(incompatible(
                AutomaticPairIncompatibility::MissingStiffnessDiagnostic,
            ));
        }
        match explicit_tableau.real_stability_radius() {
            None => return Err(SolveError::InvalidTableau),
            Some(radius) if !radius.is_finite() || radius <= 0.0 => {
                return Err(SolveError::InvalidTableau);
            }
            Some(_) => {}
        }

        let initial_branch = config.initial_branch();
        let explicit = (initial_branch == AutomaticBranch::NonStiff).then(|| {
            ExplicitKernel::new_for_automatic(ResourceTableau(explicit_tableau), dimension)
        });
        let stiff = if initial_branch == AutomaticBranch::Stiff {
            Some(ExtendedRosenbrockKernel::<A>::new(dimension)?)
        } else {
            None
        };
        Ok(Self {
            explicit_tableau,
            explicit,
            stiff,
            explicit_initialized: false,
            stiff_initialized: false,
            detector: AutoSwitchDetector::new(config),
            attempt_branch: initial_branch,
            transition: KernelTransition::identity(),
            switches: 0,
            dimension,
        })
    }

    fn initialize_active<F, P>(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError>
    where
        F: crate::OdeFunction<P>,
    {
        match self.detector.branch() {
            AutomaticBranch::NonStiff => {
                let kernel = self.explicit.get_or_insert_with(|| {
                    ExplicitKernel::new_for_automatic(
                        ResourceTableau(self.explicit_tableau),
                        self.dimension,
                    )
                });
                if !self.explicit_initialized {
                    kernel.initialize(problem, state, time, stats)?;
                    self.explicit_initialized = true;
                }
            }
            AutomaticBranch::Stiff => {
                if self.stiff.is_none() {
                    self.stiff = Some(ExtendedRosenbrockKernel::<A>::new(self.dimension)?);
                }
                if !self.stiff_initialized {
                    self.stiff
                        .as_mut()
                        .expect("stiff automatic branch must be allocated")
                        .initialize(problem, state, time, stats)?;
                    self.stiff_initialized = true;
                }
            }
        }
        self.attempt_branch = self.detector.branch();
        Ok(())
    }

    fn register_switch(&mut self, branch: AutomaticBranch) {
        self.switches = self.switches.saturating_add(1);
        match branch {
            AutomaticBranch::NonStiff => self.explicit_initialized = false,
            AutomaticBranch::Stiff => self.stiff_initialized = false,
        }
        let factor = match branch {
            AutomaticBranch::NonStiff => 1.0 / self.detector.config().switch_step_factor(),
            AutomaticBranch::Stiff => self.detector.config().switch_step_factor(),
        };
        self.transition = KernelTransition {
            step_multiplier: factor,
            reset_controller: true,
        };
    }

    fn normalized_stiffness(&self, branch: AutomaticBranch, step: f64) -> f64 {
        let estimate = match branch {
            AutomaticBranch::NonStiff => self
                .explicit
                .as_ref()
                .and_then(ExplicitKernel::stiffness_estimate),
            AutomaticBranch::Stiff => self
                .stiff
                .as_ref()
                .and_then(ExtendedRosenbrockKernel::stiffness_estimate),
        };
        let radius = self
            .explicit_tableau
            .real_stability_radius()
            .expect("automatic pair validation requires a stability radius");
        estimate.map_or(f64::NAN, |estimate| step.abs() * estimate / radius)
    }

    fn active_capabilities<F, P>(&self) -> KernelCapabilities
    where
        F: crate::OdeFunction<P>,
    {
        match self.detector.branch() {
            AutomaticBranch::NonStiff => {
                <ExplicitKernel<ResourceTableau> as StepKernel<F, P>>::capabilities(
                    self.explicit
                        .as_ref()
                        .expect("active explicit branch must be prepared"),
                )
                .recover_non_finite_derivatives()
            }
            AutomaticBranch::Stiff => {
                <ExtendedRosenbrockKernel<A> as StepKernel<F, P>>::capabilities(
                    self.stiff
                        .as_ref()
                        .expect("active stiff branch must be prepared"),
                )
            }
        }
    }
}

impl<F, P, A> StepKernel<F, P> for SwitchingKernel<A>
where
    F: crate::OdeFunction<P>,
    A: AutomaticStiffAlgorithm,
{
    fn capabilities(&self) -> KernelCapabilities {
        self.active_capabilities::<F, P>()
    }

    fn prepare_attempt(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        self.initialize_active(problem, state, time, stats)
    }

    fn take_transition(&mut self) -> KernelTransition {
        std::mem::take(&mut self.transition)
    }

    fn note_accepted_step(&mut self, stats: &mut SolverStats) {
        match self.attempt_branch {
            AutomaticBranch::NonStiff => stats.nonstiff_accepted_steps += 1,
            AutomaticBranch::Stiff => stats.stiff_accepted_steps += 1,
        }
    }

    fn finalize_stats(&self, stats: &mut SolverStats) {
        stats.algorithm_switches = self.switches;
        stats.final_automatic_branch = Some(self.detector.branch());
    }

    fn has_custom_dense_output(&self) -> bool {
        true
    }

    fn initialize(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        stats: &mut SolverStats,
    ) -> Result<(), SolveError> {
        self.initialize_active(problem, state, time, stats)
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
        match self.detector.branch() {
            AutomaticBranch::NonStiff => self
                .explicit
                .as_mut()
                .expect("active explicit branch must be initialized")
                .estimate_initial_step(
                    problem,
                    state,
                    time,
                    direction,
                    maximum_step,
                    candidate,
                    options,
                    stats,
                ),
            AutomaticBranch::Stiff => self
                .stiff
                .as_mut()
                .expect("active stiff branch must be initialized")
                .estimate_initial_step(
                    problem,
                    state,
                    time,
                    direction,
                    maximum_step,
                    candidate,
                    options,
                    stats,
                ),
        }
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
        self.attempt_branch = self.detector.branch();
        match self.attempt_branch {
            AutomaticBranch::NonStiff => self
                .explicit
                .as_mut()
                .expect("active explicit branch must be initialized")
                .attempt_step(problem, state, time, step, candidate, options, stats),
            AutomaticBranch::Stiff => self
                .stiff
                .as_mut()
                .expect("active stiff branch must be initialized")
                .attempt_step(problem, state, time, step, candidate, options, stats),
        }
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
        match self.attempt_branch {
            AutomaticBranch::NonStiff => self
                .explicit
                .as_mut()
                .expect("attempted explicit branch must exist")
                .apply_step_callbacks(
                    problem,
                    previous_state,
                    previous_time,
                    state,
                    time,
                    state_before_effect,
                    event_tolerance,
                    stats,
                ),
            AutomaticBranch::Stiff => self
                .stiff
                .as_mut()
                .expect("attempted stiff branch must exist")
                .apply_step_callbacks(
                    problem,
                    previous_state,
                    previous_time,
                    state,
                    time,
                    state_before_effect,
                    event_tolerance,
                    stats,
                ),
        }
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
        match self.attempt_branch {
            AutomaticBranch::NonStiff => self
                .explicit
                .as_mut()
                .expect("attempted explicit branch must exist")
                .record_dense_step(
                    problem,
                    previous_state,
                    state,
                    previous_time,
                    attempted_time,
                    time,
                    final_time,
                    recorder,
                    stats,
                ),
            AutomaticBranch::Stiff => self
                .stiff
                .as_mut()
                .expect("attempted stiff branch must exist")
                .record_dense_step(
                    problem,
                    previous_state,
                    state,
                    previous_time,
                    attempted_time,
                    time,
                    final_time,
                    recorder,
                    stats,
                ),
        }
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
        let diagnostic = self.normalized_stiffness(self.attempt_branch, accepted_step);
        match self.attempt_branch {
            AutomaticBranch::NonStiff => self
                .explicit
                .as_mut()
                .expect("accepted explicit branch must exist")
                .accept_step(
                    problem,
                    previous_state,
                    state,
                    time,
                    accepted_step,
                    callback_applied,
                    stats,
                )?,
            AutomaticBranch::Stiff => self
                .stiff
                .as_mut()
                .expect("accepted stiff branch must exist")
                .accept_step(
                    problem,
                    previous_state,
                    state,
                    time,
                    accepted_step,
                    callback_applied,
                    stats,
                )?,
        }

        if callback_applied {
            self.detector.reset_evidence();
        } else if let Some(branch) = self.detector.observe_accepted(diagnostic) {
            self.register_switch(branch);
        }
        Ok(())
    }

    fn reject_step(&mut self) {
        match self.attempt_branch {
            AutomaticBranch::NonStiff => {
                <ExplicitKernel<ResourceTableau> as StepKernel<F, P>>::reject_step(
                    self.explicit
                        .as_mut()
                        .expect("rejected explicit branch must exist"),
                )
            }
            AutomaticBranch::Stiff => {
                <ExtendedRosenbrockKernel<A> as StepKernel<F, P>>::reject_step(
                    self.stiff
                        .as_mut()
                        .expect("rejected stiff branch must exist"),
                )
            }
        }
    }

    fn reject_step_with_reason(&mut self, reason: RejectionReason) {
        <Self as StepKernel<F, P>>::reject_step(self);
        let switched = match reason {
            RejectionReason::Domain => {
                self.detector.reset_evidence();
                None
            }
            RejectionReason::AttemptFailure(_) | RejectionReason::ErrorEstimate(_) => {
                self.detector.observe_rejected()
            }
        };
        if let Some(branch) = switched {
            self.register_switch(branch);
        }
    }
}

fn incompatible(reason: AutomaticPairIncompatibility) -> SolveError {
    SolveError::IncompatibleAutomaticPair { reason }
}
