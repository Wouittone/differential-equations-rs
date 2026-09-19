use std::cell::Cell;
use std::rc::Rc;

use super::{
    ControllerConfig, ControllerState, KernelCapabilities, KernelTransition, StepEstimate,
    StepKernel, integrate, step_factor, step_factor_with_history,
};
use crate::{CallbackAction, OdeProblem, SaveMode, SolveError, SolveOptions, SolverStats};

struct MockKernel {
    errors: Vec<f64>,
    failures: Vec<Option<SolveError>>,
    recover_failures: bool,
    failed_attempt_factor: f64,
    attempts: usize,
    initialize_calls: usize,
    accept_calls: usize,
    reject_calls: usize,
    first_candidate: Option<*const f64>,
    second_candidate: Option<*const f64>,
    unexpected_candidate: bool,
    attempted_steps: Vec<f64>,
    controller: ControllerConfig,
    transitions: Vec<KernelTransition>,
    transition_index: usize,
    capability_queries: Cell<usize>,
}

impl MockKernel {
    fn fixed() -> Self {
        Self::with_errors(vec![0.0])
    }

    fn with_errors(errors: Vec<f64>) -> Self {
        Self {
            errors,
            failures: Vec::new(),
            recover_failures: false,
            failed_attempt_factor: 0.2,
            attempts: 0,
            initialize_calls: 0,
            accept_calls: 0,
            reject_calls: 0,
            first_candidate: None,
            second_candidate: None,
            unexpected_candidate: false,
            attempted_steps: Vec::new(),
            controller: ControllerConfig::proportional(1, 0.9, 0.2, 10.0, 0.2),
            transitions: Vec::new(),
            transition_index: 0,
            capability_queries: Cell::new(0),
        }
    }

    fn with_controller(mut self, controller: ControllerConfig) -> Self {
        self.controller = controller;
        self
    }

    fn with_transitions(mut self, transitions: Vec<KernelTransition>) -> Self {
        self.transitions = transitions;
        self
    }

    fn with_failures(failures: Vec<Option<SolveError>>) -> Self {
        let mut kernel = Self::with_errors(vec![0.0]);
        kernel.failures = failures;
        kernel.recover_failures = true;
        kernel
    }

    fn observe_candidate(&mut self, pointer: *const f64) {
        if self.first_candidate.is_none() {
            self.first_candidate = Some(pointer);
        } else if self.first_candidate != Some(pointer) && self.second_candidate.is_none() {
            self.second_candidate = Some(pointer);
        } else if self.first_candidate != Some(pointer) && self.second_candidate != Some(pointer) {
            self.unexpected_candidate = true;
        }
    }
}

impl<F, P> StepKernel<F, P> for &mut MockKernel
where
    F: crate::OdeFunction<P>,
{
    fn capabilities(&self) -> KernelCapabilities {
        self.capability_queries
            .set(self.capability_queries.get() + 1);
        let mut controller = self.controller;
        controller.failed_attempt_factor = self.failed_attempt_factor;
        let capabilities = KernelCapabilities::with_controller(true, controller);
        if self.recover_failures {
            capabilities.recover_nonlinear_and_singular_failures()
        } else {
            capabilities
        }
    }

    fn take_transition(&mut self) -> KernelTransition {
        let transition = self
            .transitions
            .get(self.transition_index)
            .copied()
            .unwrap_or_default();
        self.transition_index += 1;
        transition
    }

    fn initialize(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: f64,
        _: &mut SolverStats,
    ) -> Result<(), SolveError> {
        self.initialize_calls += 1;
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
        Ok(maximum_step.min(0.25))
    }

    fn attempt_step(
        &mut self,
        _: &OdeProblem<F, P>,
        state: &[f64],
        _: f64,
        step: f64,
        candidate: &mut [f64],
        _: &SolveOptions,
        _: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError> {
        self.observe_candidate(candidate.as_ptr());
        self.attempted_steps.push(step);
        let attempt = self.attempts;
        self.attempts += 1;
        if let Some(error) = self.failures.get(attempt).copied().flatten() {
            candidate.fill(f64::NAN);
            return Err(error);
        }
        for (candidate, state) in candidate.iter_mut().zip(state) {
            *candidate = state + step;
        }
        let error = self
            .errors
            .get(attempt)
            .copied()
            .unwrap_or_else(|| *self.errors.last().unwrap());
        Ok(StepEstimate::new(error))
    }

    fn accept_step(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: &[f64],
        _: f64,
        _: f64,
        _: bool,
        _: &mut SolverStats,
    ) -> Result<(), SolveError> {
        self.accept_calls += 1;
        Ok(())
    }

    fn reject_step(&mut self) {
        self.reject_calls += 1;
    }
}

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn unit_problem(span: (f64, f64), initial: f64) -> OdeProblem<TestRhs, ()> {
    fn unit_rate(du: &mut [f64], _: &[f64], _: &(), _: f64) {
        du[0] = 1.0;
    }
    OdeProblem::new(unit_rate, vec![initial], span, ())
}

fn fixed_options(step: f64) -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        save: SaveMode::EveryStep,
        ..SolveOptions::default()
    }
}

#[test]
fn rejection_has_no_callback_or_save_side_effects() {
    let effects = Rc::new(Cell::new(0));
    let effect_count = Rc::clone(&effects);
    let problem = unit_problem((0.0, 0.5), 0.0).with_discrete_callback(
        |_, _, time| time > 0.0,
        move |_, _, _| {
            effect_count.set(effect_count.get() + 1);
            CallbackAction::Continue
        },
    );
    let mut kernel = MockKernel::with_errors(vec![4.0, 0.0]);
    let options = SolveOptions {
        initial_step: Some(0.5),
        save: SaveMode::EveryStep,
        ..SolveOptions::default()
    };

    let solution = integrate(&problem, &options, &mut kernel).unwrap();

    assert_eq!(kernel.reject_calls, 1);
    assert_eq!(effects.get(), solution.stats().accepted_steps);
    assert_eq!(solution.times().len(), solution.stats().accepted_steps + 1);
    assert_eq!(solution.stats().rejected_steps, 1);
}

#[test]
fn rejection_transition_scales_the_retry_step() {
    let problem = unit_problem((0.0, 0.5), 0.0);
    let transition = KernelTransition {
        step_multiplier: 0.5,
        reset_controller: true,
    };
    let mut kernel = MockKernel::with_errors(vec![4.0, 0.0]).with_transitions(vec![transition]);
    let options = SolveOptions {
        initial_step: Some(0.4),
        ..SolveOptions::default()
    };

    integrate(&problem, &options, &mut kernel).unwrap();

    assert!((kernel.attempted_steps[1] - 0.045).abs() < 1.0e-15);
}

#[test]
fn accepted_transition_resets_history_and_scales_the_proposal() {
    let problem = unit_problem((0.0, 3.0), 0.0);
    let controller =
        ControllerConfig::proportional(1, 1.0, 0.1, 10.0, 0.2).with_integral_exponent(0.5);
    let transition = KernelTransition {
        step_multiplier: 0.5,
        reset_controller: true,
    };
    let mut kernel = MockKernel::with_errors(vec![0.25])
        .with_controller(controller)
        .with_transitions(vec![KernelTransition::identity(), transition]);
    let options = SolveOptions {
        initial_step: Some(0.1),
        ..SolveOptions::default()
    };

    integrate(&problem, &options, &mut kernel).unwrap();

    assert!((kernel.attempted_steps[1] - 0.4).abs() < 1.0e-15);
    assert!((kernel.attempted_steps[2] - 0.8).abs() < 1.0e-15);
}

#[test]
fn fixed_transition_multiplier_changes_only_the_next_attempt() {
    let problem = unit_problem((0.0, 0.5), 0.0);
    let transition = KernelTransition {
        step_multiplier: 0.5,
        reset_controller: true,
    };
    let mut kernel = MockKernel::fixed().with_transitions(vec![transition]);

    integrate(&problem, &fixed_options(0.1), &mut kernel).unwrap();

    assert!((kernel.attempted_steps[0] - 0.1).abs() < 1.0e-15);
    assert!((kernel.attempted_steps[1] - 0.05).abs() < 1.0e-15);
    assert!((kernel.attempted_steps[2] - 0.1).abs() < 1.0e-15);
}

#[test]
fn callback_step_request_overrides_the_transition_multiplier() {
    let problem = unit_problem((0.0, 1.0), 0.0).with_preset_time_callback([0.1], |_, _, _| {
        CallbackAction::ContinueUnmodifiedWithStepSize(0.3)
    });
    let transition = KernelTransition {
        step_multiplier: 0.5,
        reset_controller: false,
    };
    let mut kernel = MockKernel::fixed().with_transitions(vec![transition]);

    integrate(&problem, &fixed_options(0.1), &mut kernel).unwrap();

    assert!((kernel.attempted_steps[1] - 0.3).abs() < 1.0e-15);
    assert!((kernel.attempted_steps[2] - 0.1).abs() < 1.0e-15);
}

#[test]
fn capabilities_are_queried_for_each_attempt() {
    let problem = unit_problem((0.0, 1.0), 0.0);
    let mut kernel = MockKernel::fixed();

    integrate(&problem, &fixed_options(0.3), &mut kernel).unwrap();

    assert_eq!(kernel.capability_queries.get(), kernel.attempts);
}

#[test]
fn integrates_backward_and_clips_the_endpoint() {
    let problem = unit_problem((1.0, 0.0), 1.0);
    let mut kernel = MockKernel::fixed();
    let solution = integrate(&problem, &fixed_options(0.3), &mut kernel).unwrap();

    assert_eq!(solution.times().last(), Some(&0.0));
    assert!((solution.last_state()[0]).abs() < 1.0e-15);
    assert_eq!(solution.stats().accepted_steps, 4);
}

#[test]
fn fixed_steps_hit_time_stops_then_resume_the_configured_step() {
    let problem = unit_problem((0.0, 1.0), 0.0);
    let options = fixed_options(0.4).with_time_stops([0.25, 0.5]);
    let mut kernel = MockKernel::fixed();

    let solution = integrate(&problem, &options, &mut kernel).unwrap();

    assert_eq!(solution.times(), &[0.0, 0.25, 0.5, 0.9, 1.0]);
    assert_eq!(solution.values(), solution.times());
}

#[test]
fn backward_time_stops_follow_the_integration_direction() {
    let problem = unit_problem((1.0, 0.0), 1.0);
    let options = fixed_options(0.4).with_time_stops([0.75, 0.2]);
    let mut kernel = MockKernel::fixed();

    let solution = integrate(&problem, &options, &mut kernel).unwrap();

    assert_eq!(solution.times(), &[1.0, 0.75, 0.35, 0.2, 0.0]);
    assert!(solution.last_state()[0].abs() < 1.0e-15);
}

#[test]
fn time_stops_do_not_force_solution_output() {
    let problem = unit_problem((0.0, 1.0), 0.0);
    let options = fixed_options(0.4)
        .with_save(SaveMode::Endpoints)
        .with_time_stops([0.25, 0.5]);
    let mut kernel = MockKernel::fixed();

    let solution = integrate(&problem, &options, &mut kernel).unwrap();

    assert_eq!(solution.times(), &[0.0, 1.0]);
    assert_eq!(solution.stats().accepted_steps, 4);
}

#[test]
fn discrete_callbacks_can_act_at_exact_time_stops() {
    let effects = Rc::new(Cell::new(0));
    let effect_count = Rc::clone(&effects);
    let problem = unit_problem((0.0, 1.0), 0.0).with_discrete_callback(
        |_, _, time| time == 0.3,
        move |state, _, _| {
            effect_count.set(effect_count.get() + 1);
            state[0] += 1.0;
            CallbackAction::Continue
        },
    );
    let options = fixed_options(0.4).with_time_stops([0.3]);
    let mut kernel = MockKernel::fixed();

    let solution = integrate(&problem, &options, &mut kernel).unwrap();

    assert_eq!(effects.get(), 1);
    assert_eq!(solution.stats().callback_invocations, 1);
    assert!((solution.last_state()[0] - 2.0).abs() < 1.0e-15);
}

#[test]
fn callbacks_record_pre_effect_samples_and_force_the_effect_state() {
    let problem = unit_problem((0.0, 1.0), 0.0).with_discrete_callback(
        |_, _, time| time >= 0.6,
        |state, _, _| {
            state[0] = 10.0;
            CallbackAction::Continue
        },
    );
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.6),
        save: SaveMode::Endpoints,
        save_at: vec![0.2, 0.5],
        ..SolveOptions::default()
    };
    let mut kernel = MockKernel::fixed();

    let solution = integrate(&problem, &options, &mut kernel).unwrap();

    assert_eq!(solution.times()[..3], [0.2, 0.5, 0.6]);
    assert!((solution.state(0).unwrap()[0] - 0.2).abs() < 1.0e-15);
    assert!((solution.state(1).unwrap()[0] - 0.5).abs() < 1.0e-15);
    assert_eq!(solution.state(2), Some([10.0].as_slice()));
}

#[test]
fn terminating_effect_returns_before_the_kernel_accept_hook() {
    let problem = unit_problem((0.0, 1.0), 0.0).with_continuous_callback(
        |state, _, _| state[0] - 0.5,
        |state, _, _| {
            state[0] = 42.0;
            CallbackAction::Terminate
        },
    );
    let mut kernel = MockKernel::fixed();

    let solution = integrate(&problem, &fixed_options(1.0), &mut kernel).unwrap();

    assert_eq!(solution.last_state(), &[42.0]);
    assert_eq!(kernel.accept_calls, 0);
    assert_eq!(kernel.attempts, 1);
}

#[test]
fn initial_termination_never_initializes_the_kernel() {
    let problem = unit_problem((0.0, 1.0), 0.0).with_discrete_callback(
        |_, _, time| time == 0.0,
        |state, _, _| {
            state[0] = 7.0;
            CallbackAction::Terminate
        },
    );
    let mut kernel = MockKernel::fixed();

    let solution = integrate(&problem, &fixed_options(0.25), &mut kernel).unwrap();

    assert_eq!(solution.last_state(), &[7.0]);
    assert_eq!(kernel.initialize_calls, 0);
    assert_eq!(kernel.attempts, 0);
}

#[test]
fn save_at_uses_the_accepted_segment() {
    let problem = unit_problem((0.0, 1.0), 0.0);
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.4),
        save_at: vec![0.1, 0.7, 1.0],
        ..SolveOptions::default()
    };
    let mut kernel = MockKernel::fixed();
    let solution = integrate(&problem, &options, &mut kernel).unwrap();

    assert_eq!(solution.times(), &[0.1, 0.7, 1.0]);
    assert_eq!(solution.values(), &[0.1, 0.7, 1.0]);
}

#[test]
fn reports_step_underflow_before_attempting_the_kernel() {
    let problem = unit_problem((1.0, 2.0), 0.0);
    let mut kernel = MockKernel::fixed();
    let result = integrate(&problem, &fixed_options(f64::MIN_POSITIVE), &mut kernel);

    assert_eq!(result, Err(SolveError::StepSizeUnderflow));
    assert_eq!(kernel.attempts, 0);
}

#[test]
fn max_steps_counts_rejected_and_accepted_attempts() {
    let problem = unit_problem((0.0, 1.0), 0.0);
    let options = SolveOptions {
        initial_step: Some(0.5),
        max_steps: 2,
        ..SolveOptions::default()
    };
    let mut kernel = MockKernel::with_errors(vec![4.0, 0.0]);

    assert_eq!(
        integrate(&problem, &options, &mut kernel),
        Err(SolveError::MaxStepsExceeded)
    );
    assert_eq!(kernel.attempts, 2);
}

#[test]
fn recoverable_attempt_failures_reject_without_using_the_candidate() {
    let effects = Rc::new(Cell::new(0));
    let effect_count = Rc::clone(&effects);
    let problem = unit_problem((0.0, 0.5), 0.0).with_discrete_callback(
        |_, _, time| time > 0.0,
        move |_, _, _| {
            effect_count.set(effect_count.get() + 1);
            CallbackAction::Continue
        },
    );
    let options = SolveOptions {
        initial_step: Some(0.5),
        save: SaveMode::EveryStep,
        ..SolveOptions::default()
    };
    let mut kernel = MockKernel::with_failures(vec![
        Some(SolveError::NonlinearSolveFailed),
        Some(SolveError::SingularLinearSystem),
        None,
    ]);

    let solution = integrate(&problem, &options, &mut kernel).unwrap();

    assert_eq!(kernel.reject_calls, 2);
    assert_eq!(solution.stats().rejected_steps, 2);
    assert_eq!(effects.get(), solution.stats().accepted_steps);
    assert_eq!(solution.times().len(), solution.stats().accepted_steps + 1);
    assert_eq!(solution.last_state(), &[0.5]);
}

#[test]
fn failed_attempt_shrink_is_checked_for_underflow_before_retry() {
    let problem = unit_problem((1.0, 2.0), 0.0);
    let options = SolveOptions {
        initial_step: Some(f64::EPSILON),
        ..SolveOptions::default()
    };
    let mut kernel = MockKernel::with_failures(vec![Some(SolveError::NonlinearSolveFailed)]);

    assert_eq!(
        integrate(&problem, &options, &mut kernel),
        Err(SolveError::StepSizeUnderflow)
    );
    assert_eq!(kernel.attempts, 1);
    assert_eq!(kernel.reject_calls, 1);
    assert_eq!(kernel.accept_calls, 0);
}

#[test]
fn recoverable_failure_policy_is_terminal_in_fixed_step_mode() {
    let problem = unit_problem((0.0, 1.0), 0.0);
    let mut kernel = MockKernel::with_failures(vec![Some(SolveError::SingularLinearSystem)]);

    assert_eq!(
        integrate(&problem, &fixed_options(0.25), &mut kernel),
        Err(SolveError::SingularLinearSystem)
    );
    assert_eq!(kernel.attempts, 1);
    assert_eq!(kernel.reject_calls, 0);
    assert_eq!(kernel.accept_calls, 0);
}

#[test]
fn pi_controller_metadata_uses_previous_error_without_changing_defaults() {
    let proportional = ControllerConfig::proportional(5, 0.9, 0.2, 10.0, 0.2);
    assert_eq!(
        step_factor_with_history(0.25, Some(0.5), proportional),
        step_factor_with_history(0.25, Some(0.5), proportional.with_integral_exponent(0.0))
    );
    let pi = proportional.with_integral_exponent(0.2);
    assert!(step_factor_with_history(0.25, Some(0.5), pi) < step_factor(0.25, pi));
    assert_eq!(
        step_factor_with_history(0.25, None, pi),
        step_factor(0.25, pi)
    );
    let mut state = ControllerState::default();
    state.accepted(0.5, pi);
    assert!(state.factor(0.25, pi) < step_factor(0.25, pi));
    state.reset();
    assert_eq!(state.factor(0.25, pi), step_factor(0.25, pi));
}

#[test]
fn standard_pi_uses_pre_accept_history_and_current_error_only_on_rejection() {
    let controller = ControllerConfig::standard_pi(4);
    let mut state = ControllerState::default();
    let first = state.factor(0.25, controller);
    let expected_first = 0.9 * 0.25_f64.powf(-0.7 / 4.0) * 1.0e-4_f64.powf(0.4 / 4.0);
    assert!((first - expected_first).abs() < 1.0e-15);

    state.accepted(0.5, controller);
    let next = state.factor(0.25, controller);
    let expected_next = 0.9 * 0.25_f64.powf(-0.7 / 4.0) * 0.5_f64.powf(0.4 / 4.0);
    assert!((next - expected_next).abs() < 1.0e-15);

    let rejection = state.rejection_factor(0.25, controller);
    let expected_rejection = 0.9 * 0.25_f64.powf(-0.7 / 4.0);
    assert!((rejection - expected_rejection).abs() < 1.0e-15);
    state.rejected(0.25, controller);
    assert_eq!(state.previous_error, Some(0.5));
}

#[test]
fn unmodified_step_request_still_commits_controller_history() {
    let problem = unit_problem((0.0, 1.0), 0.0).with_preset_time_callback([0.25], |_, _, _| {
        CallbackAction::ContinueUnmodifiedWithStepSize(0.25)
    });
    let controller = ControllerConfig::standard_pi(4);
    let mut kernel = MockKernel::with_errors(vec![0.25]).with_controller(controller);
    let options = SolveOptions {
        initial_step: Some(0.25),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };

    integrate(&problem, &options, &mut kernel).unwrap();

    let expected_factor = 0.9 * 0.25_f64.powf(-0.7 / 4.0) * 0.25_f64.powf(0.4 / 4.0);
    assert!((kernel.attempted_steps[2] - 0.25 * expected_factor).abs() < 1.0e-14);
}

#[test]
fn pid_rejects_invalid_errors_and_shifts_only_accepted_history() {
    let controller = ControllerConfig::pid(3, [0.64, -0.31, 0.04], 0.81);
    let mut state = ControllerState::default();
    assert!(!state.accepts(f64::NAN, controller));
    assert!(!state.accepts(f64::INFINITY, controller));
    assert!(!state.accepts(-1.0, controller));

    let before = state.factor(0.25, controller);
    state.rejected(0.25, controller);
    assert_eq!(state, ControllerState::default());
    assert_eq!(state.factor(0.25, controller), before);

    state.accepted(0.25, controller);
    assert_eq!(state.previous_error, Some(0.25));
    assert_eq!(state.older_error, None);
    state.accepted(0.0, controller);
    assert_eq!(state.previous_error, Some(f64::EPSILON));
    assert_eq!(state.older_error, Some(0.25));
    state.reset();
    assert_eq!(state, ControllerState::default());
}

#[test]
fn reuses_exactly_two_driver_state_buffers_across_steps() {
    let problem = unit_problem((0.0, 1.0), 0.0);
    let mut kernel = MockKernel::fixed();
    let solution = integrate(&problem, &fixed_options(0.01), &mut kernel).unwrap();

    assert_eq!(solution.stats().accepted_steps, 100);
    assert!(kernel.first_candidate.is_some());
    assert!(kernel.second_candidate.is_some());
    assert!(!kernel.unexpected_candidate);
}
