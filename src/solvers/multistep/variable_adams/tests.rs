use std::cell::Cell;
use std::f64::consts::E;
use std::rc::Rc;

use super::algorithms::{Vcab3, Vcab4, Vcab5, Vcabm3, Vcabm4, Vcabm5};
use super::fixed::VariableAdamsKernel;
use super::history::{Workspace, update_g};
use super::method::VCAB5_METHOD;
use crate::integrator::StepKernel;
use crate::{CallbackAction, OdeProblem, SaveMode, SolveOptions, SolverStats, solve};

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn exponential(time_span: (f64, f64)) -> OdeProblem<TestRhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
        du[0] = u[0];
    }
    OdeProblem::new(rhs as TestRhs, vec![1.0], time_span, ())
}

fn options() -> SolveOptions {
    SolveOptions {
        absolute_tolerance: 1.0e-10,
        relative_tolerance: 1.0e-10,
        initial_step: Some(0.013),
        max_step: 0.17,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

#[test]
fn coefficients_cover_equal_and_unequal_step_histories() {
    let mut workspace = Workspace::new(1, 3);
    workspace.trial_steps.copy_from_slice(&[0.2, 0.2, 0.2]);
    update_g(&mut workspace, 3);

    // `g` multiplies backward differences. Expanding these three values
    // into raw derivatives produces the classical 23/-16/5 weights.
    assert!((workspace.g[0] - 0.2).abs() < 1.0e-15);
    assert!((workspace.g[1] - 0.5 * 0.2).abs() < 1.0e-15);
    assert!((workspace.g[2] - 5.0 / 12.0 * 0.2).abs() < 1.0e-15);

    workspace.trial_steps.copy_from_slice(&[0.2, 0.1, 0.3]);
    update_g(&mut workspace, 3);
    assert!((workspace.g[2] - 7.0 / 90.0).abs() < 1.0e-15);
}

#[test]
fn all_variable_adams_methods_solve_forward_and_backward() {
    macro_rules! check {
        ($algorithm:expr) => {{
            let forward = solve(&exponential((0.0, 1.0)), $algorithm, &options()).unwrap();
            let backward_problem = OdeProblem::new(
                |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
                vec![E],
                (1.0, 0.0),
                (),
            );
            let backward = solve(&backward_problem, $algorithm, &options()).unwrap();
            assert!(
                (forward.last_state()[0] - E).abs() < 2.0e-6,
                "forward endpoint: {}",
                forward.last_state()[0]
            );
            assert!(
                (backward.last_state()[0] - 1.0).abs() < 2.0e-6,
                "backward endpoint: {}",
                backward.last_state()[0]
            );
            assert!(forward.stats().accepted_steps > 5);
        }};
    }
    check!(Vcab3);
    check!(Vcab4);
    check!(Vcab5);
    check!(Vcabm3);
    check!(Vcabm4);
    check!(Vcabm5);
}

#[test]
fn callback_resets_multistep_history_and_save_at_is_honored() {
    let problem = exponential((0.0, 1.0)).with_continuous_callback(
        |_: &[f64], _: &(), time| time - 0.5,
        |state: &mut [f64], _: &(), _: f64| {
            state[0] *= 0.5;
            CallbackAction::Continue
        },
    );
    let options = SolveOptions {
        save_at: vec![0.0, 0.25, 0.5, 0.75, 1.0],
        ..options()
    };
    let solution = solve(&problem, Vcab5, &options).unwrap();

    assert_eq!(solution.times(), &[0.0, 0.25, 0.5, 0.5, 0.75, 1.0]);
    assert!(solution.state(3).unwrap()[0] < solution.state(2).unwrap()[0]);
    assert_eq!(solution.stats().callback_invocations, 1);
    assert!(
        (solution.last_state()[0] - 0.5 * E).abs() < 5.0e-6,
        "callback endpoint: {}",
        solution.last_state()[0]
    );
}

#[test]
fn tighter_tolerances_reduce_error() {
    let loose = SolveOptions {
        absolute_tolerance: 1.0e-5,
        relative_tolerance: 1.0e-5,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let tight = SolveOptions {
        absolute_tolerance: 1.0e-10,
        relative_tolerance: 1.0e-10,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let loose_solution = solve(&exponential((0.0, 4.0)), Vcab4, &loose).unwrap();
    let tight_solution = solve(&exponential((0.0, 4.0)), Vcab4, &tight).unwrap();
    let exact = 4.0_f64.exp();

    assert!(
        (tight_solution.last_state()[0] - exact).abs()
            < (loose_solution.last_state()[0] - exact).abs()
    );
    assert!(tight_solution.stats().accepted_steps > loose_solution.stats().accepted_steps);
}

#[test]
fn fixed_step_runs_recover_each_methods_design_order() {
    macro_rules! ratio {
        ($algorithm:expr, $order:expr) => {{
            let run = |step| {
                let options = SolveOptions {
                    adaptive: false,
                    initial_step: Some(step),
                    save: SaveMode::Endpoints,
                    ..SolveOptions::default()
                };
                (solve(&exponential((0.0, 1.0)), $algorithm, &options)
                    .unwrap()
                    .last_state()[0]
                    - E)
                    .abs()
            };
            let observed = run(0.05) / run(0.025);
            let minimum = 2.0_f64.powi($order - 1);
            assert!(observed > minimum, "order {} ratio: {}", $order, observed);
        }};
    }

    ratio!(Vcab3, 3);
    ratio!(Vcab4, 4);
    ratio!(Vcab5, 5);
    ratio!(Vcabm3, 3);
    ratio!(Vcabm4, 4);
    ratio!(Vcabm5, 5);
}

#[test]
fn rejected_attempt_does_not_advance_or_contaminate_history() {
    let problem = exponential((0.0, 1.0));
    let options = SolveOptions {
        absolute_tolerance: 1.0e-12,
        relative_tolerance: 1.0e-12,
        ..SolveOptions::default()
    };
    let mut rejected_kernel = VariableAdamsKernel::new(&VCAB5_METHOD);
    let mut fresh_kernel = VariableAdamsKernel::new(&VCAB5_METHOD);
    let mut rejected_stats = SolverStats::default();
    let mut fresh_stats = SolverStats::default();
    let state = [1.0];
    let mut rejected_candidate = [0.0];
    let mut fresh_candidate = [0.0];

    <VariableAdamsKernel as StepKernel<TestRhs, ()>>::initialize(
        &mut rejected_kernel,
        &problem,
        &state,
        0.0,
        &mut rejected_stats,
    )
    .unwrap();
    let rejected = <VariableAdamsKernel as StepKernel<TestRhs, ()>>::attempt_step(
        &mut rejected_kernel,
        &problem,
        &state,
        0.0,
        1.0,
        &mut rejected_candidate,
        &options,
        &mut rejected_stats,
    )
    .unwrap();
    assert!(rejected.error_norm > 1.0);
    <VariableAdamsKernel as StepKernel<TestRhs, ()>>::reject_step(&mut rejected_kernel);

    let accepted_workspace = rejected_kernel.workspace.as_ref().unwrap();
    assert_eq!(rejected_kernel.step_number, 1);
    assert!(
        accepted_workspace
            .accepted_steps
            .iter()
            .all(|step| *step == 0.0)
    );
    assert!(
        accepted_workspace
            .phi_previous
            .iter()
            .flatten()
            .all(|value| *value == 0.0)
    );
    assert_eq!(accepted_workspace.derivative, [1.0]);

    <VariableAdamsKernel as StepKernel<TestRhs, ()>>::initialize(
        &mut fresh_kernel,
        &problem,
        &state,
        0.0,
        &mut fresh_stats,
    )
    .unwrap();
    let after_rejection = <VariableAdamsKernel as StepKernel<TestRhs, ()>>::attempt_step(
        &mut rejected_kernel,
        &problem,
        &state,
        0.0,
        1.0e-4,
        &mut rejected_candidate,
        &options,
        &mut rejected_stats,
    )
    .unwrap();
    let fresh = <VariableAdamsKernel as StepKernel<TestRhs, ()>>::attempt_step(
        &mut fresh_kernel,
        &problem,
        &state,
        0.0,
        1.0e-4,
        &mut fresh_candidate,
        &options,
        &mut fresh_stats,
    )
    .unwrap();

    assert_eq!(after_rejection, fresh);
    assert_eq!(rejected_candidate, fresh_candidate);
}

#[test]
fn terminating_callback_skips_accepted_history_derivative() {
    let calls = Rc::new(Cell::new(0));
    let terminated = Rc::new(Cell::new(false));
    let calls_after_termination = Rc::new(Cell::new(0));
    let rhs_calls = Rc::clone(&calls);
    let rhs_terminated = Rc::clone(&terminated);
    let rhs_calls_after_termination = Rc::clone(&calls_after_termination);
    let effect_terminated = Rc::clone(&terminated);
    let problem = OdeProblem::new(
        move |du: &mut [f64], u: &[f64], _: &(), _: f64| {
            rhs_calls.set(rhs_calls.get() + 1);
            if rhs_terminated.get() {
                rhs_calls_after_termination.set(rhs_calls_after_termination.get() + 1);
            }
            du[0] = u[0];
        },
        vec![1.0],
        (0.0, 1.0),
        (),
    )
    .with_continuous_callback(
        |_: &[f64], _: &(), time| time - 0.625,
        move |_: &mut [f64], _: &(), _: f64| {
            effect_terminated.set(true);
            CallbackAction::Terminate
        },
    );
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.25),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };

    let solution = solve(&problem, Vcab3, &options).unwrap();

    assert_eq!(solution.stats().accepted_steps, 3);
    assert_eq!(solution.stats().rhs_evaluations, 12);
    assert_eq!(calls.get(), 12);
    assert_eq!(calls_after_termination.get(), 0);
}
