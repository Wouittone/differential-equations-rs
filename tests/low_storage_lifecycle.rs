use std::cell::Cell;
use std::rc::Rc;

use differential_equations::solvers::explicit::low_storage_rk::*;
use differential_equations::{
    CallbackAction, OdeAlgorithm, OdeProblem, SaveMode, SolveOptions, solve,
};

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn problem(time_span: (f64, f64), initial: f64) -> OdeProblem<TestRhs, ()> {
    fn rhs(derivative: &mut [f64], state: &[f64], _: &(), time: f64) {
        derivative[0] = state[0] + time;
    }
    OdeProblem::new(rhs, [initial], time_span, ())
}

fn options(step: f64) -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

fn endpoint<A: OdeAlgorithm>(algorithm: A, step: f64) -> f64 {
    solve(&problem((0.0, 1.0), 1.0), algorithm, &options(step))
        .unwrap()
        .last_state()[0]
}

fn observed_order<A: OdeAlgorithm + Copy>(algorithm: A) -> f64 {
    let exact = 2.0 * std::f64::consts::E - 2.0;
    let coarse = (endpoint(algorithm, 0.1) - exact).abs();
    let fine = (endpoint(algorithm, 0.05) - exact).abs();
    (coarse / fine).log2()
}

#[test]
fn representative_methods_recover_every_claimed_design_order() {
    for (name, observed, minimum) in [
        ("Ork256", observed_order(Ork256), 1.9),
        (
            "ParsaniKetchesonDeconinck3S32",
            observed_order(ParsaniKetchesonDeconinck3S32),
            1.8,
        ),
        (
            "ParsaniKetchesonDeconinck3S53",
            observed_order(ParsaniKetchesonDeconinck3S53),
            2.8,
        ),
        (
            "ParsaniKetchesonDeconinck3S173",
            observed_order(ParsaniKetchesonDeconinck3S173),
            2.8,
        ),
        (
            "ParsaniKetchesonDeconinck3S105",
            observed_order(ParsaniKetchesonDeconinck3S105),
            4.7,
        ),
        (
            "ParsaniKetchesonDeconinck3S82",
            observed_order(ParsaniKetchesonDeconinck3S82),
            1.8,
        ),
        (
            "ParsaniKetchesonDeconinck3S94",
            observed_order(ParsaniKetchesonDeconinck3S94),
            3.75,
        ),
        (
            "ParsaniKetchesonDeconinck3S184",
            observed_order(ParsaniKetchesonDeconinck3S184),
            3.75,
        ),
        (
            "ParsaniKetchesonDeconinck3S205",
            observed_order(ParsaniKetchesonDeconinck3S205),
            4.7,
        ),
        ("Dglddrk73C", observed_order(Dglddrk73C), 2.9),
        (
            "CarpenterKennedy2N54",
            observed_order(CarpenterKennedy2N54),
            3.75,
        ),
        ("Dglddrk84C", observed_order(Dglddrk84C), 3.75),
        ("Dglddrk84F", observed_order(Dglddrk84F), 3.75),
        ("Ndblsrk124", observed_order(Ndblsrk124), 3.75),
        ("Ndblsrk134", observed_order(Ndblsrk134), 3.75),
        ("Ndblsrk144", observed_order(Ndblsrk144), 3.75),
        ("RDPK3Sp35", observed_order(RDPK3Sp35), 2.8),
        ("RDPK3SpFSAL510", observed_order(RDPK3SpFSAL510), 4.7),
        ("CKLLSRK43_2", observed_order(CKLLSRK43_2), 2.8),
        ("CKLLSRK95_4M", observed_order(CKLLSRK95_4M), 4.7),
    ] {
        assert!(
            observed > minimum,
            "{name} observed order was {observed}, expected more than {minimum}"
        );
    }

    // Upstream also marks SHLDDRK64's order check broken because its published
    // decimal coefficients have limited precision. Its exact recurrence is
    // still covered by resource fingerprints and execution tests.
    assert!(endpoint(Shlddrk64, 0.01).is_finite());
}

#[test]
fn backward_save_at_and_callback_lifecycle_use_shared_driver_semantics() {
    let backward = problem((1.0, 0.0), 2.0 * std::f64::consts::E - 2.0);
    let backward_options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.01),
        save_at: vec![1.0, 0.5, 0.0],
        ..SolveOptions::default()
    };

    macro_rules! check_backward {
        ($algorithm:expr, $tolerance:expr) => {{
            let solution = solve(&backward, $algorithm, &backward_options).unwrap();
            assert_eq!(solution.times(), &[1.0, 0.5, 0.0]);
            assert!((solution.last_state()[0] - 1.0).abs() < $tolerance);
        }};
    }
    check_backward!(CarpenterKennedy2N54, 1.0e-8);
    check_backward!(ParsaniKetchesonDeconinck3S32, 2.0e-3);
    check_backward!(ParsaniKetchesonDeconinck3S53, 2.0e-5);
    check_backward!(ParsaniKetchesonDeconinck3S105, 2.0e-8);

    let terminating = problem((0.0, 1.0), 1.0)
        .with_continuous_callback(|_, _, time| time - 0.5, |_, _, _| CallbackAction::Terminate);
    let solution = solve(&terminating, Dglddrk73C, &options(0.1)).unwrap();
    assert!((solution.times().last().unwrap() - 0.5).abs() < 1.0e-14);
    assert_eq!(solution.stats().callback_invocations, 1);
}

#[test]
fn terminating_callbacks_do_not_trigger_post_effect_rhs_work() {
    let rhs_calls = Rc::new(Cell::new(0));
    let rhs_counter = Rc::clone(&rhs_calls);
    let problem = OdeProblem::new(
        move |derivative: &mut [f64], state: &[f64], _: &(), _: f64| {
            rhs_counter.set(rhs_counter.get() + 1);
            derivative[0] = state[0];
        },
        [1.0],
        (0.0, 1.0),
        (),
    )
    .with_discrete_callback(
        |_, _, time| time >= 0.25,
        |_, _, _| CallbackAction::Terminate,
    );
    let solution = solve(&problem, Dglddrk73C, &options(0.25)).unwrap();
    assert_eq!(solution.stats().rhs_evaluations, 7);
    assert_eq!(rhs_calls.get(), 7);

    let initial_rhs_calls = Rc::new(Cell::new(0));
    let initial_rhs_counter = Rc::clone(&initial_rhs_calls);
    let initially_terminating = OdeProblem::new(
        move |derivative: &mut [f64], state: &[f64], _: &(), _: f64| {
            initial_rhs_counter.set(initial_rhs_counter.get() + 1);
            derivative[0] = state[0];
        },
        [1.0],
        (0.0, 1.0),
        (),
    )
    .with_discrete_callback(|_, _, _| true, |_, _, _| CallbackAction::Terminate);
    let solution = solve(&initially_terminating, Ork256, &options(0.25)).unwrap();
    assert_eq!(solution.stats().rhs_evaluations, 0);
    assert_eq!(initial_rhs_calls.get(), 0);
}
