#[path = "../src/stabilized.rs"]
mod stabilized;

use differential_equations::{OdeProblem, SaveMode, SolveOptions, solve};

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn exponential_rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
    du[0] = -u[0];
}

fn stiff_decay_rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
    du[0] = -100.0 * u[0];
}

fn exponential() -> OdeProblem<TestRhs, ()> {
    OdeProblem::new(exponential_rhs, vec![1.0], (0.0, 1.0), ())
}

fn stiff_decay() -> OdeProblem<TestRhs, ()> {
    OdeProblem::new(stiff_decay_rhs, vec![1.0], (0.0, 1.0), ())
}

fn fixed_options() -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(0.05),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

fn adaptive_options() -> SolveOptions {
    SolveOptions {
        absolute_tolerance: 1.0e-8,
        relative_tolerance: 1.0e-8,
        initial_step: Some(0.01),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

#[test]
fn all_stabilized_public_names_solve_regular_odes() {
    let fixed = fixed_options();
    let adaptive = adaptive_options();

    let fixed_solutions = [
        solve(&exponential(), stabilized::ESERK4, &fixed).unwrap(),
        solve(&exponential(), stabilized::ESERK5, &fixed).unwrap(),
        solve(&exponential(), stabilized::TSRKC2, &fixed).unwrap(),
        solve(&exponential(), stabilized::TSRKC3, &fixed).unwrap(),
    ];
    let adaptive_solutions = [
        solve(&exponential(), stabilized::RKC, &adaptive).unwrap(),
        solve(&exponential(), stabilized::RKG1, &adaptive).unwrap(),
        solve(&exponential(), stabilized::RKG2, &adaptive).unwrap(),
        solve(&exponential(), stabilized::RKL1, &adaptive).unwrap(),
        solve(&exponential(), stabilized::RKL2, &adaptive).unwrap(),
        solve(&exponential(), stabilized::RKMC2, &adaptive).unwrap(),
        solve(&exponential(), stabilized::ROCK2, &adaptive).unwrap(),
        solve(&exponential(), stabilized::ROCK4, &adaptive).unwrap(),
        solve(&exponential(), stabilized::SERK2, &adaptive).unwrap(),
        solve(&exponential(), stabilized::IRKC, &adaptive).unwrap(),
    ];

    for solution in fixed_solutions.iter().chain(adaptive_solutions.iter()) {
        assert_eq!(solution.dimension(), 1);
        assert!((solution.last_state()[0] - (-1.0_f64).exp()).abs() < 1.0e-3);
        assert!(solution.stats().accepted_steps > 0);
    }
}

#[test]
fn explicit_stabilized_names_remain_bounded_on_a_stiff_decay_slice() {
    let options = fixed_options();
    let solutions = [
        solve(&stiff_decay(), stabilized::ESERK4, &options).unwrap(),
        solve(&stiff_decay(), stabilized::ESERK5, &options).unwrap(),
        solve(&stiff_decay(), stabilized::TSRKC2, &options).unwrap(),
        solve(&stiff_decay(), stabilized::TSRKC3, &options).unwrap(),
    ];

    for solution in solutions {
        assert!(solution.last_state()[0].is_finite());
        assert!(solution.last_state()[0].abs() < 1.0e-6);
    }
}

#[test]
fn adaptive_stabilized_names_are_stable_on_a_stiff_decay_slice() {
    let options = SolveOptions {
        initial_step: Some(0.05),
        max_step: 0.05,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let solutions = [
        solve(&stiff_decay(), stabilized::RKC, &options).unwrap(),
        solve(&stiff_decay(), stabilized::RKG1, &options).unwrap(),
        solve(&stiff_decay(), stabilized::RKG2, &options).unwrap(),
        solve(&stiff_decay(), stabilized::RKL1, &options).unwrap(),
        solve(&stiff_decay(), stabilized::RKL2, &options).unwrap(),
        solve(&stiff_decay(), stabilized::RKMC2, &options).unwrap(),
        solve(&stiff_decay(), stabilized::ROCK2, &options).unwrap(),
        solve(&stiff_decay(), stabilized::ROCK4, &options).unwrap(),
        solve(&stiff_decay(), stabilized::SERK2, &options).unwrap(),
        solve(&stiff_decay(), stabilized::IRKC, &options).unwrap(),
    ];

    for solution in solutions {
        assert!(solution.last_state()[0].is_finite());
        assert!(solution.last_state()[0].abs() < 1.0e-4);
    }
}
