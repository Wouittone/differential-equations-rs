use std::f64::consts::E;

use super::{
    Alshina2, Alshina3, Alshina6, Bs3, Dp5, Euler, Heun, Midpoint, Ralston, Ralston4, Rk4, Rkm,
    SspRk22, SspRk33, SspRk43,
};
use super::{Bs5, OwrenZen3, OwrenZen4, OwrenZen5};
use crate::{OdeProblem, SaveMode, SolveError, SolveOptions, solve};

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

crate::tableau::define_explicit_rk_from_file!(
    DualEstimatorHeun,
    "tests/resources/dual_estimator_heun.json",
    crate = crate
);

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
fn secondary_error_estimator_controls_adaptation() {
    let options = SolveOptions {
        absolute_tolerance: 1.0e-9,
        relative_tolerance: 1.0e-9,
        initial_step: Some(1.0),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };

    let solution = solve(&exponential(), DualEstimatorHeun, &options).unwrap();

    assert!(solution.stats().rejected_steps > 0);
}

#[test]
fn fixed_methods_have_expected_convergence() {
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.001),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let euler_error = (solve(&exponential(), Euler, &options).unwrap().last_state()[0] - E).abs();
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
