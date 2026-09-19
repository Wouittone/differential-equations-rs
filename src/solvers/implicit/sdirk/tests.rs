use super::{ExtendedKind, Sdirk2, Sfsdirk4, extended_resource};
use crate::tableau::load_tableau;
use crate::{OdeProblem, SaveMode, SolveError, SolveOptions, solve};

#[test]
fn adaptive_decay_is_stable() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -20.0 * u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions {
        absolute_tolerance: 1.0e-8,
        relative_tolerance: 1.0e-8,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let solution = solve(&problem, Sdirk2, &options).unwrap();
    assert!((solution.last_state()[0] - (-20.0_f64).exp()).abs() < 2.0e-7);
    assert!(solution.stats().accepted_steps > 0);
    assert!(solution.stats().nonlinear_iterations > 0);
}

#[test]
fn fixed_step_has_second_order_convergence() {
    fn error(step: f64) -> f64 {
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
            vec![1.0],
            (0.0, 1.0),
            (),
        );
        let options = SolveOptions {
            adaptive: false,
            initial_step: Some(step),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        };
        (solve(&problem, Sdirk2, &options).unwrap().last_state()[0] - std::f64::consts::E).abs()
    }
    let coarse = error(0.1);
    let fine = error(0.05);
    assert!(coarse / fine > 3.5, "observed ratio {}", coarse / fine);
}

#[test]
fn extended_inventory_catalog_is_finite_and_dimensionally_valid() {
    let kinds = [
        ExtendedKind::Ars222,
        ExtendedKind::Ars232,
        ExtendedKind::Ars343,
        ExtendedKind::Ars443,
        ExtendedKind::Bhr553,
        ExtendedKind::Cfnlirk3,
        ExtendedKind::Esdirk325,
        ExtendedKind::Esdirk436,
        ExtendedKind::Esdirk437,
        ExtendedKind::Esdirk547,
        ExtendedKind::Esdirk54,
        ExtendedKind::Esdirk659,
        ExtendedKind::Hairer4,
        ExtendedKind::Hairer42,
        ExtendedKind::ImexSsp222,
        ExtendedKind::ImexSsp2322,
        ExtendedKind::ImexSsp3332,
        ExtendedKind::ImexSsp3433,
        ExtendedKind::KenCarp3,
        ExtendedKind::KenCarp4,
        ExtendedKind::KenCarp47,
        ExtendedKind::KenCarp5,
        ExtendedKind::KenCarp58,
        ExtendedKind::Kvaerno3,
        ExtendedKind::Kvaerno4,
        ExtendedKind::Kvaerno5,
        ExtendedKind::Sdirk22,
        ExtendedKind::Sfsdirk4,
        ExtendedKind::Sfsdirk5,
        ExtendedKind::Sfsdirk6,
        ExtendedKind::Sfsdirk7,
        ExtendedKind::Sfsdirk8,
        ExtendedKind::SspSdirk2,
    ];
    for kind in kinds {
        let tableau = load_tableau(extended_resource(kind)).unwrap();
        let has_embedded_estimator = !matches!(
            kind,
            ExtendedKind::Ars222
                | ExtendedKind::Ars232
                | ExtendedKind::Ars343
                | ExtendedKind::Ars443
                | ExtendedKind::Bhr553
                | ExtendedKind::Cfnlirk3
                | ExtendedKind::ImexSsp222
                | ExtendedKind::ImexSsp2322
                | ExtendedKind::ImexSsp3332
                | ExtendedKind::ImexSsp3433
                | ExtendedKind::Sdirk22
                | ExtendedKind::Sfsdirk4
                | ExtendedKind::Sfsdirk5
                | ExtendedKind::Sfsdirk6
                | ExtendedKind::Sfsdirk7
                | ExtendedKind::Sfsdirk8
                | ExtendedKind::SspSdirk2
        );
        assert!((2..=9).contains(&tableau.a().len()));
        assert_eq!(tableau.a().len(), tableau.c().len());
        assert_eq!(tableau.a().len(), tableau.b().len());
        assert_eq!(tableau.error().is_some(), has_embedded_estimator);
        if let Some(error) = tableau.error() {
            assert_eq!(tableau.a().len(), error.len());
        }
        assert!(
            tableau
                .a()
                .iter()
                .flat_map(|row| row.iter())
                .chain(tableau.c().iter())
                .chain(tableau.b().iter())
                .chain(tableau.error().into_iter().flatten())
                .all(|value| value.is_finite())
        );
    }
}

#[test]
fn methods_without_embedded_estimators_reject_adaptive_stepping() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -u[0],
        vec![1.0],
        (0.0, 0.1),
        (),
    );
    assert_eq!(
        solve(&problem, Sfsdirk4, &SolveOptions::default()).unwrap_err(),
        SolveError::AdaptiveStepUnsupported
    );
}

#[test]
fn extended_kernel_integrates_stiff_decay() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -30.0 * u[0],
        vec![1.0],
        (0.0, 0.1),
        (),
    );
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.01),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let solution = solve(&problem, Sfsdirk4, &options).unwrap();
    assert!(solution.last_state()[0].is_finite());
    assert!(solution.last_state()[0] > 0.0 && solution.last_state()[0] < 0.1);
    assert!(solution.stats().nonlinear_iterations > 0);
}
