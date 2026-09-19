use super::algorithms::{AdaptiveRadau, GaussLegendre, RadauIIA3, RadauIIA5, RadauIIA9};
use super::tableau::{Family, Tableau};
use crate::{OdeProblem, SaveMode, SolveOptions, solve};

#[test]
fn generated_tableaus_have_collocation_moments() {
    for (family, stages) in [
        (Family::Radau, 2),
        (Family::Radau, 3),
        (Family::Radau, 5),
        (Family::Radau, 7),
        (Family::Gauss, 2),
    ] {
        let tableau = Tableau::generate(family, stages);
        assert!((tableau.b.iter().sum::<f64>() - 1.0).abs() < 2.0e-12);
        for i in 0..stages {
            assert!(
                (tableau.a[i * stages..(i + 1) * stages].iter().sum::<f64>() - tableau.c[i]).abs()
                    < 2.0e-12
            );
        }
    }
}

#[test]
fn fixed_radau_methods_integrate_stiff_decay() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -40.0 * u[0],
        vec![1.0],
        (0.0, 0.2),
        (),
    );
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.01),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    for value in [
        solve(&problem, RadauIIA3, &options).unwrap().last_state()[0],
        solve(&problem, RadauIIA5, &options).unwrap().last_state()[0],
        solve(&problem, RadauIIA9, &options).unwrap().last_state()[0],
    ] {
        assert!((value - (-8.0_f64).exp()).abs() < 3.0e-6, "{value}");
    }
}

#[test]
fn adaptive_firk_variants_honor_tolerances() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
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
    for value in [
        solve(&problem, AdaptiveRadau::default(), &options)
            .unwrap()
            .last_state()[0],
        solve(&problem, GaussLegendre::default(), &options)
            .unwrap()
            .last_state()[0],
    ] {
        assert!((value - std::f64::consts::E).abs() < 2.0e-7, "{value}");
    }
}
