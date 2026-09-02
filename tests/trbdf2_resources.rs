use differential_equations::solvers::multistep::Trbdf2;
use differential_equations::{OdeProblem, SolveOptions, solve};

#[test]
fn nonlinear_trajectories_match_before_resource_migration() {
    // Recorded before replacing the embedded Rust coefficients.
    let expected = [
        [
            [0.4496186965040221, 0.20349829671594857],
            [0.32694453058575684, 0.2410568846004235],
        ],
        [
            [0.4496257147403323, 0.2034962111496054],
            [0.3269361755086722, 0.2410622561301531],
        ],
    ];
    for (adaptive, expected) in [false, true].into_iter().zip(expected) {
        for (span, expected) in [(0.0, 0.5), (0.5, 0.0)].into_iter().zip(expected) {
            let problem = OdeProblem::new(
                |du: &mut [f64], u: &[f64], _: &(), t: f64| {
                    du[0] = t.sin() - u[0] * u[0] + 0.1 * u[1];
                    du[1] = -2.0 * u[1] + u[0];
                },
                [0.4, 0.2],
                span,
                (),
            );
            let options = SolveOptions::new()
                .with_adaptive(adaptive)
                .with_initial_step(0.025)
                .with_tolerances(1e-7, 1e-7);
            let solution = solve(&problem, Trbdf2, &options).unwrap();
            for (actual, expected) in solution.last_state().iter().zip(expected) {
                assert!(
                    (actual - expected).abs() < 1e-11,
                    "adaptive={adaptive}, span={span:?}: {actual} != {expected}"
                );
            }
        }
    }
}

#[test]
fn tableau_preserves_legacy_bits_and_the_companions_order_conditions() {
    let tableau = Trbdf2.tableau().unwrap();
    assert!(std::ptr::eq(tableau, Trbdf2.tableau().unwrap()));
    assert_eq!(tableau.order(), 2);
    assert_eq!(tableau.embedded_order(), Some(3));
    assert!(tableau.fsal());
    let error = tableau.error().unwrap();
    let predictor = tableau.stage_predictor(2).unwrap();
    let coefficients = [
        tableau.c()[1],
        tableau.a()[1][1],
        tableau.a()[2][0],
        error[0],
        error[1],
        error[2],
        predictor[0],
        predictor[1],
    ];
    let mut hash = 0xcbf29ce484222325_u64;
    for value in coefficients {
        for byte in value.to_bits().to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
    }
    assert_eq!(hash, 0x5d69b6a7aafc2307);
    for (row, node) in tableau.a().iter().zip(tableau.c()) {
        assert!((row.iter().sum::<f64>() - node).abs() < 1e-14);
    }
    let b_hat: Vec<_> = tableau.b().iter().zip(error).map(|(b, e)| b + e).collect();
    // TR-BDF2 keeps SciML's direct error convention b_hat - b.
    for (degree, expected) in [(0, 1.0), (1, 0.5), (2, 1.0 / 3.0)] {
        let moment: f64 = b_hat
            .iter()
            .zip(tableau.c())
            .map(|(b, c)| b * c.powi(degree))
            .sum();
        assert!((moment - expected).abs() < 1e-14);
    }
    let tree: f64 = b_hat
        .iter()
        .zip(tableau.a())
        .map(|(b, row)| b * row.iter().zip(tableau.c()).map(|(a, c)| a * c).sum::<f64>())
        .sum();
    assert!((tree - 1.0 / 6.0).abs() < 1e-14);
}

#[test]
fn the_same_decay_problem_preserves_scalar_vector_and_matrix_shapes() {
    use differential_equations::ndarray::{ArrayViewD, ArrayViewMutD, arr0, array};
    for adaptive in [false, true] {
        let options = SolveOptions::new()
            .with_adaptive(adaptive)
            .with_initial_step(0.01)
            .with_tolerances(1e-8, 1e-8)
            .with_dense_output(true);
        for span in [(0.0, 0.2), (0.2, 0.0)] {
            for initial in [
                arr0(1.0).into_dyn(),
                array![1.0, 2.0, 3.0].into_dyn(),
                array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn(),
            ] {
                let shaped = OdeProblem::from_array(
                    |mut du: ArrayViewMutD<'_, f64>, u: ArrayViewD<'_, f64>, _: &(), _| {
                        du.zip_mut_with(&u, |du, u| *du = -*u)
                    },
                    initial.clone(),
                    span,
                    (),
                );
                let flat = OdeProblem::new(
                    |du: &mut [f64], u: &[f64], _: &(), _| {
                        for (du, u) in du.iter_mut().zip(u) {
                            *du = -*u;
                        }
                    },
                    initial.iter().copied().collect::<Vec<_>>(),
                    span,
                    (),
                );
                let actual = solve(&shaped, Trbdf2, &options).unwrap();
                let expected = solve(&flat, Trbdf2, &options).unwrap();
                assert_eq!(actual.last_state(), expected.last_state());
                assert_eq!(actual.stats(), expected.stats());
                assert_eq!(actual.last_state_array().shape(), initial.shape());
                let sample = actual.interpolate_array(0.105).unwrap();
                assert_eq!(sample.shape(), initial.shape());
                for (value, initial) in actual.last_state().iter().zip(initial.iter()) {
                    assert!((value / initial - (span.0 - span.1).exp()).abs() < 1e-5);
                }
                for (value, initial) in sample.iter().zip(initial.iter()) {
                    assert!((value / initial - (span.0 - 0.105).exp()).abs() < 1e-5);
                }
            }
        }
    }
}
