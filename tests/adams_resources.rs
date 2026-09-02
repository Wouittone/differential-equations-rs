use differential_equations::solvers::multistep::{Ab3, Ab4, Ab5, Abm32, Abm43, Abm54};
use differential_equations::{OdeAlgorithm, OdeProblem, SolveOptions, solve};

fn check_trajectory<A: OdeAlgorithm + Copy>(algorithm: A, expected: [f64; 2]) {
    for (span, expected) in [(0.0, 0.5), (0.5, 0.0)].into_iter().zip(expected) {
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), t: f64| du[0] = t.sin() - u[0] * u[0],
            [0.4],
            span,
            (),
        );
        let options = SolveOptions::new()
            .with_adaptive(false)
            .with_initial_step(0.025);
        let solution = solve(&problem, algorithm, &options).unwrap();
        assert!((solution.last_state()[0] - expected).abs() < 1e-13);
    }
}

#[test]
fn named_adams_trajectories_match_before_resource_migration() {
    // Recorded with the original coefficient banks and startup formulas.
    check_trajectory(Ab3, [0.44144781514396936, 0.339_586_492_561_600_9]);
    check_trajectory(Ab4, [0.44144357695869724, 0.339_587_101_577_082_4]);
    check_trajectory(Ab5, [0.44144339752045514, 0.33958724207281094]);
    check_trajectory(Abm32, [0.441_444_615_587_718_2, 0.33958489532427233]);
    check_trajectory(Abm43, [0.44144340983552666, 0.33958728544220873]);
    check_trajectory(Abm54, [0.44144342316563934, 0.339_587_272_868_976_1]);
}

#[test]
fn fixed_and_multirate_methods_share_the_same_formula_storage() {
    use differential_equations::SolveError;
    use differential_equations::solvers::multirate::MRAB;
    for (order, predictor, corrected_predictor, corrector) in [
        (
            3,
            Ab3.tableau().unwrap(),
            Abm32.predictor_tableau().unwrap(),
            Abm32.tableau().unwrap(),
        ),
        (
            4,
            Ab4.tableau().unwrap(),
            Abm43.predictor_tableau().unwrap(),
            Abm43.tableau().unwrap(),
        ),
        (
            5,
            Ab5.tableau().unwrap(),
            Abm54.predictor_tableau().unwrap(),
            Abm54.tableau().unwrap(),
        ),
    ] {
        assert!(std::ptr::eq(predictor, corrected_predictor));
        assert!(std::ptr::eq(
            predictor,
            MRAB::new(order, 8).tableau().unwrap()
        ));
        assert_eq!(corrector.order(), order);
        assert_eq!(corrector.beta().len(), order);
        assert_eq!(predictor.steps(), order);
        assert_eq!(corrector.steps(), order - 1);
        assert!(predictor.is_explicit());
        assert!(!corrector.is_explicit());
    }
    for order in [0, 6, usize::MAX] {
        assert_eq!(
            MRAB::new(order, 8).tableau(),
            Err(SolveError::InvalidMultistepOrder)
        );
    }
}

fn check_shapes<A: OdeAlgorithm + Copy>(algorithm: A) {
    use differential_equations::ndarray::{ArrayViewD, ArrayViewMutD, arr0, array};
    let options = SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(0.01)
        .with_dense_output(true);
    for span in [(0.0, 0.2), (0.2, 0.0)] {
        for initial in [
            arr0(1.0).into_dyn(),
            array![1.0, 2.0].into_dyn(),
            array![[1.0, 2.0], [3.0, 4.0]].into_dyn(),
        ] {
            let problem = OdeProblem::from_array(
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
            let actual = solve(&problem, algorithm, &options).unwrap();
            let expected = solve(&flat, algorithm, &options).unwrap();
            assert_eq!(actual.last_state(), expected.last_state());
            assert_eq!(actual.stats(), expected.stats());
            assert_eq!(actual.state_shape(), initial.shape());
            assert_eq!(actual.last_state_array().shape(), initial.shape());
            assert_eq!(
                actual.interpolate_array(0.105).unwrap().shape(),
                initial.shape()
            );
            for (value, initial) in actual.last_state().iter().zip(initial.iter()) {
                assert!((value - initial * (span.0 - span.1).exp()).abs() < 1e-4);
            }
        }
    }
}

#[test]
fn one_decay_problem_preserves_every_state_shape_for_all_fixed_adams_methods() {
    check_shapes(Ab3);
    check_shapes(Ab4);
    check_shapes(Ab5);
    check_shapes(Abm32);
    check_shapes(Abm43);
    check_shapes(Abm54);
}

#[test]
fn every_multirate_adams_order_preserves_scalar_vector_and_matrix_states() {
    use differential_equations::SplitOdeProblem;
    use differential_equations::ndarray::{ArrayViewD, ArrayViewMutD, arr0, array};
    use differential_equations::solvers::explicit::solve_split;
    use differential_equations::solvers::multirate::MRAB;
    let options = SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(0.01);
    for order in 1..=5 {
        for span in [(0.0, 0.2), (0.2, 0.0)] {
            for initial in [
                arr0(1.0).into_dyn(),
                array![1.0, 2.0].into_dyn(),
                array![[1.0, 2.0], [3.0, 4.0]].into_dyn(),
            ] {
                let problem = SplitOdeProblem::from_array(
                    |mut du: ArrayViewMutD<'_, f64>, u: ArrayViewD<'_, f64>, _: &(), _| {
                        du.zip_mut_with(&u, |du, u| *du = -*u)
                    },
                    |mut du: ArrayViewMutD<'_, f64>, _: ArrayViewD<'_, f64>, _: &(), _| {
                        du.fill(0.0)
                    },
                    initial.clone(),
                    span,
                    (),
                );
                let result = solve_split(&problem, MRAB::new(order, 8), &options).unwrap();
                assert_eq!(result.last_state_array().shape(), initial.shape());
                for (value, initial) in result.last_state().iter().zip(initial.iter()) {
                    assert!((value - initial * (span.0 - span.1).exp()).abs() < 1e-3);
                }
            }
        }
    }
}

mod downstream {
    use differential_equations::tableau::define_multistep_tableau_from_file;
    define_multistep_tableau_from_file!(pub FORMULA, "Ab2", "src/tableau/resources/multistep/ab2.json");
}

mod renamed {
    use diffeq::tableau::define_multistep_tableau_from_file;
    use differential_equations as diffeq;
    define_multistep_tableau_from_file!(pub FORMULA, "Ab2", "src/tableau/resources/multistep/ab2.json", crate = diffeq);
}

#[test]
fn downstream_macros_support_original_and_renamed_dependencies() {
    use differential_equations::solvers::multirate::MRAB;
    use differential_equations::tableau::load_tableau;
    assert_eq!(
        load_tableau(&downstream::FORMULA).unwrap(),
        MRAB::new(2, 8).tableau().unwrap()
    );
    assert_eq!(
        load_tableau(&renamed::FORMULA).unwrap(),
        load_tableau(&downstream::FORMULA).unwrap()
    );
}
