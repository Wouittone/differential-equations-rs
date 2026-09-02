use differential_equations::solvers::multistep::{Fbdf, Qbdf, Qbdf1, Qbdf2, Qndf, Qndf1, Qndf2};
use differential_equations::{OdeAlgorithm, OdeProblem, SolveOptions, solve};

fn check_trajectory<A: OdeAlgorithm + Copy>(algorithm: A, expected: [[f64; 2]; 2]) {
    for (adaptive, expected) in [false, true].into_iter().zip(expected) {
        for (span, expected) in [(0.0, 0.2), (0.2, 0.0)].into_iter().zip(expected) {
            let problem = OdeProblem::new(
                |du: &mut [f64], u: &[f64], _: &(), t: f64| du[0] = t.sin() - u[0] * u[0],
                [0.4],
                span,
                (),
            );
            let options = SolveOptions::new()
                .with_adaptive(adaptive)
                .with_initial_step(0.01)
                .with_tolerances(1e-6, 1e-6);
            let solution = solve(&problem, algorithm, &options).unwrap();
            assert!(
                (solution.last_state()[0] - expected).abs() < 1e-10,
                "{} adaptive={adaptive}, span={span:?}: {} != {expected}",
                std::any::type_name::<A>(),
                solution.last_state()[0]
            );
        }
    }
}

#[test]
fn named_bdf_trajectories_match_before_resource_migration() {
    // Recorded from the original Rust coefficient banks before migration.
    check_trajectory(
        Qndf,
        [
            [0.390_178_681_219_978, 0.4133200532737164],
            [0.389_318_947_710_582, 0.4125216097404597],
        ],
    );
    check_trajectory(
        Qbdf,
        [
            [0.390_273_864_765_58, 0.4136531751733465],
            [0.3893216344314741, 0.4125210105461292],
        ],
    );
    check_trajectory(
        Fbdf,
        [
            [0.390_273_864_765_58, 0.4136531751733464],
            [0.3893219452545725, 0.4125167399943631],
        ],
    );
    check_trajectory(
        Qndf1,
        [
            [0.3899424985623454, 0.4132569703074265],
            [0.389_476_737_353_14, 0.4126978338640954],
        ],
    );
    check_trajectory(
        Qbdf1,
        [
            [0.3902738647655799, 0.4136531751733464],
            [0.3895040435884016, 0.412_730_230_052_365],
        ],
    );
    check_trajectory(
        Qndf2,
        [
            [0.3894396542027261, 0.4126588504651569],
            [0.3894130416401093, 0.4126225679261209],
        ],
    );
    check_trajectory(
        Qbdf2,
        [
            [0.3894373427338242, 0.4126616023734037],
            [0.3894096510668247, 0.4126184850201107],
        ],
    );
}

#[test]
fn all_orders_preserve_legacy_coefficient_bits_and_share_storage() {
    let fingerprints = [
        0x7188c2b3808da131_u64,
        0xb3650b15e6d3dfa0,
        0xf4fe55cf4011be35,
        0x8ee4493491eb22bf,
        0xafbceb4c5ec22ab0,
    ];
    for (order, fingerprint) in (1..=5).zip(fingerprints) {
        let tableau = Qndf.tableau(order).unwrap();
        assert_eq!(tableau.order(), order);
        assert_eq!(tableau.steps(), order);
        assert!(std::ptr::eq(tableau, Qbdf.tableau(order).unwrap()));
        assert!(std::ptr::eq(tableau, Fbdf.tableau(order).unwrap()));
        let mut hash = 0xcbf29ce484222325_u64;
        for value in tableau.alpha().iter().copied().chain(tableau.ndf_kappa()) {
            for byte in value.to_bits().to_le_bytes() {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
            }
        }
        assert_eq!(hash, fingerprint, "order {order}");
    }
    for (order, fixed_ndf, fixed_bdf) in [
        (1, Qndf1.tableau().unwrap(), Qbdf1.tableau().unwrap()),
        (2, Qndf2.tableau().unwrap(), Qbdf2.tableau().unwrap()),
    ] {
        assert!(std::ptr::eq(fixed_ndf, fixed_bdf));
        assert!(std::ptr::eq(fixed_ndf, Qndf.tableau(order).unwrap()));
    }
    for order in [0, 6, usize::MAX] {
        for result in [
            Qndf.tableau(order),
            Qbdf.tableau(order),
            Fbdf.tableau(order),
        ] {
            assert_eq!(
                result,
                Err(differential_equations::SolveError::InvalidMultistepOrder)
            );
        }
    }
}

fn check_shapes<A: OdeAlgorithm + Copy>(algorithm: A) {
    use differential_equations::ndarray::{ArrayViewD, ArrayViewMutD, arr0, array};
    for adaptive in [false, true] {
        let options = SolveOptions::new()
            .with_adaptive(adaptive)
            .with_initial_step(0.002)
            .with_tolerances(1e-7, 1e-7)
            .with_dense_output(true);
        for span in [(0.0, 0.1), (0.1, 0.0)] {
            for initial in [
                arr0(1.0).into_dyn(),
                array![1.0, 2.0, 3.0].into_dyn(),
                array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn(),
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
                assert_eq!(actual.last_state_array().shape(), initial.shape());
                assert_eq!(
                    actual.interpolate_array(0.051).unwrap().shape(),
                    initial.shape()
                );
                for (value, initial) in actual.last_state().iter().zip(initial.iter()) {
                    // Fixed stepping retains first-order startup error. Check
                    // it per unit initial magnitude so larger matrix entries
                    // face the same accuracy requirement as the scalar case.
                    assert!(
                        (value / initial - (span.0 - span.1).exp()).abs() < 5e-4,
                        "{} adaptive={adaptive}, span={span:?}, initial={initial}, actual={value}",
                        std::any::type_name::<A>()
                    );
                }
            }
        }
    }
}

#[test]
fn one_decay_problem_supports_all_state_shapes_for_every_bdf_method() {
    check_shapes(Qndf);
    check_shapes(Qbdf);
    check_shapes(Fbdf);
    check_shapes(Qndf1);
    check_shapes(Qbdf1);
    check_shapes(Qndf2);
    check_shapes(Qbdf2);
}
