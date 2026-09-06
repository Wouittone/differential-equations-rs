use diffeq::ndarray::{ArrayViewD, ArrayViewMutD, arr0, array};
use diffeq::tableau::{LowStorageRungeKuttaLayout, define_low_storage_rk_from_file};
use diffeq::{OdeProblem, SaveMode, SolveOptions, solve};
use differential_equations as diffeq;

define_low_storage_rk_from_file!(
    pub FileLowStorage,
    "tests/resources/file_low_storage.json",
    crate = diffeq
);

#[test]
fn renamed_dependency_can_define_and_inspect_a_low_storage_solver() {
    let tableau = FileLowStorage.tableau().unwrap();
    assert_eq!(tableau.name(), "FileLowStorage");
    assert_eq!(tableau.order(), 2);
    let LowStorageRungeKuttaLayout::TwoN(coefficients) = tableau.layout() else {
        panic!("expected a 2N recurrence")
    };
    assert_eq!(coefficients.a(), &[-0.5]);
    assert_eq!(coefficients.b(), &[0.5, 1.0]);
}

#[test]
fn file_defined_solver_preserves_scalar_vector_and_matrix_shapes() {
    for initial in [
        arr0(1.0).into_dyn(),
        array![1.0, 2.0].into_dyn(),
        array![[1.0, 2.0], [3.0, 4.0]].into_dyn(),
    ] {
        let shape = initial.shape().to_vec();
        let problem = OdeProblem::from_array(
            |mut du: ArrayViewMutD<'_, f64>, u: ArrayViewD<'_, f64>, _: &(), _| {
                du.zip_mut_with(&u, |du, u| *du = -*u);
            },
            initial.clone(),
            (0.0, 0.2),
            (),
        );
        let options = SolveOptions::new()
            .with_adaptive(false)
            .with_initial_step(0.001)
            .with_save(SaveMode::Endpoints);
        let solution = solve(&problem, FileLowStorage, &options).unwrap();
        assert_eq!(solution.state_shape(), shape);
        for (actual, initial) in solution.last_state().iter().zip(initial) {
            assert!((actual - initial * (-0.2_f64).exp()).abs() < 2.0e-7);
        }
    }
}
