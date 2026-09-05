use diffeq::ndarray::{Array, ArrayD, ArrayViewD, arr0, array};
use diffeq::solvers::second_order::{SecondOrderOdeProblem, solve_second_order};
use diffeq::tableau::{define_irkn_tableau_from_file, define_rkn_from_file, load_tableau};
use diffeq::{SaveMode, SolveOptions};
use differential_equations as diffeq;

define_rkn_from_file!(pub FileRkn, "tests/resources/file_rkn.json", crate = diffeq);
define_irkn_tableau_from_file!(
    pub FILE_IRKN_TABLEAU,
    "FileIrkn",
    "tests/resources/file_irkn.json",
    crate = diffeq
);

fn options() -> SolveOptions {
    SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(0.002)
        .with_save(SaveMode::Endpoints)
}

fn oscillator(
    initial: ArrayD<f64>,
) -> SecondOrderOdeProblem<impl diffeq::solvers::second_order::SecondOrderFunction<()>, ()> {
    SecondOrderOdeProblem::from_array_out_of_place(
        |_: ArrayViewD<'_, f64>, position: ArrayViewD<'_, f64>, _: &(), _| -&position,
        Array::zeros(initial.raw_dim()),
        initial,
        (0.0, 0.2),
        (),
    )
    .unwrap()
}

#[test]
fn one_resource_defined_solver_preserves_scalar_vector_and_matrix_states() {
    for initial in [
        arr0(1.0).into_dyn(),
        array![1.0, 2.0].into_dyn(),
        array![[1.0, 2.0], [3.0, 4.0]].into_dyn(),
    ] {
        let shape = initial.shape().to_vec();
        let solution =
            solve_second_order(&oscillator(initial.clone()), FileRkn, &options()).unwrap();
        assert_eq!(solution.state_shape(), shape);
        for (actual, initial) in solution.last_position().iter().zip(initial) {
            assert!((actual - initial * 0.2_f64.cos()).abs() < 5.0e-5);
        }
    }
}

#[test]
fn downstream_and_renamed_paths_expose_typed_tableaus() {
    let rkn = FileRkn.tableau().unwrap();
    assert_eq!(rkn.name(), "FileRkn");
    assert_eq!(rkn.a()[1], [0.125, 0.0]);

    let irkn = load_tableau(&FILE_IRKN_TABLEAU).unwrap();
    assert_eq!(irkn.name(), "FileIrkn");
    assert_eq!(irkn.a(), &[0.125]);
}
