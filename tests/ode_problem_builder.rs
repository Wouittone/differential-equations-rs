use differential_equations::ndarray::{
    ArrayView0, ArrayView1, ArrayView2, ArrayViewMut0, ArrayViewMut1, ArrayViewMut2, arr0, array,
};
use differential_equations::solvers::explicit::Tsit5;
use differential_equations::{OdeProblem, SolveOptions, solve};

#[test]
fn builder_preserves_scalar_vector_and_matrix_shapes() {
    let scalar = OdeProblem::builder()
        .initial_state(arr0(1.0))
        .parameters(-1.0)
        .time_span((0.0, 1.0))
        .build_with_in_place_rhs(
            |mut du: ArrayViewMut0<'_, f64>, u: ArrayView0<'_, f64>, rate: &f64, _time| {
                du[()] = rate * u[()];
            },
        );
    let vector = OdeProblem::builder()
        .time_span((0.0, 1.0))
        .initial_state(array![1.0, 2.0])
        .parameters(-1.0)
        .build_with_in_place_rhs(
            |mut du: ArrayViewMut1<'_, f64>, u: ArrayView1<'_, f64>, rate: &f64, _time| {
                du.zip_mut_with(&u, |du, u| *du = rate * *u);
            },
        );
    let matrix = OdeProblem::builder()
        .parameters(-1.0)
        .initial_state(array![[1.0, 2.0], [3.0, 4.0]])
        .time_span((0.0, 1.0))
        .build_with_in_place_rhs(
            |mut du: ArrayViewMut2<'_, f64>, u: ArrayView2<'_, f64>, rate: &f64, _time| {
                du.zip_mut_with(&u, |du, u| *du = rate * *u);
            },
        );

    let options = SolveOptions::default();
    let scalar_solution = solve(&scalar, Tsit5, &options).unwrap();
    let vector_solution = solve(&vector, Tsit5, &options).unwrap();
    let matrix_solution = solve(&matrix, Tsit5, &options).unwrap();

    assert!(scalar_solution.last_state_array().shape().is_empty());
    assert_eq!(vector_solution.last_state_array().shape(), &[2]);
    assert_eq!(matrix_solution.last_state_array().shape(), &[2, 2]);
    for value in scalar_solution
        .last_state()
        .iter()
        .chain(vector_solution.last_state())
        .chain(matrix_solution.last_state())
    {
        assert!(value.is_finite());
    }
}

#[test]
fn builder_supports_out_of_place_rhs_and_unit_parameters_by_default() {
    let problem = OdeProblem::builder()
        .initial_state(array![[1.0, 2.0], [3.0, 4.0]])
        .time_span((0.0, 1.0))
        .build_with_out_of_place_rhs(|u: ArrayView2<'_, f64>, _: &(), _time| -&u);

    let solution = solve(&problem, Tsit5, &SolveOptions::default()).unwrap();
    assert_eq!(solution.last_state_array().shape(), &[2, 2]);
}
