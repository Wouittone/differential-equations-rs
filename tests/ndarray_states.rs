use differential_equations::SplitOdeProblem;
use differential_equations::ndarray::{
    ArrayView0, ArrayView1, ArrayView2, ArrayViewD, ArrayViewMut0, ArrayViewMut1, ArrayViewMut2,
    ArrayViewMutD, arr0, array,
};
use differential_equations::solvers::explicit::Tsit5;
use differential_equations::solvers::explicit::split_euler::{SplitEuler, solve_split};
use differential_equations::solvers::rosenbrock::Rodas5P;
use differential_equations::{
    CallbackAction, OdeAlgorithm, OdeProblem, SaveMode, SolveOptions, solve,
};

fn options() -> SolveOptions {
    SolveOptions::new()
        .with_tolerances(1.0e-10, 1.0e-10)
        .with_save(SaveMode::Endpoints)
        .with_dense_output(true)
}

fn assert_decay_is_shape_invariant<A>(algorithm: impl Fn() -> A)
where
    A: OdeAlgorithm,
{
    let scalar = OdeProblem::from_array(
        |mut derivative: ArrayViewMut0<'_, f64>, state: ArrayView0<'_, f64>, _: &(), _: f64| {
            derivative[[]] = -state[[]];
        },
        arr0(1.0),
        (0.0, 1.0),
        (),
    );
    let vector = OdeProblem::from_array(
        |mut derivative: ArrayViewMut1<'_, f64>, state: ArrayView1<'_, f64>, _: &(), _: f64| {
            derivative.zip_mut_with(&state, |derivative, state| *derivative = -*state);
        },
        array![1.0, 1.0],
        (0.0, 1.0),
        (),
    );
    let matrix = OdeProblem::from_array(
        |mut derivative: ArrayViewMut2<'_, f64>, state: ArrayView2<'_, f64>, _: &(), _: f64| {
            derivative.zip_mut_with(&state, |derivative, state| *derivative = -*state);
        },
        array![[1.0, 1.0], [1.0, 1.0]],
        (0.0, 1.0),
        (),
    );

    let scalar_solution = solve(&scalar, algorithm(), &options()).unwrap();
    let vector_solution = solve(&vector, algorithm(), &options()).unwrap();
    let matrix_solution = solve(&matrix, algorithm(), &options()).unwrap();
    let scalar_endpoint = scalar_solution.last_state()[0];

    assert!(scalar_solution.state_shape().is_empty());
    assert_eq!(vector_solution.state_shape(), &[2]);
    assert_eq!(matrix_solution.state_shape(), &[2, 2]);
    for endpoint in vector_solution
        .last_state()
        .iter()
        .chain(matrix_solution.last_state())
    {
        assert!((*endpoint - scalar_endpoint).abs() < 1.0e-12);
    }
    assert!((scalar_endpoint - (-1.0_f64).exp()).abs() < 1.0e-9);
}

#[test]
fn one_decay_ode_is_shape_invariant_for_explicit_and_stiff_solvers() {
    assert_decay_is_shape_invariant(|| Tsit5);
    assert_decay_is_shape_invariant(|| Rodas5P);
}

#[test]
fn scalar_array_state_retains_zero_dimensional_shape() {
    let problem = OdeProblem::from_array(
        |mut derivative: ArrayViewMut0<'_, f64>, state: ArrayView0<'_, f64>, _: &(), _: f64| {
            derivative[[]] = -state[[]];
        },
        arr0(1.0),
        (0.0, 1.0),
        (),
    );

    assert!(problem.state_shape().is_empty());
    assert!(problem.initial_state_array().shape().is_empty());

    let solution = solve(&problem, Tsit5, &options()).unwrap();
    assert!(solution.state_shape().is_empty());
    let endpoint = solution.last_state_array();
    assert!((*endpoint.first().unwrap() - (-1.0_f64).exp()).abs() < 1.0e-9);

    let midpoint = solution.interpolate_array(0.5).unwrap();
    assert!((*midpoint.first().unwrap() - (-0.5_f64).exp()).abs() < 1.0e-8);
}

#[test]
fn vector_array_state_preserves_vector_indexing() {
    let problem = OdeProblem::from_array(
        |mut derivative: ArrayViewMut1<'_, f64>, state: ArrayView1<'_, f64>, _: &(), _: f64| {
            derivative.assign(&state);
        },
        array![1.0, 2.0],
        (0.0, 0.25),
        (),
    );

    assert_eq!(problem.state_shape(), &[2]);
    assert_eq!(problem.initial_state_array(), array![1.0, 2.0].into_dyn());

    let solution = solve(&problem, Tsit5, &options()).unwrap();
    assert_eq!(solution.state_shape(), &[2]);
    assert_eq!(solution.last_state_array().len(), 2);
}

#[test]
fn matrix_array_state_preserves_rows_and_columns() {
    let problem = OdeProblem::from_array(
        |mut derivative: ArrayViewMut2<'_, f64>, state: ArrayView2<'_, f64>, _: &(), _: f64| {
            derivative.zip_mut_with(&state, |derivative, state| *derivative = -*state);
        },
        array![[1.0, 2.0], [3.0, 4.0]],
        (0.0, 1.0),
        (),
    );

    assert_eq!(problem.state_shape(), &[2, 2]);
    assert_eq!(problem.initial_state_array()[[1, 0]], 3.0);

    let solution = solve(&problem, Tsit5, &options()).unwrap();
    assert_eq!(solution.state_shape(), &[2, 2]);
    let endpoint = solution.last_state_array();
    assert_eq!(endpoint.shape(), &[2, 2]);
    assert!((endpoint[[1, 0]] - 3.0 * (-1.0_f64).exp()).abs() < 1.0e-9);

    let midpoint = solution.interpolate_array(0.5).unwrap();
    assert_eq!(midpoint.shape(), &[2, 2]);
    assert!((midpoint[[0, 1]] - 2.0 * (-0.5_f64).exp()).abs() < 1.0e-8);
}

#[test]
fn ndarray_jacobians_and_callbacks_keep_matrix_indexing() {
    let problem = OdeProblem::from_array(
        |mut derivative: ArrayViewMut2<'_, f64>, state: ArrayView2<'_, f64>, _: &(), _: f64| {
            derivative.zip_mut_with(&state, |derivative, state| *derivative = -*state);
        },
        array![[1.0, 2.0], [3.0, 4.0]],
        (0.0, 0.1),
        (),
    )
    .with_array_jacobian(
        |mut jacobian: ArrayViewMut2<'_, f64>, _: ArrayViewD<'_, f64>, _: &(), _: f64| {
            jacobian.fill(0.0);
            jacobian.diag_mut().fill(-1.0);
        },
    )
    .with_array_discrete_callback(
        |_: ArrayViewD<'_, f64>, _: &(), time| time == 0.0,
        |mut state: ArrayViewMutD<'_, f64>, _: &(), _: f64| {
            state[[0, 1]] = 7.0;
            CallbackAction::Continue
        },
    );

    let solution = solve(&problem, Rodas5P, &options()).unwrap();
    assert_eq!(solution.state_shape(), &[2, 2]);
    assert!(solution.last_state_array()[[0, 1]] < 7.0);
}

#[test]
fn split_array_problems_preserve_matrix_shape() {
    let problem = SplitOdeProblem::from_array(
        |mut derivative: ArrayViewMut2<'_, f64>, state: ArrayView2<'_, f64>, _: &(), _: f64| {
            derivative.zip_mut_with(&state, |derivative, state| *derivative = -0.5 * *state);
        },
        |mut derivative: ArrayViewMut2<'_, f64>, state: ArrayView2<'_, f64>, _: &(), _: f64| {
            derivative.zip_mut_with(&state, |derivative, state| *derivative = -0.5 * *state);
        },
        array![[1.0, 2.0], [3.0, 4.0]],
        (0.0, 0.1),
        (),
    );
    let options = SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(0.01)
        .with_save(SaveMode::Endpoints);

    let solution = solve_split(&problem, SplitEuler, &options).unwrap();
    assert_eq!(solution.state_shape(), &[2, 2]);
    assert_eq!(solution.last_state_array().shape(), &[2, 2]);
}
