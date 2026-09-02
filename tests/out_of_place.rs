use std::cell::Cell;

use differential_equations::callbacks::TerminateSteadyState;
use differential_equations::ndarray::{
    Array, ArrayView0, ArrayView1, ArrayView2, ArrayViewD, ArrayViewMutD, Axis, IxDyn,
    ShapeBuilder, arr0, array,
};
use differential_equations::solvers::explicit::{
    Anas5, CarpenterKennedy2N54, Euler, Frk65, SSPRKMSVS32, SplitEuler, SplitOdeAlgorithm,
    SspRk432, Tsit5, solve_split,
};
use differential_equations::solvers::exponential::NorsettEuler;
use differential_equations::solvers::extrapolation::ExtrapolationMidpointDeuflhard;
use differential_equations::solvers::implicit::{
    Cash4, ImplicitEuler, Kvaerno3, PDIRK44, RadauIIA5,
};
use differential_equations::solvers::linear::LinearExponential;
use differential_equations::solvers::multirate::MRIGARKERK22a;
use differential_equations::solvers::multistep::{
    AN5, Ab3, Abdf2, IMEXEuler, Mebdf2, QNDF, Qndf1, Qndf2, Trbdf2, Vcab3,
};
use differential_equations::solvers::rosenbrock::{AMF, Rodas5P, Rosenbrock23};
use differential_equations::solvers::stabilized::{IRKC, RKC};
use differential_equations::solvers::taylor::ExplicitTaylor2;
use differential_equations::{
    CallbackAction, OdeAlgorithm, OdeFunction, OdeProblem, SaveMode, SolveError, SolveOptions,
    SplitOdeProblem, solve,
};

fn options() -> SolveOptions {
    SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(0.02)
        .with_save(SaveMode::Endpoints)
}

fn assert_ordinary<A: OdeAlgorithm + Copy>(algorithm: A) {
    assert_ordinary_checked(algorithm, &[true, false]);
}

fn assert_ordinary_checked<A: OdeAlgorithm + Copy>(algorithm: A, failure_modes: &[bool]) {
    for initial in [
        arr0(1.0).into_dyn(),
        array![1.0, 2.0].into_dyn(),
        array![[1.0, 2.0], [3.0, 4.0]].into_dyn(),
    ] {
        for span in [(0.0, 0.1), (0.1, 0.0)] {
            let out = OdeProblem::from_array_out_of_place(
                |u: ArrayViewD<'_, f64>, _: &(), _| -&u,
                initial.clone(),
                span,
                (),
            );
            let inplace = OdeProblem::from_array(
                |mut du: ArrayViewMutD<'_, f64>, u: ArrayViewD<'_, f64>, _: &(), _| {
                    du.zip_mut_with(&u, |du, u| *du = -*u);
                },
                initial.clone(),
                span,
                (),
            );
            let expected = solve(&inplace, algorithm, &options()).unwrap();
            let actual = solve(&out, algorithm, &options()).unwrap();
            assert_eq!(actual.state_shape(), expected.state_shape());
            assert_eq!(actual.times(), expected.times());
            assert_eq!(actual.values(), expected.values());
            assert_eq!(actual.stats(), expected.stats());
        }
    }
    for &fail_at_start in failure_modes {
        let problem = OdeProblem::from_array_out_of_place(
            move |u: ArrayViewD<'_, f64>, _: &(), t| {
                if fail_at_start || t != 0.0 {
                    Array::zeros(IxDyn(&[1, 4]))
                } else {
                    -&u
                }
            },
            array![[1.0, 2.0], [3.0, 4.0]].into_dyn(),
            (0.0, 0.1),
            (),
        );
        assert_eq!(
            solve(&problem, algorithm, &options()),
            Err(SolveError::DerivativeShapeMismatch)
        );
    }
}

#[test]
fn returned_arrays_match_in_place_states_and_statistics_across_solver_families() {
    assert_ordinary(Euler);
    assert_ordinary(Tsit5);
    assert_ordinary(Anas5::default());
    assert_ordinary(Frk65::default());
    assert_ordinary(CarpenterKennedy2N54);
    assert_ordinary(SspRk432);
    assert_ordinary(SSPRKMSVS32::default());
    assert_ordinary(ImplicitEuler);
    assert_ordinary(Kvaerno3);
    assert_ordinary(Cash4);
    assert_ordinary(PDIRK44);
    assert_ordinary(RadauIIA5);
    assert_ordinary(Rosenbrock23);
    assert_ordinary(Rodas5P);
    assert_ordinary(AMF::default());
    assert_ordinary(Ab3);
    assert_ordinary(Vcab3);
    assert_ordinary(Abdf2);
    assert_ordinary(Qndf1);
    assert_ordinary(Qndf2);
    assert_ordinary(QNDF::default());
    assert_ordinary(Trbdf2);
    assert_ordinary(Mebdf2);
    assert_ordinary(AN5);
    assert_ordinary(RKC);
    assert_ordinary(ExplicitTaylor2);
    assert_ordinary(ExtrapolationMidpointDeuflhard::default());
    assert_ordinary(NorsettEuler);
    // This autonomous specialization freezes its operator at initialization;
    // later-time RHS branches are deliberately never evaluated.
    assert_ordinary_checked(LinearExponential, &[true]);
}

fn assert_split<A: SplitOdeAlgorithm + Copy>(algorithm: A) {
    for initial in [
        arr0(1.0).into_dyn(),
        array![1.0, 2.0].into_dyn(),
        array![[1.0, 2.0], [3.0, 4.0]].into_dyn(),
    ] {
        for span in [(0.0, 0.1), (0.1, 0.0)] {
            let problem = SplitOdeProblem::from_array_out_of_place(
                |u: ArrayViewD<'_, f64>, _: &(), _| -&u,
                |u: ArrayViewD<'_, f64>, _: &(), _| -&u,
                initial.clone(),
                span,
                (),
            );
            let baseline = SplitOdeProblem::new(
                |du: &mut [f64], u: &[f64], _: &(), _| {
                    for (du, u) in du.iter_mut().zip(u) {
                        *du = -*u;
                    }
                },
                |du: &mut [f64], u: &[f64], _: &(), _| {
                    for (du, u) in du.iter_mut().zip(u) {
                        *du = -*u;
                    }
                },
                initial.iter().copied().collect::<Vec<_>>(),
                span,
                (),
            );
            let expected = solve_split(&baseline, algorithm, &options()).unwrap();
            let actual = solve_split(&problem, algorithm, &options()).unwrap();
            assert_eq!(actual.state_shape(), initial.shape());
            assert_eq!(actual.values(), expected.values());
            assert_eq!(actual.stats(), expected.stats());
        }
    }
    for explicit_fails in [true, false] {
        let problem = SplitOdeProblem::from_array_out_of_place(
            move |u: ArrayView1<'_, f64>, _: &(), _| if explicit_fails { array![0.0] } else { -&u },
            move |u: ArrayView1<'_, f64>, _: &(), _| {
                if !explicit_fails { array![0.0] } else { -&u }
            },
            array![1.0, 2.0],
            (0.0, 0.1),
            (),
        );
        assert_eq!(
            solve_split(&problem, algorithm, &options()),
            Err(SolveError::DerivativeShapeMismatch)
        );
    }
}

#[test]
fn every_split_driver_accepts_returned_arrays_and_propagates_each_components_errors() {
    assert_split(SplitEuler);
    assert_split(MRIGARKERK22a::new(4));
    assert_split(IMEXEuler);
    assert_split(IRKC::default());
}

#[test]
fn static_dimensions_and_nonstandard_layouts_keep_logical_index_order() {
    let scalar = OdeProblem::from_array_out_of_place(
        |u: ArrayView0<'_, f64>, _: &(), _| -&u,
        arr0(1.0),
        (0.0, 0.1),
        (),
    );
    assert!(
        solve(&scalar, Tsit5, &options())
            .unwrap()
            .state_shape()
            .is_empty()
    );
    let initial = Array::from_shape_vec((2, 2).f(), vec![1.0, 3.0, 2.0, 4.0]).unwrap();
    let matrix = OdeProblem::from_array_out_of_place(
        |u: ArrayView2<'_, f64>, _: &(), _| {
            let mut result = Array::zeros((2, 2).f());
            result.invert_axis(Axis(0));
            result.assign(&(-&u));
            assert!(!result.is_standard_layout());
            result
        },
        initial,
        (0.0, 0.1),
        (),
    );
    let solution = solve(&matrix, Tsit5, &options()).unwrap();
    for (actual, initial) in solution.last_state().iter().zip([1.0, 2.0, 3.0, 4.0]) {
        assert!((actual - initial * (-0.1_f64).exp()).abs() < 1e-9);
    }
}

#[test]
fn callbacks_and_dense_output_keep_matrix_shape_and_see_parameter_changes() {
    let problem = OdeProblem::from_array_out_of_place(
        |u: ArrayView2<'_, f64>, rate: &Cell<f64>, _| &u * rate.get(),
        array![[1.0, 2.0], [3.0, 4.0]],
        (0.0, 0.1),
        Cell::new(-1.0),
    )
    .with_array_preset_time_callback([0.04], |_, rate, _| {
        rate.set(0.0);
        CallbackAction::Continue
    });
    let solution = solve(&problem, Tsit5, &options().with_dense_output(true)).unwrap();
    assert_eq!(solution.last_state_array().shape(), &[2, 2]);
    let sampled = solution.interpolate_array(0.02).unwrap();
    assert_eq!(sampled.shape(), &[2, 2]);
    assert!((sampled[[0, 0]] - (-0.02_f64).exp()).abs() < 1e-9);
    assert!((solution.last_state()[0] - (-0.04_f64).exp()).abs() < 1e-9);
}

#[test]
fn derivative_shape_errors_propagate_through_steady_state_and_jacobian_checks() {
    let bad = OdeProblem::from_array_out_of_place(
        |_: ArrayView2<'_, f64>, _: &(), _| Array::zeros((1, 4)),
        Array::zeros((2, 2)),
        (0.0, 1.0),
        (),
    )
    .with_callback_set(TerminateSteadyState::new().into_callback_set().unwrap());
    assert_eq!(
        solve(&bad, Tsit5, &options()),
        Err(SolveError::DerivativeShapeMismatch)
    );
    let perturbed = OdeProblem::from_array_out_of_place(
        |u: ArrayView1<'_, f64>, _: &(), _| if u[0] > 1.0 { array![0.0, 0.0] } else { -&u },
        array![1.0],
        (0.0, 0.1),
        (),
    );
    assert_eq!(
        solve(&perturbed, Rodas5P, &options()),
        Err(SolveError::DerivativeShapeMismatch)
    );
}

#[test]
fn caller_defined_fallible_functions_keep_their_error_without_panicking() {
    struct InvalidDerivative;
    impl OdeFunction<()> for InvalidDerivative {
        fn evaluate(&self, _: &mut [f64], _: &[f64], _: &(), _: f64) -> Result<(), SolveError> {
            Err(SolveError::DerivativeShapeMismatch)
        }
    }
    let problem = OdeProblem::new(InvalidDerivative, [1.0], (0.0, 1.0), ());
    assert_eq!(
        solve(&problem, Tsit5, &options()),
        Err(SolveError::DerivativeShapeMismatch)
    );
}

#[test]
fn adaptive_returned_arrays_match_the_in_place_problem() {
    let out = OdeProblem::from_array_out_of_place(
        |u: ArrayView1<'_, f64>, _: &(), _| -&u,
        array![1.0, 2.0],
        (0.0, 1.0),
        (),
    );
    let inplace = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _| {
            for (du, u) in du.iter_mut().zip(u) {
                *du = -*u;
            }
        },
        [1.0, 2.0],
        (0.0, 1.0),
        (),
    );
    let options = options().with_adaptive(true).with_tolerances(1e-10, 1e-10);
    for (actual, expected) in [
        (
            solve(&out, Tsit5, &options).unwrap(),
            solve(&inplace, Tsit5, &options).unwrap(),
        ),
        (
            solve(&out, Rodas5P, &options).unwrap(),
            solve(&inplace, Rodas5P, &options).unwrap(),
        ),
    ] {
        assert_eq!(actual.values(), expected.values());
        assert_eq!(actual.stats(), expected.stats());
    }
}

#[test]
fn nonfinite_returned_values_are_reported_separately_from_shape_errors() {
    let problem = OdeProblem::from_array_out_of_place(
        |_: ArrayView0<'_, f64>, _: &(), _| arr0(f64::NAN),
        arr0(1.0),
        (0.0, 1.0),
        (),
    );
    assert_eq!(
        solve(&problem, Tsit5, &options()),
        Err(SolveError::NonFiniteDerivative)
    );
}
