use std::cell::Cell;

use differential_equations::callbacks::TerminateSteadyState;
use differential_equations::ndarray::{
    Array, ArrayD, ArrayView0, ArrayView1, ArrayView2, ArrayViewD, ArrayViewMutD, Axis, IxDyn,
    ShapeBuilder, arr0, array,
};
use differential_equations::solvers::second_order::*;
use differential_equations::{
    CallbackAction, CallbackSave, ConfigurationError, EventCrossing, EventDirection,
    InterpolationError, SaveMode, SolveError, SolveOptions,
};

fn options() -> SolveOptions {
    SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(0.01)
        .with_save(SaveMode::Endpoints)
        .with_dense_output(true)
}

fn states() -> [ArrayD<f64>; 3] {
    [
        arr0(1.0).into_dyn(),
        array![1.0, 1.0].into_dyn(),
        array![[1.0, 1.0], [1.0, 1.0]].into_dyn(),
    ]
}

fn out_of_place(
    initial: ArrayD<f64>,
    span: (f64, f64),
) -> SecondOrderOdeProblem<impl SecondOrderFunction<()>, ()> {
    SecondOrderOdeProblem::from_array_out_of_place(
        |_: ArrayViewD<'_, f64>, q: ArrayViewD<'_, f64>, _: &(), _| -&q,
        Array::zeros(initial.raw_dim()),
        initial,
        span,
        (),
    )
    .unwrap()
}

fn assert_second_order<A: SecondOrderOdeAlgorithm + Copy>(algorithm: A) {
    for span in [(0.0, 0.2), (0.2, 0.0)] {
        let baseline = solve_second_order(
            &out_of_place(arr0(1.0).into_dyn(), span),
            algorithm,
            &options(),
        )
        .unwrap();
        for initial in states() {
            let out = out_of_place(initial.clone(), span);
            let inplace = SecondOrderOdeProblem::from_array(
                |mut a: ArrayViewMutD<'_, f64>,
                 _: ArrayViewD<'_, f64>,
                 q: ArrayViewD<'_, f64>,
                 _: &(),
                 _| a.zip_mut_with(&q, |a, q| *a = -*q),
                Array::zeros(initial.raw_dim()),
                initial.clone(),
                span,
                (),
            )
            .unwrap();
            let actual = solve_second_order(&out, algorithm, &options()).unwrap();
            let expected = solve_second_order(&inplace, algorithm, &options()).unwrap();
            assert_eq!(actual, expected);
            assert_eq!(actual.state_shape(), initial.shape());
            assert_eq!(out.initial_velocity_array().shape(), initial.shape());
            assert_eq!(out.initial_position_array(), initial);
            for (v, q) in actual.last_velocity().iter().zip(actual.last_position()) {
                assert!((v - baseline.last_velocity()[0]).abs() < 1e-11);
                assert!((q - baseline.last_position()[0]).abs() < 1e-11);
                assert!((q - 0.2_f64.cos()).abs() < 3e-3);
                assert!((v + (span.1 - span.0).sin()).abs() < 3e-3);
            }
            assert_eq!(actual.last_position_array().shape(), initial.shape());
            assert_eq!(actual.last_velocity_array().shape(), initial.shape());
            assert_eq!(actual.position_array(0).unwrap(), initial);
            assert_eq!(actual.velocity_array(0).unwrap().shape(), initial.shape());
            assert!(actual.position_array(usize::MAX).is_none());
            assert!(actual.velocity_array(usize::MAX).is_none());
            let (v, q) = actual.interpolate_array(0.105).unwrap();
            let (flat_v, flat_q) = actual.try_interpolate(0.105).unwrap();
            assert_eq!(v.shape(), initial.shape());
            assert_eq!(q.shape(), initial.shape());
            assert_eq!(v.as_slice().unwrap(), flat_v);
            assert_eq!(q.as_slice().unwrap(), flat_q);
        }
    }
    for fail_at_start in [true, false] {
        let bad = SecondOrderOdeProblem::from_array_out_of_place(
            move |_: ArrayView2<'_, f64>, q: ArrayView2<'_, f64>, _: &(), t| {
                if fail_at_start || t != 0.0 {
                    Array::zeros((1, 4))
                } else {
                    -&q
                }
            },
            Array::zeros((2, 2)),
            Array::ones((2, 2)),
            (0.0, 0.2),
            (),
        )
        .unwrap();
        assert_eq!(
            solve_second_order(&bad, algorithm, &options()),
            Err(SolveError::DerivativeShapeMismatch.into())
        );
    }
}

#[test]
fn one_oscillator_is_shape_invariant_across_all_second_order_drivers() {
    assert_second_order(SymplecticEuler);
    assert_second_order(VelocityVerlet);
    assert_second_order(VerletLeapfrog);
    assert_second_order(LeapfrogDriftKickDrift);
    assert_second_order(Nystrom4);
    assert_second_order(Nystrom4VelocityIndependent);
    assert_second_order(Nystrom5VelocityIndependent);
    assert_second_order(Rkn4);
    assert_second_order(Dprkn4);
    assert_second_order(Dprkn5);
    assert_second_order(Dprkn6);
    assert_second_order(Dprkn6Fm);
    assert_second_order(Dprkn8);
    assert_second_order(Dprkn12);
    assert_second_order(Erkn4);
    assert_second_order(Erkn5);
    assert_second_order(Erkn7);
    assert_second_order(FineRkn4);
    assert_second_order(FineRkn5);
    assert_second_order(Irkn3);
    assert_second_order(Irkn4);
    assert_second_order(NewmarkBeta::default());
    assert_second_order(GeneralizedAlpha::default());
}

fn assert_symplectic<A: SymplecticAlgorithm>(algorithm: A) {
    for span in [(0.0, 0.2), (0.2, 0.0)] {
        for initial in states() {
            let problem = out_of_place(initial.clone(), span);
            let flat = SecondOrderOdeProblem::new(
                |a: &mut [f64], _: &[f64], q: &[f64], _: &(), _| {
                    for (a, q) in a.iter_mut().zip(q) {
                        *a = -*q;
                    }
                },
                vec![0.0; initial.len()],
                vec![1.0; initial.len()],
                span,
                (),
            );
            let actual = solve_symplectic(&problem, algorithm, &options()).unwrap();
            let expected = solve_symplectic(&flat, algorithm, &options()).unwrap();
            assert_eq!(actual.position_values(), expected.position_values());
            assert_eq!(actual.velocity_values(), expected.velocity_values());
            assert_eq!(actual.rhs_evaluations(), expected.rhs_evaluations());
            assert_eq!(actual.state_shape(), initial.shape());
            assert_eq!(actual.last_position_array().shape(), initial.shape());
            assert_eq!(actual.last_velocity_array().shape(), initial.shape());
            assert_eq!(actual.position_array(0).unwrap(), initial);
            assert_eq!(actual.velocity_array(0).unwrap().shape(), initial.shape());
            assert!(actual.position_array(usize::MAX).is_none());
            assert!(actual.velocity_array(usize::MAX).is_none());
            let (q, v) = actual.interpolate_array(0.105).unwrap();
            let (flat_q, flat_v) = expected.try_interpolate(0.105).unwrap();
            assert_eq!(q.shape(), initial.shape());
            assert_eq!(v.shape(), initial.shape());
            assert_eq!(q.as_slice().unwrap(), flat_q);
            assert_eq!(v.as_slice().unwrap(), flat_v);
        }
    }
    let bad = SecondOrderOdeProblem::from_array_out_of_place(
        |_: ArrayView2<'_, f64>, _: ArrayView2<'_, f64>, _: &(), _| Array::zeros((1, 4)),
        Array::zeros((2, 2)),
        Array::ones((2, 2)),
        (0.0, 0.2),
        (),
    )
    .unwrap();
    assert_eq!(
        solve_symplectic(&bad, algorithm, &options()),
        Err(SolveError::DerivativeShapeMismatch.into())
    );
}

#[test]
fn every_symplectic_composition_preserves_shapes_and_errors() {
    assert_symplectic(PseudoVerletLeapfrog);
    assert_symplectic(McAte2);
    assert_symplectic(Ruth3);
    assert_symplectic(McAte3);
    assert_symplectic(CandyRoz4);
    assert_symplectic(McAte4);
    assert_symplectic(CalvoSanz4);
    assert_symplectic(McAte42);
    assert_symplectic(McAte5);
    assert_symplectic(Yoshida6);
    assert_symplectic(KahanLi6);
    assert_symplectic(McAte8);
    assert_symplectic(KahanLi8);
    assert_symplectic(SofSpa10);
}

#[test]
fn nonstandard_input_and_returned_layouts_preserve_distinct_logical_indices() {
    let q = Array::from_shape_vec((2, 2).f(), vec![1.0, 3.0, 2.0, 4.0]).unwrap();
    let problem = SecondOrderOdeProblem::from_array_out_of_place(
        |_: ArrayView2<'_, f64>, q: ArrayView2<'_, f64>, _: &(), _| {
            let mut a = Array::zeros((2, 2).f());
            a.invert_axis(Axis(0));
            a.assign(&(-&q));
            assert!(!a.is_standard_layout());
            a
        },
        Array::zeros((2, 2)),
        q,
        (0.0, 0.2),
        (),
    )
    .unwrap();
    assert_eq!(problem.initial_position(), &[1.0, 2.0, 3.0, 4.0]);
    let result = solve_second_order(&problem, Dprkn6, &options()).unwrap();
    for ((q, v), initial) in result
        .last_position()
        .iter()
        .zip(result.last_velocity())
        .zip([1.0, 2.0, 3.0, 4.0])
    {
        assert!((q - initial * 0.2_f64.cos()).abs() < 1e-10);
        assert!((v + initial * 0.2_f64.sin()).abs() < 1e-10);
    }
}

#[test]
fn mismatched_shapes_empty_partitions_and_invalid_query_indices_are_safe() {
    let mismatch = SecondOrderOdeProblem::from_array_out_of_place(
        |_: ArrayViewD<'_, f64>, q: ArrayViewD<'_, f64>, _: &(), _| -&q,
        Array::zeros(IxDyn(&[1, 4])),
        Array::ones(IxDyn(&[2, 2])),
        (0.0, 0.2),
        (),
    );
    assert!(matches!(
        mismatch,
        Err(ConfigurationError::DimensionMismatch { .. })
    ));
    let empty = out_of_place(Array::zeros(IxDyn(&[0, 2])), (0.0, 0.2));
    assert_eq!(
        solve_second_order(&empty, Dprkn6, &options()),
        Err(SolveError::EmptyState.into())
    );
    let flat = SecondOrderOdeProblem::new(
        |_: &mut [f64], _: &[f64], _: &[f64], _: &(), _| {},
        [0.0],
        [0.0, 1.0],
        (0.0, 0.2),
        (),
    );
    assert_eq!(flat.initial_velocity_array().shape(), &[1]);
    assert_eq!(
        solve_second_order(&flat, Dprkn6, &options()),
        Err(SecondOrderSolveError::StateDimensionMismatch)
    );
    let scalar = solve_second_order(
        &out_of_place(arr0(1.0).into_dyn(), (0.0, 0.2)),
        Dprkn6,
        &options(),
    )
    .unwrap();
    assert!(scalar.position_array(usize::MAX).is_none());
    assert_eq!(
        scalar.interpolate_array(f64::NAN),
        Err(InterpolationError::NonFiniteTime)
    );
    assert_eq!(
        scalar.interpolate_array(0.3),
        Err(InterpolationError::OutsideTimeSpan)
    );
}

#[test]
fn adaptive_oscillator_keeps_both_evaluation_forms_identical() {
    fn check<A: SecondOrderOdeAlgorithm + Copy>(algorithm: A) {
        let returned = out_of_place(array![[1.0, 2.0], [3.0, 4.0]].into_dyn(), (0.0, 0.2));
        let inplace = SecondOrderOdeProblem::from_array(
            |mut a: ArrayViewMutD<'_, f64>,
             _: ArrayViewD<'_, f64>,
             q: ArrayViewD<'_, f64>,
             _: &(),
             _| a.zip_mut_with(&q, |a, q| *a = -*q),
            Array::zeros(IxDyn(&[2, 2])),
            array![[1.0, 2.0], [3.0, 4.0]].into_dyn(),
            (0.0, 0.2),
            (),
        )
        .unwrap();
        let options = options().with_adaptive(true).with_tolerances(1e-9, 1e-9);
        assert_eq!(
            solve_second_order(&returned, algorithm, &options),
            solve_second_order(&inplace, algorithm, &options)
        );
    }
    check(Dprkn6);
    check(NewmarkBeta::default());
    check(GeneralizedAlpha::default());
}

#[test]
fn shape_errors_survive_steady_state_and_structural_jacobian_evaluations() {
    let bad = SecondOrderOdeProblem::from_array_out_of_place(
        |_: ArrayView1<'_, f64>, _: ArrayView1<'_, f64>, _: &(), _| array![0.0, 0.0],
        array![0.0],
        array![1.0],
        (0.0, 0.2),
        (),
    )
    .unwrap()
    .with_callback_set(
        TerminateSteadyState::new()
            .into_second_order_callback_set()
            .unwrap(),
    );
    assert_eq!(
        solve_second_order(&bad, Dprkn6, &options()),
        Err(SolveError::DerivativeShapeMismatch.into())
    );
    assert_eq!(
        solve_symplectic(&bad, Yoshida6, &options()),
        Err(SolveError::DerivativeShapeMismatch.into())
    );
    // The zero initial acceleration keeps the predictor at q = 1. The
    // time-dependent force gives a nonzero residual, requiring a Jacobian;
    // its acceleration perturbation is the first evaluation with q > 1.
    let jacobian = SecondOrderOdeProblem::from_array_out_of_place(
        |_: ArrayView1<'_, f64>, q: ArrayView1<'_, f64>, calls: &Cell<usize>, t| {
            calls.set(calls.get() + 1);
            if q[0] > 1.0 {
                array![0.0, 0.0]
            } else {
                array![if t == 0.0 { 0.0 } else { 1.0 }]
            }
        },
        array![0.0],
        array![1.0],
        (0.0, 0.2),
        Cell::new(0),
    )
    .unwrap();
    assert_eq!(
        solve_second_order(&jacobian, NewmarkBeta::default(), &options()),
        Err(SolveError::DerivativeShapeMismatch.into())
    );
    assert_eq!(jacobian.parameters().get(), 3);
}

#[test]
fn preset_callbacks_preserve_matrix_shapes_mutate_parameters_and_save_both_sides() {
    let rate = Cell::new(0.0);
    let problem = SecondOrderOdeProblem::from_array_out_of_place(
        |_: ArrayView2<'_, f64>, q: ArrayView2<'_, f64>, rate: &&Cell<f64>, _| {
            Array::from_elem(q.raw_dim(), rate.get())
        },
        Array::ones((2, 2)),
        Array::zeros((2, 2)),
        (0.0, 0.2),
        &rate,
    )
    .unwrap()
    .with_array_preset_time_callback_saving(
        [0.05],
        CallbackSave::Both,
        |mut v, mut q, rate, _| {
            assert_eq!(v.shape(), &[2, 2]);
            assert_eq!(q.shape(), &[2, 2]);
            v.fill(0.0);
            q[[1, 0]] = 3.0;
            rate.set(2.0);
            CallbackAction::Continue
        },
    );
    let result = solve_second_order(&problem, Dprkn6, &options()).unwrap();
    let duplicate = result
        .times()
        .windows(2)
        .position(|ts| ts[0] == 0.05 && ts[1] == 0.05)
        .unwrap();
    assert!((result.position_array(duplicate).unwrap()[[1, 0]] - 0.05).abs() < 1e-12);
    assert_eq!(result.position_array(duplicate + 1).unwrap()[[1, 0]], 3.0);
    let (v, q) = result.interpolate_array(0.05).unwrap();
    assert_eq!(v[[0, 0]], 0.0);
    assert_eq!(q[[1, 0]], 3.0);
    assert!((result.last_position_array()[[1, 0]] - 3.0225).abs() < 1e-11);
}

#[test]
fn scalar_and_vector_continuous_callbacks_keep_partition_order() {
    let scalar = SecondOrderOdeProblem::from_array_out_of_place(
        |_: ArrayView0<'_, f64>, _: ArrayView0<'_, f64>, _: &(), _| arr0(0.0),
        arr0(1.0),
        arr0(0.0),
        (0.0, 0.2),
        (),
    )
    .unwrap()
    .with_array_continuous_callback_direction(
        EventDirection::Rising,
        |v, q, _, _| {
            assert_eq!(v[[]], 1.0);
            q[[]] - 0.055
        },
        |mut v, q, _, _| {
            assert!((q[[]] - 0.055).abs() < 1e-10);
            v[[]] = 2.0;
            CallbackAction::Terminate
        },
    );
    let result = solve_second_order(&scalar, VelocityVerlet, &options()).unwrap();
    assert!(result.state_shape().is_empty());
    assert_eq!(result.last_velocity_array()[[]], 2.0);
    let vector = SecondOrderOdeProblem::from_array_out_of_place(
        |_: ArrayView1<'_, f64>, _: ArrayView1<'_, f64>, _: &(), _| array![0.0, 0.0],
        array![1.0, 1.0],
        array![0.0, 0.0],
        (0.0, 0.2),
        (),
    )
    .unwrap()
    .with_array_vector_continuous_callback(
        2,
        |mut output, v, q, _, _| {
            assert_eq!(v.shape(), &[2]);
            output[0] = q[0] - 0.055;
            output[1] = 0.055 - q[1];
        },
        |mut v, q, _, _, mask| {
            assert_eq!(mask, &[EventCrossing::Rising, EventCrossing::Falling]);
            assert_eq!(q.shape(), &[2]);
            v.fill(0.0);
            CallbackAction::Terminate
        },
    );
    let result = solve_symplectic(&vector, Yoshida6, &options()).unwrap();
    assert_eq!(result.state_shape(), &[2]);
    assert_eq!(result.last_velocity(), &[0.0, 0.0]);
    assert!((result.last_position()[0] - 0.055).abs() < 1e-10);
}

#[test]
fn initial_discrete_termination_preserves_shapes_without_calling_acceleration() {
    let problem = SecondOrderOdeProblem::from_array_out_of_place(
        |_: ArrayView2<'_, f64>, _: ArrayView2<'_, f64>, _: &(), _| {
            panic!("terminated before acceleration")
        },
        Array::zeros((2, 2)),
        Array::ones((2, 2)),
        (0.0, 0.2),
        (),
    )
    .unwrap()
    .with_array_discrete_callback(
        |_, _, _, _| true,
        |mut v, mut q, _, _| {
            v.fill(2.0);
            q[[1, 1]] = 3.0;
            CallbackAction::Terminate
        },
    );
    let result = solve_second_order(&problem, NewmarkBeta::default(), &options()).unwrap();
    assert_eq!(result.state_shape(), &[2, 2]);
    assert_eq!(result.last_position_array()[[1, 1]], 3.0);
    assert_eq!(result.stats().rhs_evaluations, 0);
    let result = solve_symplectic(&problem, Yoshida6, &options()).unwrap();
    assert_eq!(result.state_shape(), &[2, 2]);
    assert_eq!(result.last_velocity_array()[[0, 1]], 2.0);
    assert_eq!(result.rhs_evaluations(), 0);
}

#[test]
fn user_functions_and_nonfinite_values_return_typed_errors() {
    struct Bad;
    impl SecondOrderFunction<()> for Bad {
        fn evaluate(
            &self,
            _: &mut [f64],
            _: &[f64],
            _: &[f64],
            _: &(),
            _: f64,
        ) -> Result<(), SolveError> {
            Err(SolveError::DerivativeShapeMismatch)
        }
    }
    let problem = SecondOrderOdeProblem::new(Bad, [0.0], [1.0], (0.0, 0.2), ());
    assert_eq!(
        solve_second_order(&problem, Dprkn6, &options()),
        Err(SolveError::DerivativeShapeMismatch.into())
    );
    let nan = SecondOrderOdeProblem::from_array_out_of_place(
        |_: ArrayView0<'_, f64>, _: ArrayView0<'_, f64>, _: &(), _| arr0(f64::NAN),
        arr0(0.0),
        arr0(1.0),
        (0.0, 0.2),
        (),
    )
    .unwrap();
    assert_eq!(
        solve_second_order(&nan, Dprkn6, &options()),
        Err(SolveError::NonFiniteDerivative.into())
    );
}
