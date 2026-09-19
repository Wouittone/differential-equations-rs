use differential_equations::solvers::explicit::Euler;
use differential_equations::solvers::exponential::RKIP;
use differential_equations::solvers::rosenbrock::AmfOperator;
use differential_equations::solvers::second_order::{NewmarkBeta, SecondOrderSolution};
use differential_equations::tableau::{TableauErrorKind, parse_tableau};
use differential_equations::{
    ConfigurationError, DEFAULT_EVENT_TOLERANCE, InterpolationError, LieGroupProblem,
    LinearOperatorProblem, OdeProblem, SemilinearOdeProblem, SolveError, SolveOptions,
    SplitOdeProblem, solve,
};
use std::error::Error as _;

#[test]
fn tableau_resource_failures_preserve_category_and_source_chain() {
    let resource_error = parse_tableau("{", "Broken").unwrap_err();
    assert_eq!(resource_error.kind(), TableauErrorKind::JsonSyntax);
    assert!(resource_error.source().is_some());

    let solve_error = SolveError::from(resource_error);
    assert!(matches!(&solve_error, SolveError::TableauResource(_)));
    assert!(solve_error.source().is_some());
}

#[test]
fn public_constructors_report_typed_configuration_errors() {
    assert!(matches!(
        AmfOperator::from_split(2, vec![vec![0.0; 3]]),
        Err(ConfigurationError::DimensionMismatch {
            context: "AMF factor collection"
        })
    ));

    let overflow = LieGroupProblem::matrix(
        |_: &mut [f64], _: &[f64], _: &(), _: f64| {},
        Vec::new(),
        usize::MAX,
        (0.0, 1.0),
        (),
    );
    assert!(matches!(
        overflow,
        Err(ConfigurationError::DimensionOverflow {
            context: "Lie-group matrix"
        })
    ));

    let semilinear = SemilinearOdeProblem::new(
        vec![f64::NAN],
        |_: &mut [f64], _: &[f64], _: &(), _: f64| {},
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    assert!(matches!(
        semilinear,
        Err(ConfigurationError::NonFiniteData {
            context: "semilinear linear operator"
        })
    ));

    assert!(matches!(
        RKIP::new(1.0, 0.5, 1),
        Err(ConfigurationError::InvalidBounds {
            context: "RKIP cache",
            ..
        })
    ));
    assert!(matches!(
        NewmarkBeta::new(-0.1, 0.5),
        Err(ConfigurationError::InvalidParameter {
            parameter: "Newmark beta",
            ..
        })
    ));
}

#[test]
fn interpolation_queries_preserve_failure_reasons() {
    let problem = OdeProblem::new(
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = 1.0,
        vec![0.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(0.5);
    let solution = solve(&problem, Euler, &options).unwrap();

    assert_eq!(
        solution.try_interpolate(f64::NAN),
        Err(InterpolationError::NonFiniteTime)
    );
    assert_eq!(
        solution.try_interpolate(2.0),
        Err(InterpolationError::OutsideTimeSpan)
    );
    assert_eq!(solution.try_interpolate(0.25).unwrap(), vec![0.25]);
    assert_eq!(solution.interpolate(2.0), None);

    let mut ordinary_output = [0.0];
    solution
        .try_interpolate_into(0.25, &mut ordinary_output)
        .unwrap();
    assert_eq!(ordinary_output, [0.25]);
    assert_eq!(
        solution.try_interpolate_into(0.25, &mut []),
        Err(InterpolationError::DimensionMismatch)
    );

    let partitioned = SecondOrderSolution::from_saved(
        vec![0.0, 1.0],
        vec![0.0, 2.0],
        vec![1.0, 3.0],
        &[1],
        Default::default(),
    )
    .unwrap();
    let mut velocity = [0.0];
    let mut position = [0.0];
    partitioned
        .try_interpolate_into(0.5, &mut velocity, &mut position)
        .unwrap();
    assert_eq!(velocity, [1.0]);
    assert_eq!(position, [2.0]);
    assert_eq!(
        partitioned.try_interpolate_into(0.5, &mut [], &mut position),
        Err(InterpolationError::DimensionMismatch)
    );
}

#[test]
fn default_event_tolerance_is_a_named_stable_constant() {
    assert_eq!(
        SolveOptions::default().event_tolerance,
        DEFAULT_EVENT_TOLERANCE
    );
}

#[test]
fn public_operator_and_semilinear_evaluations_are_checked() {
    let linear = LinearOperatorProblem::new(
        |operator: &mut [f64], _: &[f64], _: &(), _: f64| operator.fill(1.0),
        [1.0, 2.0],
        (0.0, 1.0),
        (),
    )
    .unwrap();
    assert_eq!(
        linear.evaluate_operator(&mut [0.0; 3], &[1.0, 2.0], 0.0),
        Err(SolveError::EvaluationDimensionMismatch)
    );
    assert_eq!(
        linear.evaluate_operator(&mut [0.0; 4], &[1.0], 0.0),
        Err(SolveError::EvaluationDimensionMismatch)
    );

    let nonfinite_group = LieGroupProblem::vector(
        |operator: &mut [f64], _: &[f64], _: &(), _: f64| operator.fill(f64::NAN),
        [1.0, 0.0],
        (0.0, 1.0),
        (),
    )
    .unwrap();
    assert_eq!(
        nonfinite_group.evaluate_operator(&mut [0.0; 4], &[1.0, 0.0], 0.0),
        Err(SolveError::NonFiniteDerivative)
    );

    let semilinear = SemilinearOdeProblem::new(
        [-1.0],
        |output: &mut [f64], _: &[f64], _: &(), _: f64| output[0] = f64::INFINITY,
        [1.0],
        (0.0, 1.0),
        (),
    )
    .unwrap();
    assert_eq!(
        semilinear.evaluate(&mut [], &[1.0], 0.0),
        Err(SolveError::EvaluationDimensionMismatch)
    );
    assert_eq!(
        semilinear.evaluate_nonlinear(&mut [0.0], &[1.0], 0.0),
        Err(SolveError::NonFiniteDerivative)
    );
}

#[test]
fn public_split_evaluations_check_dimensions_and_finiteness() {
    let problem = SplitOdeProblem::new(
        |output: &mut [f64], _: &[f64], _: &(), _: f64| output.fill(f64::NAN),
        |output: &mut [f64], _: &[f64], _: &(), _: f64| output.fill(f64::INFINITY),
        [1.0, 2.0],
        (0.0, 1.0),
        (),
    );

    assert_eq!(
        problem.evaluate_explicit(&mut [0.0], &[1.0, 2.0], 0.0),
        Err(SolveError::EvaluationDimensionMismatch)
    );
    assert_eq!(
        problem.evaluate_implicit(&mut [0.0, 0.0], &[1.0], 0.0),
        Err(SolveError::EvaluationDimensionMismatch)
    );
    assert_eq!(
        problem.evaluate_explicit(&mut [0.0, 0.0], &[1.0, 2.0], 0.0),
        Err(SolveError::NonFiniteDerivative)
    );
    assert_eq!(
        problem.evaluate_implicit(&mut [0.0, 0.0], &[1.0, 2.0], 0.0),
        Err(SolveError::NonFiniteDerivative)
    );
}
