use differential_equations::algorithms::explicit::Euler;
use differential_equations::algorithms::second_order::NewmarkBeta;
use differential_equations::{
    AmfOperator, ConfigurationError, DEFAULT_EVENT_TOLERANCE, InterpolationError, LieGroupProblem,
    MassMatrixOdeProblem, OdeProblem, RKIP, SemilinearOdeProblem, SolveOptions, solve,
};

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

    let mass_matrix = MassMatrixOdeProblem::new(
        |_: &mut [f64], _: &[f64], _: &(), _: f64| {},
        vec![1.0, 2.0],
        (0.0, 1.0),
        (),
        vec![1.0; 3],
    );
    assert!(matches!(
        mass_matrix,
        Err(ConfigurationError::DimensionMismatch {
            context: "mass matrix"
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
}

#[test]
fn default_event_tolerance_is_a_named_stable_constant() {
    assert_eq!(
        SolveOptions::default().event_tolerance,
        DEFAULT_EVENT_TOLERANCE
    );
}
