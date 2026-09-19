use super::{JacobianProvider, OdeProblem, SplitOdeProblem};

#[test]
fn jacobian_provider_reports_analytic_callbacks() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    )
    .with_jacobian(|jacobian: &mut [f64], _: &[f64], _: &(), _: f64| jacobian[0] = 2.0);
    let provider = JacobianProvider::new(&problem);
    let mut jacobian = [0.0];
    assert!(provider.is_analytic());
    assert!(provider.evaluate(&mut jacobian, &[1.0], 0.0));
    assert_eq!(jacobian, [2.0]);
}

#[test]
fn split_representation_preserves_components() {
    let split = SplitOdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let mut explicit = [0.0];
    let mut implicit = [0.0];
    split.evaluate_explicit(&mut explicit, &[2.0], 0.0).unwrap();
    split.evaluate_implicit(&mut implicit, &[2.0], 0.0).unwrap();
    assert_eq!(explicit, [2.0]);
    assert_eq!(implicit, [-2.0]);
}
