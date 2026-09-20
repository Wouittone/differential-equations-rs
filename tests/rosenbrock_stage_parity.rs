//! Public stage-convention parity against independent native Rosenbrock dispatch.
use differential_equations::{solvers::rosenbrock::*, stepping::RosenbrockStepper, *};
use std::convert::Infallible;
fn rhs(out: &mut [f64], y: &[f64], _: &(), t: f64) {
    out[0] = -3. * y[0] + 7. * y[1] + t;
    out[1] = -2. * y[0] - 11. * y[1] - 2. * t;
}
fn compare<A: OdeAlgorithm>(algorithm: A, tableau: &tableau::RosenbrockTableau) {
    let problem = OdeProblem::new(rhs, vec![1., -0.25], (0., 0.125), ())
        .with_jacobian(|j, _, _, _| j.copy_from_slice(&[-3., 7., -2., -11.]));
    let native = solve(
        &problem,
        algorithm,
        &SolveOptions::default()
            .with_adaptive(false)
            .with_initial_step(0.125)
            .with_save(SaveMode::Endpoints),
    )
    .unwrap();
    let mut stepper = RosenbrockStepper::new(tableau, 0., &[1., -0.25]).unwrap();
    let mut f = |t, y: &[f64], out: &mut [f64]| {
        rhs(out, y, &(), t);
        Ok::<_, Infallible>(())
    };
    let mut jac = |_: f64, _: &[f64], out: &mut [f64]| {
        out.copy_from_slice(&[-3., 7., -2., -11.]);
        Ok::<_, Infallible>(())
    };
    let mut time = |_: f64, _: &[f64], out: &mut [f64]| {
        out.copy_from_slice(&[1., -2.]);
        Ok::<_, Infallible>(())
    };
    let view = stepper
        .attempt(0.125, &mut f, Some(&mut jac), Some(&mut time))
        .unwrap();
    for (actual, expected) in view.candidate.iter().zip(native.last_state()) {
        assert!(
            (actual - expected).abs() < 2e-9,
            "{}: {actual} vs {expected}",
            tableau.name()
        );
    }
}
#[test]
fn transformed_rosenbrock_fixed_endpoints_match_native_nonsymmetric_forcing() {
    macro_rules! method {
        ($method:expr) => {
            compare($method, $method.tableau().unwrap());
        };
    }
    method!(Rodas4);
    method!(Rodas5P);
    method!(Rodas5Pe);
    method!(Rodas6P);
    method!(Ros2);
    method!(Ros34Pw1a);
    method!(Ros34Pw1b);
    method!(RosenbrockW6S4OS);
}
#[test]
fn ros34pw1a_raw_embedded_error_has_scalar_linear_blind_spot() {
    let mut stepper = RosenbrockStepper::new(Ros34Pw1a.tableau().unwrap(), 0., &[1.]).unwrap();
    let mut f = |_: f64, y: &[f64], out: &mut [f64]| {
        out[0] = -y[0];
        Ok::<_, Infallible>(())
    };
    let mut jac = |_: f64, _: &[f64], out: &mut [f64]| {
        out[0] = -1.;
        Ok::<_, Infallible>(())
    };
    let mut time = |_: f64, _: &[f64], out: &mut [f64]| {
        out[0] = 0.;
        Ok::<_, Infallible>(())
    };
    let view = stepper
        .attempt(0.5, &mut f, Some(&mut jac), Some(&mut time))
        .unwrap();
    assert!(view.component_error.unwrap()[0].abs() < 1e-14);
    assert!((view.candidate[0] - (-0.5_f64).exp()).abs() > 1e-5);
}
