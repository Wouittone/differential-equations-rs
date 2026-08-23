use differential_equations::algorithms::*;
use differential_equations::*;

type ScalarRhs = fn(&mut [f64], &[f64], &(), f64);

fn exponential() -> OdeProblem<ScalarRhs, ()> {
    OdeProblem::new(
        (|du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0]) as ScalarRhs,
        vec![1.0],
        (0.0, 1.0),
        (),
    )
}

fn fixed_options(step: f64) -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        ..SolveOptions::default()
    }
}

fn adaptive_options() -> SolveOptions {
    SolveOptions {
        absolute_tolerance: 1.0e-9,
        relative_tolerance: 1.0e-9,
        ..SolveOptions::default()
    }
}

fn assert_fixed_endpoint<A: OdeAlgorithm + Copy>(algorithm: A) {
    let endpoint = solve(&exponential(), algorithm, &fixed_options(0.01))
        .unwrap()
        .last_state()[0];
    assert!(
        (endpoint - std::f64::consts::E).abs() < 5.0e-5,
        "{} endpoint={endpoint:.17e}",
        std::any::type_name::<A>()
    );
}

#[test]
fn remaining_rosenbrock_tableaus_integrate_fixed_exponential() {
    assert_fixed_endpoint(Rodas3P);
    assert_fixed_endpoint(Ros2Pr);
    assert_fixed_endpoint(Ros2S);
    assert_fixed_endpoint(Ros34Pw1a);
    assert_fixed_endpoint(Ros4LStab);
    assert_fixed_endpoint(RosShamp4);
    assert_fixed_endpoint(Scholz4_7);
    assert_fixed_endpoint(Veldd4);
    assert_fixed_endpoint(Velds4);
    assert_fixed_endpoint(Tsit5DA);
}

#[test]
fn remaining_rosenbrock_tableaus_support_adaptive_jacobians() {
    let problem = exponential()
        .with_jacobian(|jacobian: &mut [f64], _: &[f64], _: &(), _: f64| jacobian[0] = 1.0);

    macro_rules! assert_adaptive {
        ($algorithm:expr, $expects_jacobian:expr) => {
            let solution = solve(&problem, $algorithm, &adaptive_options())
                .unwrap_or_else(|error| panic!("{}: {error:?}", stringify!($algorithm)));
            assert!(
                (solution.last_state()[0] - std::f64::consts::E).abs() < 2.0e-7,
                "{} endpoint={:.17e}",
                stringify!($algorithm),
                solution.last_state()[0]
            );
            assert_eq!(
                solution.stats().jacobian_evaluations > 0,
                $expects_jacobian,
                "{} Jacobian usage",
                stringify!($algorithm)
            );
        };
    }
    assert_adaptive!(Rodas3P, true);
    assert_adaptive!(Ros2Pr, true);
    assert_adaptive!(Ros2S, true);
    assert_adaptive!(Ros34Pw1a, true);
    assert_adaptive!(Ros4LStab, true);
    assert_adaptive!(RosShamp4, true);
    assert_adaptive!(Scholz4_7, true);
    assert_adaptive!(Veldd4, true);
    assert_adaptive!(Velds4, true);
    assert_adaptive!(Tsit5DA, false);
}

#[test]
fn tsit5da_alias_is_the_hybrid_driver_instantiation() {
    let aliased = solve(&exponential(), Tsit5DA, &fixed_options(0.02))
        .unwrap()
        .last_state()[0];
    let generic = solve(
        &exponential(),
        HybridExplicitImplicitRK,
        &fixed_options(0.02),
    )
    .unwrap()
    .last_state()[0];
    assert_eq!(aliased, generic);
}
