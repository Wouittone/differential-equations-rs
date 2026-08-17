use differential_equations::{
    OdeAlgorithm, OdeProblem, RKC, RKG1, RKG2, RKL1, RKL2, RKMC2, Rk4, SaveMode, SolveOptions,
    solve,
};

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn decay_rhs(derivative: &mut [f64], state: &[f64], _: &(), _: f64) {
    derivative[0] = -state[0];
}

fn stiff_decay_rhs(derivative: &mut [f64], state: &[f64], _: &(), _: f64) {
    derivative[0] = -100.0 * state[0];
}

fn problem(rhs: TestRhs, end: f64) -> OdeProblem<TestRhs, ()> {
    OdeProblem::new(rhs, vec![1.0], (0.0, end), ())
}

fn fixed_options(step: f64) -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

fn final_value<A: OdeAlgorithm>(algorithm: A, problem: &OdeProblem<TestRhs, ()>, step: f64) -> f64 {
    solve(problem, algorithm, &fixed_options(step))
        .unwrap()
        .last_state()[0]
}

fn adaptive_final_value<A: OdeAlgorithm>(algorithm: A, problem: &OdeProblem<TestRhs, ()>) -> f64 {
    let options = SolveOptions {
        absolute_tolerance: 1.0e-7,
        relative_tolerance: 1.0e-7,
        initial_step: Some(0.05),
        max_step: 0.1,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    solve(problem, algorithm, &options).unwrap().last_state()[0]
}

#[test]
fn stabilized_recurrences_remain_bounded_beyond_rk4s_real_axis_interval() {
    let problem = problem(stiff_decay_rhs, 0.1);
    let stabilized = [
        final_value(RKC, &problem, 0.1),
        final_value(RKL1, &problem, 0.1),
        final_value(RKL2, &problem, 0.1),
        final_value(RKG1, &problem, 0.1),
        final_value(RKG2, &problem, 0.1),
        final_value(RKMC2, &problem, 0.1),
    ];

    for value in stabilized {
        assert!(value.is_finite());
        assert!(
            value.abs() <= 1.0,
            "stabilized value {value} escaped its real-axis interval"
        );
    }

    let rk4 = final_value(Rk4, &problem, 0.1);
    assert!(
        rk4.abs() > 100.0,
        "the control method must be outside its stability interval"
    );
}

#[test]
fn stabilized_families_have_distinguishable_stability_polynomials() {
    let problem = problem(stiff_decay_rhs, 0.1);
    let values = [
        final_value(RKC, &problem, 0.1),
        final_value(RKL1, &problem, 0.1),
        final_value(RKL2, &problem, 0.1),
        final_value(RKG1, &problem, 0.1),
        final_value(RKG2, &problem, 0.1),
        final_value(RKMC2, &problem, 0.1),
    ];

    for left in 0..values.len() {
        for right in left + 1..values.len() {
            assert!(
                (values[left] - values[right]).abs() > 1.0e-10,
                "methods {left} and {right} produced the same stability polynomial value"
            );
        }
    }
}

#[test]
fn implemented_stabilized_methods_converge_on_smooth_decay() {
    let problem = problem(decay_rhs, 1.0);
    let expected = (-1.0_f64).exp();
    let values = [
        ("RKC", final_value(RKC, &problem, 0.01)),
        ("RKL1", final_value(RKL1, &problem, 0.01)),
        ("RKL2", final_value(RKL2, &problem, 0.01)),
        ("RKG1", final_value(RKG1, &problem, 0.01)),
        ("RKG2", final_value(RKG2, &problem, 0.01)),
        ("RKMC2", final_value(RKMC2, &problem, 0.01)),
    ];

    for (method, value) in values {
        assert!(
            (value - expected).abs() < 1.0e-2,
            "{method} produced unexpected smooth-decay error: {value}"
        );
    }
}

#[test]
fn stabilized_recurrences_support_adaptive_driver_control() {
    let problem = problem(decay_rhs, 1.0);
    let expected = (-1.0_f64).exp();
    let values = [
        ("RKC", adaptive_final_value(RKC, &problem)),
        ("RKL1", adaptive_final_value(RKL1, &problem)),
        ("RKL2", adaptive_final_value(RKL2, &problem)),
        ("RKG1", adaptive_final_value(RKG1, &problem)),
        ("RKG2", adaptive_final_value(RKG2, &problem)),
        ("RKMC2", adaptive_final_value(RKMC2, &problem)),
    ];

    for (method, value) in values {
        assert!(
            (value - expected).abs() < 1.0e-3,
            "{method} adaptive solve produced {value}"
        );
    }
}
