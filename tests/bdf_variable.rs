use differential_equations::solvers::multistep::*;
use differential_equations::*;

fn fixed(step: f64) -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

type TrackingRhs = fn(&mut [f64], &[f64], &(), f64);

fn stiff_tracking_problem() -> OdeProblem<TrackingRhs, ()> {
    fn stiff_rhs(du: &mut [f64], u: &[f64], _: &(), t: f64) {
        du[0] = -15.0 * (u[0] - t.cos()) - t.sin();
    }
    OdeProblem::new(stiff_rhs as TrackingRhs, vec![1.0], (0.0, 1.0), ())
}

#[test]
fn variable_order_methods_converge_under_fixed_refinement() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let exact = (-1.0_f64).exp();
    for (name, coarse, fine) in [
        (
            "QNDF",
            solve(&problem, QNDF, &fixed(0.05)).unwrap().last_state()[0],
            solve(&problem, QNDF, &fixed(0.025)).unwrap().last_state()[0],
        ),
        (
            "QBDF",
            solve(&problem, QBDF, &fixed(0.05)).unwrap().last_state()[0],
            solve(&problem, QBDF, &fixed(0.025)).unwrap().last_state()[0],
        ),
        (
            "FBDF",
            solve(&problem, FBDF, &fixed(0.05)).unwrap().last_state()[0],
            solve(&problem, FBDF, &fixed(0.025)).unwrap().last_state()[0],
        ),
    ] {
        let coarse_error = (coarse - exact).abs();
        let fine_error = (fine - exact).abs();
        assert!(
            fine_error < coarse_error,
            "{name}: {coarse_error} -> {fine_error}"
        );
        assert!(fine_error < 6.0e-3, "{name}: {fine_error}");
    }
}

#[test]
fn adaptive_methods_track_a_stiff_forced_mode() {
    let problem = stiff_tracking_problem();
    let options = SolveOptions {
        absolute_tolerance: 1.0e-8,
        relative_tolerance: 1.0e-8,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    for (name, endpoint, accepted) in [
        {
            let solution = solve(&problem, QNDF, &options).unwrap();
            (
                "QNDF",
                solution.last_state()[0],
                solution.stats().accepted_steps,
            )
        },
        {
            let solution = solve(&problem, QBDF, &options).unwrap();
            (
                "QBDF",
                solution.last_state()[0],
                solution.stats().accepted_steps,
            )
        },
        {
            let solution = solve(&problem, FBDF, &options).unwrap();
            (
                "FBDF",
                solution.last_state()[0],
                solution.stats().accepted_steps,
            )
        },
    ] {
        assert!(
            (endpoint - 1.0_f64.cos()).abs() < 2.0e-5,
            "{name}: {endpoint}"
        );
        assert!(accepted > 0, "{name}");
    }
}

#[test]
fn qndf_and_qbdf_are_distinct_at_nonzero_kappa_orders() {
    let problem = stiff_tracking_problem();
    let qndf = solve(&problem, QNDF, &fixed(0.1)).unwrap();
    let qbdf = solve(&problem, QBDF, &fixed(0.1)).unwrap();
    assert!((qndf.last_state()[0] - qbdf.last_state()[0]).abs() > 1.0e-10);
}
