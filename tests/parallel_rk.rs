use differential_equations::solvers::explicit::*;
use differential_equations::*;

type ScalarRhs = fn(&mut [f64], &[f64], &(), f64);

fn exponential() -> OdeProblem<ScalarRhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
        du[0] = u[0];
    }
    OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
}

fn fixed(step: f64) -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

#[test]
fn public_names_implement_the_solver_contract() {
    fn assert_algorithm<A: OdeAlgorithm>() {}
    assert_algorithm::<KuttaPRK2p5>();
    assert_algorithm::<QPRK98>();
}

#[test]
fn tableaus_have_consistent_rows_and_moments() {
    fn check(algorithm: ResourceExplicitRungeKutta) {
        let tableau = algorithm.tableau().unwrap();
        assert_eq!(tableau.c().len(), tableau.b().len());
        assert_eq!(tableau.a().len(), tableau.b().len());
        for (index, (row, node)) in tableau.a().iter().zip(tableau.c()).enumerate() {
            assert_eq!(row[index..], vec![0.0; row.len() - index]);
            assert!((row.iter().sum::<f64>() - node).abs() < 2.0e-10);
        }
        assert!((tableau.b().iter().sum::<f64>() - 1.0).abs() < 2.0e-10);
    }
    check(KuttaPRK2p5());
    check(QPRK98());
}

#[test]
fn fixed_step_rhs_counts_match_real_stage_counts() {
    assert_eq!(
        solve(&exponential(), KuttaPRK2p5(), &fixed(1.0))
            .unwrap()
            .stats()
            .rhs_evaluations,
        6
    );
    assert_eq!(
        solve(&exponential(), QPRK98(), &fixed(1.0))
            .unwrap()
            .stats()
            .rhs_evaluations,
        16
    );
}

#[test]
fn kutta_prk_has_fifth_order_convergence_and_rejects_adaptive_mode() {
    let exact = 1.0_f64.exp();
    let coarse = solve(&exponential(), KuttaPRK2p5(), &fixed(0.25))
        .unwrap()
        .last_state()[0];
    let fine = solve(&exponential(), KuttaPRK2p5(), &fixed(0.125))
        .unwrap()
        .last_state()[0];
    assert!((coarse - exact).abs() / (fine - exact).abs() > 25.0);

    assert_eq!(
        solve(&exponential(), KuttaPRK2p5(), &SolveOptions::default()).unwrap_err(),
        SolveError::AdaptiveStepUnsupported
    );
}

#[test]
fn qprk98_solves_fixed_and_adaptive() {
    let exact = 1.0_f64.exp();
    let fixed_endpoint = solve(&exponential(), QPRK98(), &fixed(0.25))
        .unwrap()
        .last_state()[0];
    // QPRK98 was constructed for quadruple precision. Its very large,
    // cancelling coefficients lose several digits in f64 arithmetic even
    // though the exact 9(8) tableau is being used.
    assert!((fixed_endpoint - exact).abs() < 2.0e-8);

    let adaptive = SolveOptions {
        absolute_tolerance: 1.0e-10,
        relative_tolerance: 1.0e-10,
        initial_step: Some(0.25),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let adaptive_endpoint = solve(&exponential(), QPRK98(), &adaptive)
        .unwrap()
        .last_state()[0];
    assert!((adaptive_endpoint - exact).abs() < 2.0e-8);
}
