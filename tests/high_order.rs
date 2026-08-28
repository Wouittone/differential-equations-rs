use differential_equations::solvers::explicit::*;
use differential_equations::*;

type ScalarRhs = fn(&mut [f64], &[f64], &(), f64);

fn exponential() -> OdeProblem<ScalarRhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
        du[0] = u[0];
    }
    OdeProblem::new(rhs as ScalarRhs, vec![1.0], (0.0, 1.0), ())
}

#[test]
fn all_remaining_high_order_names_solve_fixed_and_adaptive() {
    let fixed = SolveOptions {
        adaptive: false,
        initial_step: Some(0.1),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let adaptive = SolveOptions {
        absolute_tolerance: 1.0e-10,
        relative_tolerance: 1.0e-10,
        initial_step: Some(0.2),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let exact = 1.0_f64.exp();

    macro_rules! check {
        ($algorithm:expr) => {{
            let fixed_endpoint = solve(&exponential(), $algorithm, &fixed)
                .unwrap()
                .last_state()[0];
            let adaptive_endpoint = solve(&exponential(), $algorithm, &adaptive)
                .unwrap()
                .last_state()[0];
            assert!(
                (fixed_endpoint - exact).abs() < 1.0e-8,
                "{} fixed endpoint={fixed_endpoint:.17e}",
                stringify!($algorithm)
            );
            assert!(
                (adaptive_endpoint - exact).abs() < 2.0e-8,
                "{} adaptive endpoint={adaptive_endpoint:.17e}",
                stringify!($algorithm)
            );
        }};
    }

    check!(DP8);
    check!(Feagin10);
    check!(Feagin12);
    check!(Feagin14);
    check!(PFRK87);
    check!(RKV76IIa);
    check!(TanYam7);
    check!(TsitPap8);
}

#[test]
fn names_implement_the_solver_contract() {
    fn assert_algorithm<A: OdeAlgorithm>() {}

    assert_algorithm::<DP8>();
    assert_algorithm::<Feagin10>();
    assert_algorithm::<Feagin12>();
    assert_algorithm::<Feagin14>();
    assert_algorithm::<PFRK87>();
    assert_algorithm::<RKV76IIa>();
    assert_algorithm::<TanYam7>();
    assert_algorithm::<TsitPap8>();
}

#[test]
fn fixed_step_rhs_counts_match_each_methods_stage_count() {
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(1.0),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };

    macro_rules! check {
        ($algorithm:expr, $stages:expr) => {
            assert_eq!(
                solve(&exponential(), $algorithm, &options)
                    .unwrap()
                    .stats()
                    .rhs_evaluations,
                $stages,
                "{} did not use its own tableau",
                stringify!($algorithm)
            );
        };
    }

    check!(TanYam7, 10);
    check!(TsitPap8, 13);
    check!(DP8, 13);
    check!(PFRK87, 13);
    check!(Feagin10, 17);
    check!(Feagin12, 25);
    check!(Feagin14, 35);
    check!(RKV76IIa, 10);
}
