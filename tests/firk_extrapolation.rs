use differential_equations::algorithms::extrapolation::{
    AitkenNeville, ExtrapolationMidpointDeuflhard, ExtrapolationMidpointHairerWanner,
    ImplicitDeuflhardExtrapolation, ImplicitEulerBarycentricExtrapolation,
    ImplicitEulerExtrapolation, ImplicitHairerWannerExtrapolation,
};
use differential_equations::algorithms::implicit::fully_implicit::{
    AdaptiveRadau, GaussLegendre, RadauIIA3, RadauIIA5, RadauIIA9,
};
use differential_equations::{
    CallbackAction, OdeAlgorithm, OdeProblem, SaveMode, SolveOptions, solve,
};

fn adaptive_endpoint<A: OdeAlgorithm>(algorithm: A) -> (f64, differential_equations::SolverStats) {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -3.0 * u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions {
        absolute_tolerance: 1.0e-7,
        relative_tolerance: 1.0e-7,
        max_step: 0.25,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let solution = solve(&problem, algorithm, &options).unwrap();
    (solution.last_state()[0], solution.stats())
}

#[test]
fn all_twelve_public_algorithms_solve_the_reference_problem() {
    let firk = [
        adaptive_endpoint(RadauIIA3).0,
        adaptive_endpoint(RadauIIA5).0,
        adaptive_endpoint(RadauIIA9).0,
        adaptive_endpoint(AdaptiveRadau::default()).0,
        adaptive_endpoint(GaussLegendre::default()).0,
    ];
    let extrapolation = [
        adaptive_endpoint(AitkenNeville::default()).0,
        adaptive_endpoint(ExtrapolationMidpointDeuflhard::default()).0,
        adaptive_endpoint(ExtrapolationMidpointHairerWanner::default()).0,
        adaptive_endpoint(ImplicitEulerExtrapolation::default()).0,
        adaptive_endpoint(ImplicitDeuflhardExtrapolation::default()).0,
        adaptive_endpoint(ImplicitHairerWannerExtrapolation::default()).0,
        adaptive_endpoint(ImplicitEulerBarycentricExtrapolation::default()).0,
    ];
    let exact = (-3.0_f64).exp();
    for value in firk.into_iter().chain(extrapolation) {
        assert!(
            (value - exact).abs() < 2.0e-5,
            "endpoint {value}, exact {exact}"
        );
    }
}

#[test]
fn implicit_families_report_nonlinear_and_linear_work() {
    let (_, radau) = adaptive_endpoint(RadauIIA5);
    assert!(radau.nonlinear_iterations > 0);
    assert!(radau.linear_factorizations > 0);
    assert!(radau.linear_solves > 0);

    let (_, extrapolation) = adaptive_endpoint(ImplicitEulerExtrapolation::default());
    assert!(extrapolation.jacobian_evaluations > 0);
    assert!(extrapolation.linear_factorizations > 0);
    assert!(extrapolation.linear_solves > 0);
}

#[test]
fn backward_integration_and_callbacks_use_family_dense_segments() {
    let backward = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
        vec![std::f64::consts::E],
        (1.0, 0.0),
        (),
    );
    let fixed = SolveOptions {
        adaptive: false,
        initial_step: Some(0.05),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    assert!((solve(&backward, RadauIIA5, &fixed).unwrap().last_state()[0] - 1.0).abs() < 1.0e-8);
    assert!(
        (solve(&backward, ExtrapolationMidpointDeuflhard::default(), &fixed)
            .unwrap()
            .last_state()[0]
            - 1.0)
            .abs()
            < 1.0e-8
    );

    let callback_problem = OdeProblem::new(
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du[0] = 1.0,
        vec![0.0],
        (0.0, 1.0),
        (),
    )
    .with_continuous_callback(
        |u: &[f64], _: &(), _: f64| u[0] - 0.4,
        |u: &mut [f64], _: &(), _: f64| {
            u[0] = 2.0;
            CallbackAction::Terminate
        },
    );
    let callback_options = SolveOptions {
        adaptive: false,
        initial_step: Some(1.0),
        save: SaveMode::EveryStep,
        save_at: vec![0.2, 0.4],
        ..SolveOptions::default()
    };
    for solution in [
        solve(&callback_problem, RadauIIA5, &callback_options).unwrap(),
        solve(
            &callback_problem,
            AitkenNeville::default(),
            &callback_options,
        )
        .unwrap(),
    ] {
        assert!((solution.times().last().unwrap() - 0.4).abs() < 2.0e-12);
        assert_eq!(solution.last_state(), &[2.0]);
        assert_eq!(solution.stats().callback_invocations, 1);
    }
}
