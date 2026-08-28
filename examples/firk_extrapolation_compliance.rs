use differential_equations::solvers::extrapolation::{
    AitkenNeville, ExtrapolationMidpointDeuflhard, ExtrapolationMidpointHairerWanner,
    ImplicitDeuflhardExtrapolation, ImplicitEulerBarycentricExtrapolation,
    ImplicitEulerExtrapolation, ImplicitHairerWannerExtrapolation,
};
use differential_equations::solvers::implicit::firk::{
    AdaptiveRadau, GaussLegendre, RadauIIA3, RadauIIA5, RadauIIA9,
};
use differential_equations::{OdeAlgorithm, OdeProblem, SaveMode, SolveOptions, solve};

fn endpoint<A: OdeAlgorithm>(algorithm: A) -> f64 {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.1),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    solve(&problem, algorithm, &options).unwrap().last_state()[0]
}

fn main() {
    let results = [
        ("radau_iia3", endpoint(RadauIIA3)),
        ("radau_iia5", endpoint(RadauIIA5)),
        ("radau_iia9", endpoint(RadauIIA9)),
        ("adaptive_radau", endpoint(AdaptiveRadau::new(5, 5))),
        ("gauss_legendre", endpoint(GaussLegendre::new(2))),
        ("aitken_neville", endpoint(AitkenNeville::default())),
        (
            "midpoint_deuflhard",
            endpoint(ExtrapolationMidpointDeuflhard::default()),
        ),
        (
            "midpoint_hairer_wanner",
            endpoint(ExtrapolationMidpointHairerWanner::default()),
        ),
        (
            "implicit_euler",
            endpoint(ImplicitEulerExtrapolation::default()),
        ),
        (
            "implicit_deuflhard",
            endpoint(ImplicitDeuflhardExtrapolation::default()),
        ),
        (
            "implicit_hairer_wanner",
            endpoint(ImplicitHairerWannerExtrapolation::default()),
        ),
        (
            "implicit_euler_barycentric",
            endpoint(ImplicitEulerBarycentricExtrapolation::default()),
        ),
    ];
    for (name, value) in results {
        println!("{name},{value:.17e}");
    }
}
