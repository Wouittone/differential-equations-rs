use super::{
    AitkenNeville, ExtrapolationMidpointDeuflhard, ExtrapolationMidpointHairerWanner,
    ImplicitDeuflhardExtrapolation, ImplicitEulerBarycentricExtrapolation,
    ImplicitEulerExtrapolation, ImplicitHairerWannerExtrapolation,
};
use crate::{OdeProblem, SaveMode, SolveOptions, solve};

#[test]
fn explicit_extrapolation_families_are_accurate() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.2),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    for value in [
        solve(&problem, AitkenNeville::default(), &options)
            .unwrap()
            .last_state()[0],
        solve(
            &problem,
            ExtrapolationMidpointDeuflhard::default(),
            &options,
        )
        .unwrap()
        .last_state()[0],
        solve(
            &problem,
            ExtrapolationMidpointHairerWanner::default(),
            &options,
        )
        .unwrap()
        .last_state()[0],
    ] {
        assert!((value - std::f64::consts::E).abs() < 2.0e-7, "{value}");
    }
}

#[test]
fn implicit_extrapolation_families_handle_stiff_decay() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -40.0 * u[0],
        vec![1.0],
        (0.0, 0.2),
        (),
    );
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.01),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    for value in [
        solve(&problem, ImplicitEulerExtrapolation::default(), &options)
            .unwrap()
            .last_state()[0],
        solve(
            &problem,
            ImplicitDeuflhardExtrapolation::default(),
            &options,
        )
        .unwrap()
        .last_state()[0],
        solve(
            &problem,
            ImplicitHairerWannerExtrapolation::default(),
            &options,
        )
        .unwrap()
        .last_state()[0],
        solve(
            &problem,
            ImplicitEulerBarycentricExtrapolation::default(),
            &options,
        )
        .unwrap()
        .last_state()[0],
    ] {
        assert!((value - (-8.0_f64).exp()).abs() < 2.0e-5, "{value}");
    }
}
