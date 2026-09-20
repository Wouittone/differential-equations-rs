//! Analytic eccentric Kepler orbits isolate norm/controller/Jacobian effects.
use differential_equations::solvers::rosenbrock::Rodas4;
use differential_equations::stepping::{AdaptiveController, ControllerConfig, RosenbrockStepper};
use differential_equations::tolerances::{ErrorNorm, Tolerances};
use std::convert::Infallible;

fn force(_: f64, y: &[f64], d: &mut [f64]) -> Result<(), Infallible> {
    let r2 = y[0] * y[0] + y[1] * y[1];
    let inverse_cube = r2.powf(-1.5);
    d[0] = y[2];
    d[1] = y[3];
    d[2] = -y[0] * inverse_cube;
    d[3] = -y[1] * inverse_cube;
    Ok(())
}
fn jacobian(_: f64, y: &[f64], j: &mut [f64]) -> Result<(), Infallible> {
    j.fill(0.0);
    j[2] = 1.0;
    j[7] = 1.0;
    let r2 = y[0] * y[0] + y[1] * y[1];
    for row in 0..2 {
        for col in 0..2 {
            j[(row + 2) * 4 + col] = 3.0 * y[row] * y[col] * r2.powf(-2.5)
                - if row == col { r2.powf(-1.5) } else { 0.0 };
        }
    }
    Ok(())
}
fn orbit(step: f64, adaptive: Option<(f64, ErrorNorm, bool)>, numerical: bool) -> (f64, usize) {
    let eccentricity: f64 = 0.2;
    let mut solver = RosenbrockStepper::new(
        Rodas4.tableau().unwrap(),
        0.0,
        &[
            1.0 - eccentricity,
            0.0,
            0.0,
            ((1.0 + eccentricity) / (1.0 - eccentricity)).sqrt(),
        ],
    )
    .unwrap();
    // A shorter order-study arc avoids endpoint cancellation between leading
    // global error terms near a full revolution. Adaptive studies cover 6 rad.
    let end = step.signum() * if adaptive.is_some() { 6.0 } else { 1.0 };
    let tolerance = adaptive.map_or(1e-8, |x| x.0);
    let tolerances = Tolerances::scalar(4, tolerance, tolerance).unwrap();
    let mut config = ControllerConfig::proportional(4).unwrap();
    if adaptive.is_some_and(|x| x.2) {
        config.beta = [0.7 / 4.0, 0.4 / 4.0, 0.0];
    }
    let mut controller = AdaptiveController::new(config, step).unwrap();
    let mut proposal = step;
    for _ in 0..100_000 {
        if solver.time() == end {
            break;
        }
        let h = proposal.signum() * proposal.abs().min((end - solver.time()).abs());
        let mut zero_partial = |_: f64, _: &[f64], d: &mut [f64]| {
            d.fill(0.0);
            Ok::<_, Infallible>(())
        };
        let mut analytic_jacobian = jacobian;
        let view = solver
            .attempt(
                h,
                &mut force,
                if numerical {
                    None
                } else {
                    Some(&mut analytic_jacobian)
                },
                if numerical {
                    None
                } else {
                    Some(&mut zero_partial)
                },
            )
            .unwrap();
        let accept = if let Some((_, norm, _)) = adaptive {
            let error = tolerances
                .error_norm(
                    view.previous_state,
                    view.candidate,
                    view.component_error.unwrap(),
                    norm,
                )
                .unwrap();
            let decision = controller.assess(h, error).unwrap();
            proposal = decision.next_step;
            decision.accepted
        } else {
            true
        };
        if accept {
            solver.accept().unwrap();
        } else {
            solver.reject().unwrap();
        }
    }
    assert_eq!(solver.time(), end);
    let mut anomaly = end;
    for _ in 0..10 {
        anomaly -=
            (anomaly - eccentricity * anomaly.sin() - end) / (1.0 - eccentricity * anomaly.cos());
    }
    let scale = (1.0 - eccentricity * eccentricity).sqrt();
    let radius = 1.0 - eccentricity * anomaly.cos();
    let exact = [
        anomaly.cos() - eccentricity,
        scale * anomaly.sin(),
        -anomaly.sin() / radius,
        scale * anomaly.cos() / radius,
    ];
    let error = solver
        .state()
        .iter()
        .zip(exact)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    (error, solver.statistics().rhs_evaluations)
}

#[test]
fn kepler_orbit_has_fourth_order_with_analytic_and_numerical_jacobians() {
    for numerical in [false, true] {
        for direction in [-1.0, 1.0] {
            let coarse = orbit(direction * 0.025, None, numerical).0;
            let fine = orbit(direction * 0.0125, None, numerical).0;
            assert!(
                coarse / fine > 12.0 && coarse / fine < 22.0,
                "numeric={numerical} direction={direction} errors={coarse:e},{fine:e}"
            );
        }
    }
}

#[test]
fn tightening_tolerance_improves_orbits_across_norm_and_controller_choices() {
    for norm in [ErrorNorm::Max, ErrorNorm::Rms] {
        for pi in [false, true] {
            let coarse = orbit(0.1, Some((1e-6, norm, pi)), false);
            let fine = orbit(0.1, Some((1e-9, norm, pi)), false);
            eprintln!("norm={norm:?},pi={pi},coarse={coarse:?},fine={fine:?}");
            assert!(fine.0 < coarse.0 / 50.0);
            assert!(fine.0 < 2e-7);
            assert!(fine.1 > coarse.1);
        }
    }
}
