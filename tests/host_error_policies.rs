//! Independent physical-state and variational checks for external controllers.
use differential_equations::solvers::explicit::Tsit5;
use differential_equations::stepping::ExplicitRungeKuttaStepper;
use differential_equations::tolerances::{ErrorNorm, Tolerances};
use std::convert::Infallible;

#[test]
fn max_and_rms_control_analytic_state_and_rectangular_sensitivities() {
    // x'=v, v'=-x. The remaining two columns solve the same linear system:
    // Phi(0)=I and S(0)=[2,-3]. They are independent analytic oracles.
    for norm in [ErrorNorm::Max, ErrorNorm::Rms] {
        for (tolerance, relative) in [(1e-3, 1e-9), (1e-6, 1e-12)] {
            for direction in [-1.0, 1.0] {
                let mut solver = ExplicitRungeKuttaStepper::new(
                    Tsit5.tableau().unwrap(), 0.0,
                    &[7e6, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, -3.0],
                ).unwrap();
                let tolerances = Tolerances::scalar(8, tolerance, relative).unwrap()
                    .with_block(2..6, tolerance * 0.5, relative * 0.5).unwrap()
                    .with_block(6..8, tolerance * 2.0, relative).unwrap();
                let end = direction * 8.0;
                let mut step = direction;
                let mut rhs = |_: f64, y: &[f64], d: &mut [f64]| {
                    for (source, destination) in y.chunks_exact(2).zip(d.chunks_exact_mut(2)) {
                        destination[0] = source[1];
                        destination[1] = -source[0];
                    }
                    Ok::<_, Infallible>(())
                };
                for _ in 0..10000 {
                    if solver.time() == end { break; }
                    let old: [f64; 8] = solver.state().try_into().unwrap();
                    let view = solver.attempt_to(end, step, &mut rhs).unwrap();
                    let error = tolerances.error_norm(&old, view.candidate,
                        view.component_error.unwrap(), norm).unwrap();
                    if error <= 1.0 { solver.accept().unwrap(); }
                    else { solver.reject().unwrap(); }
                    step *= if error == 0.0 { 4.0 } else { (0.9 * error.powf(-0.2)).clamp(0.2, 4.0) };
                }
                assert_eq!(solver.time(), end);
                assert!(solver.statistics().rejected_steps > 0);
                let (sin, cos) = end.sin_cos();
                let exact = [7e6*cos, -7e6*sin, cos, -sin, sin, cos, 2.0*cos-3.0*sin, -2.0*sin-3.0*cos];
                for (&actual, expected) in solver.state().iter().zip(exact) {
                    assert!((actual-expected).abs() < (tolerance + relative * expected.abs()) * 60.0,
                        "{norm:?}, tolerance={tolerance}, direction={direction}: {actual} vs {expected}");
                }
            }
        }
    }
}

#[test]
fn host_decisions_match_independent_max_and_rms_formulas() {
    let old: [f64; 7] = [7e6, -2e6, 1.0, 7e3, -300.0, 1.0, 0.0];
    let new: [f64; 7] = [7e6+12.0, -2e6+5.0, 0.5, 7e3-2.0, -300.5, 0.99, 1e-5];
    let error: [f64; 7] = [2e-3, -3e-4, 1e-5, 3e-6, -1e-6, 2e-7, 4e-8];
    for (absolute, relative) in [(1e-3,1e-9),(1e-6,1e-12)] {
        let tolerances = Tolerances::scalar(7,absolute,relative).unwrap();
        let scaled: Vec<_> = (0..7).map(|i| error[i].abs()/(absolute+relative*old[i].abs().max(new[i].abs()))).collect();
        let rms = (scaled.iter().map(|e|e*e).sum::<f64>()/7.0).sqrt();
        let max = scaled.iter().copied().fold(0.0,f64::max);
        for (norm, expected) in [(ErrorNorm::Max,max),(ErrorNorm::Rms,rms)] {
            let actual=tolerances.error_norm(&old,&new,&error,norm).unwrap();
            assert!((actual-expected).abs() <= 1e-14*expected.max(1.0));
            assert_eq!(actual <= 1.0, expected <= 1.0);
        }
    }
}
