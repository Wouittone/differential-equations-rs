//! Independent analytic regression oracles for absolute-epoch differentiation.
use differential_equations::solvers::rosenbrock::Rodas4;
use differential_equations::{OdeProblem, SaveMode, SolveOptions, solve};

#[test]
fn shifted_cubic_forcing_is_independent_of_absolute_epoch() {
    for origin in [0.0, 1.0e9, 1.0e12] {
        for direction in [-1.0, 1.0] {
            let problem = OdeProblem::new(
                |dy: &mut [f64], _: &[f64], origin: &f64, t: f64| {
                    dy[0] = 3.0 * (t - origin).powi(2);
                },
                [0.0], (origin, origin + direction), origin,
            ).with_jacobian(|j: &mut [f64], _: &[f64], _: &f64, _: f64| j[0] = 0.0);
            let solution = solve(&problem, Rodas4, &SolveOptions::new()
                .with_adaptive(false).with_initial_step(0.0625)
                .with_save(SaveMode::Endpoints)).unwrap();
            let error = (solution.last_state()[0] - direction).abs();
            // At 1e12, binary64 time quantization is 1.22e-4 s. This bound
            // includes that input uncertainty; it does not scale with epoch.
            assert!(error < 5e-4, "origin={origin} direction={direction} error={error:e}");
        }
    }
}

#[test]
fn time_differentiation_does_not_probe_outside_the_step_domain() {
    for direction in [-1.0, 1.0] {
        let start: f64 = 1.0e9;
        let end = start + direction * 0.5;
        let lower = start.min(end);
        let upper = start.max(end);
        let problem = OdeProblem::new(
            move |dy: &mut [f64], _: &[f64], _: &(), t: f64| {
                assert!((lower..=upper).contains(&t), "probe {t} outside [{lower},{upper}]");
                dy[0] = t - start;
            }, [0.0], (start, end), (),
        ).with_jacobian(|j: &mut [f64], _: &[f64], _: &(), _: f64| j[0] = 0.0);
        let solution = solve(&problem, Rodas4, &SolveOptions::new()
            .with_adaptive(false).with_initial_step(0.125)
            .with_save(SaveMode::Endpoints)).unwrap();
        assert!((solution.last_state()[0] - 0.125).abs() < 1e-7);
    }
}
