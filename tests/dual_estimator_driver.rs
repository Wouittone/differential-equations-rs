use differential_equations::{
    OdeProblem, SaveMode, SolveOptions, solve,
    solvers::explicit::ResourceExplicitRungeKutta,
    stepping::*,
    tableau::{LazyTableau, parse_tableau},
};
use std::{convert::Infallible, sync::LazyLock};
static PRIMARY: LazyTableau = LazyLock::new(|| {
    parse_tableau(
        r#"{"name":"Primary","description":"Heun with primary estimator dominant","kind":"explicit-runge-kutta","order":2,"embedded_order":1,"A":[[0,0],[1,0]],"b":[0.5,0.5],"c":[0,1],"error":[-0.5,0.5],"second_error":[-0.0005,0.0005]}"#,
        "Primary",
    )
});
static SECONDARY: LazyTableau = LazyLock::new(|| {
    parse_tableau(
        r#"{"name":"Secondary","description":"Heun with secondary estimator dominant","kind":"explicit-runge-kutta","order":2,"embedded_order":1,"A":[[0,0],[1,0]],"b":[0.5,0.5],"c":[0,1],"error":[0,0],"second_error":[-0.5,0.5]}"#,
        "Secondary",
    )
});
#[test]
fn both_estimator_dominance_cases_reproduce_native_acceptance_sequences() {
    for resource in [&PRIMARY, &SECONDARY] {
        let method = ResourceExplicitRungeKutta::new(resource);
        let tab = method.tableau().unwrap();
        let p = OdeProblem::new(
            |d: &mut [f64], y: &[f64], _: &(), _: f64| d[0] = y[0],
            [1.],
            (0., 0.25),
            (),
        );
        let options = SolveOptions::new()
            .with_tolerances(1e-6, 1e-6)
            .with_initial_step(0.125)
            .with_save(SaveMode::EveryStep);
        let options = options.with_max_step(0.125);
        let native = solve(&p, method, &options).unwrap();
        let mut s = ExplicitRungeKuttaStepper::new(tab, 0., &[1.]).unwrap();
        let mut config = ControllerConfig::proportional(2).unwrap();
        config.maximum_step = 0.125;
        let mut c = AdaptiveController::new(config, 0.125).unwrap();
        let mut times = vec![0.];
        let mut norm_calls = 0;
        integrate_rk(
            &mut s,
            &mut c,
            0.25,
            &[],
            1000,
            &mut |_: f64, y: &[f64], d: &mut [f64]| {
                d[0] = y[0];
                Ok::<_, Infallible>(())
            },
            &mut |old: &[f64], new: &[f64], e: &[f64]| {
                norm_calls += 1;
                Ok(e[0].abs() / (1e-6 + 1e-6 * old[0].abs().max(new[0].abs())))
            },
            &mut |o: Observation<'_>| {
                times.push(o.time);
                Ok(ObserverAction::Continue)
            },
        )
        .unwrap();
        assert!(native.stats().rejected_steps > 0);
        assert_eq!(s.statistics().accepted_steps, native.stats().accepted_steps);
        assert_eq!(s.statistics().rejected_steps, native.stats().rejected_steps);
        assert_eq!(norm_calls, 2 * s.statistics().attempts);
        assert_eq!(times.len(), native.times().len());
        for (a, b) in times.iter().zip(native.times()) {
            assert!((a - b).abs() < 1e-11, "{} {a} {b}", tab.name());
        }
        assert!((s.state()[0] - native.last_state()[0]).abs() < 1e-12);
    }
}
#[test]
fn dual_estimator_events_evaluate_condition_once_per_attempt() {
    let tab = ResourceExplicitRungeKutta::new(&SECONDARY)
        .tableau()
        .unwrap();
    let mut s = ExplicitRungeKuttaStepper::new(tab, 0., &[0.]).unwrap();
    let mut c = AdaptiveController::new(ControllerConfig::proportional(2).unwrap(), 0.7).unwrap();
    let mut conditions = 0;
    let mut norms = 0;
    let result = integrate_rk_until_event(
        &mut s,
        &mut c,
        1.,
        &[],
        100,
        RootOptions::default(),
        &mut |_: f64, _: &[f64], d: &mut [f64]| {
            d[0] = 1.;
            Ok::<_, Infallible>(())
        },
        &mut |_: &[f64], _: &[f64], _: &[f64]| {
            norms += 1;
            Ok(0.)
        },
        &mut |_: f64, y: &[f64]| {
            conditions += 1;
            Ok(y[0] - 0.3)
        },
        &mut |_| Ok(ObserverAction::Continue),
    )
    .unwrap();
    assert!(result.event_value.is_some());
    assert_eq!(conditions, s.statistics().attempts + 1);
    assert_eq!(norms, 2 * s.statistics().attempts);
}
#[test]
fn invalid_or_fallible_secondary_norm_is_never_hidden_by_primary() {
    let tab = ResourceExplicitRungeKutta::new(&SECONDARY)
        .tableau()
        .unwrap();
    for invalid in [f64::NAN, -1.] {
        let mut s = ExplicitRungeKuttaStepper::new(tab, 0., &[1.]).unwrap();
        let mut c =
            AdaptiveController::new(ControllerConfig::proportional(2).unwrap(), 0.1).unwrap();
        let mut calls = 0;
        let result = integrate_rk(
            &mut s,
            &mut c,
            1.,
            &[],
            10,
            &mut |_: f64, y: &[f64], d: &mut [f64]| {
                d[0] = y[0];
                Ok::<_, Infallible>(())
            },
            &mut |_: &[f64], _: &[f64], _: &[f64]| {
                calls += 1;
                Ok(if calls == 1 { 0. } else { invalid })
            },
            &mut |_| Ok(ObserverAction::Continue),
        );
        assert!(matches!(
            result,
            Err(IntegrationError::Controller(ControllerError::Configuration))
        ));
        assert_eq!(s.time(), 0.);
        assert_eq!(s.accept(), Err(StepFailure::NoCandidate));
    }
    let mut s = ExplicitRungeKuttaStepper::new(tab, 0., &[1.]).unwrap();
    let mut c = AdaptiveController::new(ControllerConfig::proportional(2).unwrap(), 0.1).unwrap();
    let mut calls = 0;
    let result = integrate_rk(
        &mut s,
        &mut c,
        1.,
        &[],
        10,
        &mut |_: f64, y: &[f64], d: &mut [f64]| {
            d[0] = y[0];
            Ok(())
        },
        &mut |_: &[f64], _: &[f64], _: &[f64]| {
            calls += 1;
            if calls == 1 { Ok(0.) } else { Err("secondary") }
        },
        &mut |_| Ok(ObserverAction::Continue),
    );
    assert!(matches!(
        result,
        Err(IntegrationError::Step(StepError::User("secondary")))
    ));
    assert_eq!(s.accept(), Err(StepFailure::NoCandidate));
}
