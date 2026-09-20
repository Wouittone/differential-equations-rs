use differential_equations::solvers::explicit::Tsit5;
use differential_equations::stepping::*;
use std::convert::Infallible;

#[test]
fn terminal_event_keeps_requested_output_and_localizes_backward_too() {
    for direction in [-1.0,1.0] {
        let mut state=[1.0,0.0];
        let mut solver=ExplicitRungeKuttaStepper::from_buffer(Tsit5.tableau().unwrap(),0.0,&mut state).unwrap();
        let mut controller=AdaptiveController::new(ControllerConfig::proportional(5).unwrap(),direction*0.25).unwrap();
        let requested=[direction*0.5,direction*1.0,direction*2.0];
        let mut saved=Vec::new();
        let result=integrate_rk_until_event(&mut solver,&mut controller,direction*3.0,
            &requested,1000,RootOptions::default(),
            &mut |_:f64,y:&[f64],d:&mut[f64]|{d[0]=y[1];d[1]=-y[0];Ok::<_,Infallible>(())},
            &mut |_:&[f64],_:&[f64],e:&[f64]|Ok(e[0].abs().max(e[1].abs())/1e-11),
            &mut |_:f64,y:&[f64]|Ok(y[0]),
            &mut |o:Observation<'_>|{if o.requested{saved.push(o.time);}Ok(ObserverAction::Continue)}).unwrap();
        assert!(result.integration.interrupted);
        assert!(result.event_value.unwrap().abs()<1e-9);
        assert!((solver.time()-direction*std::f64::consts::FRAC_PI_2).abs()<1e-9);
        assert_eq!(saved,requested[..2]);
        assert!(solver.state()[0].abs()<1e-9);
        assert!((solver.state()[1]+direction).abs()<1e-9);
    }
}

#[test]
fn condition_payload_is_preserved_and_failed_attempt_is_not_committed() {
    let mut solver=ExplicitRungeKuttaStepper::new(Tsit5.tableau().unwrap(),0.0,&[1.0]).unwrap();
    let mut controller=AdaptiveController::new(ControllerConfig::proportional(5).unwrap(),0.1).unwrap();
    let result=integrate_rk_until_event(&mut solver,&mut controller,1.0,&[],100,RootOptions::default(),
        &mut |_:f64,y:&[f64],d:&mut[f64]|{d[0]=-y[0];Ok(())},
        &mut |_:&[f64],_:&[f64],_:&[f64]|Ok(0.0),
        &mut |t:f64,_:&[f64]| if t==0.0 {Ok(1.0)}else{Err("event payload")},
        &mut |_|Ok(ObserverAction::Continue));
    assert!(matches!(result,Err(RootError::Integration(IntegrationError::Step(StepError::User("event payload"))))));
    assert_eq!(solver.time(),0.0);
    assert_eq!(solver.state(),&[1.0]);
    assert_eq!(solver.accept(),Err(StepFailure::NoCandidate));
}

#[test]
fn direction_filter_and_initial_zero_are_explicit() {
    for rising in [false,true] {
        let mut solver=ExplicitRungeKuttaStepper::new(Tsit5.tableau().unwrap(),0.0,&[-0.5]).unwrap();
        let mut controller=AdaptiveController::new(ControllerConfig::proportional(5).unwrap(),0.75).unwrap();
        let result=integrate_rk_until_event(&mut solver,&mut controller,1.0,&[],100,
            RootOptions{direction:if rising{RootDirection::Rising}else{RootDirection::Falling},..Default::default()},
            &mut |_:f64,_:&[f64],d:&mut[f64]|{d[0]=1.0;Ok::<_,Infallible>(())},
            &mut |_:&[f64],_:&[f64],_:&[f64]|Ok(0.0),
            &mut |_:f64,y:&[f64]|Ok(y[0]), &mut |_|Ok(ObserverAction::Continue)).unwrap();
        assert_eq!(result.event_value.is_some(),rising);
        assert!((solver.time()-if rising{0.5}else{1.0}).abs()<1e-9);
    }
}
