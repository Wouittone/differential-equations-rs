use std::convert::Infallible;

#[test]
fn borrowed_state_typed_errors_and_continuation_are_public() {
    use diffeq::stepping::{AdaptiveController, ControllerConfig, ExplicitRungeKuttaStepper, ObserverAction, integrate_rk};
    use diffeq::tolerances::{ErrorNorm, Tolerances};
    let mut state=[1.0];
    let tableau=diffeq::solvers::explicit::Tsit5.tableau().unwrap();
    let tolerances=Tolerances::scalar(1,1e-11,1e-11).unwrap();
    let mut stepper=ExplicitRungeKuttaStepper::from_buffer(tableau,0.0,&mut state).unwrap();
    let mut controller=AdaptiveController::new(ControllerConfig::proportional(5).unwrap(),0.1).unwrap();
    let mut calls=0;
    let mut rhs=|_:f64,y:&[f64],d:&mut[f64]|{calls+=1;d[0]=-y[0];Ok::<_,Infallible>(())};
    let mut norm=|old:&[f64],new:&[f64],error:&[f64]|Ok(tolerances.error_norm(old,new,error,ErrorNorm::Rms).unwrap());
    for end in [0.5,1.0] {
        integrate_rk(&mut stepper,&mut controller,end,&[],1000,&mut rhs,&mut norm,&mut |_|Ok(ObserverAction::Continue)).unwrap();
    }
    assert!((stepper.state()[0]-(-1.0_f64).exp()).abs()<1e-10);
    assert!(calls>0);
    drop(stepper);
    assert!((state[0]-(-1.0_f64).exp()).abs()<1e-10);
}

#[test]
fn fixed_history_method_is_public() {
    let mut solver=diffeq::GaussJackson8::new(0.0,&[1.0],&[0.0],0.05,Default::default()).unwrap();
    let mut force=|_:f64,q:&[f64],_:&[f64],a:&mut[f64]|{a[0]=-q[0];Ok::<_,Infallible>(())};
    while solver.time()<1.0 { solver.try_step_to(1.0,&mut force).unwrap(); }
    assert!((solver.position()[0]-1.0_f64.cos()).abs()<1e-10);
    assert!(solver.statistics().accepted_steps>8);
    let owned=diffeq::solvers::explicit::Tsit5.tableau().unwrap();
    assert!(owned.stages()>0);
}
