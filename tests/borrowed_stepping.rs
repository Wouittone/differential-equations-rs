use differential_equations::solvers::{explicit::Tsit5, second_order::FineRkn4};
use differential_equations::stepping::{AccelerationPolicy, ExplicitRungeKuttaStepper, RknStepper};
use std::convert::Infallible;

#[test]
fn array_state_is_borrowed_and_committed_only_on_acceptance() {
    let mut state = [1.0, 0.0];
    let pointer = state.as_ptr();
    {
        let mut solver = ExplicitRungeKuttaStepper::from_buffer(Tsit5.tableau().unwrap(), 0.0, &mut state).unwrap();
        assert_eq!(solver.state().as_ptr(), pointer);
        let mut rhs = |_: f64, y: &[f64], d: &mut [f64]| {
            d[0]=y[1]; d[1]=-y[0]; Ok::<_,Infallible>(())
        };
        solver.attempt(0.1,&mut rhs).unwrap();
        assert_eq!(solver.state(), &[1.0,0.0]);
        solver.reject().unwrap();
        solver.attempt(0.05,&mut rhs).unwrap();
        solver.accept().unwrap();
        assert_eq!(solver.state().as_ptr(), pointer);
    }
    assert!((state[0]-0.05f64.cos()).abs()<1e-11);
    assert!((state[1]+0.05f64.sin()).abs()<1e-11);
}

#[test]
fn dynamic_partitions_remain_caller_owned_through_restart() {
    let mut position=vec![1.0,2.0];
    let mut velocity=vec![0.0,1.0];
    let pointers=(position.as_ptr(),velocity.as_ptr());
    {
        let mut solver=RknStepper::from_buffers(FineRkn4.tableau().unwrap(),
            AccelerationPolicy::VelocityDependent,0.0,&mut position,&mut velocity).unwrap();
        let mut rhs=|_:f64,q:&[f64],v:&[f64],a:&mut[f64]| {
            for i in 0..q.len(){a[i]=-q[i]-v[i];} Ok::<_,Infallible>(())
        };
        solver.attempt(0.1,&mut rhs).unwrap();
        solver.reject().unwrap();
        assert_eq!(solver.position(),&[1.0,2.0]);
        assert_eq!(solver.velocity(),&[0.0,1.0]);
        solver.reset(1.0,&[3.0,4.0],&[5.0,6.0]).unwrap();
        assert_eq!((solver.position().as_ptr(),solver.velocity().as_ptr()),pointers);
        solver.attempt(0.0,&mut rhs).unwrap();
        solver.accept().unwrap();
    }
    assert_eq!(position,[3.0,4.0]);
    assert_eq!(velocity,[5.0,6.0]);
}
