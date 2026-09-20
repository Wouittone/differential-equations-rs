use differential_equations::{
    ScopedOdeProblem, ScopedSecondOrderProblem,
    solvers::{explicit::Tsit5, rosenbrock::Rodas4},
    stepping::*,
};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};
use std::{alloc::System, convert::Infallible};
#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;
fn controller(order: usize) -> AdaptiveController {
    AdaptiveController::new(ControllerConfig::proportional(order).unwrap(), 0.1).unwrap()
}
#[test]
fn scoped_rk_rkn_and_rosenbrock_solves_reuse_all_storage() {
    let mut rk_problem = ScopedOdeProblem::new(
        |_: f64, y: &[f64], dy: &mut [f64]| {
            dy[0] = -y[0];
            Ok::<_, Infallible>(())
        },
        [1.0],
        (0.0, 1.0),
    );
    let mut rk = ExplicitRungeKuttaStepper::new(Tsit5.tableau().unwrap(), 0.0, &[1.0]).unwrap();
    let mut rk_control = controller(5);
    let table = differential_equations::tableau::parse_rkn_tableau(
        include_str!("../src/tableau/resources/second_order/fine-rkn4.json"),
        "FineRkn4",
    )
    .unwrap();
    let mut rkn_problem = ScopedSecondOrderProblem::new(
        |_: f64, q: &[f64], _: &[f64], a: &mut [f64]| {
            a[0] = -q[0];
            Ok::<_, Infallible>(())
        },
        [0.0],
        [1.0],
        (0.0, 1.0),
    );
    let mut rkn = RknStepper::new(
        &table,
        AccelerationPolicy::VelocityDependent,
        0.0,
        &[0.0],
        &[1.0],
    )
    .unwrap();
    let mut rkn_control = controller(4);
    let mut ros_problem = ScopedOdeProblem::new(
        |_: f64, y: &[f64], dy: &mut [f64]| {
            dy[0] = -y[0];
            Ok::<_, Infallible>(())
        },
        [1.0],
        (0.0, 1.0),
    )
    .with_jacobian(|_: f64, _: &[f64], j: &mut [f64]| {
        j[0] = -1.0;
        Ok(())
    });
    let mut ros = RosenbrockStepper::new(Rodas4.tableau().unwrap(), 0.0, &[1.0]).unwrap();
    let mut ros_control = controller(4);
    let region = Region::new(GLOBAL);
    for _ in 0..3 {
        rk_problem
            .integrate(
                &mut rk,
                &mut rk_control,
                &[0.25, 0.5, 1.0],
                1000,
                &mut |_: &[f64], _: &[f64], e: &[f64]| Ok(e[0].abs() / 1e-10),
            )
            .unwrap();
        rkn_problem
            .integrate(
                &mut rkn,
                &mut rkn_control,
                &[0.25, 0.5, 1.0],
                1000,
                &mut |v: &RknStepView<'_>| {
                    Ok(v.position_error.unwrap()[0]
                        .abs()
                        .max(v.velocity_error.unwrap()[0].abs())
                        / 1e-10)
                },
            )
            .unwrap();
        ros_problem
            .integrate_rosenbrock(
                &mut ros,
                &mut ros_control,
                &[0.25, 0.5, 1.0],
                1000,
                None,
                &mut |v: &RosenbrockStepView<'_>| Ok(v.component_error.unwrap()[0].abs() / 1e-10),
            )
            .unwrap();
    }
    assert_eq!(region.change().allocations, 0);
    assert_eq!(region.change().reallocations, 0);
    assert!((rk.state()[0] - (-1.0f64).exp()).abs() < 1e-8);
    assert!((rkn.position()[0] - 1.0f64.sin()).abs() < 1e-8);
    assert!((ros.state()[0] - (-1.0f64).exp()).abs() < 1e-8);
}
