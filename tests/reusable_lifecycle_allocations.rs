use differential_equations::{
    solvers::explicit::{Tsit5, Vern9},
    stepping::{
        AdaptiveController, ControllerConfig, ExplicitRungeKuttaStepper, ObserverAction,
        integrate_rk,
    },
};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};
use std::{alloc::System, convert::Infallible, hint::black_box};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

fn scalar_rhs(_: f64, state: &[f64], derivative: &mut [f64]) -> Result<(), Infallible> {
    derivative[0] = -state[0];
    Ok(())
}

fn scalar_norm(_: &[f64], _: &[f64], error: &[f64]) -> Result<f64, Infallible> {
    Ok(error[0].abs() / 1.0e-10)
}

/// Velocity-coupled orbital acceleration: gravity plus a drag term that
/// depends on speed, matching the benchmark's workload so this regression
/// exercises the same state/velocity-dependent RHS shape.
fn orbit_rhs(_: f64, state: &[f64], derivative: &mut [f64]) -> Result<(), Infallible> {
    let (x, y, z, vx, vy, vz) = (state[0], state[1], state[2], state[3], state[4], state[5]);
    let radius = (x * x + y * y + z * z).sqrt().max(1.0e-12);
    let scale = 1.0 / (radius * radius * radius);
    let speed = (vx * vx + vy * vy + vz * vz).sqrt();
    let drag = 0.1;
    derivative[0] = vx;
    derivative[1] = vy;
    derivative[2] = vz;
    derivative[3] = -scale * x - drag * speed * vx;
    derivative[4] = -scale * y - drag * speed * vy;
    derivative[5] = -scale * z - drag * speed * vz;
    Ok(())
}

fn orbit_norm(_: &[f64], _: &[f64], error: &[f64]) -> Result<f64, Infallible> {
    Ok(error
        .iter()
        .copied()
        .fold(0.0_f64, |worst, component| worst.max(component.abs()))
        / 1.0e-10)
}

fn measure_scalar_arc(
    tableau: &differential_equations::tableau::RungeKuttaTableau,
    endpoint: f64,
) -> (usize, usize) {
    let mut state = [1.0];
    let mut stepper = ExplicitRungeKuttaStepper::from_buffer(tableau, 0.0, &mut state).unwrap();
    let mut controller =
        AdaptiveController::new(ControllerConfig::proportional(5).unwrap(), 0.1).unwrap();
    let run = |stepper: &mut ExplicitRungeKuttaStepper<'_>, controller: &mut AdaptiveController| {
        integrate_rk(
            stepper,
            controller,
            endpoint,
            &[],
            100_000,
            &mut scalar_rhs,
            &mut scalar_norm,
            &mut |_| Ok::<_, Infallible>(ObserverAction::Continue),
        )
        .unwrap();
        black_box(stepper.state()[0]);
    };
    run(&mut stepper, &mut controller);
    stepper.reset(0.0, &[1.0]).unwrap();
    controller.reset(0.1).unwrap();
    let region = Region::new(GLOBAL);
    run(&mut stepper, &mut controller);
    let change = region.change();
    (
        change.allocations + change.reallocations,
        change.bytes_allocated,
    )
}

fn measure_orbit_arc(
    tableau: &differential_equations::tableau::RungeKuttaTableau,
    endpoint: f64,
) -> (usize, usize) {
    // Matches the benchmark's ORBIT_INITIAL_STATE: unit radius with a
    // tangential velocity component so this exercises a genuine (drag-decaying)
    // orbit rather than radial free-fall.
    let mut state = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let mut stepper = ExplicitRungeKuttaStepper::from_buffer(tableau, 0.0, &mut state).unwrap();
    let mut controller =
        AdaptiveController::new(ControllerConfig::proportional(5).unwrap(), 0.1).unwrap();
    let run = |stepper: &mut ExplicitRungeKuttaStepper<'_>, controller: &mut AdaptiveController| {
        integrate_rk(
            stepper,
            controller,
            endpoint,
            &[],
            100_000,
            &mut orbit_rhs,
            &mut orbit_norm,
            &mut |_| Ok::<_, Infallible>(ObserverAction::Continue),
        )
        .unwrap();
        black_box(stepper.state()[0] + stepper.state()[3]);
    };
    run(&mut stepper, &mut controller);
    stepper.reset(0.0, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
    controller.reset(0.1).unwrap();
    let region = Region::new(GLOBAL);
    run(&mut stepper, &mut controller);
    let change = region.change();
    (
        change.allocations + change.reallocations,
        change.bytes_allocated,
    )
}

fn minimum_measurement(
    endpoint: f64,
    measure: impl Fn(&differential_equations::tableau::RungeKuttaTableau, f64) -> (usize, usize),
    tableau: &differential_equations::tableau::RungeKuttaTableau,
) -> (usize, usize) {
    (0..3)
        .map(|_| measure(tableau, endpoint))
        .min_by_key(|&(allocations, bytes)| (allocations, bytes))
        .unwrap()
}

#[test]
fn reusable_downstream_workspace_has_zero_steady_state_allocations() {
    for tableau in [Tsit5.tableau().unwrap(), Vern9.tableau().unwrap()] {
        let short = minimum_measurement(1.0, measure_scalar_arc, tableau);
        let long = minimum_measurement(10.0, measure_scalar_arc, tableau);
        assert_eq!(short, (0, 0));
        assert_eq!(long, (0, 0));
    }

    for tableau in [Tsit5.tableau().unwrap(), Vern9.tableau().unwrap()] {
        let orbit = minimum_measurement(1.0, measure_orbit_arc, tableau);
        assert_eq!(orbit, (0, 0));
    }
}
