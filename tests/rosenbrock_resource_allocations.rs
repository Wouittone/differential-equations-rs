use differential_equations::solvers::explicit::Rk4;
use differential_equations::solvers::rosenbrock::*;
use differential_equations::{OdeAlgorithm, OdeProblem, SaveMode, SolveOptions, solve};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};
use std::alloc::System;
use std::hint::black_box;

#[path = "support/allocation.rs"]
mod allocation_support;

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

fn allocations<A: OdeAlgorithm + Copy>(method: A, step: f64) -> usize {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _| du[0] = -u[0],
        [1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(step)
        .with_save(SaveMode::Endpoints);
    allocation_support::minimum_measurement(|| {
        let region = Region::new(GLOBAL);
        black_box(solve(&problem, method, &options).unwrap());
        region.change().allocations
    })
}

#[test]
fn rosenbrock_resources_are_individually_lazy_and_cached() {
    let construction = allocation_support::minimum_measurement(|| {
        let region = Region::new(GLOBAL);
        black_box(Ros2);
        black_box(Rodas6P);
        region.change().allocations
    });
    assert_eq!(construction, 0);
    let first = Region::new(GLOBAL);
    black_box(Ros2.tableau().unwrap());
    assert!(first.change().allocations > 0);
    // Parsing one Rosenbrock method must not parse other Rosenbrock methods.
    let second = Region::new(GLOBAL);
    black_box(Rodas6P.tableau().unwrap());
    assert!(second.change().allocations > 0);
    let unrelated = Region::new(GLOBAL);
    black_box(Rk4.tableau().unwrap());
    assert!(unrelated.change().allocations > 0);
    let primary = Region::new(GLOBAL);
    black_box(Rodas5P.tableau().unwrap());
    assert!(primary.change().allocations > 0);
    let repeated = allocation_support::minimum_measurement(|| {
        let region = Region::new(GLOBAL);
        for _ in 0..1000 {
            black_box(Ros2.tableau().unwrap());
            black_box(Rodas6P.tableau().unwrap());
            black_box(Rodas5Pr.tableau().unwrap()); // Shares Rodas5P.
        }
        region.change().allocations
    });
    assert_eq!(repeated, 0);
    macro_rules! check_steps {
        ($($method:ident),+ $(,)?) => {$({
            black_box($method.tableau().unwrap()); // Exclude first-use parsing.
            assert_eq!(allocations($method, 0.01), allocations($method, 0.001), stringify!($method));
        })+};
    }
    check_steps!(
        Ros2,
        Rodas3,
        Rodas3d,
        Ros3,
        Ros3Pr,
        Ros3Prl,
        Ros3Prl2,
        Ros3p,
        Ros34Prw,
        Ros34Pw3,
        Grk4a,
        Grk4t,
        Rok4a,
        Ros34Pw1b,
        Ros34Pw2,
        Rodas4,
        Rodas42,
        Rodas4P,
        Rodas4P2,
        Rodas4PW,
        Rodas5,
        Rodas5P,
        Rodas5Pe,
        Rodas6P,
        RosenbrockW6S4OS,
        Rodas23W,
        Rodas3P,
        Ros2Pr,
        Ros2S,
        Ros34Pw1a,
        Ros4LStab,
        RosShamp4,
        Scholz4_7,
        Veldd4,
        Velds4,
        Rodas5Pr
    );
}
