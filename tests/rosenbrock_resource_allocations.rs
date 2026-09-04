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
        black_box(Tsit5DA);
        region.change().allocations
    });
    assert_eq!(construction, 0);
    let pair = Region::new(GLOBAL);
    black_box(Rosenbrock23.tableau().unwrap());
    assert!(pair.change().allocations > 0);
    let pair_cached = Region::new(GLOBAL);
    black_box(Rosenbrock32.tableau().unwrap());
    assert_eq!(pair_cached.change().allocations, 0);
    assert!(std::ptr::eq(
        Rosenbrock23.tableau().unwrap(),
        Rosenbrock32.tableau().unwrap()
    ));
    let first = Region::new(GLOBAL);
    black_box(Ros2.tableau().unwrap());
    assert!(first.change().allocations > 0);
    // Parsing one Rosenbrock method must not parse other Rosenbrock methods.
    let second = Region::new(GLOBAL);
    black_box(Rodas6P.tableau().unwrap());
    assert!(second.change().allocations > 0);
    let hybrid = Region::new(GLOBAL);
    black_box(Tsit5DA.tableau().unwrap());
    assert!(hybrid.change().allocations > 0);
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
            black_box(HybridExplicitImplicitRK.tableau().unwrap());
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
        Tsit5DA,
        Rodas5Pr
    );
    check_hybrid_memory_scaling();
}

fn hybrid_solve_bytes(shape: &[usize], adaptive: bool, dense: bool) -> usize {
    use differential_equations::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD, IxDyn};
    let problem = OdeProblem::from_array(
        |mut du: ArrayViewMutD<'_, f64>, u: ArrayViewD<'_, f64>, _: &(), _| {
            du.zip_mut_with(&u, |du, u| *du = -*u);
        },
        ArrayD::from_elem(IxDyn(shape), 1.0),
        (0.0, 0.02),
        (),
    )
    .with_jacobian(|_, _, _, _| panic!("hybrid ODE solve must not request a Jacobian"));
    let options = SolveOptions::new()
        .with_adaptive(adaptive)
        .with_initial_step(0.01)
        .with_max_step(0.01)
        .with_tolerances(1e-8, 1e-8)
        .with_save(SaveMode::Endpoints)
        .with_dense_output(dense);
    allocation_support::minimum_measurement(|| {
        let region = Region::new(GLOBAL);
        let solution = solve(&problem, Tsit5DA, &options).unwrap();
        let bytes = region.change().bytes_allocated;
        assert_eq!(solution.last_state_array().shape(), shape);
        assert_eq!(solution.stats().accepted_steps, 2);
        assert_eq!(solution.stats().rejected_steps, 0);
        assert_eq!(solution.stats().jacobian_evaluations, 0);
        assert_eq!(solution.stats().linear_factorizations, 0);
        for value in solution.last_state() {
            assert!((value - (-0.02_f64).exp()).abs() < 1e-8);
        }
        if dense {
            let sample = solution.interpolate_array(0.015).unwrap();
            assert_eq!(sample.shape(), shape);
            for value in &sample {
                assert!((value - (-0.015_f64).exp()).abs() < 1e-5);
            }
        }
        bytes
    })
}

fn check_hybrid_memory_scaling() {
    // One decay ODE in scalar, vector, and non-square matrix forms. Count
    // allocated bytes, not just allocation calls: two dense n*n matrices
    // pass a step-count test but make large explicit ODE states impractical.
    for adaptive in [false, true] {
        for dense in [false, true] {
            black_box(hybrid_solve_bytes(&[], adaptive, dense));
            for (small_shape, large_shape) in [(vec![128], vec![256]), (vec![4, 32], vec![8, 32])] {
                let small = hybrid_solve_bytes(&small_shape, adaptive, dense);
                let large = hybrid_solve_bytes(&large_shape, adaptive, dense);
                assert!(
                    large <= 2 * small + 4096,
                    "hybrid memory is not linear: {small} -> {large} bytes, adaptive={adaptive}, dense={dense}"
                );
            }
        }
    }
}
