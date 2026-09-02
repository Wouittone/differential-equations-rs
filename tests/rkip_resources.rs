use differential_equations::solvers::exponential::{RKIP, solve_rkip};
use differential_equations::{SemilinearOdeProblem, SolveOptions};

type NonlinearFunction = fn(&mut [f64], &[f64], &(), f64);

fn coupled_problem(span: (f64, f64)) -> SemilinearOdeProblem<NonlinearFunction, ()> {
    SemilinearOdeProblem::<NonlinearFunction, ()>::new(
        vec![-1.0, 0.0, 0.0, -2.0],
        |out: &mut [f64], u: &[f64], _: &(), t: f64| {
            out[0] = t.sin() + u[1] * u[1];
            out[1] = 0.2 * u[0] * u[1];
        },
        vec![0.2, 0.4],
        span,
        (),
    )
    .unwrap()
}

fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < tolerance,
            "{actual} != {expected}"
        );
    }
}

#[test]
fn resource_migration_preserves_coupled_nonautonomous_trajectories() {
    // Samples recorded with the original Rust coefficient constants.
    for (adaptive, span, endpoint, midpoint) in [
        (
            false,
            (0.0, 0.5),
            [0.25101517307900234, 0.1503189910021122],
            [0.20651454091632512, 0.24505397106451157],
        ),
        (
            false,
            (0.5, 0.0),
            [-0.14425035384330737, 1.0807305355155892],
            [0.07586012538075602, 0.6548501743473275],
        ),
        (
            true,
            (0.0, 0.5),
            [0.25101517308573773, 0.15031899100344584],
            [0.2065169849955922, 0.24505219526826122],
        ),
    ] {
        let problem = coupled_problem(span);
        let algorithm = RKIP::new(0.05, 0.1, 2).unwrap().with_clamping(false, false);
        let options = SolveOptions::new()
            .with_adaptive(adaptive)
            .with_initial_step(0.05)
            .with_tolerances(1e-10, 1e-10)
            .with_max_steps(200)
            .with_dense_output(true);
        let solution = solve_rkip(&problem, &algorithm, &options).unwrap();
        assert_close(solution.last_state(), &endpoint, 1e-12);
        // Adaptive dense segments may shift slightly with platform rounding.
        assert_close(
            &solution.interpolate(0.25).unwrap(),
            &midpoint,
            if adaptive { 1e-8 } else { 1e-12 },
        );
    }
}

#[test]
fn adaptive_retries_can_shrink_below_cache_nodes_and_the_lower_cache_bound() {
    for span in [(0.0, 0.5), (0.5, 0.0)] {
        let problem = coupled_problem(span);
        let options = SolveOptions::new()
            .with_initial_step(0.1)
            .with_tolerances(1e-10, 1e-10)
            .with_max_steps(200);
        // A tight solve entirely below this uncapped grid provides a reference
        // without step snapping. The sparse grid previously stalled backward.
        let reference = solve_rkip(
            &problem,
            &RKIP::new(1.0, 2.0, 2).unwrap().with_clamping(false, false),
            &options.clone().with_tolerances(1e-13, 1e-13),
        )
        .unwrap();
        for clamp_lower in [false, true] {
            let algorithm = RKIP::new(0.1, 0.2, 2)
                .unwrap()
                .with_clamping(clamp_lower, true);
            let result = solve_rkip(&problem, &algorithm, &options).unwrap();
            assert_eq!(result.times().last(), Some(&span.1));
            assert!(result.stats().rejected_steps > 0);
            assert_close(result.last_state(), reference.last_state(), 2e-9);
        }
    }
}

#[test]
fn repeated_resource_nodes_share_exponentials_and_reuse_them_across_solves() {
    let problem = SemilinearOdeProblem::new(
        vec![-1.0],
        |du: &mut [f64], _: &[f64], _: &(), _: f64| du.fill(0.0),
        vec![1.0],
        (0.0, 0.1),
        (),
    )
    .unwrap();
    let algorithm = RKIP::new(0.1, 0.2, 2).unwrap();
    let tableau = algorithm.tableau().unwrap();
    assert_eq!(tableau.order(), algorithm.order());
    assert_eq!(tableau.embedded_order(), Some(algorithm.adaptive_order()));
    assert!(std::ptr::eq(tableau, RKIP::default().tableau().unwrap()));
    let options = SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(0.1);
    let first = solve_rkip(&problem, &algorithm, &options).unwrap();
    let before = algorithm.cache_stats();
    // Six distinct nonzero nodes, for each exponential sign; the duplicate
    // endpoint node and the two endpoint actions reuse those slots.
    assert_eq!(before.exponentials_built, 12);
    assert_eq!(before.cached_step_sizes, 1);
    assert_close(first.last_state(), &[(-0.1_f64).exp()], 1e-14);
    assert_eq!(solve_rkip(&problem, &algorithm, &options).unwrap(), first);
    assert_eq!(
        algorithm.cache_stats().exponentials_built,
        before.exponentials_built
    );
    assert!(algorithm.cache_stats().cache_hits > before.cache_hits);
}

mod canonical {
    use differential_equations::tableau::define_explicit_rk_from_file;
    define_explicit_rk_from_file!(pub RKIP, "src/tableau/resources/explicit/rkip.json");
}

#[test]
fn the_same_resource_defines_a_regular_explicit_solver() {
    use differential_equations::ndarray::{ArrayViewD, arr0, array};
    use differential_equations::{OdeProblem, solve};
    fn nonlinear(du: &mut [f64], u: &[f64], _: &(), t: f64) {
        for (du, u) in du.iter_mut().zip(u) {
            *du = u * (1.0 - u) + t.sin();
        }
    }
    let options = SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(0.05);
    for initial in [
        arr0(0.2).into_dyn(),
        array![0.2, 0.4].into_dyn(),
        array![[0.2, 0.3], [0.4, 0.5]].into_dyn(),
    ] {
        let ordinary = OdeProblem::from_array_out_of_place(
            |u: ArrayViewD<'_, f64>, _: &(), t: f64| u.mapv(|u| u * (1.0 - u) + t.sin()),
            initial.clone(),
            (0.0, 0.5),
            (),
        );
        // The same componentwise ODE in RKIP's flat semilinear representation.
        let semilinear = SemilinearOdeProblem::new(
            vec![0.0; initial.len() * initial.len()],
            nonlinear,
            initial.iter().copied().collect::<Vec<_>>(),
            (0.0, 0.5),
            (),
        )
        .unwrap();
        let explicit = solve(&ordinary, canonical::RKIP, &options).unwrap();
        let interaction =
            solve_rkip(&semilinear, &RKIP::new(0.05, 0.1, 2).unwrap(), &options).unwrap();
        assert_eq!(explicit.state_shape(), initial.shape());
        assert_eq!(explicit.last_state_array().shape(), initial.shape());
        assert_eq!(explicit.times(), interaction.times());
        assert_close(explicit.last_state(), interaction.last_state(), 1e-14);
    }
}
