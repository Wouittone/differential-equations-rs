use differential_equations::{OdeAlgorithm, OdeProblem, SaveMode, SolveOptions, solve, solvers};

fn assert_algorithm<T: OdeAlgorithm>() {}

fn assert_same_type<T>(_: T, _: T) {}

#[test]
fn family_facades_reexport_the_implementation_types() {
    assert_same_type(solvers::explicit::Tsit5, solvers::explicit::tsit5::Tsit5);
    assert_same_type(
        solvers::explicit::RDPK3Sp35,
        solvers::explicit::low_storage_rk::RDPK3Sp35,
    );
    assert_same_type(
        solvers::implicit::Kvaerno5,
        solvers::implicit::sdirk::Kvaerno5,
    );
    assert_same_type(
        solvers::rosenbrock::Rodas3P,
        solvers::rosenbrock::rosenbrock_extended::Rodas3P,
    );
    assert_same_type(
        solvers::second_order::VelocityVerlet,
        solvers::second_order::general::VelocityVerlet,
    );
    assert_same_type(
        solvers::explicit::SspRkMsvs43,
        solvers::explicit::ssprk_msvs::SspRkMsvs43,
    );
}

#[test]
fn canonical_solver_paths_include_the_implementation_module() {
    assert_same_type(solvers::explicit::Rk4, solvers::explicit::general::Rk4);
    assert_same_type(solvers::implicit::Sdirk2, solvers::implicit::sdirk::Sdirk2);
    assert_same_type(solvers::multistep::Qndf1, solvers::multistep::qndf1::Qndf1);
    assert_same_type(
        solvers::rosenbrock::Rosenbrock23,
        solvers::rosenbrock::general::Rosenbrock23,
    );
}

#[test]
fn namespaced_exports_are_real_ode_algorithms() {
    assert_algorithm::<solvers::explicit::CKLLSRK54_3C>();
    assert_algorithm::<solvers::explicit::RDPK3SpFSAL510>();
    assert_algorithm::<solvers::implicit::Esdirk436L2Sa2>();
    assert_algorithm::<solvers::implicit::KenCarp4>();
    assert_algorithm::<solvers::implicit::Kvaerno5>();
    assert_algorithm::<solvers::rosenbrock::Rodas3P>();
    assert_algorithm::<solvers::rosenbrock::Ros4LStab>();
    assert_algorithm::<solvers::rosenbrock::Tsit5DA>();
    assert_algorithm::<solvers::explicit::SSPRKMSVS43>();
    assert_algorithm::<solvers::multistep::QNDF>();
    assert_algorithm::<solvers::multistep::QBDF>();
    assert_algorithm::<solvers::multistep::FBDF>();
}

#[test]
fn namespaced_algorithm_runs_through_the_public_driver() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -u[0],
        [1.0],
        (0.0, 0.1),
        (),
    );
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.01),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };

    let solution = solve(
        &problem,
        solvers::explicit::low_storage_rk::RDPK3Sp35,
        &options,
    )
    .expect("the concrete low-storage method should solve through its namespace");

    assert_eq!(solution.dimension(), 1);
    assert!((solution.last_state()[0] - (-0.1_f64).exp()).abs() < 1.0e-6);
}
