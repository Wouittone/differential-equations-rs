use differential_equations::{
    OdeAlgorithm, OdeProblem, SaveMode, SolveOptions, algorithms, solve, solvers,
};

fn assert_algorithm<T: OdeAlgorithm>() {}

fn assert_same_type<T>(_: T, _: T) {}

#[test]
fn family_namespaces_reexport_the_flat_algorithm_types() {
    assert_same_type(algorithms::Tsit5, algorithms::explicit::general::Tsit5);
    assert_same_type(
        algorithms::RDPK3Sp35,
        algorithms::explicit::low_storage::RDPK3Sp35,
    );
    assert_same_type(
        algorithms::Kvaerno5,
        algorithms::implicit::diagonally_implicit::Kvaerno5,
    );
    assert_same_type(algorithms::Rodas3P, algorithms::rosenbrock::Rodas3P);
    assert_same_type(
        algorithms::VelocityVerlet,
        algorithms::second_order::VelocityVerlet,
    );
    assert_same_type(
        algorithms::SspRkMsvs43,
        algorithms::explicit::ssp::SspRkMsvs43,
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
    assert_algorithm::<algorithms::CKLLSRK54_3C>();
    assert_algorithm::<algorithms::RDPK3SpFSAL510>();
    assert_algorithm::<algorithms::Esdirk436L2Sa2>();
    assert_algorithm::<algorithms::KenCarp4>();
    assert_algorithm::<algorithms::Kvaerno5>();
    assert_algorithm::<algorithms::Rodas3P>();
    assert_algorithm::<algorithms::Ros4LStab>();
    assert_algorithm::<algorithms::Tsit5DA>();
    assert_algorithm::<algorithms::SSPRKMSVS43>();
    assert_algorithm::<algorithms::QNDF>();
    assert_algorithm::<algorithms::QBDF>();
    assert_algorithm::<algorithms::FBDF>();
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
        algorithms::explicit::low_storage::RDPK3Sp35,
        &options,
    )
    .expect("the concrete low-storage method should solve through its namespace");

    assert_eq!(solution.dimension(), 1);
    assert!((solution.last_state()[0] - (-0.1_f64).exp()).abs() < 1.0e-6);
}
