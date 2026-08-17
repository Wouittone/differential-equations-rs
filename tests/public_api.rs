use differential_equations::{OdeAlgorithm, OdeProblem, SaveMode, SolveOptions, algorithms, solve};

fn assert_algorithm<T: OdeAlgorithm>() {}

fn assert_same_type<T>(_: T, _: T) {}

#[test]
fn family_namespaces_reexport_the_concrete_root_types() {
    assert_same_type(
        differential_equations::Tsit5,
        algorithms::explicit::general::Tsit5,
    );
    assert_same_type(
        differential_equations::RDPK3Sp35,
        algorithms::explicit::low_storage::RDPK3Sp35,
    );
    assert_same_type(
        differential_equations::Kvaerno5,
        algorithms::implicit::diagonally_implicit::Kvaerno5,
    );
    assert_same_type(
        differential_equations::Rodas3P,
        algorithms::rosenbrock::Rodas3P,
    );
    assert_same_type(
        differential_equations::VelocityVerlet,
        algorithms::second_order::VelocityVerlet,
    );
}

#[test]
fn recovered_root_exports_are_real_ode_algorithms() {
    assert_algorithm::<differential_equations::CKLLSRK54_3C>();
    assert_algorithm::<differential_equations::RDPK3SpFSAL510>();
    assert_algorithm::<differential_equations::Esdirk436L2Sa2>();
    assert_algorithm::<differential_equations::KenCarp4>();
    assert_algorithm::<differential_equations::Kvaerno5>();
    assert_algorithm::<differential_equations::Rodas3P>();
    assert_algorithm::<differential_equations::Ros4LStab>();
    assert_algorithm::<differential_equations::Tsit5DA>();
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
