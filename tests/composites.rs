use differential_equations::{
    AutoTsit5, AutoVern6, AutoVern7, AutoVern8, AutoVern9, ButcherTableau,
    DefaultImplicitODEAlgorithm, DefaultODEAlgorithm, Dp5, ExplicitRK, OdeProblem, Rodas5P,
    SaveMode, SolveOptions, Tsit5, Vern6, Vern7, Vern8, Vern9, solve,
};

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn problem() -> OdeProblem<TestRhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), time: f64) {
        du[0] = u[0] + time;
    }
    OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
}

fn options() -> SolveOptions {
    SolveOptions {
        absolute_tolerance: 1.0e-10,
        relative_tolerance: 1.0e-10,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

#[test]
fn automatic_and_default_facades_delegate_to_native_components() {
    let options = options();
    let pairs = [
        (
            solve(&problem(), AutoTsit5::new(Rodas5P), &options).unwrap(),
            solve(&problem(), Tsit5, &options).unwrap(),
        ),
        (
            solve(&problem(), AutoVern6::new(Rodas5P), &options).unwrap(),
            solve(&problem(), Vern6, &options).unwrap(),
        ),
        (
            solve(&problem(), AutoVern7::new(Rodas5P), &options).unwrap(),
            solve(&problem(), Vern7, &options).unwrap(),
        ),
        (
            solve(&problem(), AutoVern8::new(Rodas5P), &options).unwrap(),
            solve(&problem(), Vern8, &options).unwrap(),
        ),
        (
            solve(&problem(), AutoVern9::new(Rodas5P), &options).unwrap(),
            solve(&problem(), Vern9, &options).unwrap(),
        ),
        (
            solve(&problem(), DefaultODEAlgorithm::default(), &options).unwrap(),
            solve(&problem(), Tsit5, &options).unwrap(),
        ),
        (
            solve(&problem(), DefaultImplicitODEAlgorithm::default(), &options).unwrap(),
            solve(&problem(), Rodas5P, &options).unwrap(),
        ),
        (
            solve(&problem(), ExplicitRK::<Dp5>::new(), &options).unwrap(),
            solve(&problem(), Dp5, &options).unwrap(),
        ),
    ];
    for (facade, component) in pairs {
        assert_eq!(facade.last_state(), component.last_state());
    }
}

#[test]
fn explicit_rk_alias_exposes_the_existing_tableau_kernel() {
    fn assert_tableau<T: ButcherTableau>() {}
    assert_tableau::<Dp5>();
}
