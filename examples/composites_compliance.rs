use differential_equations::{
    AutoTsit5, AutoVern6, AutoVern7, AutoVern8, AutoVern9, DefaultImplicitODEAlgorithm,
    DefaultODEAlgorithm, Dp5, ExplicitRK, OdeProblem, Rodas5P, SaveMode, SolveOptions, solve,
};

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn problem() -> OdeProblem<TestRhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), time: f64) {
        du[0] = u[0] + time;
    }
    OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
}

fn main() {
    let options = SolveOptions {
        absolute_tolerance: 1.0e-10,
        relative_tolerance: 1.0e-10,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    for (name, endpoint) in [
        (
            "auto_tsit5",
            solve(&problem(), AutoTsit5::new(Rodas5P), &options),
        ),
        (
            "auto_vern6",
            solve(&problem(), AutoVern6::new(Rodas5P), &options),
        ),
        (
            "auto_vern7",
            solve(&problem(), AutoVern7::new(Rodas5P), &options),
        ),
        (
            "auto_vern8",
            solve(&problem(), AutoVern8::new(Rodas5P), &options),
        ),
        (
            "auto_vern9",
            solve(&problem(), AutoVern9::new(Rodas5P), &options),
        ),
        (
            "default_ode_algorithm",
            solve(&problem(), DefaultODEAlgorithm::default(), &options),
        ),
        (
            "default_implicit_ode_algorithm",
            solve(&problem(), DefaultImplicitODEAlgorithm::default(), &options),
        ),
        (
            "explicit_rk",
            solve(&problem(), ExplicitRK::<Dp5>::new(), &options),
        ),
    ] {
        println!("{name},{:.17e}", endpoint.unwrap().last_state()[0]);
    }
}
