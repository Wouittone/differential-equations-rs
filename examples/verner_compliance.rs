use differential_equations::{
    OdeAlgorithm, OdeProblem, SaveMode, SolveOptions, Vern6, Vern7, Vern8, Vern9, solve,
};

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn problem() -> OdeProblem<TestRhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), time: f64) {
        du[0] = u[1];
        du[1] = -u[0] + 0.1 * time;
    }

    OdeProblem::new(rhs, vec![1.0, 0.0], (0.0, 2.0), ())
}

fn endpoint<A: OdeAlgorithm>(algorithm: A, adaptive: bool) -> Vec<f64> {
    let options = SolveOptions {
        adaptive,
        absolute_tolerance: 1.0e-10,
        relative_tolerance: 1.0e-10,
        initial_step: Some(if adaptive { 0.5 } else { 0.05 }),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    solve(&problem(), algorithm, &options)
        .expect("Verner compliance solve failed")
        .last_state()
        .to_vec()
}

fn print_result<A: OdeAlgorithm + Copy>(name: &str, algorithm: A) {
    for (mode, values) in [
        ("adaptive", endpoint(algorithm, true)),
        ("fixed", endpoint(algorithm, false)),
    ] {
        println!("{name}_{mode},{:.17e},{:.17e}", values[0], values[1]);
    }
}

fn main() {
    print_result("vern6", Vern6);
    print_result("vern7", Vern7);
    print_result("vern8", Vern8);
    print_result("vern9", Vern9);
}
