use differential_equations::{
    OdeAlgorithm, OdeProblem, Rodas3, Rodas4, Rodas5P, Ros2, Ros3, Ros3Pr, Rosenbrock32,
    RosenbrockW6S4OS, SaveMode, SolveOptions, solve,
};

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn stiff_problem() -> OdeProblem<TestRhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), time: f64) {
        du[0] = -1000.0 * (u[0] - time.cos()) - time.sin();
    }
    OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
}

fn nonstiff_problem() -> OdeProblem<TestRhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
        du[0] = u[0];
    }
    OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
}

fn endpoint<A: OdeAlgorithm>(algorithm: A, adaptive: bool) -> f64 {
    let options = SolveOptions {
        adaptive,
        absolute_tolerance: 1.0e-8,
        relative_tolerance: 1.0e-8,
        initial_step: Some(if adaptive { 0.1 } else { 0.01 }),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let problem = if adaptive {
        stiff_problem()
    } else {
        nonstiff_problem()
    };
    solve(&problem, algorithm, &options)
        .expect("extended Rosenbrock compliance solve failed")
        .last_state()[0]
}

fn print_result<A: OdeAlgorithm + Copy>(name: &str, algorithm: A) {
    println!("{name}_adaptive,{:.17e}", endpoint(algorithm, true));
    println!("{name}_fixed,{:.17e}", endpoint(algorithm, false));
}

fn print_adaptive<A: OdeAlgorithm>(name: &str, algorithm: A) {
    println!("{name}_adaptive,{:.17e}", endpoint(algorithm, true));
}

fn main() {
    print_result("ros2", Ros2);
    print_result("rodas3", Rodas3);
    print_adaptive("ros3", Ros3);
    print_result("ros3pr", Ros3Pr);
    print_result("rosenbrock32", Rosenbrock32);
    print_result("rodas4", Rodas4);
    print_result("rodas5p", Rodas5P);
    println!(
        "rosenbrockw6s4os_fixed,{:.17e}",
        endpoint(RosenbrockW6S4OS, false)
    );
}
