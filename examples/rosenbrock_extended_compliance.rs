use differential_equations::{
    Grk4a, Grk4t, OdeAlgorithm, OdeProblem, Rodas3, Rodas3d, Rodas4, Rodas4P, Rodas4PW, Rodas5,
    Rodas5P, Rodas6P, Rodas23W, Rodas42, Ros2, Ros3, Ros3Pr, Ros3Prl, Ros3p, Ros34Prw, Ros34Pw1b,
    Ros34Pw2, Rosenbrock32, RosenbrockW6S4OS, SaveMode, SolveOptions, solve,
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
    print_result("rodas23w", Rodas23W);
    print_result("rodas3", Rodas3);
    print_result("rodas3d", Rodas3d);
    print_adaptive("ros3", Ros3);
    print_result("ros3pr", Ros3Pr);
    print_result("ros3prl", Ros3Prl);
    print_result("ros3p", Ros3p);
    print_result("ros34prw", Ros34Prw);
    print_result("rosenbrock32", Rosenbrock32);
    print_result("grk4t", Grk4t);
    print_result("ros34pw1b", Ros34Pw1b);
    print_result("ros34pw2", Ros34Pw2);
    print_result("rodas4", Rodas4);
    print_result("rodas42", Rodas42);
    print_result("rodas4p", Rodas4P);
    print_result("rodas5", Rodas5);
    print_result("rodas4pw", Rodas4PW);
    print_result("grk4a", Grk4a);
    print_result("rodas5p", Rodas5P);
    print_result("rodas6p", Rodas6P);
    println!(
        "rosenbrockw6s4os_fixed,{:.17e}",
        endpoint(RosenbrockW6S4OS, false)
    );
}
