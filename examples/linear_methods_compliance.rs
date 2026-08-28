use differential_equations::solvers::linear::general::{
    LinearOperatorAlgorithm, solve_lie_group, solve_linear_operator,
};
use differential_equations::solvers::linear::*;
use differential_equations::{LieGroupProblem, LinearOperatorProblem, SaveMode, SolveOptions};

fn options() -> SolveOptions {
    SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(0.2)
        .with_save(SaveMode::Endpoints)
}

fn generator(output: &mut [f64], _: &[f64], _: &(), _: f64) {
    output.copy_from_slice(&[0.0, -1.0, 1.0, 0.0]);
}

fn endpoint<A: LinearOperatorAlgorithm>(algorithm: A) -> Vec<f64> {
    let problem = LinearOperatorProblem::new(generator, vec![1.0, 0.0], (0.0, 1.0), ()).unwrap();
    solve_linear_operator(&problem, algorithm, &options())
        .unwrap()
        .last_state()
        .to_vec()
}

fn print_endpoint<A: LinearOperatorAlgorithm>(name: &str, algorithm: A) {
    let state = endpoint(algorithm);
    println!("{name},{:.17e},{:.17e}", state[0], state[1]);
}

fn main() {
    print_endpoint("lie_euler", LieEuler);
    print_endpoint("linear_exponential", LinearExponential);
    print_endpoint("magnus_midpoint", MagnusMidpoint);
    print_endpoint("magnus_leapfrog", MagnusLeapfrog);
    print_endpoint("rkmk2", RKMK2);
    print_endpoint("rkmk4", RKMK4);
    print_endpoint("lie_rk4", LieRK4);
    print_endpoint("cg2", CG2);
    print_endpoint("cg3", CG3);
    print_endpoint("cg4a", CG4a);
    print_endpoint("magnus_adapt4", MagnusAdapt4);
    print_endpoint("magnus_gauss4", MagnusGauss4);
    print_endpoint("magnus_gl4", MagnusGL4);
    print_endpoint("magnus_gl6", MagnusGL6);
    print_endpoint("magnus_nc6", MagnusNC6);
    print_endpoint("magnus_gl8", MagnusGL8);
    print_endpoint("magnus_nc8", MagnusNC8);

    let problem =
        LieGroupProblem::matrix(generator, vec![2.0, 0.5, 0.5, -1.0], 2, (0.0, 1.0), ()).unwrap();
    let state = solve_lie_group(&problem, CayleyEuler, &options())
        .unwrap()
        .last_state()
        .to_vec();
    println!(
        "cayley_euler,{:.17e},{:.17e},{:.17e},{:.17e}",
        state[0], state[1], state[2], state[3]
    );
}
