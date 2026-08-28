use differential_equations::solvers::rosenbrock::*;
use differential_equations::*;

fn main() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), time: f64| {
            du[0] = -1000.0 * (u[0] - time.cos()) - time.sin();
        },
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions {
        absolute_tolerance: 1.0e-7,
        relative_tolerance: 1.0e-7,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let solution = solve(&problem, Rosenbrock23, &options).unwrap();

    println!(
        "rosenbrock23,{:.17e},{}",
        solution.last_state()[0],
        solution.stats().rhs_evaluations
    );
}
