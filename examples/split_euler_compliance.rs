use differential_equations::solvers::explicit::*;
use differential_equations::*;

fn main() {
    let problem = SplitOdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
        |du: &mut [f64], _: &[f64], _: &(), time: f64| du[0] = time,
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.01),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let solution = solve_split_euler(&problem, SplitEuler, &options).unwrap();
    println!("split_euler,{:.17e}", solution.last_state()[0]);
}
