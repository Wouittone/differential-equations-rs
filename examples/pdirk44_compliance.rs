use differential_equations::solvers::implicit::PDIRK44;
use differential_equations::{OdeProblem, SaveMode, SolveOptions, solve};

fn main() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions::default()
        .with_adaptive(false)
        .with_initial_step(0.05)
        .with_save(SaveMode::Endpoints);
    let solution = solve(&problem, PDIRK44, &options).unwrap();
    println!("pdirk44_fixed,{:.17e}", solution.last_state()[0]);
}
