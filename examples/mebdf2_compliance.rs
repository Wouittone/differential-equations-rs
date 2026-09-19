use differential_equations::solvers::multistep::*;
use differential_equations::*;

fn main() {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), t: f64| {
            du[0] = -15.0 * (u[0] - t.cos()) - t.sin();
        },
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions::default()
        .with_adaptive(false)
        .with_initial_step(0.01)
        .with_save(SaveMode::Endpoints);
    let solution = solve(&problem, Mebdf2, &options).expect("MEBDF2 solve");
    println!(
        "mebdf2,{:.17e},{}",
        solution.last_state()[0],
        solution.stats().accepted_steps
    );
}
