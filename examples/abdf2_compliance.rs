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
    let fixed_options = SolveOptions::default()
        .with_adaptive(false)
        .with_initial_step(0.01)
        .with_save(SaveMode::Endpoints);
    let fixed = solve(&problem, Abdf2, &fixed_options).expect("ABDF2 fixed solve");
    let adaptive_options = SolveOptions::default()
        .with_absolute_tolerance(1.0e-8)
        .with_relative_tolerance(1.0e-8)
        .with_save(SaveMode::Endpoints);
    let adaptive = solve(&problem, Abdf2, &adaptive_options).expect("ABDF2 adaptive solve");
    println!("abdf2_fixed,{:.17e}", fixed.last_state()[0]);
    println!(
        "abdf2,{:.17e},{},{}",
        adaptive.last_state()[0],
        adaptive.stats().accepted_steps,
        adaptive.stats().rejected_steps
    );
}
