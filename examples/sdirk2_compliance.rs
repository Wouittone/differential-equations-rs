use differential_equations::solvers::implicit::*;
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
        .with_absolute_tolerance(1.0e-8)
        .with_relative_tolerance(1.0e-8)
        .with_save(SaveMode::Endpoints);
    let solution = solve(&problem, Sdirk2, &options).expect("SDIRK2 solve");
    println!(
        "sdirk2,{:.6},{:.12e},{},{}",
        solution.times().last().copied().unwrap_or_default(),
        solution.last_state()[0],
        solution.stats().accepted_steps,
        solution.stats().rejected_steps
    );

    let fixed_options = (options).with_adaptive(false).with_initial_step(0.01);
    let fixed = solve(&problem, Sdirk2, &fixed_options).expect("SDIRK2 fixed solve");
    println!("sdirk2_fixed,{:.17e}", fixed.last_state()[0]);
}
