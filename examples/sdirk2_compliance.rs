use differential_equations::algorithms::*;
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
    let options = SolveOptions {
        absolute_tolerance: 1.0e-8,
        relative_tolerance: 1.0e-8,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let solution = solve(&problem, Sdirk2, &options).expect("SDIRK2 solve");
    println!(
        "sdirk2,{:.6},{:.12e},{},{}",
        solution.times().last().copied().unwrap_or_default(),
        solution.last_state()[0],
        solution.stats().accepted_steps,
        solution.stats().rejected_steps
    );

    let fixed_options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.01),
        ..options
    };
    let fixed = solve(&problem, Sdirk2, &fixed_options).expect("SDIRK2 fixed solve");
    println!("sdirk2_fixed,{:.17e}", fixed.last_state()[0]);
}
