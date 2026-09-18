use differential_equations::solvers::explicit::{SplitOdeAlgorithm, solve_split};
use differential_equations::solvers::multirate::{
    MIS, MRAB, MREEF, MRIGARKERK22a, MRIGARKERK22b, MRIGARKERK33a, MRIGARKERK45a, MRIGARKESDIRK34a,
    MRIGARKIRK21a,
};
use differential_equations::{SaveMode, SolveOptions, SplitOdeProblem};

fn endpoint<A: SplitOdeAlgorithm>(algorithm: A) -> f64 {
    let problem = SplitOdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -0.9 * u[0],
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -0.1 * u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(0.05)
        .with_save(SaveMode::Endpoints);
    solve_split(&problem, algorithm, &options)
        .unwrap()
        .last_state()[0]
}

fn main() {
    let results = [
        ("mis", endpoint(MIS::new(8).unwrap())),
        ("mrab", endpoint(MRAB::new(3, 8).unwrap())),
        ("mreef", endpoint(MREEF::default())),
        ("erk22a", endpoint(MRIGARKERK22a::new(8).unwrap())),
        ("erk22b", endpoint(MRIGARKERK22b::new(8).unwrap())),
        ("erk33a", endpoint(MRIGARKERK33a::new(8).unwrap())),
        ("erk45a", endpoint(MRIGARKERK45a::new(8).unwrap())),
        ("esdirk34a", endpoint(MRIGARKESDIRK34a::new(8).unwrap())),
        ("irk21a", endpoint(MRIGARKIRK21a::new(8).unwrap())),
    ];
    for (name, value) in results {
        println!("{name},{value:.17e}");
    }
}
