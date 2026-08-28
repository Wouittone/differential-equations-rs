use differential_equations::solvers::second_order::{
    GeneralizedAlpha, NewmarkBeta, SecondOrderOdeProblem, solve_second_order,
};
use differential_equations::{SaveMode, SolveOptions};

fn endpoint<A: differential_equations::solvers::second_order::SecondOrderOdeAlgorithm>(
    algorithm: A,
) -> (f64, f64) {
    let problem = SecondOrderOdeProblem::new(
        |acceleration: &mut [f64], _: &[f64], position: &[f64], _: &(), _: f64| {
            acceleration[0] = -position[0];
        },
        vec![1.0],
        vec![0.0],
        (0.0, 1.0),
        (),
    );
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.05),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let solution = solve_second_order(&problem, algorithm, &options).unwrap();
    (solution.last_velocity()[0], solution.last_position()[0])
}

fn main() {
    let (velocity, position) = endpoint(NewmarkBeta::default());
    println!("newmark_beta,{velocity:.17e},{position:.17e}");
    let (velocity, position) = endpoint(GeneralizedAlpha::from_spectral_radius(0.5).unwrap());
    println!("generalized_alpha,{velocity:.17e},{position:.17e}");
}
