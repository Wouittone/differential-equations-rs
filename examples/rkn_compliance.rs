use differential_equations::solvers::second_order::*;
use differential_equations::*;

type Acceleration = fn(&mut [f64], &[f64], &[f64], &(), f64);

fn endpoint<A: SecondOrderOdeAlgorithm>(algorithm: A) -> (f64, f64) {
    fn acceleration(output: &mut [f64], _: &[f64], position: &[f64], _: &(), _: f64) {
        output[0] = -position[0];
    }
    let problem = SecondOrderOdeProblem::new(
        acceleration as Acceleration,
        vec![0.0],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let solution = solve_second_order(
        &problem,
        algorithm,
        &SolveOptions {
            adaptive: false,
            initial_step: Some(0.01),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        },
    )
    .expect("RKN compliance solve");
    (solution.last_velocity()[0], solution.last_position()[0])
}

fn irkn_endpoint<A: SecondOrderOdeAlgorithm>(algorithm: A) -> (f64, f64) {
    fn acceleration(output: &mut [f64], _: &[f64], position: &[f64], _: &(), _: f64) {
        output[0] = -position[0];
    }
    let problem = SecondOrderOdeProblem::new(
        acceleration as Acceleration,
        vec![0.0],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let solution = solve_second_order(
        &problem,
        algorithm,
        &SolveOptions {
            adaptive: false,
            initial_step: Some(0.125),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        },
    )
    .expect("IRKN compliance solve");
    (solution.last_velocity()[0], solution.last_position()[0])
}

fn adaptive_endpoint<A: SecondOrderOdeAlgorithm>(algorithm: A) -> (f64, f64) {
    fn acceleration(output: &mut [f64], _: &[f64], position: &[f64], _: &(), _: f64) {
        output[0] = -position[0];
    }
    let problem = SecondOrderOdeProblem::new(
        acceleration as Acceleration,
        vec![0.0],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let solution = solve_second_order(
        &problem,
        algorithm,
        &SolveOptions {
            absolute_tolerance: 1.0e-10,
            relative_tolerance: 1.0e-10,
            initial_step: Some(0.5),
            max_step: 0.5,
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        },
    )
    .expect("adaptive RKN compliance solve");
    (solution.last_velocity()[0], solution.last_position()[0])
}

fn velocity_dependent_endpoint<A: SecondOrderOdeAlgorithm>(
    algorithm: A,
    adaptive: bool,
) -> (f64, f64) {
    fn acceleration(output: &mut [f64], velocity: &[f64], position: &[f64], _: &(), time: f64) {
        output[0] = -position[0] - 0.2 * velocity[0] + 0.1 * time;
    }
    let problem = SecondOrderOdeProblem::new(
        acceleration as Acceleration,
        vec![0.25],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let solution = solve_second_order(
        &problem,
        algorithm,
        &SolveOptions {
            adaptive,
            initial_step: Some(if adaptive { 0.5 } else { 0.01 }),
            max_step: 0.5,
            absolute_tolerance: 1.0e-10,
            relative_tolerance: 1.0e-10,
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        },
    )
    .expect("velocity-dependent Nystrom4 compliance solve");
    (solution.last_velocity()[0], solution.last_position()[0])
}

fn dprkn6_dense_midpoint() -> (f64, f64) {
    fn acceleration(output: &mut [f64], _: &[f64], position: &[f64], _: &(), _: f64) {
        output[0] = -position[0];
    }
    let problem = SecondOrderOdeProblem::new(
        acceleration as Acceleration,
        vec![0.0],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let solution = solve_second_order(
        &problem,
        Dprkn6,
        &SolveOptions {
            adaptive: false,
            initial_step: Some(1.0),
            save_at: vec![0.0, 0.5, 1.0],
            ..SolveOptions::default()
        },
    )
    .expect("DPRKN6 dense compliance solve");
    (
        solution.velocity(1).unwrap()[0],
        solution.position(1).unwrap()[0],
    )
}

fn print_endpoint(name: &str, endpoint: (f64, f64)) {
    println!("{name},{:.17e},{:.17e}", endpoint.0, endpoint.1);
}

fn main() {
    print_endpoint("nystrom4", endpoint(Nystrom4));
    print_endpoint(
        "nystrom4_velocity_independent",
        endpoint(Nystrom4VelocityIndependent),
    );
    print_endpoint(
        "nystrom5_velocity_independent",
        endpoint(Nystrom5VelocityIndependent),
    );
    print_endpoint("rkn4", endpoint(Rkn4));
    print_endpoint(
        "nystrom4_velocity_dependent",
        velocity_dependent_endpoint(Nystrom4, false),
    );
    print_endpoint("dprkn4_fixed", endpoint(Dprkn4));
    print_endpoint("dprkn5_fixed", endpoint(Dprkn5));
    print_endpoint("dprkn6_fixed", endpoint(Dprkn6));
    print_endpoint("dprkn6fm_fixed", endpoint(Dprkn6Fm));
    print_endpoint("dprkn8_fixed", endpoint(Dprkn8));
    print_endpoint("dprkn12_fixed", endpoint(Dprkn12));
    print_endpoint("erkn4_fixed", endpoint(Erkn4));
    print_endpoint("erkn5_fixed", endpoint(Erkn5));
    print_endpoint("erkn7_fixed", endpoint(Erkn7));
    print_endpoint("dprkn4_adaptive", adaptive_endpoint(Dprkn4));
    print_endpoint("dprkn5_adaptive", adaptive_endpoint(Dprkn5));
    print_endpoint("dprkn6_adaptive", adaptive_endpoint(Dprkn6));
    print_endpoint("dprkn6fm_adaptive", adaptive_endpoint(Dprkn6Fm));
    print_endpoint("dprkn8_adaptive", adaptive_endpoint(Dprkn8));
    print_endpoint("dprkn12_adaptive", adaptive_endpoint(Dprkn12));
    print_endpoint("erkn4_adaptive", adaptive_endpoint(Erkn4));
    print_endpoint("erkn5_adaptive", adaptive_endpoint(Erkn5));
    print_endpoint("erkn7_adaptive", adaptive_endpoint(Erkn7));
    print_endpoint(
        "finerkn4_fixed",
        velocity_dependent_endpoint(FineRkn4, false),
    );
    print_endpoint(
        "finerkn5_fixed",
        velocity_dependent_endpoint(FineRkn5, false),
    );
    print_endpoint(
        "finerkn4_adaptive",
        velocity_dependent_endpoint(FineRkn4, true),
    );
    print_endpoint(
        "finerkn5_adaptive",
        velocity_dependent_endpoint(FineRkn5, true),
    );
    print_endpoint("dprkn6_dense_midpoint", dprkn6_dense_midpoint());
    print_endpoint("irkn3_fixed", irkn_endpoint(Irkn3));
    print_endpoint("irkn4_fixed", irkn_endpoint(Irkn4));
}
