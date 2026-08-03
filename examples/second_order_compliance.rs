use differential_equations::{
    LeapfrogDriftKickDrift, SaveMode, SecondOrderOdeProblem, SolveOptions, SymplecticEuler,
    VelocityVerlet, VerletLeapfrog, solve_second_order,
};

fn options() -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(0.01),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

fn main() {
    let oscillator = SecondOrderOdeProblem::new(
        |output: &mut [f64], _: &[f64], position: &[f64], _: &(), _| {
            output[0] = -position[0];
        },
        vec![0.0],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    for (name, endpoint) in [
        (
            "symplectic_euler",
            solve_second_order(&oscillator, SymplecticEuler, &options()).unwrap(),
        ),
        (
            "velocity_verlet",
            solve_second_order(&oscillator, VelocityVerlet, &options()).unwrap(),
        ),
        (
            "verlet_leapfrog",
            solve_second_order(&oscillator, VerletLeapfrog, &options()).unwrap(),
        ),
        (
            "leapfrog_dkd",
            solve_second_order(&oscillator, LeapfrogDriftKickDrift, &options()).unwrap(),
        ),
    ] {
        println!(
            "{name},{:.17e},{:.17e}",
            endpoint.last_velocity()[0],
            endpoint.last_position()[0]
        );
    }

    let velocity_dependent = SecondOrderOdeProblem::new(
        |output: &mut [f64], velocity: &[f64], position: &[f64], _: &(), time| {
            output[0] = -position[0] - 0.2 * velocity[0] + 0.1 * time;
        },
        vec![0.25],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let endpoint =
        solve_second_order(&velocity_dependent, LeapfrogDriftKickDrift, &options()).unwrap();
    println!(
        "leapfrog_dkd_velocity_dependent,{:.17e},{:.17e}",
        endpoint.last_velocity()[0],
        endpoint.last_position()[0]
    );
}
