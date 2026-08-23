use differential_equations::{
    CalvoSanz4, CandyRoz4, KahanLi6, KahanLi8, LeapfrogDriftKickDrift, McAte2, McAte3, McAte4,
    McAte5, McAte8, McAte42, PseudoVerletLeapfrog, Ruth3, SaveMode, SecondOrderOdeProblem,
    SofSpa10, SolveOptions, SymplecticAlgorithm, SymplecticEuler, VelocityVerlet, VerletLeapfrog,
    Yoshida6, solve_second_order, solve_symplectic,
};

type Acceleration = fn(&mut [f64], &[f64], &[f64], &(), f64);

fn options() -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(0.01),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

fn composition_oscillator() -> SecondOrderOdeProblem<Acceleration, ()> {
    fn acceleration(output: &mut [f64], _: &[f64], position: &[f64], _: &(), _: f64) {
        output[0] = -position[0];
    }

    SecondOrderOdeProblem::new(
        acceleration as Acceleration,
        vec![0.0],
        vec![1.0],
        (0.0, 1.0),
        (),
    )
}

fn print_composition<A: SymplecticAlgorithm>(name: &str, algorithm: A) {
    let endpoint = solve_symplectic(&composition_oscillator(), algorithm, &options()).unwrap();
    println!(
        "{name},{:.17e},{:.17e}",
        endpoint.last_velocity()[0],
        endpoint.last_position()[0]
    );
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

    print_composition("pseudo_verlet_leapfrog", PseudoVerletLeapfrog);
    print_composition("mcate2", McAte2);
    print_composition("ruth3", Ruth3);
    print_composition("mcate3", McAte3);
    print_composition("candy_roz4", CandyRoz4);
    print_composition("mcate4", McAte4);
    print_composition("calvo_sanz4", CalvoSanz4);
    print_composition("mcate42", McAte42);
    print_composition("mcate5", McAte5);
    print_composition("yoshida6", Yoshida6);
    print_composition("kahan_li6", KahanLi6);
    print_composition("mcate8", McAte8);
    print_composition("kahan_li8", KahanLi8);
    print_composition("sof_spa10", SofSpa10);
}
