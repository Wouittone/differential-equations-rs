use differential_equations::algorithms::*;
use differential_equations::*;

fn main() {
    let options = SolveOptions {
        absolute_tolerance: 1.0e-9,
        relative_tolerance: 1.0e-9,
        initial_step: Some(0.013),
        max_step: 0.2,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };

    macro_rules! run {
        ($name:literal, $algorithm:expr) => {{
            let exponential = OdeProblem::new(
                |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
                vec![1.0],
                (0.0, 1.0),
                (),
            );
            let forward = solve(&exponential, $algorithm, &options).unwrap();
            println!(
                "{},forward,{:.17e},{},{}",
                $name,
                forward.last_state()[0],
                forward.stats().accepted_steps,
                forward.stats().rejected_steps
            );

            let nonautonomous = OdeProblem::new(
                |du: &mut [f64], u: &[f64], _: &(), time: f64| {
                    du[0] = -0.4 * u[0] + time.sin();
                    du[1] = u[0] - 0.2 * u[1] + time.cos();
                },
                vec![0.3, -0.7],
                (0.0, 2.0),
                (),
            );
            let vector = solve(&nonautonomous, $algorithm, &options).unwrap();
            println!(
                "{},vector,{:.17e};{:.17e},{},{}",
                $name,
                vector.last_state()[0],
                vector.last_state()[1],
                vector.stats().accepted_steps,
                vector.stats().rejected_steps
            );
        }};
    }

    run!("vcab3", Vcab3);
    run!("vcab4", Vcab4);
    run!("vcab5", Vcab5);
    run!("vcabm3", Vcabm3);
    run!("vcabm4", Vcabm4);
    run!("vcabm5", Vcabm5);
}
