use differential_equations::solvers::explicit::*;
use differential_equations::*;

fn options() -> SolveOptions {
    SolveOptions {
        absolute_tolerance: 1.0e-10,
        relative_tolerance: 1.0e-10,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

fn print_endpoint(name: &str, solution: &Solution) {
    print!("{name}");
    for value in solution.last_state() {
        print!(",{value:.17e}");
    }
    println!();
}

fn main() {
    let exponential = OdeProblem::new(
        |du: &mut [f64], u: &[f64], rate: &f64, _: f64| {
            du[0] = rate * u[0];
        },
        vec![0.5],
        (0.0, 1.0),
        1.01,
    );
    print_endpoint(
        "exponential",
        &solve(&exponential, Tsit5, &options()).expect("exponential solve failed"),
    );

    let oscillator = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| {
            du[0] = u[1];
            du[1] = -u[0];
        },
        vec![1.0, 0.0],
        (0.0, std::f64::consts::TAU),
        (),
    );
    print_endpoint(
        "oscillator",
        &solve(&oscillator, Tsit5, &options()).expect("oscillator solve failed"),
    );

    let logistic = OdeProblem::new(
        |du: &mut [f64], u: &[f64], parameters: &(f64, f64), _: f64| {
            let (rate, capacity) = *parameters;
            du[0] = rate * u[0] * (1.0 - u[0] / capacity);
        },
        vec![0.25],
        (0.0, 5.0),
        (1.3, 10.0),
    );
    print_endpoint(
        "logistic",
        &solve(&logistic, Tsit5, &options()).expect("logistic solve failed"),
    );

    let lorenz = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| {
            du[0] = 10.0 * (u[1] - u[0]);
            du[1] = u[0] * (28.0 - u[2]) - u[1];
            du[2] = u[0] * u[1] - (8.0 / 3.0) * u[2];
        },
        vec![1.0, 0.0, 0.0],
        (0.0, 1.0),
        (),
    );
    print_endpoint(
        "lorenz",
        &solve(&lorenz, Tsit5, &options()).expect("Lorenz solve failed"),
    );
}
