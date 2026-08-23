use differential_equations::algorithms::*;
use differential_equations::*;

fn main() {
    let stiff = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), time: f64| {
            du[0] = -1000.0 * (u[0] - time.cos()) - time.sin();
        },
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let adaptive = SolveOptions {
        absolute_tolerance: 1.0e-8,
        relative_tolerance: 1.0e-8,
        initial_step: Some(0.1),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let adaptive_endpoint = solve(&stiff, Ros34Pw3, &adaptive).unwrap().last_state()[0];

    let fixed = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let fixed_options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.01),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let fixed_endpoint = solve(&fixed, Ros34Pw3, &fixed_options)
        .unwrap()
        .last_state()[0];

    println!("ros34pw3_adaptive,{adaptive_endpoint:.17e}");
    println!("ros34pw3_fixed,{fixed_endpoint:.17e}");
}
