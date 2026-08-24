use differential_equations::algorithms::exponential::*;
use differential_equations::{OdeAlgorithm, OdeProblem, SaveMode, SolveOptions, solve};

fn endpoint<A: OdeAlgorithm>(algorithm: A) -> f64 {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -2.0 * u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    )
    .with_jacobian(|jacobian, _, _, _| jacobian[0] = -2.0);
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.2),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    solve(&problem, algorithm, &options).unwrap().last_state()[0]
}

fn main() {
    macro_rules! print_endpoint {
        ($name:literal, $algorithm:expr) => {
            println!(concat!($name, ",{:0.17e}"), endpoint($algorithm));
        };
    }
    print_endpoint!("lawson_euler", LawsonEuler);
    print_endpoint!("norsett_euler", NorsettEuler);
    print_endpoint!("etd1", ETD1);
    print_endpoint!("etdrk2", ETDRK2);
    print_endpoint!("etdrk3", ETDRK3);
    print_endpoint!("etdrk4", ETDRK4);
    print_endpoint!("hoch_ost4", HochOst4);
    print_endpoint!("exp4", Exp4);
    print_endpoint!("epirk4s3a", EPIRK4s3A);
    print_endpoint!("epirk4s3b", EPIRK4s3B);
    print_endpoint!("epirk5s3", EPIRK5s3);
    print_endpoint!("exprb53s3", EXPRB53s3);
    print_endpoint!("epirk5p1", EPIRK5P1);
    print_endpoint!("epirk5p2", EPIRK5P2);
    print_endpoint!("etd2", ETD2);
    print_endpoint!("exprb32", Exprb32);
    print_endpoint!("exprb43", Exprb43);
}
