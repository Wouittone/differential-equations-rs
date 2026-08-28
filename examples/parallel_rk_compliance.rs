use differential_equations::solvers::explicit::*;
use differential_equations::*;

type ScalarRhs = fn(&mut [f64], &[f64], &(), f64);
type VectorRhs = fn(&mut [f64], &[f64], &(), f64);

fn nonautonomous() -> OdeProblem<ScalarRhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), time: f64) {
        du[0] = u[0] + time;
    }
    OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
}

fn oscillator() -> OdeProblem<VectorRhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), time: f64) {
        du[0] = u[1];
        du[1] = -u[0] + 0.1 * time;
    }
    OdeProblem::new(rhs, vec![1.0, 0.0], (0.0, 2.0), ())
}

fn fixed_endpoint<A: OdeAlgorithm>(algorithm: A) -> f64 {
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.01),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    solve(&nonautonomous(), algorithm, &options)
        .unwrap()
        .last_state()[0]
}

fn main() {
    println!("kutta_prk2p5_fixed,{:.17e}", fixed_endpoint(KuttaPRK2p5()));
    println!("qprk98_fixed,{:.17e}", fixed_endpoint(QPRK98()));

    let options = SolveOptions {
        absolute_tolerance: 1.0e-10,
        relative_tolerance: 1.0e-10,
        initial_step: Some(0.25),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let solution = solve(&oscillator(), QPRK98(), &options).unwrap();
    println!(
        "qprk98_adaptive,{:.17e},{:.17e}",
        solution.last_state()[0],
        solution.last_state()[1]
    );
}
