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
    let options = SolveOptions::default()
        .with_adaptive(false)
        .with_initial_step(0.01)
        .with_save(SaveMode::Endpoints);
    solve(&nonautonomous(), algorithm, &options)
        .unwrap()
        .last_state()[0]
}

fn main() {
    println!("kutta_prk2p5_fixed,{:.17e}", fixed_endpoint(KuttaPRK2p5()));
    println!("qprk98_fixed,{:.17e}", fixed_endpoint(QPRK98()));

    let options = SolveOptions::default()
        .with_absolute_tolerance(1.0e-10)
        .with_relative_tolerance(1.0e-10)
        .with_initial_step(0.25)
        .with_save(SaveMode::Endpoints);
    let solution = solve(&oscillator(), QPRK98(), &options).unwrap();
    println!(
        "qprk98_adaptive,{:.17e},{:.17e}",
        solution.last_state()[0],
        solution.last_state()[1]
    );
}
