use differential_equations::solvers::explicit::*;
use differential_equations::*;

type ScalarRhs = fn(&mut [f64], &[f64], &(), f64);
type VectorRhs = fn(&mut [f64], &[f64], &(), f64);

fn nonautonomous(initial: f64, span: (f64, f64)) -> OdeProblem<ScalarRhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), time: f64) {
        du[0] = u[0] + time;
    }
    OdeProblem::new(rhs as ScalarRhs, vec![initial], span, ())
}

fn oscillator() -> OdeProblem<VectorRhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), time: f64) {
        du[0] = u[1];
        du[1] = -u[0] + 0.1 * time;
    }
    OdeProblem::new(rhs as VectorRhs, vec![1.0, 0.0], (0.0, 2.0), ())
}

fn exponential() -> OdeProblem<ScalarRhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
        du[0] = u[0];
    }
    OdeProblem::new(rhs as ScalarRhs, vec![1.0], (0.0, 2.0), ())
}

fn fixed_options(step: f64) -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

fn adaptive_options(tolerance: f64) -> SolveOptions {
    SolveOptions {
        absolute_tolerance: tolerance,
        relative_tolerance: tolerance,
        initial_step: Some(0.5),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

fn convergence_ratio<A: OdeAlgorithm + Copy>(algorithm: A) -> f64 {
    let exact = 2.0_f64.exp();
    let coarse = solve(&exponential(), algorithm, &fixed_options(1.0))
        .unwrap()
        .last_state()[0];
    let fine = solve(&exponential(), algorithm, &fixed_options(0.5))
        .unwrap()
        .last_state()[0];
    (coarse - exact).abs() / (fine - exact).abs()
}

fn print_result<A: OdeAlgorithm + Copy>(name: &str, algorithm: A, adaptive_tolerance: f64) {
    let forward = solve(
        &nonautonomous(1.0, (0.0, 2.0)),
        algorithm,
        &fixed_options(0.25),
    )
    .unwrap()
    .last_state()[0];
    let backward_initial = 2.0 * 2.0_f64.exp() - 3.0;
    let backward = solve(
        &nonautonomous(backward_initial, (2.0, 0.0)),
        algorithm,
        &fixed_options(0.25),
    )
    .unwrap()
    .last_state()[0];
    let adaptive = solve(
        &oscillator(),
        algorithm,
        &adaptive_options(adaptive_tolerance),
    )
    .unwrap();
    let adaptive = adaptive.last_state();

    println!("{name}_fixed_forward,{forward:.17e}");
    println!("{name}_fixed_backward,{backward:.17e}");
    println!(
        "{name}_adaptive_vector,{:.17e},{:.17e}",
        adaptive[0], adaptive[1]
    );
    println!(
        "{name}_convergence_ratio,{:.17e}",
        convergence_ratio(algorithm)
    );
}

fn main() {
    print_result("dp8", DP8, 1.0e-10);
    print_result("feagin10", Feagin10, 1.0e-10);
    print_result("feagin12", Feagin12, 1.0e-10);
    print_result("feagin14", Feagin14, 1.0e-10);
    // OrdinaryDiffEq's PFRK87 estimator stalls at tighter tolerances.
    print_result("pfrk87", PFRK87, 1.0e-4);
    print_result("rkv76iia", RKV76IIa, 1.0e-10);
    print_result("tanyam7", TanYam7, 1.0e-10);
    print_result("tsitpap8", TsitPap8, 1.0e-10);
}
