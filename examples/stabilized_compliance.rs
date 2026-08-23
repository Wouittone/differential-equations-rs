use differential_equations::{
    ESERK4, ESERK5, OdeAlgorithm, OdeProblem, RKC, RKG1, RKG2, RKL1, RKL2, RKMC2, ROCK2, ROCK4,
    SERK2, SaveMode, SolveOptions, TSRKC2, TSRKC3, solve,
};

type ScalarRhs = fn(&mut [f64], &[f64], &(), f64);

fn stiff_forced() -> OdeProblem<ScalarRhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), time: f64) {
        du[0] = -40.0 * (u[0] - time.cos()) - time.sin();
    }
    OdeProblem::new(rhs as ScalarRhs, vec![1.0], (0.0, 1.0), ())
}

fn nonautonomous() -> OdeProblem<ScalarRhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), time: f64) {
        du[0] = u[0] + time;
    }
    OdeProblem::new(rhs as ScalarRhs, vec![1.0], (0.0, 1.0), ())
}

fn exponential() -> OdeProblem<ScalarRhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
        du[0] = u[0];
    }
    OdeProblem::new(rhs as ScalarRhs, vec![1.0], (0.0, 1.0), ())
}

fn fixed_options(step: f64) -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

fn adaptive_options() -> SolveOptions {
    SolveOptions {
        absolute_tolerance: 1.0e-7,
        relative_tolerance: 1.0e-7,
        initial_step: Some(0.1),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

fn convergence_ratio<A: OdeAlgorithm + Copy>(algorithm: A) -> f64 {
    let exact = std::f64::consts::E;
    let coarse = solve(&exponential(), algorithm, &fixed_options(0.1))
        .expect("coarse stabilized convergence solve failed")
        .last_state()[0];
    let fine = solve(&exponential(), algorithm, &fixed_options(0.05))
        .expect("fine stabilized convergence solve failed")
        .last_state()[0];
    (coarse - exact).abs() / (fine - exact).abs()
}

fn print_result<A: OdeAlgorithm + Copy>(name: &str, algorithm: A) {
    let fixed = solve(&stiff_forced(), algorithm, &fixed_options(0.05))
        .expect("fixed stabilized compliance solve failed")
        .last_state()[0];
    let adaptive = solve(&nonautonomous(), algorithm, &adaptive_options())
        .expect("adaptive stabilized compliance solve failed")
        .last_state()[0];
    println!("{name}_fixed_stiff,{fixed:.17e}");
    println!("{name}_adaptive_nonautonomous,{adaptive:.17e}");
    println!(
        "{name}_convergence_ratio,{:.17e}",
        convergence_ratio(algorithm)
    );
}

fn main() {
    print_result("eserk4", ESERK4);
    print_result("eserk5", ESERK5);
    print_result("rkc", RKC);
    print_result("rkl1", RKL1);
    print_result("rkl2", RKL2);
    print_result("rkg1", RKG1);
    print_result("rkg2", RKG2);
    print_result("rkmc2", RKMC2);
    print_result("rock2", ROCK2);
    print_result("rock4", ROCK4);
    print_result("serk2", SERK2);
    print_result("tsrkc2", TSRKC2);
    print_result("tsrkc3", TSRKC3);
}
