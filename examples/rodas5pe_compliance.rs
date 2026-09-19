use differential_equations::solvers::rosenbrock::*;
use differential_equations::*;

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn problem() -> OdeProblem<TestRhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
        du[0] = u[0];
    }

    OdeProblem::new(rhs as TestRhs, vec![1.0], (0.0, 1.0), ())
}

fn main() {
    let fixed = SolveOptions::default()
        .with_adaptive(false)
        .with_initial_step(0.125)
        .with_save(SaveMode::Endpoints);
    let adaptive = SolveOptions::default()
        .with_absolute_tolerance(1.0e-9)
        .with_relative_tolerance(1.0e-9)
        .with_save(SaveMode::Endpoints);
    let fixed_solution = solve(&problem(), Rodas5Pe, &fixed).unwrap();
    let adaptive_solution = solve(&problem(), Rodas5Pe, &adaptive).unwrap();
    println!("rodas5pe_fixed,{:.17e}", fixed_solution.last_state()[0]);
    println!(
        "rodas5pe_adaptive,{:.17e}",
        adaptive_solution.last_state()[0]
    );
}
