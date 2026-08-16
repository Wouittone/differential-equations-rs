use differential_equations::{OdeProblem, Rodas5Pe, SaveMode, SolveOptions, solve};

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn problem() -> OdeProblem<TestRhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
        du[0] = u[0];
    }

    OdeProblem::new(rhs as TestRhs, vec![1.0], (0.0, 1.0), ())
}

fn main() {
    let fixed = SolveOptions {
        adaptive: false,
        initial_step: Some(0.125),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let adaptive = SolveOptions {
        absolute_tolerance: 1.0e-9,
        relative_tolerance: 1.0e-9,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let fixed_solution = solve(&problem(), Rodas5Pe, &fixed).unwrap();
    let adaptive_solution = solve(&problem(), Rodas5Pe, &adaptive).unwrap();
    println!("rodas5pe_fixed,{:.17e}", fixed_solution.last_state()[0]);
    println!(
        "rodas5pe_adaptive,{:.17e}",
        adaptive_solution.last_state()[0]
    );
}
