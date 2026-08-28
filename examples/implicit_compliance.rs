use differential_equations::solvers::implicit::*;
use differential_equations::*;

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn problem() -> OdeProblem<TestRhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
        du[0] = -10.0 * u[0] + u[1];
        du[1] = -u[1];
    }
    OdeProblem::new(rhs, vec![1.0, 1.0], (0.0, 1.0), ())
}

fn options() -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(0.01),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

fn print_endpoint(name: &str, values: &[f64]) {
    println!("{name},{:.17e},{:.17e}", values[0], values[1]);
}

fn main() {
    let euler = solve(&problem(), ImplicitEuler, &options()).unwrap();
    print_endpoint("implicit_euler", euler.last_state());

    let midpoint = solve(&problem(), ImplicitMidpoint, &options()).unwrap();
    print_endpoint("implicit_midpoint", midpoint.last_state());

    let trapezoid = solve(&problem(), Trapezoid, &options()).unwrap();
    print_endpoint("trapezoid", trapezoid.last_state());
}
