use differential_equations::solvers::explicit::*;
use differential_equations::*;

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn problem() -> OdeProblem<TestRhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
        du[0] = u[0];
    }

    OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
}

fn adaptive_options() -> SolveOptions {
    SolveOptions::default()
        .with_absolute_tolerance(1.0e-9)
        .with_relative_tolerance(1.0e-9)
        .with_save(SaveMode::Endpoints)
}

fn fixed_options() -> SolveOptions {
    SolveOptions::default()
        .with_adaptive(false)
        .with_initial_step(0.05)
        .with_save(SaveMode::Endpoints)
}

fn main() {
    let adaptive = adaptive_options();
    let fixed = fixed_options();

    for (name, adaptive_value, fixed_value) in [
        (
            "owren_zen3",
            solve(&problem(), OwrenZen3, &adaptive)
                .unwrap()
                .last_state()[0],
            solve(&problem(), OwrenZen3, &fixed).unwrap().last_state()[0],
        ),
        (
            "owren_zen4",
            solve(&problem(), OwrenZen4, &adaptive)
                .unwrap()
                .last_state()[0],
            solve(&problem(), OwrenZen4, &fixed).unwrap().last_state()[0],
        ),
        (
            "owren_zen5",
            solve(&problem(), OwrenZen5, &adaptive)
                .unwrap()
                .last_state()[0],
            solve(&problem(), OwrenZen5, &fixed).unwrap().last_state()[0],
        ),
        (
            "bs5",
            solve(&problem(), Bs5, &adaptive).unwrap().last_state()[0],
            solve(&problem(), Bs5, &fixed).unwrap().last_state()[0],
        ),
    ] {
        println!("{name}_adaptive,{adaptive_value:.17e}");
        println!("{name}_fixed,{fixed_value:.17e}");
    }
}
