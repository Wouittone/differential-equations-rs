use differential_equations::solvers::explicit::solve_split;
use differential_equations::solvers::multistep::*;
use differential_equations::*;

type Rhs = fn(&mut [f64], &[f64], &(), f64);

fn variable_problem() -> OdeProblem<Rhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), time: f64) {
        du[0] = u[0] + time;
    }
    OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
}

fn split_problem() -> SplitOdeProblem<Rhs, Rhs, ()> {
    fn explicit(du: &mut [f64], u: &[f64], _: &(), time: f64) {
        du[0] = 0.5 * u[0] + time.sin();
    }
    fn implicit(du: &mut [f64], u: &[f64], _: &(), _: f64) {
        du[0] = -2.0 * u[0];
    }
    SplitOdeProblem::new(explicit, implicit, vec![1.0], (0.0, 1.0), ())
}

fn main() {
    let adaptive = SolveOptions {
        absolute_tolerance: 1.0e-9,
        relative_tolerance: 1.0e-9,
        initial_step: Some(0.001),
        max_step: 0.05,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    let vcabm = solve(&variable_problem(), VCABM, &adaptive).unwrap();
    println!("vcabm,{:.17e}", vcabm.last_state()[0]);

    let fixed = SolveOptions {
        adaptive: false,
        initial_step: Some(0.0025),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    macro_rules! row {
        ($name:literal, $algorithm:expr) => {{
            let solution = solve_split(&split_problem(), $algorithm, &fixed).unwrap();
            println!(concat!($name, ",{:.17e}"), solution.last_state()[0]);
        }};
    }
    row!("imex_euler", IMEXEuler);
    row!("imex_euler_ark", IMEXEulerARK);
    row!("sbdf", SBDF::new(2));
    row!("sbdf2", SBDF2);
    row!("sbdf3", SBDF3);
    row!("sbdf4", SBDF4);
    row!("cnab2", CNAB2);
    row!("cnlf2", CNLF2);
}
