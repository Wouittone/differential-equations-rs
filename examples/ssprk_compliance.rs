use differential_equations::solvers::explicit::*;
use differential_equations::*;

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn problem() -> OdeProblem<TestRhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
        du[0] = u[0];
    }
    OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
}

fn fixed_options() -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(0.01),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

fn adaptive_options() -> SolveOptions {
    SolveOptions {
        absolute_tolerance: 1.0e-9,
        relative_tolerance: 1.0e-9,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

fn main() {
    let rk22 = solve(&problem(), SspRk22, &fixed_options()).unwrap();
    println!("ssprk22,{:.17e}", rk22.last_state()[0]);

    let prrk22 = solve(&problem(), Prrk22::default(), &fixed_options()).unwrap();
    println!("prrk22,{:.17e}", prrk22.last_state()[0]);

    let prrk33 = solve(&problem(), Prrk33::default(), &fixed_options()).unwrap();
    println!("prrk33,{:.17e}", prrk33.last_state()[0]);

    let prrk54 = solve(&problem(), Prrk54::default(), &fixed_options()).unwrap();
    println!("prrk54,{:.17e}", prrk54.last_state()[0]);

    let rk33 = solve(&problem(), SspRk33, &fixed_options()).unwrap();
    println!("ssprk33,{:.17e}", rk33.last_state()[0]);

    let rk43 = solve(&problem(), SspRk43, &adaptive_options()).unwrap();
    println!("ssprk43,{:.17e}", rk43.last_state()[0]);

    let rk432 = solve(&problem(), SspRk432, &adaptive_options()).unwrap();
    println!("ssprk432,{:.17e}", rk432.last_state()[0]);

    let kyk2014 = solve(&problem(), Kyk2014DgSsprk3S2, &fixed_options()).unwrap();
    println!("kyk2014dgssprk_3s2,{:.17e}", kyk2014.last_state()[0]);

    let kyk42 = solve(&problem(), KYKSSPRK42::default(), &fixed_options()).unwrap();
    println!("kykssprk42,{:.17e}", kyk42.last_state()[0]);
}
