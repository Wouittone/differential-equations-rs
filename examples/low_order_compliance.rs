use differential_equations::{
    Alshina2, Alshina3, Alshina6, Anas5, Bs3, Dp5, Euler, Frk65, Heun, Midpoint, Msrk5, Msrk6,
    OdeProblem, Psrk3p5q4, Ralston, Ralston4, Rk4, Rkm, SaveMode, SolveOptions, solve,
};

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn problem() -> OdeProblem<TestRhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), _: f64) {
        du[0] = u[0];
    }

    OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
}

fn adaptive_options() -> SolveOptions {
    SolveOptions {
        absolute_tolerance: 1.0e-9,
        relative_tolerance: 1.0e-9,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

fn fixed_options(step: f64) -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

fn main() {
    let euler = solve(&problem(), Euler, &fixed_options(0.001)).unwrap();
    println!("euler,{:.17e}", euler.last_state()[0]);

    let rk4 = solve(&problem(), Rk4, &fixed_options(0.01)).unwrap();
    println!("rk4,{:.17e}", rk4.last_state()[0]);

    let rkm = solve(&problem(), Rkm, &fixed_options(0.01)).unwrap();
    println!("rkm,{:.17e}", rkm.last_state()[0]);

    let ralston4 = solve(&problem(), Ralston4, &fixed_options(0.01)).unwrap();
    println!("ralston4,{:.17e}", ralston4.last_state()[0]);

    let alshina2 = solve(&problem(), Alshina2, &fixed_options(0.01)).unwrap();
    println!("alshina2,{:.17e}", alshina2.last_state()[0]);

    let alshina3 = solve(&problem(), Alshina3, &fixed_options(0.01)).unwrap();
    println!("alshina3,{:.17e}", alshina3.last_state()[0]);

    let alshina6 = solve(&problem(), Alshina6, &fixed_options(0.01)).unwrap();
    println!("alshina6,{:.17e}", alshina6.last_state()[0]);

    let anas5 = solve(&problem(), Anas5::default(), &fixed_options(0.01)).unwrap();
    println!("anas5,{:.17e}", anas5.last_state()[0]);

    let msrk5 = solve(&problem(), Msrk5, &fixed_options(0.01)).unwrap();
    println!("msrk5,{:.17e}", msrk5.last_state()[0]);

    let msrk6 = solve(&problem(), Msrk6, &fixed_options(0.01)).unwrap();
    println!("msrk6,{:.17e}", msrk6.last_state()[0]);

    let frk65 = solve(&problem(), Frk65::default(), &fixed_options(0.01)).unwrap();
    println!("frk65,{:.17e}", frk65.last_state()[0]);

    let psrk3p5q4 = solve(&problem(), Psrk3p5q4, &fixed_options(0.01)).unwrap();
    println!("psrk3p5q4,{:.17e}", psrk3p5q4.last_state()[0]);

    let midpoint = solve(&problem(), Midpoint, &adaptive_options()).unwrap();
    println!("midpoint,{:.17e}", midpoint.last_state()[0]);

    let heun = solve(&problem(), Heun, &adaptive_options()).unwrap();
    println!("heun,{:.17e}", heun.last_state()[0]);

    let ralston = solve(&problem(), Ralston, &adaptive_options()).unwrap();
    println!("ralston,{:.17e}", ralston.last_state()[0]);

    let bs3 = solve(&problem(), Bs3, &adaptive_options()).unwrap();
    println!("bs3,{:.17e}", bs3.last_state()[0]);

    let dp5 = solve(&problem(), Dp5, &adaptive_options()).unwrap();
    println!("dp5,{:.17e}", dp5.last_state()[0]);
}
