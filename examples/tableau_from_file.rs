use differential_equations::{OdeProblem, SolveOptions, solve};

differential_equations::tableau::define_explicit_rk_from_file!(
    pub FileHeun,
    "examples/resources/file_heun.json"
);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let problem = OdeProblem::new(
        |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -u[0],
        vec![1.0],
        (0.0, 1.0),
        (),
    );
    let solution = solve(&problem, FileHeun, &SolveOptions::default())?;
    println!("u(1) = {}", solution.last_state()[0]);
    Ok(())
}
