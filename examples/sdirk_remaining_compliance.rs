use differential_equations::{
    Ars222, Ars232, Ars343, Ars443, Bhr553, Cfnlirk3, Esdirk54I8L2Sa, Esdirk325L2Sa,
    Esdirk436L2Sa2, Esdirk437L2Sa, Esdirk547L2Sa2, Esdirk659L2Sa, Hairer4, Hairer42, ImexSsp222,
    ImexSsp2322, ImexSsp3332, ImexSsp3433, KenCarp3, KenCarp4, KenCarp5, KenCarp47, KenCarp58,
    Kvaerno3, Kvaerno4, Kvaerno5, OdeAlgorithm, OdeProblem, SaveMode, Sdirk22, Sfsdirk4, Sfsdirk5,
    Sfsdirk6, Sfsdirk7, Sfsdirk8, SolveOptions, SspSdirk2, solve,
};

type TestRhs = fn(&mut [f64], &[f64], &(), f64);

fn problem() -> OdeProblem<TestRhs, ()> {
    fn rhs(du: &mut [f64], u: &[f64], _: &(), time: f64) {
        du[0] = u[0] + time;
    }
    OdeProblem::new(rhs, vec![1.0], (0.0, 1.0), ())
}

fn endpoint<A: OdeAlgorithm>(algorithm: A) -> f64 {
    let options = SolveOptions {
        adaptive: false,
        initial_step: Some(0.01),
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    };
    solve(&problem(), algorithm, &options).unwrap().last_state()[0]
}

fn main() {
    println!("ars222,{:.17e}", endpoint(Ars222));
    println!("ars232,{:.17e}", endpoint(Ars232));
    println!("ars343,{:.17e}", endpoint(Ars343));
    println!("ars443,{:.17e}", endpoint(Ars443));
    println!("bhr553,{:.17e}", endpoint(Bhr553));
    println!("cfnlirk3,{:.17e}", endpoint(Cfnlirk3));
    println!("esdirk325l2sa,{:.17e}", endpoint(Esdirk325L2Sa));
    println!("esdirk436l2sa2,{:.17e}", endpoint(Esdirk436L2Sa2));
    println!("esdirk437l2sa,{:.17e}", endpoint(Esdirk437L2Sa));
    println!("esdirk547l2sa2,{:.17e}", endpoint(Esdirk547L2Sa2));
    println!("esdirk54i8l2sa,{:.17e}", endpoint(Esdirk54I8L2Sa));
    println!("esdirk659l2sa,{:.17e}", endpoint(Esdirk659L2Sa));
    println!("hairer4,{:.17e}", endpoint(Hairer4));
    println!("hairer42,{:.17e}", endpoint(Hairer42));
    println!("imexssp222,{:.17e}", endpoint(ImexSsp222));
    println!("imexssp2322,{:.17e}", endpoint(ImexSsp2322));
    println!("imexssp3332,{:.17e}", endpoint(ImexSsp3332));
    println!("imexssp3433,{:.17e}", endpoint(ImexSsp3433));
    println!("kencarp3,{:.17e}", endpoint(KenCarp3));
    println!("kencarp4,{:.17e}", endpoint(KenCarp4));
    println!("kencarp47,{:.17e}", endpoint(KenCarp47));
    println!("kencarp5,{:.17e}", endpoint(KenCarp5));
    println!("kencarp58,{:.17e}", endpoint(KenCarp58));
    println!("kvaerno3,{:.17e}", endpoint(Kvaerno3));
    println!("kvaerno4,{:.17e}", endpoint(Kvaerno4));
    println!("kvaerno5,{:.17e}", endpoint(Kvaerno5));
    println!("sdirk22,{:.17e}", endpoint(Sdirk22));
    println!("sfsdirk4,{:.17e}", endpoint(Sfsdirk4));
    println!("sfsdirk5,{:.17e}", endpoint(Sfsdirk5));
    println!("sfsdirk6,{:.17e}", endpoint(Sfsdirk6));
    println!("sfsdirk7,{:.17e}", endpoint(Sfsdirk7));
    println!("sfsdirk8,{:.17e}", endpoint(Sfsdirk8));
    println!("sspsdirk2,{:.17e}", endpoint(SspSdirk2));
}
