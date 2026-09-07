use diffeq::tableau::{
    define_rock2_tableau_from_file, define_rock4_tableau_from_file, define_serk2_tableau_from_file,
    load_tableau,
};
use differential_equations as diffeq;

define_rock2_tableau_from_file!(
    pub FILE_ROCK2_DEGREE_2,
    "ROCK2",
    2,
    "tests/resources/file_rock2.json",
    crate = diffeq
);

define_rock4_tableau_from_file!(
    pub FILE_ROCK4_DEGREE_1,
    "ROCK4",
    1,
    "tests/resources/file_rock4.json",
    crate = diffeq
);

define_serk2_tableau_from_file!(
    pub FILE_SERK2_DEGREE_2,
    "SERK2",
    2,
    "tests/resources/file_serk2.json",
    crate = diffeq
);

#[test]
fn renamed_dependency_path_loads_a_degree_specific_rock2_tableau() {
    let first = load_tableau(&FILE_ROCK2_DEGREE_2).unwrap();
    let second = load_tableau(&FILE_ROCK2_DEGREE_2).unwrap();

    assert!(std::ptr::eq(first, second));
    assert_eq!(first.name(), "ROCK2");
    assert_eq!(first.order(), 2);
    assert_eq!(first.degree(), 2);
    assert_eq!(first.recurrence().stages().len(), 1);
    assert_eq!(
        first.finish_first().to_bits(),
        0.3889624104727243f64.to_bits()
    );
    assert_eq!(
        first.finish_second().to_bits(),
        0.4219428123056774f64.to_bits()
    );
}

#[test]
fn renamed_dependency_path_loads_a_degree_specific_rock4_tableau() {
    let first = load_tableau(&FILE_ROCK4_DEGREE_1).unwrap();
    let second = load_tableau(&FILE_ROCK4_DEGREE_1).unwrap();

    assert!(std::ptr::eq(first, second));
    assert_eq!(first.name(), "ROCK4");
    assert_eq!(first.order(), 4);
    assert_eq!(first.embedded_order(), 3);
    assert_eq!(first.degree(), 1);
    assert!(first.recurrence().stages().is_empty());
    assert_eq!(
        first.recurrence().first_stage().to_bits(),
        0.1762962957651941_f64.to_bits()
    );
    assert_eq!(
        first.finishing_a()[1][0].to_bits(),
        (-0.149352078672699_f64).to_bits()
    );
    assert_eq!(first.b()[0].to_bits(), 0.934502625489809_f64.to_bits());
    assert_eq!(first.b_hat()[4].to_bits(), 0.109256697110981_f64.to_bits());
}

#[test]
fn renamed_dependency_path_loads_a_degree_specific_serk2_tableau() {
    let first = load_tableau(&FILE_SERK2_DEGREE_2).unwrap();
    let second = load_tableau(&FILE_SERK2_DEGREE_2).unwrap();

    assert!(std::ptr::eq(first, second));
    assert_eq!(first.name(), "SERK2");
    assert_eq!(first.order(), 2);
    assert_eq!(first.degree(), 2);
    assert_eq!(first.subdivisions(), 1);
    assert_eq!(first.internal_degree(), 2);
    assert_eq!(first.weights(), [1.32, -0.96, 0.64]);
}
