use diffeq::tableau::{define_rock2_tableau_from_file, load_tableau};
use differential_equations as diffeq;

define_rock2_tableau_from_file!(
    pub FILE_ROCK2_DEGREE_2,
    "ROCK2",
    2,
    "tests/resources/file_rock2.json",
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
