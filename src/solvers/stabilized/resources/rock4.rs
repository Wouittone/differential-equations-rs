use crate::tableau::{
    LazyRock4Tableau, Rock4Tableau, TableauError, define_rock4_tableau_from_file, load_tableau,
};

macro_rules! define_rock4_resources {
    ($(($degree:literal, $static_name:ident, $path:literal)),+ $(,)?) => {
        $(
            define_rock4_tableau_from_file!(
                pub(super) $static_name,
                "ROCK4",
                $degree,
                $path,
                crate = crate
            );
        )+

        pub(super) static ROCK4_RESOURCES: &[(usize, &LazyRock4Tableau)] = &[
            $(($degree, &$static_name),)+
        ];
    };
}

#[rustfmt::skip]
define_rock4_resources!(
    (1, ROCK4_001, "src/tableau/resources/methods/stabilized/rock4/degree-001.json"),
    (2, ROCK4_002, "src/tableau/resources/methods/stabilized/rock4/degree-002.json"),
    (3, ROCK4_003, "src/tableau/resources/methods/stabilized/rock4/degree-003.json"),
    (4, ROCK4_004, "src/tableau/resources/methods/stabilized/rock4/degree-004.json"),
    (5, ROCK4_005, "src/tableau/resources/methods/stabilized/rock4/degree-005.json"),
    (6, ROCK4_006, "src/tableau/resources/methods/stabilized/rock4/degree-006.json"),
    (7, ROCK4_007, "src/tableau/resources/methods/stabilized/rock4/degree-007.json"),
    (8, ROCK4_008, "src/tableau/resources/methods/stabilized/rock4/degree-008.json"),
    (9, ROCK4_009, "src/tableau/resources/methods/stabilized/rock4/degree-009.json"),
    (10, ROCK4_010, "src/tableau/resources/methods/stabilized/rock4/degree-010.json"),
    (11, ROCK4_011, "src/tableau/resources/methods/stabilized/rock4/degree-011.json"),
    (12, ROCK4_012, "src/tableau/resources/methods/stabilized/rock4/degree-012.json"),
    (13, ROCK4_013, "src/tableau/resources/methods/stabilized/rock4/degree-013.json"),
    (14, ROCK4_014, "src/tableau/resources/methods/stabilized/rock4/degree-014.json"),
    (15, ROCK4_015, "src/tableau/resources/methods/stabilized/rock4/degree-015.json"),
    (16, ROCK4_016, "src/tableau/resources/methods/stabilized/rock4/degree-016.json"),
    (17, ROCK4_017, "src/tableau/resources/methods/stabilized/rock4/degree-017.json"),
    (18, ROCK4_018, "src/tableau/resources/methods/stabilized/rock4/degree-018.json"),
    (19, ROCK4_019, "src/tableau/resources/methods/stabilized/rock4/degree-019.json"),
    (20, ROCK4_020, "src/tableau/resources/methods/stabilized/rock4/degree-020.json"),
    (22, ROCK4_022, "src/tableau/resources/methods/stabilized/rock4/degree-022.json"),
    (24, ROCK4_024, "src/tableau/resources/methods/stabilized/rock4/degree-024.json"),
    (26, ROCK4_026, "src/tableau/resources/methods/stabilized/rock4/degree-026.json"),
    (28, ROCK4_028, "src/tableau/resources/methods/stabilized/rock4/degree-028.json"),
    (30, ROCK4_030, "src/tableau/resources/methods/stabilized/rock4/degree-030.json"),
    (32, ROCK4_032, "src/tableau/resources/methods/stabilized/rock4/degree-032.json"),
    (34, ROCK4_034, "src/tableau/resources/methods/stabilized/rock4/degree-034.json"),
    (36, ROCK4_036, "src/tableau/resources/methods/stabilized/rock4/degree-036.json"),
    (38, ROCK4_038, "src/tableau/resources/methods/stabilized/rock4/degree-038.json"),
    (41, ROCK4_041, "src/tableau/resources/methods/stabilized/rock4/degree-041.json"),
    (44, ROCK4_044, "src/tableau/resources/methods/stabilized/rock4/degree-044.json"),
    (47, ROCK4_047, "src/tableau/resources/methods/stabilized/rock4/degree-047.json"),
    (50, ROCK4_050, "src/tableau/resources/methods/stabilized/rock4/degree-050.json"),
    (53, ROCK4_053, "src/tableau/resources/methods/stabilized/rock4/degree-053.json"),
    (56, ROCK4_056, "src/tableau/resources/methods/stabilized/rock4/degree-056.json"),
    (59, ROCK4_059, "src/tableau/resources/methods/stabilized/rock4/degree-059.json"),
    (63, ROCK4_063, "src/tableau/resources/methods/stabilized/rock4/degree-063.json"),
    (67, ROCK4_067, "src/tableau/resources/methods/stabilized/rock4/degree-067.json"),
    (71, ROCK4_071, "src/tableau/resources/methods/stabilized/rock4/degree-071.json"),
    (76, ROCK4_076, "src/tableau/resources/methods/stabilized/rock4/degree-076.json"),
    (81, ROCK4_081, "src/tableau/resources/methods/stabilized/rock4/degree-081.json"),
    (86, ROCK4_086, "src/tableau/resources/methods/stabilized/rock4/degree-086.json"),
    (92, ROCK4_092, "src/tableau/resources/methods/stabilized/rock4/degree-092.json"),
    (98, ROCK4_098, "src/tableau/resources/methods/stabilized/rock4/degree-098.json"),
    (105, ROCK4_105, "src/tableau/resources/methods/stabilized/rock4/degree-105.json"),
    (112, ROCK4_112, "src/tableau/resources/methods/stabilized/rock4/degree-112.json"),
    (120, ROCK4_120, "src/tableau/resources/methods/stabilized/rock4/degree-120.json"),
    (129, ROCK4_129, "src/tableau/resources/methods/stabilized/rock4/degree-129.json"),
    (138, ROCK4_138, "src/tableau/resources/methods/stabilized/rock4/degree-138.json"),
    (148, ROCK4_148, "src/tableau/resources/methods/stabilized/rock4/degree-148.json"),
);

pub(in crate::solvers::stabilized) fn rock4_tableau_for_degree(
    requested_degree: usize,
) -> Result<&'static Rock4Tableau, TableauError> {
    load_tableau(rock4_resource_for_degree(requested_degree))
}

pub(super) fn rock4_resource_for_degree(requested_degree: usize) -> &'static LazyRock4Tableau {
    let index = ROCK4_RESOURCES
        .partition_point(|(degree, _)| *degree < requested_degree)
        .min(ROCK4_RESOURCES.len() - 1);
    ROCK4_RESOURCES[index].1
}

pub(in crate::solvers::stabilized) fn rock4_available_degrees()
-> impl ExactSizeIterator<Item = usize> {
    ROCK4_RESOURCES.iter().map(|(degree, _)| *degree)
}
