use crate::tableau::{
    LazySerk2Tableau, Serk2Tableau, TableauError, define_serk2_tableau_from_file, load_tableau,
};

macro_rules! define_serk2_resources {
    ($(($degree:literal, $static_name:ident, $path:literal)),+ $(,)?) => {
        $(
            define_serk2_tableau_from_file!(
                pub(super) $static_name,
                "SERK2",
                $degree,
                $path,
                crate = crate
            );
        )+

        pub(super) static SERK2_RESOURCES: &[(usize, &LazySerk2Tableau)] = &[
            $(($degree, &$static_name),)+
        ];
    };
}

#[rustfmt::skip]
define_serk2_resources!(
    (10, SERK2_010, "src/tableau/resources/methods/stabilized/serk2/degree-010.json"),
    (20, SERK2_020, "src/tableau/resources/methods/stabilized/serk2/degree-020.json"),
    (30, SERK2_030, "src/tableau/resources/methods/stabilized/serk2/degree-030.json"),
    (40, SERK2_040, "src/tableau/resources/methods/stabilized/serk2/degree-040.json"),
    (50, SERK2_050, "src/tableau/resources/methods/stabilized/serk2/degree-050.json"),
    (60, SERK2_060, "src/tableau/resources/methods/stabilized/serk2/degree-060.json"),
    (80, SERK2_080, "src/tableau/resources/methods/stabilized/serk2/degree-080.json"),
    (100, SERK2_100, "src/tableau/resources/methods/stabilized/serk2/degree-100.json"),
    (150, SERK2_150, "src/tableau/resources/methods/stabilized/serk2/degree-150.json"),
    (200, SERK2_200, "src/tableau/resources/methods/stabilized/serk2/degree-200.json"),
    (250, SERK2_250, "src/tableau/resources/methods/stabilized/serk2/degree-250.json"),
);

pub(in crate::solvers::stabilized) fn serk2_tableau_for_degree(
    requested_degree: usize,
) -> Result<&'static Serk2Tableau, TableauError> {
    load_tableau(serk2_resource_for_degree(requested_degree))
}

pub(super) fn serk2_resource_for_degree(requested_degree: usize) -> &'static LazySerk2Tableau {
    let index = SERK2_RESOURCES
        .partition_point(|(degree, _)| *degree < requested_degree)
        .min(SERK2_RESOURCES.len() - 1);
    SERK2_RESOURCES[index].1
}

pub(in crate::solvers::stabilized) fn serk2_available_degrees()
-> impl ExactSizeIterator<Item = usize> {
    SERK2_RESOURCES.iter().map(|(degree, _)| *degree)
}
