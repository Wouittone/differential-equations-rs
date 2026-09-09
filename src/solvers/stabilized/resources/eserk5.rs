use crate::tableau::{
    EserkTableau, LazyEserkTableau, TableauError, define_eserk_tableau_from_file, load_tableau,
};

macro_rules! define_resources {
    ($(($degree:literal, $static_name:ident, $path:literal)),+ $(,)?) => {
        $(
            define_eserk_tableau_from_file!(
                pub(super) $static_name,
                "ESERK5",
                5,
                $degree,
                $path,
                crate = crate
            );
        )+

        pub(super) static RESOURCES: &[(usize, &LazyEserkTableau)] = &[
            $(($degree, &$static_name),)+
        ];
    };
}

#[rustfmt::skip]
define_resources!(
    (1, ESERK5_0001, "src/tableau/resources/methods/stabilized/eserk5/degree-0001.json"),
    (2, ESERK5_0002, "src/tableau/resources/methods/stabilized/eserk5/degree-0002.json"),
    (3, ESERK5_0003, "src/tableau/resources/methods/stabilized/eserk5/degree-0003.json"),
    (4, ESERK5_0004, "src/tableau/resources/methods/stabilized/eserk5/degree-0004.json"),
    (5, ESERK5_0005, "src/tableau/resources/methods/stabilized/eserk5/degree-0005.json"),
    (6, ESERK5_0006, "src/tableau/resources/methods/stabilized/eserk5/degree-0006.json"),
    (7, ESERK5_0007, "src/tableau/resources/methods/stabilized/eserk5/degree-0007.json"),
    (8, ESERK5_0008, "src/tableau/resources/methods/stabilized/eserk5/degree-0008.json"),
    (9, ESERK5_0009, "src/tableau/resources/methods/stabilized/eserk5/degree-0009.json"),
    (10, ESERK5_0010, "src/tableau/resources/methods/stabilized/eserk5/degree-0010.json"),
    (11, ESERK5_0011, "src/tableau/resources/methods/stabilized/eserk5/degree-0011.json"),
    (12, ESERK5_0012, "src/tableau/resources/methods/stabilized/eserk5/degree-0012.json"),
    (13, ESERK5_0013, "src/tableau/resources/methods/stabilized/eserk5/degree-0013.json"),
    (14, ESERK5_0014, "src/tableau/resources/methods/stabilized/eserk5/degree-0014.json"),
    (15, ESERK5_0015, "src/tableau/resources/methods/stabilized/eserk5/degree-0015.json"),
    (16, ESERK5_0016, "src/tableau/resources/methods/stabilized/eserk5/degree-0016.json"),
    (17, ESERK5_0017, "src/tableau/resources/methods/stabilized/eserk5/degree-0017.json"),
    (18, ESERK5_0018, "src/tableau/resources/methods/stabilized/eserk5/degree-0018.json"),
    (19, ESERK5_0019, "src/tableau/resources/methods/stabilized/eserk5/degree-0019.json"),
    (20, ESERK5_0020, "src/tableau/resources/methods/stabilized/eserk5/degree-0020.json"),
    (25, ESERK5_0025, "src/tableau/resources/methods/stabilized/eserk5/degree-0025.json"),
    (30, ESERK5_0030, "src/tableau/resources/methods/stabilized/eserk5/degree-0030.json"),
    (35, ESERK5_0035, "src/tableau/resources/methods/stabilized/eserk5/degree-0035.json"),
    (40, ESERK5_0040, "src/tableau/resources/methods/stabilized/eserk5/degree-0040.json"),
    (45, ESERK5_0045, "src/tableau/resources/methods/stabilized/eserk5/degree-0045.json"),
    (50, ESERK5_0050, "src/tableau/resources/methods/stabilized/eserk5/degree-0050.json"),
    (60, ESERK5_0060, "src/tableau/resources/methods/stabilized/eserk5/degree-0060.json"),
    (70, ESERK5_0070, "src/tableau/resources/methods/stabilized/eserk5/degree-0070.json"),
    (80, ESERK5_0080, "src/tableau/resources/methods/stabilized/eserk5/degree-0080.json"),
    (90, ESERK5_0090, "src/tableau/resources/methods/stabilized/eserk5/degree-0090.json"),
    (100, ESERK5_0100, "src/tableau/resources/methods/stabilized/eserk5/degree-0100.json"),
    (150, ESERK5_0150, "src/tableau/resources/methods/stabilized/eserk5/degree-0150.json"),
    (200, ESERK5_0200, "src/tableau/resources/methods/stabilized/eserk5/degree-0200.json"),
    (250, ESERK5_0250, "src/tableau/resources/methods/stabilized/eserk5/degree-0250.json"),
    (300, ESERK5_0300, "src/tableau/resources/methods/stabilized/eserk5/degree-0300.json"),
    (350, ESERK5_0350, "src/tableau/resources/methods/stabilized/eserk5/degree-0350.json"),
    (400, ESERK5_0400, "src/tableau/resources/methods/stabilized/eserk5/degree-0400.json"),
    (450, ESERK5_0450, "src/tableau/resources/methods/stabilized/eserk5/degree-0450.json"),
    (500, ESERK5_0500, "src/tableau/resources/methods/stabilized/eserk5/degree-0500.json"),
    (600, ESERK5_0600, "src/tableau/resources/methods/stabilized/eserk5/degree-0600.json"),
    (700, ESERK5_0700, "src/tableau/resources/methods/stabilized/eserk5/degree-0700.json"),
    (800, ESERK5_0800, "src/tableau/resources/methods/stabilized/eserk5/degree-0800.json"),
    (900, ESERK5_0900, "src/tableau/resources/methods/stabilized/eserk5/degree-0900.json"),
    (1000, ESERK5_1000, "src/tableau/resources/methods/stabilized/eserk5/degree-1000.json"),
    (1200, ESERK5_1200, "src/tableau/resources/methods/stabilized/eserk5/degree-1200.json"),
    (1400, ESERK5_1400, "src/tableau/resources/methods/stabilized/eserk5/degree-1400.json"),
    (1600, ESERK5_1600, "src/tableau/resources/methods/stabilized/eserk5/degree-1600.json"),
    (1800, ESERK5_1800, "src/tableau/resources/methods/stabilized/eserk5/degree-1800.json"),
    (2000, ESERK5_2000, "src/tableau/resources/methods/stabilized/eserk5/degree-2000.json"),
);

pub(crate) fn eserk5_tableau_for_degree(
    requested_degree: usize,
) -> Result<&'static EserkTableau, TableauError> {
    load_tableau(resource_for_degree(requested_degree))
}

pub(super) fn resource_for_degree(requested_degree: usize) -> &'static LazyEserkTableau {
    let index = RESOURCES
        .partition_point(|(degree, _)| *degree < requested_degree)
        .min(RESOURCES.len() - 1);
    RESOURCES[index].1
}

pub(crate) fn eserk5_available_degrees() -> impl ExactSizeIterator<Item = usize> {
    RESOURCES.iter().map(|(degree, _)| *degree)
}
