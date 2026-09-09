use crate::tableau::{
    EserkTableau, LazyEserkTableau, TableauError, define_eserk_tableau_from_file, load_tableau,
};

macro_rules! define_resources {
    ($(($degree:literal, $static_name:ident, $path:literal)),+ $(,)?) => {
        $(
            define_eserk_tableau_from_file!(
                pub(super) $static_name,
                "ESERK4",
                4,
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
    (2, ESERK4_0002, "src/tableau/resources/methods/stabilized/eserk4/degree-0002.json"),
    (4, ESERK4_0004, "src/tableau/resources/methods/stabilized/eserk4/degree-0004.json"),
    (6, ESERK4_0006, "src/tableau/resources/methods/stabilized/eserk4/degree-0006.json"),
    (8, ESERK4_0008, "src/tableau/resources/methods/stabilized/eserk4/degree-0008.json"),
    (10, ESERK4_0010, "src/tableau/resources/methods/stabilized/eserk4/degree-0010.json"),
    (12, ESERK4_0012, "src/tableau/resources/methods/stabilized/eserk4/degree-0012.json"),
    (14, ESERK4_0014, "src/tableau/resources/methods/stabilized/eserk4/degree-0014.json"),
    (16, ESERK4_0016, "src/tableau/resources/methods/stabilized/eserk4/degree-0016.json"),
    (18, ESERK4_0018, "src/tableau/resources/methods/stabilized/eserk4/degree-0018.json"),
    (20, ESERK4_0020, "src/tableau/resources/methods/stabilized/eserk4/degree-0020.json"),
    (30, ESERK4_0030, "src/tableau/resources/methods/stabilized/eserk4/degree-0030.json"),
    (40, ESERK4_0040, "src/tableau/resources/methods/stabilized/eserk4/degree-0040.json"),
    (50, ESERK4_0050, "src/tableau/resources/methods/stabilized/eserk4/degree-0050.json"),
    (60, ESERK4_0060, "src/tableau/resources/methods/stabilized/eserk4/degree-0060.json"),
    (70, ESERK4_0070, "src/tableau/resources/methods/stabilized/eserk4/degree-0070.json"),
    (80, ESERK4_0080, "src/tableau/resources/methods/stabilized/eserk4/degree-0080.json"),
    (90, ESERK4_0090, "src/tableau/resources/methods/stabilized/eserk4/degree-0090.json"),
    (100, ESERK4_0100, "src/tableau/resources/methods/stabilized/eserk4/degree-0100.json"),
    (150, ESERK4_0150, "src/tableau/resources/methods/stabilized/eserk4/degree-0150.json"),
    (200, ESERK4_0200, "src/tableau/resources/methods/stabilized/eserk4/degree-0200.json"),
    (250, ESERK4_0250, "src/tableau/resources/methods/stabilized/eserk4/degree-0250.json"),
    (300, ESERK4_0300, "src/tableau/resources/methods/stabilized/eserk4/degree-0300.json"),
    (350, ESERK4_0350, "src/tableau/resources/methods/stabilized/eserk4/degree-0350.json"),
    (400, ESERK4_0400, "src/tableau/resources/methods/stabilized/eserk4/degree-0400.json"),
    (450, ESERK4_0450, "src/tableau/resources/methods/stabilized/eserk4/degree-0450.json"),
    (500, ESERK4_0500, "src/tableau/resources/methods/stabilized/eserk4/degree-0500.json"),
    (600, ESERK4_0600, "src/tableau/resources/methods/stabilized/eserk4/degree-0600.json"),
    (700, ESERK4_0700, "src/tableau/resources/methods/stabilized/eserk4/degree-0700.json"),
    (800, ESERK4_0800, "src/tableau/resources/methods/stabilized/eserk4/degree-0800.json"),
    (900, ESERK4_0900, "src/tableau/resources/methods/stabilized/eserk4/degree-0900.json"),
    (1000, ESERK4_1000, "src/tableau/resources/methods/stabilized/eserk4/degree-1000.json"),
    (1200, ESERK4_1200, "src/tableau/resources/methods/stabilized/eserk4/degree-1200.json"),
    (1400, ESERK4_1400, "src/tableau/resources/methods/stabilized/eserk4/degree-1400.json"),
    (1600, ESERK4_1600, "src/tableau/resources/methods/stabilized/eserk4/degree-1600.json"),
    (1800, ESERK4_1800, "src/tableau/resources/methods/stabilized/eserk4/degree-1800.json"),
    (2000, ESERK4_2000, "src/tableau/resources/methods/stabilized/eserk4/degree-2000.json"),
    (2200, ESERK4_2200, "src/tableau/resources/methods/stabilized/eserk4/degree-2200.json"),
    (2400, ESERK4_2400, "src/tableau/resources/methods/stabilized/eserk4/degree-2400.json"),
    (2600, ESERK4_2600, "src/tableau/resources/methods/stabilized/eserk4/degree-2600.json"),
    (2800, ESERK4_2800, "src/tableau/resources/methods/stabilized/eserk4/degree-2800.json"),
    (3000, ESERK4_3000, "src/tableau/resources/methods/stabilized/eserk4/degree-3000.json"),
    (3200, ESERK4_3200, "src/tableau/resources/methods/stabilized/eserk4/degree-3200.json"),
    (3400, ESERK4_3400, "src/tableau/resources/methods/stabilized/eserk4/degree-3400.json"),
    (3600, ESERK4_3600, "src/tableau/resources/methods/stabilized/eserk4/degree-3600.json"),
    (3800, ESERK4_3800, "src/tableau/resources/methods/stabilized/eserk4/degree-3800.json"),
    (4000, ESERK4_4000, "src/tableau/resources/methods/stabilized/eserk4/degree-4000.json"),
);

pub(crate) fn eserk4_tableau_for_degree(
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

pub(crate) fn eserk4_available_degrees() -> impl ExactSizeIterator<Item = usize> {
    RESOURCES.iter().map(|(degree, _)| *degree)
}
