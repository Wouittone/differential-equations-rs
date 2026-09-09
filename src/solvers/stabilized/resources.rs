use crate::tableau::{
    LazyRock2Tableau, LazyRock4Tableau, LazySerk2Tableau, Rock2Tableau, Rock4Tableau, Serk2Tableau,
    TableauError, define_rock2_tableau_from_file, define_rock4_tableau_from_file,
    define_serk2_tableau_from_file, load_tableau,
};

mod eserk4;
mod eserk5;

pub(super) use eserk4::{eserk4_available_degrees, eserk4_tableau_for_degree};
pub(super) use eserk5::{eserk5_available_degrees, eserk5_tableau_for_degree};

define_rock2_tableau_from_file!(
    pub(super) ROCK2_001,
    "ROCK2",
    1,
    "src/tableau/resources/methods/stabilized/rock2/degree-001.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_002,
    "ROCK2",
    2,
    "src/tableau/resources/methods/stabilized/rock2/degree-002.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_003,
    "ROCK2",
    3,
    "src/tableau/resources/methods/stabilized/rock2/degree-003.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_004,
    "ROCK2",
    4,
    "src/tableau/resources/methods/stabilized/rock2/degree-004.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_005,
    "ROCK2",
    5,
    "src/tableau/resources/methods/stabilized/rock2/degree-005.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_006,
    "ROCK2",
    6,
    "src/tableau/resources/methods/stabilized/rock2/degree-006.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_007,
    "ROCK2",
    7,
    "src/tableau/resources/methods/stabilized/rock2/degree-007.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_008,
    "ROCK2",
    8,
    "src/tableau/resources/methods/stabilized/rock2/degree-008.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_009,
    "ROCK2",
    9,
    "src/tableau/resources/methods/stabilized/rock2/degree-009.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_010,
    "ROCK2",
    10,
    "src/tableau/resources/methods/stabilized/rock2/degree-010.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_011,
    "ROCK2",
    11,
    "src/tableau/resources/methods/stabilized/rock2/degree-011.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_012,
    "ROCK2",
    12,
    "src/tableau/resources/methods/stabilized/rock2/degree-012.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_013,
    "ROCK2",
    13,
    "src/tableau/resources/methods/stabilized/rock2/degree-013.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_014,
    "ROCK2",
    14,
    "src/tableau/resources/methods/stabilized/rock2/degree-014.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_015,
    "ROCK2",
    15,
    "src/tableau/resources/methods/stabilized/rock2/degree-015.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_016,
    "ROCK2",
    16,
    "src/tableau/resources/methods/stabilized/rock2/degree-016.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_017,
    "ROCK2",
    17,
    "src/tableau/resources/methods/stabilized/rock2/degree-017.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_018,
    "ROCK2",
    18,
    "src/tableau/resources/methods/stabilized/rock2/degree-018.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_019,
    "ROCK2",
    19,
    "src/tableau/resources/methods/stabilized/rock2/degree-019.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_020,
    "ROCK2",
    20,
    "src/tableau/resources/methods/stabilized/rock2/degree-020.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_022,
    "ROCK2",
    22,
    "src/tableau/resources/methods/stabilized/rock2/degree-022.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_024,
    "ROCK2",
    24,
    "src/tableau/resources/methods/stabilized/rock2/degree-024.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_026,
    "ROCK2",
    26,
    "src/tableau/resources/methods/stabilized/rock2/degree-026.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_028,
    "ROCK2",
    28,
    "src/tableau/resources/methods/stabilized/rock2/degree-028.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_030,
    "ROCK2",
    30,
    "src/tableau/resources/methods/stabilized/rock2/degree-030.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_033,
    "ROCK2",
    33,
    "src/tableau/resources/methods/stabilized/rock2/degree-033.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_036,
    "ROCK2",
    36,
    "src/tableau/resources/methods/stabilized/rock2/degree-036.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_039,
    "ROCK2",
    39,
    "src/tableau/resources/methods/stabilized/rock2/degree-039.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_043,
    "ROCK2",
    43,
    "src/tableau/resources/methods/stabilized/rock2/degree-043.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_047,
    "ROCK2",
    47,
    "src/tableau/resources/methods/stabilized/rock2/degree-047.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_051,
    "ROCK2",
    51,
    "src/tableau/resources/methods/stabilized/rock2/degree-051.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_056,
    "ROCK2",
    56,
    "src/tableau/resources/methods/stabilized/rock2/degree-056.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_061,
    "ROCK2",
    61,
    "src/tableau/resources/methods/stabilized/rock2/degree-061.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_066,
    "ROCK2",
    66,
    "src/tableau/resources/methods/stabilized/rock2/degree-066.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_072,
    "ROCK2",
    72,
    "src/tableau/resources/methods/stabilized/rock2/degree-072.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_078,
    "ROCK2",
    78,
    "src/tableau/resources/methods/stabilized/rock2/degree-078.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_085,
    "ROCK2",
    85,
    "src/tableau/resources/methods/stabilized/rock2/degree-085.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_093,
    "ROCK2",
    93,
    "src/tableau/resources/methods/stabilized/rock2/degree-093.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_102,
    "ROCK2",
    102,
    "src/tableau/resources/methods/stabilized/rock2/degree-102.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_112,
    "ROCK2",
    112,
    "src/tableau/resources/methods/stabilized/rock2/degree-112.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_123,
    "ROCK2",
    123,
    "src/tableau/resources/methods/stabilized/rock2/degree-123.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_135,
    "ROCK2",
    135,
    "src/tableau/resources/methods/stabilized/rock2/degree-135.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_148,
    "ROCK2",
    148,
    "src/tableau/resources/methods/stabilized/rock2/degree-148.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_163,
    "ROCK2",
    163,
    "src/tableau/resources/methods/stabilized/rock2/degree-163.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_180,
    "ROCK2",
    180,
    "src/tableau/resources/methods/stabilized/rock2/degree-180.json",
    crate = crate
);

define_rock2_tableau_from_file!(
    pub(super) ROCK2_198,
    "ROCK2",
    198,
    "src/tableau/resources/methods/stabilized/rock2/degree-198.json",
    crate = crate
);

static ROCK2_RESOURCES: [(usize, &LazyRock2Tableau); 46] = [
    (1, &ROCK2_001),
    (2, &ROCK2_002),
    (3, &ROCK2_003),
    (4, &ROCK2_004),
    (5, &ROCK2_005),
    (6, &ROCK2_006),
    (7, &ROCK2_007),
    (8, &ROCK2_008),
    (9, &ROCK2_009),
    (10, &ROCK2_010),
    (11, &ROCK2_011),
    (12, &ROCK2_012),
    (13, &ROCK2_013),
    (14, &ROCK2_014),
    (15, &ROCK2_015),
    (16, &ROCK2_016),
    (17, &ROCK2_017),
    (18, &ROCK2_018),
    (19, &ROCK2_019),
    (20, &ROCK2_020),
    (22, &ROCK2_022),
    (24, &ROCK2_024),
    (26, &ROCK2_026),
    (28, &ROCK2_028),
    (30, &ROCK2_030),
    (33, &ROCK2_033),
    (36, &ROCK2_036),
    (39, &ROCK2_039),
    (43, &ROCK2_043),
    (47, &ROCK2_047),
    (51, &ROCK2_051),
    (56, &ROCK2_056),
    (61, &ROCK2_061),
    (66, &ROCK2_066),
    (72, &ROCK2_072),
    (78, &ROCK2_078),
    (85, &ROCK2_085),
    (93, &ROCK2_093),
    (102, &ROCK2_102),
    (112, &ROCK2_112),
    (123, &ROCK2_123),
    (135, &ROCK2_135),
    (148, &ROCK2_148),
    (163, &ROCK2_163),
    (180, &ROCK2_180),
    (198, &ROCK2_198),
];

pub(super) fn rock2_tableau_for_degree(
    requested_degree: usize,
) -> Result<&'static Rock2Tableau, TableauError> {
    load_tableau(rock2_resource_for_degree(requested_degree))
}

fn rock2_resource_for_degree(requested_degree: usize) -> &'static LazyRock2Tableau {
    let index = ROCK2_RESOURCES
        .partition_point(|(degree, _)| *degree < requested_degree)
        .min(ROCK2_RESOURCES.len() - 1);
    ROCK2_RESOURCES[index].1
}

pub(super) fn rock2_available_degrees() -> impl ExactSizeIterator<Item = usize> {
    ROCK2_RESOURCES.iter().map(|(degree, _)| *degree)
}

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

        static ROCK4_RESOURCES: &[(usize, &LazyRock4Tableau)] = &[
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

pub(super) fn rock4_tableau_for_degree(
    requested_degree: usize,
) -> Result<&'static Rock4Tableau, TableauError> {
    load_tableau(rock4_resource_for_degree(requested_degree))
}

fn rock4_resource_for_degree(requested_degree: usize) -> &'static LazyRock4Tableau {
    let index = ROCK4_RESOURCES
        .partition_point(|(degree, _)| *degree < requested_degree)
        .min(ROCK4_RESOURCES.len() - 1);
    ROCK4_RESOURCES[index].1
}

pub(super) fn rock4_available_degrees() -> impl ExactSizeIterator<Item = usize> {
    ROCK4_RESOURCES.iter().map(|(degree, _)| *degree)
}

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

        static SERK2_RESOURCES: &[(usize, &LazySerk2Tableau)] = &[
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

pub(super) fn serk2_tableau_for_degree(
    requested_degree: usize,
) -> Result<&'static Serk2Tableau, TableauError> {
    load_tableau(serk2_resource_for_degree(requested_degree))
}

fn serk2_resource_for_degree(requested_degree: usize) -> &'static LazySerk2Tableau {
    let index = SERK2_RESOURCES
        .partition_point(|(degree, _)| *degree < requested_degree)
        .min(SERK2_RESOURCES.len() - 1);
    SERK2_RESOURCES[index].1
}

pub(super) fn serk2_available_degrees() -> impl ExactSizeIterator<Item = usize> {
    SERK2_RESOURCES.iter().map(|(degree, _)| *degree)
}

#[cfg(test)]
mod tests {
    use super::eserk4::{RESOURCES as ESERK4_RESOURCES, resource_for_degree as eserk4_resource};
    use super::eserk5::{RESOURCES as ESERK5_RESOURCES, resource_for_degree as eserk5_resource};
    use super::{
        ROCK2_RESOURCES, ROCK4_RESOURCES, SERK2_RESOURCES, eserk4_tableau_for_degree,
        eserk5_tableau_for_degree, rock2_resource_for_degree, rock2_tableau_for_degree,
        rock4_resource_for_degree, rock4_tableau_for_degree, serk2_resource_for_degree,
        serk2_tableau_for_degree,
    };

    fn hash_word(hash: &mut u64, word: u64) {
        for byte in word.to_le_bytes() {
            *hash ^= u64::from(byte);
            *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    #[test]
    fn rock2_registry_is_sorted_unique_and_matches_resources() {
        assert!(!ROCK2_RESOURCES.is_empty());
        assert_eq!(ROCK2_RESOURCES.len(), 46);
        assert!(ROCK2_RESOURCES.windows(2).all(|pair| pair[0].0 < pair[1].0));
        for &(degree, _) in &ROCK2_RESOURCES {
            let tableau = rock2_tableau_for_degree(degree).unwrap();
            assert_eq!(tableau.degree(), degree);
            assert_eq!(tableau.recurrence().stages().len(), degree - 1);
        }
    }

    #[test]
    fn rock2_selection_uses_ceiling_degree_and_clamps() {
        for (requested, selected) in [
            (0, 1),
            (1, 1),
            (20, 20),
            (21, 22),
            (22, 22),
            (197, 198),
            (198, 198),
            (usize::MAX, 198),
        ] {
            assert_eq!(
                rock2_tableau_for_degree(requested).unwrap().degree(),
                selected
            );
        }

        for requested in 0..=199 {
            let expected = ROCK2_RESOURCES
                .iter()
                .find(|(degree, _)| *degree >= requested)
                .unwrap_or_else(|| ROCK2_RESOURCES.last().unwrap());
            assert!(std::ptr::eq(
                rock2_resource_for_degree(requested),
                expected.1
            ));
        }
    }

    #[test]
    fn rock2_resources_match_the_pinned_coefficient_fingerprint() {
        let mut hash = 0xcbf2_9ce4_8422_2325;
        for &(degree, _) in &ROCK2_RESOURCES {
            let tableau = rock2_tableau_for_degree(degree).unwrap();
            hash_word(&mut hash, degree as u64);
            hash_word(&mut hash, tableau.finish_first().to_bits());
            hash_word(&mut hash, tableau.finish_second().to_bits());
            hash_word(&mut hash, tableau.recurrence().first_stage().to_bits());
            for stage in tableau.recurrence().stages() {
                hash_word(&mut hash, stage.mu().to_bits());
                hash_word(&mut hash, stage.kappa().to_bits());
            }
        }
        assert_eq!(hash, 0x674d_1940_faf4_7f34);
    }

    #[test]
    fn rock4_registry_is_sorted_unique_and_matches_resources() {
        assert_eq!(ROCK4_RESOURCES.len(), 50);
        assert_eq!(ROCK4_RESOURCES.first().unwrap().0, 1);
        assert_eq!(ROCK4_RESOURCES.last().unwrap().0, 148);
        assert!(ROCK4_RESOURCES.windows(2).all(|pair| pair[0].0 < pair[1].0));
        for &(degree, _) in ROCK4_RESOURCES {
            let tableau = rock4_tableau_for_degree(degree).unwrap();
            assert_eq!(tableau.degree(), degree);
            assert_eq!(tableau.order(), 4);
            assert_eq!(tableau.embedded_order(), 3);
            assert_eq!(tableau.recurrence().stages().len(), degree - 1);
            assert_eq!(
                tableau
                    .finishing_a()
                    .iter()
                    .map(Vec::len)
                    .collect::<Vec<_>>(),
                [0, 1, 2, 3]
            );
            assert_eq!(tableau.b().len(), 4);
            assert_eq!(tableau.b_hat().len(), 5);
        }
    }

    #[test]
    fn rock4_selection_uses_ceiling_degree_and_clamps() {
        for (requested, selected) in [
            (0, 1),
            (1, 1),
            (20, 20),
            (21, 22),
            (22, 22),
            (30, 30),
            (31, 32),
            (39, 41),
            (40, 41),
            (129, 129),
            (130, 138),
            (139, 148),
            (148, 148),
            (149, 148),
            (usize::MAX, 148),
        ] {
            assert_eq!(
                rock4_tableau_for_degree(requested).unwrap().degree(),
                selected
            );
        }

        for requested in 0..=149 {
            let expected = ROCK4_RESOURCES
                .iter()
                .find(|(degree, _)| *degree >= requested)
                .unwrap_or_else(|| ROCK4_RESOURCES.last().unwrap());
            assert!(std::ptr::eq(
                rock4_resource_for_degree(requested),
                expected.1
            ));
        }
    }

    #[test]
    fn rock4_resources_match_the_pinned_coefficient_fingerprint() {
        let mut hash = 0xcbf2_9ce4_8422_2325;
        for &(degree, _) in ROCK4_RESOURCES {
            let tableau = rock4_tableau_for_degree(degree).unwrap();
            hash_word(&mut hash, degree as u64);
            hash_word(&mut hash, tableau.recurrence().first_stage().to_bits());
            for stage in tableau.recurrence().stages() {
                hash_word(&mut hash, stage.mu().to_bits());
                hash_word(&mut hash, stage.kappa().to_bits());
            }
            for coefficient in tableau.finishing_a().iter().flatten() {
                hash_word(&mut hash, coefficient.to_bits());
            }
            for coefficient in tableau.b() {
                hash_word(&mut hash, coefficient.to_bits());
            }
            for coefficient in tableau.b_hat() {
                hash_word(&mut hash, coefficient.to_bits());
            }
        }
        assert_eq!(hash, 0x9ace_59aa_6696_f261);
    }

    #[test]
    fn serk2_registry_is_sorted_unique_and_matches_resources() {
        assert_eq!(SERK2_RESOURCES.len(), 11);
        assert_eq!(SERK2_RESOURCES.first().unwrap().0, 10);
        assert_eq!(SERK2_RESOURCES.last().unwrap().0, 250);
        assert!(SERK2_RESOURCES.windows(2).all(|pair| pair[0].0 < pair[1].0));
        for &(degree, _) in SERK2_RESOURCES {
            let tableau = serk2_tableau_for_degree(degree).unwrap();
            assert_eq!(tableau.degree(), degree);
            assert_eq!(tableau.order(), 2);
            assert_eq!(tableau.alpha(), 2.5 / (degree * degree) as f64);
            assert_eq!(tableau.subdivisions(), 10);
            assert_eq!(tableau.internal_degree(), degree / 10);
            assert_eq!(tableau.weights().len(), degree + 1);
        }
    }

    #[test]
    fn serk2_selection_uses_ceiling_degree_and_clamps() {
        for (requested, selected) in [
            (0, 10),
            (10, 10),
            (11, 20),
            (20, 20),
            (21, 30),
            (60, 60),
            (61, 80),
            (100, 100),
            (101, 150),
            (250, 250),
            (251, 250),
            (usize::MAX, 250),
        ] {
            assert_eq!(
                serk2_tableau_for_degree(requested).unwrap().degree(),
                selected
            );
        }

        for requested in 0..=251 {
            let expected = SERK2_RESOURCES
                .iter()
                .find(|(degree, _)| *degree >= requested)
                .unwrap_or_else(|| SERK2_RESOURCES.last().unwrap());
            assert!(std::ptr::eq(
                serk2_resource_for_degree(requested),
                expected.1
            ));
        }
    }

    #[test]
    fn serk2_resources_match_the_pinned_coefficient_fingerprint() {
        let mut hash = 0xcbf2_9ce4_8422_2325;
        for &(degree, _) in SERK2_RESOURCES {
            let tableau = serk2_tableau_for_degree(degree).unwrap();
            hash_word(&mut hash, degree as u64);
            hash_word(&mut hash, tableau.alpha().to_bits());
            hash_word(&mut hash, tableau.subdivisions() as u64);
            for coefficient in tableau.weights() {
                hash_word(&mut hash, coefficient.to_bits());
            }
        }
        assert_eq!(hash, 0xea2b_b5b4_ce6d_aa22);
    }

    fn assert_eserk_registry(
        resources: &'static [(usize, &'static crate::tableau::LazyEserkTableau)],
        order: usize,
        first_degree: usize,
        last_degree: usize,
    ) {
        assert_eq!(resources.first().unwrap().0, first_degree);
        assert_eq!(resources.last().unwrap().0, last_degree);
        assert!(resources.windows(2).all(|pair| pair[0].0 < pair[1].0));
        for &(degree, resource) in resources {
            let tableau = crate::tableau::load_tableau(resource).unwrap();
            let (
                expected_internal_degree,
                expected_alpha,
                expected_solution,
                expected_error,
                expected_denominator,
            ): (usize, f64, &[i32], &[i32], f64) = if order == 4 {
                let internal = match degree {
                    0..=20 => 2,
                    21..=100 => 10,
                    101..=500 => 25,
                    501..=1_000 => 100,
                    _ => 200,
                };
                (
                    internal,
                    2.0 / (degree * degree) as f64,
                    &[-1, 24, -81, 64],
                    &[-1, 12, -27, 16],
                    6.0,
                )
            } else {
                let internal = match degree {
                    0..=20 => 2,
                    21..=50 => 5,
                    51..=100 => 10,
                    101..=500 => 50,
                    501..=1_000 => 100,
                    _ => 200,
                };
                (
                    internal,
                    100.0 / (49 * degree * degree) as f64,
                    &[1, -64, 486, -1024, 625],
                    &[1, -32, 162, -256, 125],
                    24.0,
                )
            };
            assert_eq!(tableau.degree(), degree);
            assert_eq!(tableau.order(), order);
            assert_eq!(tableau.embedded_order(), order - 1);
            assert_eq!(tableau.subdivisions(), order);
            assert_eq!(tableau.internal_degree(), expected_internal_degree);
            assert_eq!(tableau.alpha().to_bits(), expected_alpha.to_bits());
            assert_eq!(tableau.solution_combination(), expected_solution);
            assert_eq!(tableau.error_combination(), expected_error);
            assert_eq!(tableau.combination_denominator(), expected_denominator);
            assert_eq!(tableau.solution_combination().len(), order);
            assert_eq!(tableau.error_combination().len(), order);
            assert_eq!(tableau.weights().len(), degree + 1);
        }
    }

    #[test]
    fn eserk_registries_are_sorted_unique_and_match_resources() {
        assert_eq!(ESERK4_RESOURCES.len(), 46);
        assert_eq!(ESERK5_RESOURCES.len(), 49);
        assert_eserk_registry(ESERK4_RESOURCES, 4, 2, 4_000);
        assert_eserk_registry(ESERK5_RESOURCES, 5, 1, 2_000);
    }

    #[test]
    fn eserk_selection_uses_ceiling_degree_and_clamps() {
        for (requested, selected) in [
            (0, 2),
            (2, 2),
            (3, 4),
            (20, 20),
            (21, 30),
            (1_001, 1_200),
            (4_000, 4_000),
            (usize::MAX, 4_000),
        ] {
            assert_eq!(
                eserk4_tableau_for_degree(requested).unwrap().degree(),
                selected
            );
        }
        for (requested, selected) in [
            (0, 1),
            (1, 1),
            (20, 20),
            (21, 25),
            (1_001, 1_200),
            (2_000, 2_000),
            (usize::MAX, 2_000),
        ] {
            assert_eq!(
                eserk5_tableau_for_degree(requested).unwrap().degree(),
                selected
            );
        }
        for requested in 0..=4_001 {
            let expected = ESERK4_RESOURCES
                .iter()
                .find(|(degree, _)| *degree >= requested)
                .unwrap_or_else(|| ESERK4_RESOURCES.last().unwrap());
            assert!(std::ptr::eq(eserk4_resource(requested), expected.1));
        }
        for requested in 0..=2_001 {
            let expected = ESERK5_RESOURCES
                .iter()
                .find(|(degree, _)| *degree >= requested)
                .unwrap_or_else(|| ESERK5_RESOURCES.last().unwrap());
            assert!(std::ptr::eq(eserk5_resource(requested), expected.1));
        }
    }

    fn eserk_fingerprint(
        resources: &'static [(usize, &'static crate::tableau::LazyEserkTableau)],
    ) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325;
        for &(degree, resource) in resources {
            let tableau = crate::tableau::load_tableau(resource).unwrap();
            hash_word(&mut hash, degree as u64);
            hash_word(&mut hash, tableau.internal_degree() as u64);
            hash_word(&mut hash, tableau.alpha().to_bits());
            hash_word(&mut hash, tableau.subdivisions() as u64);
            for &coefficient in tableau.solution_combination() {
                hash_word(&mut hash, coefficient as u32 as u64);
            }
            for &coefficient in tableau.error_combination() {
                hash_word(&mut hash, coefficient as u32 as u64);
            }
            hash_word(&mut hash, tableau.combination_denominator().to_bits());
            for coefficient in tableau.weights() {
                hash_word(&mut hash, coefficient.to_bits());
            }
        }
        hash
    }

    #[test]
    fn eserk_resources_match_pinned_coefficient_fingerprints() {
        assert_eq!(eserk_fingerprint(ESERK4_RESOURCES), 0x91fd_2411_206a_082a);
        assert_eq!(eserk_fingerprint(ESERK5_RESOURCES), 0x7cce_0916_0981_5847);
    }
}
