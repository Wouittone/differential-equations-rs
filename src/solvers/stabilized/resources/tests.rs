use super::eserk4::{RESOURCES as ESERK4_RESOURCES, resource_for_degree as eserk4_resource};
use super::eserk4_tableau_for_degree;
use super::eserk5::{RESOURCES as ESERK5_RESOURCES, resource_for_degree as eserk5_resource};
use super::eserk5_tableau_for_degree;
use super::rock2::{ROCK2_RESOURCES, rock2_resource_for_degree, rock2_tableau_for_degree};
use super::rock4::{ROCK4_RESOURCES, rock4_resource_for_degree, rock4_tableau_for_degree};
use super::serk2::{SERK2_RESOURCES, serk2_resource_for_degree, serk2_tableau_for_degree};

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
