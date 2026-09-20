//! Exact host data, with source revision/hash/license retained beside fixtures.
//! No host dependency, JSON parser or global method cache participates.
use differential_equations_tableau_core::{LazyDenseStageCoefficients, RungeKuttaCoefficients};
#[path = "fixtures/brahe_rkf78.rs"]
mod brahe;
#[path = "fixtures/numeris_rkv98.rs"]
mod numeris;
#[test]
fn numeris_core_embedded_and_lazy_dense_coefficients_are_exact() {
    let rows: Vec<&[f64]> = numeris::A[..16].iter().map(|row| &row[..16]).collect();
    let dense: Vec<&[f64]> = numeris::BI.iter().map(|row| row.as_slice()).collect();
    let sparse: Vec<Vec<(usize, f64)>> = numeris::A[16..]
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .filter(|(_, v)| **v != 0.)
                .map(|(i, &v)| (i, v))
                .collect()
        })
        .collect();
    let lazy: Vec<_> = sparse
        .iter()
        .enumerate()
        .map(|(i, row)| LazyDenseStageCoefficients {
            node: numeris::C[16 + i],
            coefficients: row,
        })
        .collect();
    let mut input = RungeKuttaCoefficients::explicit(
        "Exact host values, arbitrary label",
        9,
        &rows,
        &numeris::B[..16],
        &numeris::C[..16],
    );
    input.embedded_order = Some(8);
    input.b_hat = Some(&numeris::BHAT[..16]);
    input.dense = Some(&dense);
    input.lazy_dense_stages = &lazy;
    let method = input.build().unwrap();
    for stage in 0..16 {
        assert_eq!(method.a()[stage], numeris::A[stage][..16]);
        assert_eq!(method.b()[stage].to_bits(), numeris::B[stage].to_bits());
        assert_eq!(method.c()[stage].to_bits(), numeris::C[stage].to_bits());
        assert_eq!(
            method.error().unwrap()[stage].to_bits(),
            (numeris::B[stage] - numeris::BHAT[stage]).to_bits()
        );
    }
    for stage in 0..5 {
        assert_eq!(
            method.lazy_dense_stages()[stage].node().to_bits(),
            numeris::C[16 + stage].to_bits()
        );
        assert_eq!(
            method.lazy_dense_stages()[stage].coefficients(),
            sparse[stage]
        );
    }
    for stage in 0..21 {
        for degree in 0..8 {
            assert_eq!(
                method.dense().unwrap()[stage][degree].to_bits(),
                numeris::BI[stage][degree].to_bits()
            );
        }
    }
    // Independent polynomial evaluation checks the imported dense convention,
    // including the otherwise-unused extra interpolation stages.
    for theta in [0.13_f64, 0.51, 0.89] {
        let host: f64 = (0..21)
            .map(|i| {
                (i + 1) as f64
                    * (0..8)
                        .map(|k| numeris::BI[i][k] * theta.powi(k as i32 + 1))
                        .sum::<f64>()
            })
            .sum();
        let imported: f64 = method
            .dense()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, row)| {
                (i + 1) as f64 * theta * row.iter().rev().fold(0., |a, c| a * theta + c)
            })
            .sum();
        assert!((host - imported).abs() < 1e-11);
    }
}
#[test]
fn brahe_rkf78_arrays_keep_row_major_shape_and_embedded_formula() {
    let rows: Vec<_> = brahe::A_FLAT.chunks_exact(13).collect();
    let mut input = RungeKuttaCoefficients::explicit("Brahe exact", 8, &rows, &brahe::B, &brahe::C);
    input.embedded_order = Some(7);
    input.b_hat = Some(&brahe::BHAT);
    let method = input.build().unwrap();
    for i in 0..13 {
        for j in 0..13 {
            assert_eq!(
                method.a()[i][j].to_bits(),
                brahe::A_FLAT[i * 13 + j].to_bits()
            );
        }
        assert_eq!(method.b()[i].to_bits(), brahe::B[i].to_bits());
        assert_eq!(method.c()[i].to_bits(), brahe::C[i].to_bits());
        assert_eq!(
            method.error().unwrap()[i].to_bits(),
            (brahe::B[i] - brahe::BHAT[i]).to_bits()
        );
    }
}
