//! Reproducible lookup-only scaling experiment. Run after competing builds stop:
//! `cargo run --release --example dense_lookup_scaling -- samples.csv`
//! Defaults: 3 rounds x 9 samples, 100000 prevalidated queries per sample.
//! Optional environment variables DENSE_ROUNDS/DENSE_SAMPLES/DENSE_QUERIES
//! change repetition; output records each value. Setup/validation is untimed.
use differential_equations::{
    DenseSegmentData, InterpolationQuality, PortableDenseSegment, Solution, SolverStats,
};
use std::{
    hint::black_box,
    io::{self, Write},
    time::Instant,
};

fn trajectory(length: usize, backward: bool, repeats: bool, dense: bool) -> Solution {
    let mut times = Vec::new();
    let mut values = Vec::new();
    let value = |t: f64| if dense { t * t + 1.0 } else { 2.0 * t + 1.0 };
    for i in 0..length {
        let t = if backward {
            (length - 1 - i) as f64
        } else {
            i as f64
        };
        times.push(t);
        values.push(value(t));
        if repeats && i % 8 == 0 {
            times.push(t);
            values.push(value(t));
        }
    }
    let mut data = Solution::from_saved(times, values, &[1], SolverStats::default())
        .unwrap()
        .export_data()
        .unwrap();
    if dense {
        for i in 0..length.saturating_sub(1) {
            let start = if backward {
                (length - 1 - i) as f64
            } else {
                i as f64
            };
            let step = if backward { -1.0 } else { 1.0 };
            let end = start + step;
            data.segments.push(
                PortableDenseSegment::from_data(DenseSegmentData {
                    version: 1,
                    start_time: start,
                    end_time: end,
                    bound_time: end,
                    dimension: 1,
                    coefficients: vec![value(start), 2.0 * start * step, step * step],
                    end_state: vec![value(end)],
                    bound_state: None,
                    quality: InterpolationQuality::MethodSpecific,
                })
                .unwrap(),
            );
        }
    }
    Solution::from_data(data).unwrap()
}
fn count(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|v| v.parse().expect("positive integer"))
        .unwrap_or(default)
        .max(1)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rounds = count("DENSE_ROUNDS", 3);
    let samples = count("DENSE_SAMPLES", 9);
    let query_count = count("DENSE_QUERIES", 100000);
    let mut writer: Box<dyn Write> = if let Some(path) = std::env::args().nth(1) {
        Box::new(std::fs::File::create(path)?)
    } else {
        Box::new(io::stdout())
    };
    writeln!(
        writer,
        "round,sample,mode,direction,repeated_times,base_length,saved_states,queries,elapsed_ns,ns_per_query,checksum"
    )?;
    for round in 0..rounds {
        for length in [1, 16, 256, 4096] {
            for backward in [false, true] {
                for repeats in [false, true] {
                    for dense in [false, true] {
                        let solution = trajectory(length, backward, repeats, dense);
                        let queries: Vec<f64> = (0..query_count)
                            .map(|i| {
                                if length == 1 {
                                    0.0
                                } else {
                                    ((i.wrapping_mul(104729)) % (length - 1)) as f64 + 0.375
                                }
                            })
                            .collect();
                        let mut output = [0.0];
                        for &time in &queries {
                            solution.try_interpolate_into(time, &mut output)?;
                            let expected = if dense {
                                time * time + 1.0
                            } else {
                                2.0 * time + 1.0
                            };
                            assert!(
                                (output[0] - expected).abs() <= 1e-12 * expected.abs().max(1.0)
                            );
                        }
                        for sample in 0..samples {
                            let started = Instant::now();
                            let mut checksum = 0.0;
                            for &time in &queries {
                                black_box(&solution).try_interpolate_into(
                                    black_box(time),
                                    black_box(&mut output),
                                )?;
                                checksum += black_box(output[0]);
                            }
                            let elapsed = started.elapsed().as_nanos();
                            writeln!(
                                writer,
                                "{round},{sample},{},{},{repeats},{length},{},{query_count},{elapsed},{:.6},{:.17e}",
                                if dense { "polynomial" } else { "saved_linear" },
                                if backward { "backward" } else { "forward" },
                                solution.times().len(),
                                elapsed as f64 / query_count as f64,
                                black_box(checksum)
                            )?;
                        }
                    }
                }
            }
        }
    }
    writer.flush()?;
    Ok(())
}
