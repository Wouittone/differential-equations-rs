use differential_equations::{
    DenseSegmentData, InterpolationQuality, PortableDenseSegment, Solution,
};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};
use std::alloc::System;
#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;
#[test]
fn portable_and_indexed_solution_queries_allocate_nothing() {
    let segment = PortableDenseSegment::from_data(DenseSegmentData {
        version: 1,
        start_time: 0.0,
        end_time: 1.0,
        bound_time: 1.0,
        dimension: 1,
        coefficients: vec![1.0, 2.0, 3.0],
        end_state: vec![6.0],
        bound_state: None,
        quality: InterpolationQuality::MethodSpecific,
    })
    .unwrap();
    let mut data = Solution::from_saved(vec![0.0, 1.0], vec![1.0, 6.0], &[1], Default::default())
        .unwrap()
        .export_data()
        .unwrap();
    data.segments.push(segment.clone());
    let solution = Solution::from_data(data).unwrap();
    let region = Region::new(GLOBAL);
    let mut output = [0.0];
    for i in 0..1000 {
        let t = i as f64 / 1000.0;
        segment.interpolate_into(t, &mut output).unwrap();
        solution.try_interpolate_into(t, &mut output).unwrap();
    }
    assert_eq!(region.change().allocations, 0);
    assert_eq!(region.change().reallocations, 0);
}
