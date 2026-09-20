use differential_equations::{
    GaussJackson8, GaussJacksonConfig, InterpolationQuality, PortableDenseSegment,
};
use std::convert::Infallible;
#[test]
fn gj_portable_dense_survives_restart_and_round_trip() {
    for h in [0.1, -0.1] {
        let mut solver =
            GaussJackson8::new(0., &[1., 2.], &[0., 0.], h, GaussJacksonConfig::default()).unwrap();
        assert!(solver.export_dense_segment().is_err());
        let mut force = |_: f64, q: &[f64], _: &[f64], a: &mut [f64]| {
            a[0] = -q[0];
            a[1] = -q[1];
            Ok::<_, Infallible>(())
        };
        for _ in 0..12 {
            solver.try_step(&mut force).unwrap();
        }
        let segment = solver.export_dense_segment().unwrap();
        assert_eq!(segment.quality(), InterpolationQuality::MethodSpecific);
        assert_eq!(solver.dense_polynomial_degree(), 5);
        assert_eq!(segment.dimension(), 4);
        let (start, end) = segment.time_bounds();
        for x in [0., 0.1, 0.5, 0.9, 1.] {
            let t = start + x * (end - start);
            let mut q = [0.; 2];
            let mut v = [0.; 2];
            let mut state = [0.; 4];
            solver.interpolate_into(t, &mut q, &mut v).unwrap();
            segment.interpolate_into(t, &mut state).unwrap();
            for i in 0..2 {
                assert!((state[i] - q[i]).abs() < 1e-14);
                assert!((state[2 + i] - v[i]).abs() < 1e-14);
            }
        }
        let data = segment.into_data();
        solver.restart(0., &[8., 9.], &[1., 2.], h).unwrap();
        let segment = PortableDenseSegment::from_data(data).unwrap();
        let mut out = [0.; 4];
        segment.interpolate_into(end, &mut out).unwrap();
        assert!((out[0] - end.cos()).abs() < 1e-12);
        #[cfg(feature = "serde")]
        {
            let serialized = serde_json::to_string(&segment).unwrap();
            let restored: PortableDenseSegment = serde_json::from_str(&serialized).unwrap();
            assert_eq!(restored, segment);
        }
    }
}
