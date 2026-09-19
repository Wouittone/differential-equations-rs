use super::{Family, JVODE, JvodeMethod, NordsieckKernel};

#[test]
fn jvode_cache_switches_order_after_waiting() {
    let mut kernel = NordsieckKernel::<fn(), ()>::jvode(1, JVODE::adams());
    kernel.started = true;
    kernel.order = 1;
    kernel.n_wait = 0;
    kernel.previous_d = 1.0;
    kernel.current_d = 1.0;
    kernel.dts.fill(0.1);
    kernel.delta[0] = 1.0e-8;
    kernel.z[kernel.maximum_order][0] = 0.0;
    let options = crate::SolveOptions::default();
    kernel.choose_factor(1.0e-8, &[1.0], &[1.0], &options);
    assert!(kernel.next_order >= kernel.order);
    assert_eq!(kernel.family, Family::Jvode(JvodeMethod::Adams));
}
