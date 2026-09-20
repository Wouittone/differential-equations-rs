//! Conversion/allocation microbenchmark: cargo bench --bench layout.
use differential_equations::state_layout::{MatrixOrder, MatrixView};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    hint::black_box,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};
struct Count;
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Count {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOC: Count = Count;
fn main() {
    let copies = if std::env::args().any(|arg| arg == "--test") {
        2
    } else {
        1_000_000
    };
    let state = [1.0; 42];
    let mut output = [0.0; 42];
    let start_alloc = ALLOCATIONS.load(Ordering::Relaxed);
    let start = Instant::now();
    for _ in 0..copies {
        let view = MatrixView::new(black_box(&state), 6, 7, MatrixOrder::ColumnMajor).unwrap();
        view.flatten_into(black_box(&mut output), MatrixOrder::RowMajor)
            .unwrap();
        black_box(&output);
    }
    let nanos = start.elapsed().as_nanos();
    let allocations = ALLOCATIONS.load(Ordering::Relaxed) - start_alloc;
    assert_eq!(allocations, 0);
    println!("6x7 column-to-row: {nanos} ns / {copies} copies; {allocations} allocations");
}
