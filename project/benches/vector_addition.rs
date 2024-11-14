// /benches/vector_addition.rs

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use tokio::runtime::Runtime;
use project::cpu;
use project::gpu;

fn cpu_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vector Addition/CPU");

    for &size in &[1_000, 20_000, 300_000, 4_000_000, 50_000_000, 600_000_000] {
        group.bench_with_input(BenchmarkId::new("Size", size), &size, |bencher, &vec_size| {
            let a = vec![1.0; vec_size];
            let b = vec![1.0; vec_size];
            bencher.iter(|| {
                cpu::add_vectors_cpu(&a, &b)
            });
        });
    }
    group.finish();
}

async fn gpu_async_vector_addition(a: &[f32], b: &[f32]) -> Vec<f32> {
    gpu::add_vectors_gpu(a, b).await
}

fn bench_gpu_vector_addition(runtime: &Runtime, a: &[f32], b: &[f32]) -> Vec<f32> {
    runtime.block_on(async {
        gpu_async_vector_addition(a, b).await
    })
}

fn gpu_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Vector Addition/GPU");

    // Set longer measurement time for GPU to get reliable samples
    group.sample_size(50).measurement_time(std::time::Duration::from_secs(10));

    let runtime = Runtime::new().unwrap();

    for &size in &[1_000, 20_000, 300_000, 4_000_000, 50_000_000, 600_000_000] {
        group.bench_with_input(BenchmarkId::new("Size", size), &size, |bencher, &vec_size| {
            let a = vec![1.0; vec_size];
            let b = vec![1.0; vec_size];
            bencher.iter(|| {
                bench_gpu_vector_addition(&runtime, &a, &b)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, cpu_benchmark, gpu_benchmark);
criterion_main!(benches);
