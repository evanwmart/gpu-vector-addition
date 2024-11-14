// /tests/vector_addition_tests.rs
use tokio::runtime::Runtime;
use project::cpu::add_vectors_cpu;
use project::gpu::add_vectors_gpu;
use rand::Rng; // To create random vectors

/// Validates that the CPU and GPU results match within a given tolerance.
fn validate_results(cpu_result: &[f32], gpu_result: &[f32], epsilon: f32) -> Result<(), String> {
    if cpu_result.len() != gpu_result.len() {
        return Err("Results differ in length".to_string());
    }
    for (i, (c, g)) in cpu_result.iter().zip(gpu_result.iter()).enumerate() {
        if (c - g).abs() > epsilon {
            return Err(format!(
                "Results differ at index {}: CPU = {}, GPU = {}",
                i, c, g
            ));
        }
    }
    Ok(())
}

/// Generates a vector of specified size filled with random `f32` values.
fn generate_random_vector(size: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(-1000.0..1000.0)).collect()
}

#[test]
fn test_cpu_gpu_vector_addition_fixed() {
    let vec_size = 1000;
    let a = vec![1.0; vec_size];
    let b = vec![1.0; vec_size];

    let cpu_result = add_vectors_cpu(&a, &b);

    let runtime = Runtime::new().unwrap();
    let gpu_result = runtime.block_on(async { add_vectors_gpu(&a, &b).await });

    validate_results(&cpu_result, &gpu_result, 1e-6).expect("CPU and GPU results differ for fixed input");
}

#[test]
fn test_cpu_gpu_vector_addition_random() {
    let vec_size = 1000;
    let a = generate_random_vector(vec_size);
    let b = generate_random_vector(vec_size);

    let cpu_result = add_vectors_cpu(&a, &b);

    let runtime = Runtime::new().unwrap();
    let gpu_result = runtime.block_on(async { add_vectors_gpu(&a, &b).await });

    validate_results(&cpu_result, &gpu_result, 1e-6).expect("CPU and GPU results differ for random input");
}

#[test]
fn test_cpu_gpu_vector_addition_large_vector() {
    let vec_size = 1_000_000; // Large vector size to test performance and correctness at scale
    let a = vec![1.0; vec_size];
    let b = vec![1.0; vec_size];

    let cpu_result = add_vectors_cpu(&a, &b);

    let runtime = Runtime::new().unwrap();
    let gpu_result = runtime.block_on(async { add_vectors_gpu(&a, &b).await });

    validate_results(&cpu_result, &gpu_result, 1e-6).expect("CPU and GPU results differ for large vector");
}

#[test]
fn test_cpu_gpu_vector_addition_different_epsilons() {
    let vec_size = 1000;
    let a = vec![1.0; vec_size];
    let b = vec![1.0; vec_size];

    let cpu_result = add_vectors_cpu(&a, &b);

    let runtime = Runtime::new().unwrap();
    let gpu_result = runtime.block_on(async { add_vectors_gpu(&a, &b).await });

    // Test with different tolerance levels
    validate_results(&cpu_result, &gpu_result, 1e-3).expect("CPU and GPU results differ with epsilon 1e-3");
    validate_results(&cpu_result, &gpu_result, 1e-6).expect("CPU and GPU results differ with epsilon 1e-6");
    validate_results(&cpu_result, &gpu_result, 1e-9).expect("CPU and GPU results differ with epsilon 1e-9");
}
