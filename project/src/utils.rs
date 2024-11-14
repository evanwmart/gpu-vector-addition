/// Validates that the input vectors have the same length.
/// Returns an error if lengths differ.
pub fn validate_vectors(a: &[f32], b: &[f32]) -> Result<(), String> {
    if a.len() != b.len() {
        Err("Vectors must have the same length".to_string())
    } else {
        Ok(())
    }
}

/// Compares the CPU and GPU result vectors within a specified tolerance.
/// Returns an error if any elements differ by more than `epsilon`.
pub fn compare_results(cpu_result: &[f32], gpu_result: &[f32], epsilon: f32) -> Result<(), String> {
    validate_vectors(cpu_result, gpu_result)?;
    for (i, (cpu_val, gpu_val)) in cpu_result.iter().zip(gpu_result.iter()).enumerate() {
        if (cpu_val - gpu_val).abs() > epsilon {
            return Err(format!(
                "Difference found at index {}: CPU = {}, GPU = {}",
                i, cpu_val, gpu_val
            ));
        }
    }
    Ok(())
}
