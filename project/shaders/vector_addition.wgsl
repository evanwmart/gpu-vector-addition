// vector_addition.wgsl

@group(0) @binding(0) var<storage, read> first_array: array<f32>;  // Input vector A
@group(0) @binding(1) var<storage, read> second_array: array<f32>; // Input vector B
@group(0) @binding(2) var<storage, read_write> result_array: array<f32>; // Output vector

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Get the index of the current element
    let index = global_id.x;
    
    // Prevent going past bounds
    if (index >= arrayLength(&result_array)) {
        return;
    }
    
    // Perform addition
    result_array[index] = first_array[index] + second_array[index];
}