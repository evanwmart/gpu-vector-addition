// main.rs

use tokio;

mod cpu_vector_add;
mod gpu_vector_add;


#[tokio::main]
async fn main() {

    // Basic vectors for now
    let a = vec![1.0; 1000];
    let b = vec![2.0; 1000];
    
    // CPU Vector Addition
    let cpu_result = cpu_vector_add::add_vectors_cpu(&a, &b);

    // Print input and output
    println!("CPU Result: {:?}", cpu_result);

    // GPU Vector ADdition
    let gpu_future = gpu_vector_add::add_vectors_gpu(&a, &b);

    // Wait for result

    let gpu_result = gpu_future.await;

    // Print input and output 
    println!("GPU Result: {:?}", gpu_result);

}
