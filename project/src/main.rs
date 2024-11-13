// main.rs

mod cpu_vector_add;

fn main() {

    // Basic vectors for now
    let a = vec![1.0; 1000];
    let b = vec![2.0; 1000];
    
    // CPU Vector Addition
    let cpu_result = cpu_vector_add::add_vectors_cpu(&a, &b);

    // Print input and output
    println!("CPU Result: {:?}", cpu_result);

}
