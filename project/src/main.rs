mod cpu;
mod gpu;

use tokio::runtime::Runtime;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} [cpu|gpu]", args[0]);
        return;
    }

    let vec_size = 1000;
    let a = vec![1.0; vec_size];
    let b = vec![1.0; vec_size];

    match args[1].as_str() {
        "cpu" => {
            let result = cpu::add_vectors_cpu(&a, &b);
            println!("CPU result: {:?}", result);
        }
        "gpu" => {
            let runtime = Runtime::new().unwrap();
            let result = runtime.block_on(async { gpu::add_vectors_gpu(&a, &b).await });
            println!("GPU result: {:?}", result);
        }
        _ => {
            eprintln!("Unknown option: {}. Use 'cpu' or 'gpu'", args[1]);
        }
    }
}
