// src/bin/test_memory.rs
use std::fs::File;
use std::io::{BufReader, Read};
use bincode;
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
struct PartialIndexedData {
    rt_values_min: Vec<f32>,
    // Skip other fields for now
}

fn main() {
    println!("Testing memory allocation and partial deserialization...\n");
    
    // Get system memory info
    println!("System memory info:");
    if let Ok(output) = std::process::Command::new("free")
        .arg("-h")
        .output() 
    {
        println!("{}", String::from_utf8_lossy(&output.stdout));
    }
    
    let cache_path = "/wangshuaiyao/dia-bert-timstof/00.TimsTOF_Rust/jiangheng/dia-peak-group-64threads-acceleration/.timstof_cache/CAD20220207yuel_TPHP_DIA_pool1_Slot2-54_1_4382.d.ms1_indexed.cache";
    
    // Try to deserialize just the first vector
    match File::open(cache_path) {
        Ok(file) => {
            println!("Attempting to read cache file in chunks...");
            
            // First, try to read the file size from bincode header
            let mut reader = BufReader::with_capacity(1024 * 1024, file);
            
            // Read first 8 bytes (vector length)
            let mut len_bytes = [0u8; 8];
            reader.read_exact(&mut len_bytes).unwrap();
            let vec_len = u64::from_le_bytes(len_bytes);
            println!("First vector length: {} elements", vec_len);
            println!("Memory needed for first vector: {:.2} GB", 
                     (vec_len as f64 * 4.0) / 1024.0 / 1024.0 / 1024.0);
            
            // Try to allocate memory for just the first vector
            println!("\nTrying to allocate memory for {} f32 elements...", vec_len);
            match Vec::<f32>::try_reserve(vec_len as usize) {
                Ok(_) => println!("✓ Memory allocation would succeed"),
                Err(e) => println!("✗ Memory allocation would fail: {}", e),
            }
        },
        Err(e) => println!("Cannot open file: {}", e),
    }
}