// src/bin/test_bincode.rs
use std::fs::File;
use std::io::{BufReader, Read};
use bincode;
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
struct TestStruct {
    values: Vec<f32>,
}

fn main() {
    // Test 1: Create and serialize a small test structure
    let test_data = TestStruct {
        values: vec![1.0, 2.0, 3.0],
    };
    
    let serialized = bincode::serialize(&test_data).unwrap();
    println!("Test serialization successful, {} bytes", serialized.len());
    
    // Test 2: Deserialize it back
    let deserialized: TestStruct = bincode::deserialize(&serialized).unwrap();
    println!("Test deserialization successful: {:?}", deserialized);
    
    // Test 3: Try to read first part of actual cache
    let cache_path = "/wangshuaiyao/dia-bert-timstof/00.TimsTOF_Rust/jiangheng/dia-peak-group-64threads-acceleration/.timstof_cache/CAD20220207yuel_TPHP_DIA_pool1_Slot2-54_1_4382.d.ms1_indexed.cache";
    
    match File::open(cache_path) {
        Ok(mut file) => {
            // Read first 1KB
            let mut buffer = vec![0u8; 1024];
            match file.read_exact(&mut buffer) {
                Ok(_) => {
                    println!("\n✓ Read first 1KB from cache");
                    
                    // Check bincode header pattern
                    println!("First 8 bytes (bincode header): {:?}", &buffer[..8]);
                    
                    // Try to read the size of the first vector
                    let size_bytes = &buffer[..8];
                    let size = u64::from_le_bytes([
                        size_bytes[0], size_bytes[1], size_bytes[2], size_bytes[3],
                        size_bytes[4], size_bytes[5], size_bytes[6], size_bytes[7]
                    ]);
                    println!("First vector size in cache: {}", size);
                    
                    if size > 1_000_000_000 {
                        println!("⚠️  Warning: Very large vector size detected!");
                    }
                },
                Err(e) => println!("✗ Cannot read from cache: {}", e),
            }
        },
        Err(e) => println!("✗ Cannot open cache: {}", e),
    }
}