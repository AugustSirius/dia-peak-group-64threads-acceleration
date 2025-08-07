// Create a test file: src/bin/test_cache.rs
use std::fs::File;
use std::io::Read;
use std::path::Path;

fn main() {
    let cache_path = ".timstof_cache/CAD20220207yuel_TPHP_DIA_pool1_Slot2-54_1_4382.d.ms1_indexed.cache";
    
    println!("Testing cache file reading...");
    
    // Test 1: Basic file open
    match File::open(cache_path) {
        Ok(mut file) => {
            println!("✓ File opened successfully");
            
            // Test 2: Read first few bytes
            let mut buffer = [0u8; 100];
            match file.read_exact(&mut buffer) {
                Ok(_) => {
                    println!("✓ Can read first 100 bytes");
                    println!("First 20 bytes: {:?}", &buffer[..20]);
                },
                Err(e) => println!("✗ Read error: {}", e),
            }
        },
        Err(e) => println!("✗ Cannot open file: {}", e),
    }
    
    // Test 3: File metadata
    match std::fs::metadata(cache_path) {
        Ok(metadata) => {
            println!("✓ File size: {} bytes ({:.2} GB)", 
                     metadata.len(), 
                     metadata.len() as f64 / 1024.0 / 1024.0 / 1024.0);
            println!("✓ Read-only: {}", metadata.permissions().readonly());
        },
        Err(e) => println!("✗ Cannot get metadata: {}", e),
    }
}