// src/bin/test_streaming.rs
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

fn main() {
    println!("Testing streaming read of cache file...\n");
    
    let cache_path = "/wangshuaiyao/dia-bert-timstof/00.TimsTOF_Rust/jiangheng/dia-peak-group-64threads-acceleration/.timstof_cache/CAD20220207yuel_TPHP_DIA_pool1_Slot2-54_1_4382.d.ms1_indexed.cache";
    
    match File::open(cache_path) {
        Ok(mut file) => {
            // Get file size
            let file_size = file.metadata().unwrap().len();
            println!("File size: {} bytes ({:.2} GB)", file_size, file_size as f64 / 1024.0 / 1024.0 / 1024.0);
            
            // Read in 100MB chunks
            let chunk_size = 100 * 1024 * 1024; // 100MB
            let mut buffer = vec![0u8; chunk_size];
            let mut total_read = 0u64;
            let mut chunk_count = 0;
            
            println!("\nReading file in {}MB chunks...", chunk_size / 1024 / 1024);
            
            loop {
                match file.read(&mut buffer) {
                    Ok(0) => break, // EOF
                    Ok(n) => {
                        total_read += n as u64;
                        chunk_count += 1;
                        
                        if chunk_count % 10 == 0 {
                            println!("Read {} chunks, total: {:.2} GB", 
                                     chunk_count, 
                                     total_read as f64 / 1024.0 / 1024.0 / 1024.0);
                        }
                    },
                    Err(e) => {
                        println!("Error reading at chunk {}: {}", chunk_count, e);
                        println!("Total read before error: {} bytes", total_read);
                        break;
                    }
                }
            }
            
            println!("\nSuccessfully read entire file");
            println!("Total chunks: {}", chunk_count);
            println!("Total bytes read: {} ({:.2} GB)", total_read, total_read as f64 / 1024.0 / 1024.0 / 1024.0);
        },
        Err(e) => println!("Cannot open file: {}", e),
    }
}